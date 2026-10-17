"""GCN fault-localization model, metrics, robustness, and layout figures."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from grid.baseline import FaultMLP, _accuracy, _topk_accuracy
from grid.graph import GraphTopology


class FaultGCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden: int = 64,
        num_classes: int = 39,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Linear(hidden, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        return self.head(x)


def undirected_edge_pairs(edge_index: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for u, v in edge_index.T:
        u, v = int(u), int(v)
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def drop_random_undirected_edge(data: Data, rng: np.random.Generator) -> Data:
    """Remove one random undirected branch (both directed edges) from a graph."""
    ei = data.edge_index.numpy()
    pairs = undirected_edge_pairs(ei)
    if not pairs:
        return data.clone()
    drop_u, drop_v = pairs[int(rng.integers(0, len(pairs)))]
    drop_set = {(drop_u, drop_v), (drop_v, drop_u)}
    keep = [
        k
        for k in range(ei.shape[1])
        if (int(ei[0, k]), int(ei[1, k])) not in drop_set
    ]
    out = data.clone()
    out.edge_index = data.edge_index[:, keep]
    if getattr(data, "edge_attr", None) is not None:
        out.edge_attr = data.edge_attr[keep]
    return out


def build_hop_distance_matrix(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for u, v in undirected_edge_pairs(edge_index):
        g.add_edge(u, v)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    lengths = dict(nx.all_pairs_shortest_path_length(g))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            dist[i, j] = int(lengths[i].get(j, n_nodes))
    return dist


def mean_hop_error(
    y_true: np.ndarray, y_pred: np.ndarray, hop_matrix: np.ndarray
) -> float:
    hops = hop_matrix[y_true, y_pred]
    return float(hops.mean())


@dataclass
class ClassifierMetrics:
    accuracy: float
    top3: float
    mean_hop_error: float
    n_samples: int


@dataclass
class GnnRunResult:
    gnn_test: ClassifierMetrics
    gnn_test_one_edge_out: ClassifierMetrics
    mlp_test: ClassifierMetrics
    mlp_test_one_edge_out: ClassifierMetrics
    epochs_run: int
    n_train: int
    n_val: int
    n_test: int


def _edge_weight_from_data(data: Data) -> torch.Tensor | None:
    if getattr(data, "edge_attr", None) is None:
        return None
    w = data.edge_attr
    if w.dim() == 2 and w.size(-1) == 1:
        return w.squeeze(-1)
    return w


def _predict_gnn(
    model: FaultGCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    model.eval()
    preds: list[int] = []
    truths: list[int] = []
    logits_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ew = _edge_weight_from_data(batch)
            logits = model(batch.x, batch.edge_index, edge_weight=ew, batch=batch.batch)
            logits_all.append(logits.cpu())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            truths.extend(batch.y.view(-1).cpu().numpy().tolist())
    stacked = torch.cat(logits_all, dim=0)
    return (
        np.array(truths, dtype=np.int64),
        np.array(preds, dtype=np.int64),
        stacked,
    )


def _metrics_from_logits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: torch.Tensor,
    hop_matrix: np.ndarray,
) -> ClassifierMetrics:
    y_t = torch.from_numpy(y_true)
    acc = float((y_true == y_pred).mean())
    top3 = _topk_accuracy(logits, y_t, k=3)
    hops = mean_hop_error(y_true, y_pred, hop_matrix)
    return ClassifierMetrics(
        accuracy=acc, top3=top3, mean_hop_error=hops, n_samples=len(y_true)
    )


def evaluate_full_run(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    graphs: list[Data],
    x_norm: np.ndarray,
    y: np.ndarray,
    x_raw: np.ndarray,
    topology: GraphTopology,
    *,
    seed: int = 0,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> tuple[FaultGCN, GnnRunResult]:
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    hop_matrix = build_hop_distance_matrix(topology.edge_index, n_nodes=39)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    model = FaultGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    stale = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            ew = _edge_weight_from_data(batch)
            logits = model(batch.x, batch.edge_index, edge_weight=ew, batch=batch.batch)
            loss = loss_fn(logits, batch.y.view(-1))
            loss.backward()
            opt.step()

        model.eval()
        val_logits_list: list[torch.Tensor] = []
        val_y_list: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ew = _edge_weight_from_data(batch)
                logits = model(batch.x, batch.edge_index, edge_weight=ew, batch=batch.batch)
                val_logits_list.append(logits.cpu())
                val_y_list.append(batch.y.view(-1).cpu())
            val_logits = torch.cat(val_logits_list, dim=0)
            val_y = torch.cat(val_y_list, dim=0)
            val_acc = _accuracy(val_logits, val_y)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test = y[test_idx]
    _, y_pred_gnn, gnn_logits = _predict_gnn(model, test_loader, device)
    gnn_test = _metrics_from_logits(y_test, y_pred_gnn, gnn_logits, hop_matrix)

    rng = np.random.default_rng(seed)
    dropped_test = [drop_random_undirected_edge(g, rng) for g in test_graphs]
    drop_loader = DataLoader(dropped_test, batch_size=batch_size, shuffle=False)
    _, y_pred_drop, drop_logits = _predict_gnn(model, drop_loader, device)
    gnn_drop = _metrics_from_logits(y_test, y_pred_drop, drop_logits, hop_matrix)

    n_buses, n_feat = x_norm.shape[1], x_norm.shape[2]
    x_train = torch.from_numpy(x_norm[train_idx].reshape(len(train_idx), -1))
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_norm[val_idx].reshape(len(val_idx), -1))
    y_val = torch.from_numpy(y[val_idx])
    x_test = torch.from_numpy(x_norm[test_idx].reshape(len(test_idx), -1))

    mlp = FaultMLP(n_buses=n_buses, n_features=n_feat)
    train_loader_mlp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=lr)
    best_val_mlp = -1.0
    best_mlp_state = None
    stale_mlp = 0
    for epoch in range(max_epochs):
        mlp.train()
        for xb, yb in train_loader_mlp:
            opt_mlp.zero_grad()
            loss = loss_fn(mlp(xb), yb)
            loss.backward()
            opt_mlp.step()
        mlp.eval()
        with torch.no_grad():
            val_acc_mlp = _accuracy(mlp(x_val), y_val)
        if val_acc_mlp > best_val_mlp:
            best_val_mlp = val_acc_mlp
            best_mlp_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            stale_mlp = 0
        else:
            stale_mlp += 1
            if stale_mlp >= patience:
                break
    if best_mlp_state is not None:
        mlp.load_state_dict(best_mlp_state)
    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(x_test)
        mlp_pred = mlp_logits.argmax(dim=1).numpy()
    mlp_test = _metrics_from_logits(y_test, mlp_pred, mlp_logits, hop_matrix)
    mlp_drop = mlp_test  # edge outage does not change flattened MLP inputs

    result = GnnRunResult(
        gnn_test=gnn_test,
        gnn_test_one_edge_out=gnn_drop,
        mlp_test=mlp_test,
        mlp_test_one_edge_out=mlp_drop,
        epochs_run=epochs_run,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
    )
    return model, result


def plot_fault_error_layout(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    topology: GraphTopology,
    out_path: Path,
    *,
    top_mistake_pairs: int = 8,
) -> None:
    """Kamada–Kawai layout: node colour = per-bus recall; arcs = common mistakes."""
    n_buses = len(topology.bus_ids)
    confusion = Counter(
        (int(t), int(p)) for t, p in zip(y_true, y_pred, strict=True) if t != p
    )
    recall = np.zeros(n_buses, dtype=np.float64)
    support = np.bincount(y_true, minlength=n_buses)
    hits = np.bincount(y_true[y_true == y_pred], minlength=n_buses)
    for b in range(n_buses):
        if support[b] > 0:
            recall[b] = hits[b] / support[b]

    g = nx.Graph()
    for u, v in undirected_edge_pairs(topology.edge_index):
        g.add_edge(u, v)
    pos = nx.kamada_kawai_layout(g)

    fig, ax = plt.subplots(figsize=(11, 8))
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_color=recall,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        node_size=340,
        ax=ax,
    )
    nx.draw_networkx_edges(g, pos, alpha=0.25, ax=ax)
    plt.colorbar(nodes, ax=ax, label="Test recall (per true fault bus)")

    top_pairs = confusion.most_common(top_mistake_pairs)
    for (true_b, pred_b), count in top_pairs:
        if true_b not in pos or pred_b not in pos:
            continue
        x0, y0 = pos[true_b]
        x1, y1 = pos[pred_b]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="crimson",
                alpha=min(0.25 + 0.05 * count, 0.85),
                lw=1.2 + 0.15 * min(count, 5),
                connectionstyle="arc3,rad=0.15",
            ),
        )

    ax.set_title(
        "IEEE 39-bus fault localization — per-bus recall and top misroutes "
        f"({len(y_true)} test graphs)"
    )
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gnn_metrics(result: GnnRunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def load_processed_graphs(data_dir: Path) -> tuple[list[Data], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, GraphTopology]:
    npz = np.load(data_dir / "fault_dataset.npz", allow_pickle=True)
    x_norm = npz["x_norm"]
    x_raw = npz["x_raw"]
    y = npz["y"]
    train_idx = npz["train_idx"]
    val_idx = npz["val_idx"]
    test_idx = npz["test_idx"]
    topo = GraphTopology(
        edge_index=npz["edge_index"],
        edge_weight=npz["edge_weight"],
        bus_ids=npz["bus_ids"],
    )
    pt_path = data_dir / "fault_graphs.pt"
    try:
        graphs = torch.load(pt_path, weights_only=False)
    except TypeError:
        graphs = torch.load(pt_path)
    return graphs, x_norm, x_raw, y, train_idx, val_idx, test_idx, topo
