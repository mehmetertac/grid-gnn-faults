"""Graph topology, PyG objects, splits, and sanity plots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandapower as pp
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from grid.simulate import FaultSample, load_case39

FEATURE_NAMES = [
    "vm_pre_pu",
    "vm_fault_pu",
    "delta_v_pu",
    "is_slack",
    "is_pv",
    "is_pq",
    "p_mw",
    "q_mvar",
]


@dataclass(frozen=True)
class GraphTopology:
    edge_index: np.ndarray  # (2, E)
    edge_weight: np.ndarray  # (E,)
    bus_ids: np.ndarray


def _branch_z_pu(net) -> list[tuple[int, int, float]]:
    sn_mva = float(net.sn_mva)
    edges: list[tuple[int, int, float]] = []
    bus_to_pos = {int(b): i for i, b in enumerate(net.bus.index)}

    for _, row in net.line.iterrows():
        if not row.in_service:
            continue
        fb = bus_to_pos[int(row.from_bus)]
        tb = bus_to_pos[int(row.to_bus)]
        vn = float(net.bus.at[row.from_bus, "vn_kv"])
        z_base = vn**2 / sn_mva
        r = float(row.r_ohm_per_km * row.length_km)
        x = float(row.x_ohm_per_km * row.length_km)
        z_mag = max(np.hypot(r, x) / z_base, 1e-6)
        w = 1.0 / z_mag
        edges.append((fb, tb, w))
        edges.append((tb, fb, w))

    for _, row in net.trafo.iterrows():
        if not row.in_service:
            continue
        fb = bus_to_pos[int(row.hv_bus)]
        tb = bus_to_pos[int(row.lv_bus)]
        vk = float(row.vk_percent)
        vkr = float(row.vkr_percent)
        xt = float(np.sqrt(max(vk**2 - vkr**2, 1e-6)))
        z_pu = np.hypot(vkr, xt) / 100.0 * float(row.sn_mva) / sn_mva
        z_pu = max(z_pu, 1e-6)
        w = 1.0 / z_pu
        edges.append((fb, tb, w))
        edges.append((tb, fb, w))

    return edges


def build_case39_topology(net=None) -> GraphTopology:
    template = load_case39() if net is None else copy.deepcopy(net)
    pp.runpp(template, calculate_voltage_angles=True)
    bus_ids = template.bus.index.to_numpy()
    edge_triples = _branch_z_pu(template)
    if not edge_triples:
        raise ValueError("No in-service branches found for topology.")

    src = np.array([s for s, _, _ in edge_triples], dtype=np.int64)
    dst = np.array([d for _, d, _ in edge_triples], dtype=np.int64)
    w = np.array([wt for _, _, wt in edge_triples], dtype=np.float32)
    edge_index = np.stack([src, dst], axis=0)
    return GraphTopology(edge_index=edge_index, edge_weight=w, bus_ids=bus_ids)


def stratified_split(
    labels: np.ndarray,
    seed: int = 0,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    idx = np.arange(n)
    train_idx, hold_idx = train_test_split(
        idx,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels,
    )
    relative_val = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=1.0 - relative_val,
        random_state=seed,
        stratify=labels[hold_idx],
    )
    return train_idx, val_idx, test_idx


def zscore_fit(x_train: np.ndarray, std_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < std_floor, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def samples_to_arrays(
    samples: list[FaultSample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = np.stack([s.x_raw for s in samples], axis=0)
    y = np.array([s.label for s in samples], dtype=np.int64)
    scenario = np.array([s.scenario_id for s in samples], dtype=np.int64)
    return x_raw, y, scenario


def make_pyg_dataset(
    x_norm: np.ndarray,
    labels: np.ndarray,
    topology: GraphTopology,
) -> list[Data]:
    edge_index = torch.from_numpy(topology.edge_index.copy())
    edge_attr = torch.from_numpy(topology.edge_weight.copy()).unsqueeze(-1)
    graphs: list[Data] = []
    for i in range(len(labels)):
        graphs.append(
            Data(
                x=torch.from_numpy(x_norm[i]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([labels[i]], dtype=torch.long),
            )
        )
    return graphs


def save_processed_dataset(
    out_dir: Path,
    x_raw: np.ndarray,
    x_norm: np.ndarray,
    labels: np.ndarray,
    topology: GraphTopology,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    voltage_methods: np.ndarray | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "fault_dataset.npz"
    payload = dict(
        x_raw=x_raw,
        x_norm=x_norm,
        y=labels,
        edge_index=topology.edge_index,
        edge_weight=topology.edge_weight,
        bus_ids=topology.bus_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_mean=mean,
        feature_std=std,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )
    if voltage_methods is not None:
        payload["voltage_method"] = voltage_methods
    np.savez_compressed(npz_path, **payload)

    graphs = make_pyg_dataset(x_norm, labels, topology)
    pt_path = out_dir / "fault_graphs.pt"
    torch.save(graphs, pt_path)
    return npz_path, pt_path


def plot_dv_sample(
    sample: FaultSample,
    topology: GraphTopology,
    out_path: Path,
) -> None:
    dv = sample.x_raw[:, 2]
    g = nx.Graph()
    for i in range(len(topology.bus_ids)):
        g.add_node(i, dv=float(dv[i]))
    ei = topology.edge_index
    for k in range(ei.shape[1]):
        g.add_edge(int(ei[0, k]), int(ei[1, k]))

    pos = nx.kamada_kawai_layout(g)
    colors = [g.nodes[n]["dv"] for n in g.nodes]
    fig, ax = plt.subplots(figsize=(10, 8))
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_color=colors,
        cmap="viridis_r",
        node_size=320,
        ax=ax,
    )
    nx.draw_networkx_edges(g, pos, alpha=0.35, ax=ax)
    fault = sample.label
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[fault],
        node_color="red",
        node_size=420,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )
    plt.colorbar(nodes, ax=ax, label="ΔV (pu)")
    ax.set_title(f"IEEE 39-bus ΔV (fault bus index {fault}, id {topology.bus_ids[fault]})")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
