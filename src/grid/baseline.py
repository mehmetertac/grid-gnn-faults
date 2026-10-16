"""Non-graph baselines for faulted-bus classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FaultMLP(nn.Module):
    def __init__(self, n_buses: int = 39, n_features: int = 8, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_buses * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, n_buses),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.net(x_flat)


@dataclass
class BaselineMetrics:
    mlp_test_accuracy: float
    mlp_test_top3: float
    dv_heuristic_test_accuracy: float
    dv_heuristic_test_top3: float
    n_train: int
    n_val: int
    n_test: int
    epochs_run: int


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == y).float().mean().item())


def _topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 3) -> float:
    topk = logits.topk(k, dim=1).indices
    hits = (topk == y.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


def dv_heuristic_predictions(x_raw: np.ndarray) -> np.ndarray:
    """Predict fault bus as minimum during-fault voltage magnitude."""
    return np.argmin(x_raw[:, :, 1], axis=1)


def _topk_from_scores(scores: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    topk = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    hits = [y_true[i] in set(topk[i]) for i in range(len(y_true))]
    return float(np.mean(hits))


def train_mlp_baseline(
    x_norm: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    x_raw: np.ndarray,
    seed: int = 0,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> BaselineMetrics:
    torch.manual_seed(seed)
    n_buses, n_feat = x_norm.shape[1], x_norm.shape[2]

    x_train = torch.from_numpy(x_norm[train_idx].reshape(len(train_idx), -1))
    y_train = torch.from_numpy(y[train_idx])
    x_val = torch.from_numpy(x_norm[val_idx].reshape(len(val_idx), -1))
    y_val = torch.from_numpy(y[val_idx])
    x_test = torch.from_numpy(x_norm[test_idx].reshape(len(test_idx), -1))
    y_test = torch.from_numpy(y[test_idx])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    model = FaultMLP(n_buses=n_buses, n_features=n_feat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    stale = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = _accuracy(model(x_val), y_val)
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

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test)
        mlp_acc = _accuracy(test_logits, y_test)
        mlp_top3 = _topk_accuracy(test_logits, y_test, k=3)

    dv_pred = dv_heuristic_predictions(x_raw[test_idx])
    dv_acc = float((y[test_idx] == dv_pred).mean())
    dv_scores = -x_raw[test_idx][:, :, 1]
    dv_top3 = _topk_from_scores(dv_scores, y[test_idx], k=3)

    return BaselineMetrics(
        mlp_test_accuracy=mlp_acc,
        mlp_test_top3=mlp_top3,
        dv_heuristic_test_accuracy=dv_acc,
        dv_heuristic_test_top3=dv_top3,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        epochs_run=epochs_run,
    )


def save_baseline_metrics(metrics: BaselineMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
