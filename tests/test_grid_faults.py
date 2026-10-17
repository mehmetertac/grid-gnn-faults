"""Tests for IEEE 39-bus fault dataset and graph construction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from grid.baseline import FaultMLP, dv_heuristic_predictions, train_mlp_baseline
from grid.graph import (
    FEATURE_NAMES,
    build_case39_topology,
    make_pyg_dataset,
    stratified_split,
    zscore_fit,
    zscore_transform,
)
from grid.model import (
    FaultGCN,
    build_hop_distance_matrix,
    drop_random_undirected_edge,
    undirected_edge_pairs,
)
from grid.simulate import generate_fault_dataset, load_case39
import torch
from torch_geometric.data import Data


def test_feature_names_length():
    assert len(FEATURE_NAMES) == 8


def test_topology_has_39_nodes_and_bidirectional_edges():
    topo = build_case39_topology()
    assert len(topo.bus_ids) == 39
    assert topo.edge_index.shape[0] == 2
    assert (topo.edge_weight > 0).all()
    n_line = len(load_case39().line)
    n_trafo = len(load_case39().trafo)
    assert topo.edge_index.shape[1] == 2 * (n_line + n_trafo)


def test_ybus_fault_sags_at_labeled_bus():
    samples, _, _ = generate_fault_dataset(n_scenarios=1, seed=0, try_branch_sc=False)
    assert len(samples) == 39
    for sample in samples[:5]:
        vm_fault = sample.x_raw[:, 1]
        assert int(np.argmin(vm_fault)) == sample.label
        assert vm_fault[sample.label] < 0.05


def test_stratified_split_per_bus():
    labels = np.repeat(np.arange(39), 50)
    train, val, test = stratified_split(labels, seed=0)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    for split in (train, val, test):
        counts = np.bincount(labels[split], minlength=39)
        assert counts.min() == counts.max()


def test_normalization_uses_train_stats_only():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(39), 10)
    x = rng.normal(size=(len(y), 39, 8)).astype(np.float32)
    train, _, _ = stratified_split(y, seed=0)
    mean, std = zscore_fit(x[train])
    x_norm = zscore_transform(x, mean, std)
    train_flat = x_norm[train].reshape(-1, 8)
    assert np.allclose(train_flat.mean(axis=0), 0.0, atol=0.15)
    assert np.allclose(train_flat.std(axis=0), 1.0, atol=0.15)


def test_pyg_data_shapes():
    topo = build_case39_topology()
    x = np.zeros((2, 39, 8), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    graphs = make_pyg_dataset(x, y, topo)
    assert len(graphs) == 2
    assert graphs[0].x.shape == (39, 8)
    assert graphs[0].y.shape == (1,)
    assert graphs[0].edge_index.shape[0] == 2
    rev = graphs[0].edge_index.flip(0)
    pairs = set(map(tuple, graphs[0].edge_index.T.tolist()))
    for u, v in pairs:
        assert (v, u) in pairs


def test_mlp_smoke_and_heuristic():
    samples, _, _ = generate_fault_dataset(n_scenarios=2, seed=1, try_branch_sc=False)
    x_raw = np.stack([s.x_raw for s in samples])
    y = np.array([s.label for s in samples])
    n = len(y)
    train = np.arange(0, n, 3)
    val = np.arange(1, n, 3)
    test = np.arange(2, n, 3)
    mean, std = zscore_fit(x_raw[train])
    x_norm = zscore_transform(x_raw, mean, std)

    model = FaultMLP()
    opt = __import__("torch").optim.Adam(model.parameters(), lr=1e-2)
    xb = __import__("torch").from_numpy(x_norm[train].reshape(len(train), -1))
    yb = __import__("torch").from_numpy(y[train])
    for _ in range(3):
        opt.zero_grad()
        loss = __import__("torch").nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

    metrics = train_mlp_baseline(
        x_norm, y, train, val, test, x_raw, seed=1, max_epochs=5, patience=3
    )
    assert metrics.dv_heuristic_test_accuracy >= 0.99
    assert dv_heuristic_predictions(x_raw[test]).shape == (len(test),)


def test_gcn_forward_batch_shape():
    topo = build_case39_topology()
    x = torch.randn(39, 8)
    data = Data(
        x=x,
        edge_index=torch.from_numpy(topo.edge_index.copy()),
        edge_attr=torch.from_numpy(topo.edge_weight.copy()).unsqueeze(-1),
        y=torch.tensor([3], dtype=torch.long),
    )
    loader = __import__("torch_geometric.loader", fromlist=["DataLoader"]).DataLoader(
        [data, data], batch_size=2
    )
    model = FaultGCN()
    batch = next(iter(loader))
    ew = batch.edge_attr.squeeze(-1)
    logits = model(batch.x, batch.edge_index, edge_weight=ew, batch=batch.batch)
    assert logits.shape == (2, 39)


def test_hop_distance_self_and_neighbor():
    topo = build_case39_topology()
    hop = build_hop_distance_matrix(topo.edge_index, n_nodes=39)
    assert hop[0, 0] == 0
    pairs = undirected_edge_pairs(topo.edge_index)
    u, v = pairs[0]
    assert hop[u, v] == 1
    assert hop[v, u] == 1


def test_drop_random_edge_removes_one_undirected_pair():
    topo = build_case39_topology()
    data = Data(
        x=torch.zeros(39, 8),
        edge_index=torch.from_numpy(topo.edge_index.copy()),
    )
    before = len(undirected_edge_pairs(topo.edge_index))
    rng = np.random.default_rng(0)
    out = drop_random_undirected_edge(data, rng)
    after = len(undirected_edge_pairs(out.edge_index.numpy()))
    assert after == before - 1
    assert out.edge_index.shape[1] == data.edge_index.shape[1] - 2
