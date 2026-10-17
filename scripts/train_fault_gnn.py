"""Train 2-layer GCN for IEEE 39-bus fault localization and write metrics + figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from grid.model import (
    _predict_gnn,
    evaluate_full_run,
    load_processed_graphs,
    plot_fault_error_layout,
    save_gnn_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure-docs", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    graphs, x_norm, x_raw, y, train_idx, val_idx, test_idx, topo = load_processed_graphs(
        args.data
    )
    _ = x_raw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, result = evaluate_full_run(
        train_idx,
        val_idx,
        test_idx,
        graphs,
        x_norm,
        y,
        x_raw,
        topo,
        seed=args.seed,
        device=device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "gnn_metrics.json"
    save_gnn_metrics(result, metrics_path)
    torch.save(model.state_dict(), args.out / "fault_gcn.pt")

    test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=64, shuffle=False)
    _, y_pred, _ = _predict_gnn(model, test_loader, device)
    y_test = y[test_idx]
    fig_results = args.out / "fault_error_layout.png"
    fig_docs = args.figure_docs / "fault_error_layout.png"
    plot_fault_error_layout(y_test, y_pred, topo, fig_results)
    plot_fault_error_layout(y_test, y_pred, topo, fig_docs)

    print(result)
    print(f"Wrote {metrics_path}, {fig_results}, {fig_docs}")


if __name__ == "__main__":
    main()
