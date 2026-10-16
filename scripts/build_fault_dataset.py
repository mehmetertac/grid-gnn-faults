"""Generate IEEE 39-bus fault dataset, PyG graphs, baseline, and sanity figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grid.baseline import save_baseline_metrics, train_mlp_baseline
from grid.graph import (
    build_case39_topology,
    plot_dv_sample,
    save_processed_dataset,
    samples_to_arrays,
    stratified_split,
    zscore_fit,
    zscore_transform,
)
from grid.simulate import generate_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples, bus_ids, method_counts = generate_and_save(
        out_dir=args.out,
        n_scenarios=args.scenarios,
        seed=args.seed,
    )
    print(f"Generated {len(samples)} samples; voltage methods: {method_counts}")

    topology = build_case39_topology()
    x_raw, labels, _scenario = samples_to_arrays(samples)
    train_idx, val_idx, test_idx = stratified_split(labels, seed=args.seed)
    mean, std = zscore_fit(x_raw[train_idx])
    x_norm = zscore_transform(x_raw, mean, std)

    methods = np.array([s.voltage_method for s in samples], dtype=object)
    npz_path, pt_path = save_processed_dataset(
        args.out,
        x_raw,
        x_norm,
        labels,
        topology,
        train_idx,
        val_idx,
        test_idx,
        mean,
        std,
        voltage_methods=methods,
    )
    print(f"Saved {npz_path} and {pt_path}")

    plot_dv_sample(samples[0], topology, args.out / "fault_dv_sample.png")
    print(f"Wrote sanity figure to {args.out / 'fault_dv_sample.png'}")

    metrics = train_mlp_baseline(
        x_norm,
        labels,
        train_idx,
        val_idx,
        test_idx,
        x_raw,
        seed=args.seed,
    )
    metrics_path = args.out / "mlp_baseline.json"
    save_baseline_metrics(metrics, metrics_path)
    print(f"Baseline metrics: {metrics}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
