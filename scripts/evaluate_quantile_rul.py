#!/usr/bin/env python
"""Leave-one-failure-out quantile RUL evaluation and headline figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rul.evaluate import evaluate_baselines
from rul.labels import attach_rul_labels, daily_from_scored_frame
from rul.plots import write_all_band_figures
from rul.quantile import (
    DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
    evaluate_quantile_model,
    pdp_tables,
)


def _load_labeled(args: argparse.Namespace) -> pd.DataFrame:
    if args.labeled:
        return pd.read_csv(args.labeled, parse_dates=["date"])
    if not args.scored or not args.failures:
        raise SystemExit("Provide --labeled or both --scored and --failures.")
    scored = pd.read_csv(args.scored)
    failures = pd.read_csv(args.failures)
    daily = daily_from_scored_frame(scored)
    return attach_rul_labels(daily, failures, rul_max=args.rul_max)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", type=Path, help="Pre-built daily labeled CSV")
    parser.add_argument("--scored", type=Path, help="10-minute scored CSV from the twin")
    parser.add_argument("--failures", type=Path, help="Gearbox failure log CSV")
    parser.add_argument("--out", type=Path, default=Path("reports/rul"))
    parser.add_argument("--rul-max", type=int, default=90)
    parser.add_argument(
        "--maintenance-threshold",
        type=float,
        default=DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
    )
    parser.add_argument("--last-n-days", type=int, default=90)
    args = parser.parse_args()

    labeled = _load_labeled(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    baselines = evaluate_baselines(labeled, rul_max=args.rul_max)
    metrics, coverage, warnings, fold_results = evaluate_quantile_model(
        labeled,
        rul_max=args.rul_max,
        maintenance_threshold=args.maintenance_threshold,
    )
    combined = pd.concat([baselines, metrics], ignore_index=True, sort=False)

    combined.to_csv(out / "metrics.csv", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    warnings.to_csv(out / "warnings.csv", index=False)
    pdp_tables(fold_results).to_csv(out / "pdp_anomaly_mean.csv", index=False)
    write_all_band_figures(
        fold_results,
        out / "figures",
        last_n_days=args.last_n_days,
        maintenance_threshold=args.maintenance_threshold,
    )
    print(f"Wrote metrics to {out}")


if __name__ == "__main__":
    main()
