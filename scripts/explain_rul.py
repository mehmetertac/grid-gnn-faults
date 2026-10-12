#!/usr/bin/env python
"""Fit or load LOFO quantile RUL models and write SHAP sensor explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rul.explain import fit_bundle_for_fold, run_explain, select_explain_fold
from rul.labels import attach_rul_labels, daily_from_scored_frame
from rul.quantile import save_quantile_bundle


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
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--model", type=Path, help="Existing quantile bundle (skip refit)")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Where to save bundle after fit (default: OUT/rul_quantile_bundle.joblib)",
    )
    parser.add_argument("--event", type=str, default=None, help="LOFO event_id to hold out")
    parser.add_argument("--turbine", type=str, default=None, help="Held-out turbine (LOFO)")
    parser.add_argument("--rul-max", type=int, default=90)
    parser.add_argument(
        "--field-notes",
        type=Path,
        default=Path("docs/EXPLANATIONS.md"),
        help="Append field-intuition paragraph to this doc",
    )
    parser.add_argument("--no-field-notes", action="store_true")
    args = parser.parse_args()

    labeled = _load_labeled(args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model_out = args.model_out or (out / "rul_quantile_bundle.joblib")

    bundle = None
    model_path = args.model
    if model_path is None:
        fold = select_explain_fold(
            labeled,
            rul_max=args.rul_max,
            event_id=args.event,
            turbine_id=args.turbine,
        )
        bundle = fit_bundle_for_fold(fold, rul_max=args.rul_max)
        save_quantile_bundle(bundle, model_out)
        model_path = model_out
        print(f"Saved bundle to {model_out}")

    paths = run_explain(
        labeled,
        out,
        bundle=bundle,
        model_path=model_path,
        event_id=args.event,
        turbine_id=args.turbine,
        rul_max=args.rul_max,
        write_field_notes=None if args.no_field_notes else args.field_notes,
    )
    print(f"Wrote {paths['ranking_csv']}")
    print(f"Wrote {paths['shap_global']}")
    print(f"Wrote {paths['shap_local_timeline']}")


if __name__ == "__main__":
    main()
