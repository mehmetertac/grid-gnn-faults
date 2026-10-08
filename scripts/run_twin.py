#!/usr/bin/env python
"""End-to-end twin: SCADA → physics → hybrid residual → anomaly → conformal RUL."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rul.conformal import predict_conformal_rul_for_frame
from rul.labels import attach_rul_labels, catalog_failure_frame, daily_from_scored_frame
from rul.plots import plot_headline_timeline


def _import_twin():
    try:
        from wind_digital_twin.anomaly.hybrid_residual_detector import (
            fit_hybrid_residual_detector,
        )
        from wind_digital_twin.config import (
            DATA_RAW,
            DEFAULT_BUFFER_DAYS,
            THERMAL_MIN_POWER_KW,
        )
        from wind_digital_twin.data.clean import (
            clean_turbine_df,
            get_failure_for_turbine,
            healthy_training_mask,
        )
        from wind_digital_twin.data.load_edp import load_edp_dataset
        from wind_digital_twin.residual.fusion_features import build_fusion_feature_frame
    except ImportError as exc:
        raise SystemExit(
            "wind_digital_twin is not installed. Clone wind-digital-twin and "
            "run: pip install -e path/to/wind-digital-twin"
        ) from exc
    return {
        "fit_hybrid_residual_detector": fit_hybrid_residual_detector,
        "DATA_RAW": DATA_RAW,
        "DEFAULT_BUFFER_DAYS": DEFAULT_BUFFER_DAYS,
        "THERMAL_MIN_POWER_KW": THERMAL_MIN_POWER_KW,
        "clean_turbine_df": clean_turbine_df,
        "get_failure_for_turbine": get_failure_for_turbine,
        "healthy_training_mask": healthy_training_mask,
        "load_edp_dataset": load_edp_dataset,
        "build_fusion_feature_frame": build_fusion_feature_frame,
    }


def build_scored_frame(df: pd.DataFrame, pipeline, twin) -> pd.DataFrame:
    """Ten-minute scored contract for ``daily_from_scored_frame``."""
    build_fusion = twin["build_fusion_feature_frame"]
    residuals = pipeline.hybrid.dual_hybrid_residuals(df)
    scores = pipeline.score(df)
    fusion = build_fusion(df)
    common = residuals.index.intersection(scores.index).intersection(fusion.index)
    frame = pd.DataFrame(
        {
            "timestamp": common,
            "oil_residual": residuals.loc[common, "oil_residual"].values,
            "bear_residual": residuals.loc[common, "bear_residual"].values,
            "anomaly_score": scores.loc[common].values,
            "theta_oil": fusion.loc[common, "theta_oil"].values,
            "theta_bear": fusion.loc[common, "theta_bear"].values,
            "delta_bear_oil": fusion.loc[common, "delta_bear_oil"].values,
        }
    )
    return frame


def build_timeline_daily(
    df: pd.DataFrame,
    pipeline,
    rul_daily: pd.DataFrame,
    twin,
) -> pd.DataFrame:
    """Daily rows for the four-panel headline figure."""
    scored = build_scored_frame(df, pipeline, twin)
    scored["date"] = pd.to_datetime(scored["timestamp"], utc=True).dt.floor("D")
    oil_actual = df["Gear_Oil_Temp_Avg"]
    oil_physics = pipeline.hybrid.oil_model.predict(df)
    physics = pd.DataFrame(
        {
            "timestamp": oil_physics.index,
            "oil_temp_actual": oil_actual.reindex(oil_physics.index).values,
            "oil_temp_physics": oil_physics.values,
        }
    )
    physics["date"] = pd.to_datetime(physics["timestamp"], utc=True).dt.floor("D")
    phys_daily = physics.groupby("date", as_index=False).agg(
        oil_temp_actual=("oil_temp_actual", "mean"),
        oil_temp_physics=("oil_temp_physics", "mean"),
    )
    sc_daily = scored.groupby("date", as_index=False).agg(
        oil_residual=("oil_residual", "mean"),
        anomaly_score=("anomaly_score", "mean"),
    )
    timeline = phys_daily.merge(sc_daily, on="date", how="inner")
    rul_cols = [
        "date",
        "rul_p10",
        "rul_p50",
        "rul_p90",
        "cqr80_lo",
        "cqr80_hi",
        "cqr90_lo",
        "cqr90_hi",
    ]
    present = [c for c in rul_cols if c in rul_daily.columns]
    timeline = timeline.merge(rul_daily[present], on="date", how="left")
    return timeline.sort_values("date")


def run_twin(
    *,
    turbine_id: str = "T06",
    data_dir: Path | None = None,
    out_dir: Path = Path("reports/twin"),
    buffer_days: int | None = None,
    rul_max: int = 90,
) -> tuple[pd.DataFrame, Path]:
    twin = _import_twin()
    raw_dir = data_dir or twin["DATA_RAW"]
    buffer_days = buffer_days if buffer_days is not None else twin["DEFAULT_BUFFER_DAYS"]

    turbines, failures = twin["load_edp_dataset"](raw_dir)
    if turbine_id not in turbines:
        raise SystemExit(f"{turbine_id} not in dataset under {raw_dir}")

    df = twin["clean_turbine_df"](
        turbines[turbine_id],
        min_power_kw=twin["THERMAL_MIN_POWER_KW"],
    )
    failure = twin["get_failure_for_turbine"](turbine_id, failures)
    healthy_mask = twin["healthy_training_mask"](df.index, failure, buffer_days)
    train = df.loc[healthy_mask]
    if train.empty:
        raise SystemExit(f"No healthy training rows for {turbine_id}")

    pipeline = twin["fit_hybrid_residual_detector"](train)
    scored = build_scored_frame(df, pipeline, twin)
    scored["turbine_id"] = turbine_id

    daily = daily_from_scored_frame(scored)
    failure_log = catalog_failure_frame()
    labeled = attach_rul_labels(daily, failure_log, rul_max=rul_max)
    fleet = labeled.copy()
    turbine_daily = labeled.loc[labeled["turbine_id"] == turbine_id].copy()
    with_rul = predict_conformal_rul_for_frame(fleet, turbine_daily, rul_max=rul_max)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{turbine_id}_turbine_day.csv"
    with_rul.to_csv(csv_path, index=False)

    timeline = build_timeline_daily(df, pipeline, with_rul, twin)
    failure_time = failure.timestamp if failure is not None else None
    fig_path = out_dir / f"{turbine_id}_timeline.png"
    plot_headline_timeline(timeline, fig_path, failure_time=failure_time)

    return with_rul, fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbine", default="T06")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/twin"))
    parser.add_argument("--buffer-days", type=int, default=None)
    parser.add_argument("--rul-max", type=int, default=90)
    args = parser.parse_args()
    _, fig_path = run_twin(
        turbine_id=args.turbine,
        data_dir=args.data_dir,
        out_dir=args.out,
        buffer_days=args.buffer_days,
        rul_max=args.rul_max,
    )
    print(f"Wrote turbine-day CSV and {fig_path}")


if __name__ == "__main__":
    main()
