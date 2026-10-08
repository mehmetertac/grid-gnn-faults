"""Tests for conformal RUL intervals (no EDP, no wind_digital_twin import)."""

from pathlib import Path

import numpy as np
import pandas as pd
from rul.conformal import (
    CONFIDENCE_80,
    CONFIDENCE_90,
    evaluate_conformal_rul,
    score_fold_conformal,
    split_conformal_calibration,
)
from rul.evaluate import leave_one_failure_out
from rul.labels import attach_rul_labels
from rul.plots import plot_headline_timeline, plot_rul_band
from rul.quantile import evaluate_quantile_model

RUL_MAX = 10
FAST_LGBM = {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 2}


def test_calibration_disjoint_from_test_turbine_and_time_ordered():
    labeled = _labeled_panel()
    fold = next(f for f in leave_one_failure_out(labeled, rul_max=RUL_MAX) if f.turbine_id == "T_A")
    split = split_conformal_calibration(labeled, fold)
    if split.calibrated:
        assert split.train_fit["turbine_id"].nunique() >= 1
        assert fold.turbine_id not in set(split.train_fit["turbine_id"])
        assert fold.turbine_id not in set(split.calibration["turbine_id"])
        assert split.calibration["censored"].eq(False).all()
        test_start = pd.to_datetime(fold.test["date"], utc=True).min()
        assert pd.to_datetime(split.calibration["date"], utc=True).max() < test_start
        if not split.train_fit.empty and not split.calibration.empty:
            assert pd.to_datetime(split.train_fit["date"], utc=True).max() <= pd.to_datetime(
                split.calibration["date"], utc=True
            ).min()


def test_conformal_coverage_table_per_event_and_pooled():
    labeled = _labeled_panel()
    coverage, width_df, _ = evaluate_conformal_rul(
        labeled, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM
    )
    assert not coverage.empty
    assert "pooled" in set(coverage["event_id"])
    assert "raw80_coverage" in coverage.columns
    assert "cqr80_coverage" in coverage.columns
    assert "cqr90_coverage" in coverage.columns
    assert coverage.loc[coverage["event_id"] != "pooled", "raw80_nominal"].eq(CONFIDENCE_80).all()
    assert coverage.loc[coverage["event_id"] != "pooled", "cqr90_nominal"].eq(CONFIDENCE_90).all()
    assert not width_df.empty
    assert "days_to_event" in width_df.columns
    assert "cqr80_width" in width_df.columns


def test_conformal_band_figure_with_overlay(tmp_path: Path):
    labeled = _labeled_panel()
    _, _, _, q_folds = evaluate_quantile_model(labeled, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    _, _, c_folds = evaluate_conformal_rul(labeled, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    q = next(r for r in q_folds if r.fold.turbine_id == "T_A")
    c = next(r for r in c_folds if r.fold.event_id == q.fold.event_id)
    out = tmp_path / "band_cqr.png"
    plot_rul_band(q, out, conformal=c, last_n_days=RUL_MAX)
    assert out.is_file() and out.stat().st_size > 0


def test_headline_timeline_png_without_twin_import(tmp_path: Path):
    days = pd.date_range("2016-01-01", periods=30, freq="D", tz="UTC")
    timeline = pd.DataFrame(
        {
            "date": days,
            "oil_temp_actual": np.linspace(40, 45, len(days)),
            "oil_temp_physics": np.linspace(39, 44, len(days)),
            "oil_residual": np.linspace(0.0, 0.5, len(days)),
            "anomaly_score": np.linspace(0.1, 1.0, len(days)),
            "rul_p50": np.linspace(20, 5, len(days)),
            "cqr80_lo": np.linspace(15, 3, len(days)),
            "cqr80_hi": np.linspace(25, 8, len(days)),
            "cqr90_lo": np.linspace(12, 2, len(days)),
            "cqr90_hi": np.linspace(28, 10, len(days)),
        }
    )
    out = tmp_path / "timeline.png"
    plot_headline_timeline(timeline, out, failure_time=days[-1])
    assert out.is_file() and out.stat().st_size > 0


def test_score_fold_conformal_runs():
    labeled = _labeled_panel()
    fold = next(f for f in leave_one_failure_out(labeled, rul_max=RUL_MAX) if f.turbine_id == "T_A")
    result = score_fold_conformal(labeled, fold, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    assert result is not None
    if result.split.calibrated:
        assert np.isfinite(result.cqr80_lo).any()


def _labeled_panel() -> pd.DataFrame:
    start = pd.Timestamp("2016-01-01", tz="UTC")
    event = pd.Timestamp("2016-03-01", tz="UTC")
    days = pd.date_range(start, event - pd.Timedelta(days=1), freq="D", tz="UTC")
    n = len(days)
    frames = []
    for turbine, base in (("T_A", 1.0), ("T_B", 0.5)):
        anomaly = np.linspace(base, base + 1.0, n)
        frames.append(_daily_row(turbine, days, anomaly))
    healthy = pd.date_range("2016-01-01", periods=20, freq="D", tz="UTC")
    frames.append(_daily_row("T_C", healthy, np.full(20, 0.1)))
    daily = pd.concat(frames, ignore_index=True)
    failures = pd.DataFrame(
        {
            "turbine_id": ["T_A", "T_B"],
            "timestamp": [event, event],
            "remarks": ["a", "b"],
        }
    )
    return attach_rul_labels(daily, failures, rul_max=RUL_MAX)


def _daily_row(turbine: str, days: pd.DatetimeIndex, anomaly: np.ndarray) -> pd.DataFrame:
    n = len(days)
    return pd.DataFrame(
        {
            "turbine_id": turbine,
            "date": days,
            "oil_deg_mean": np.linspace(0.1, 1.0, n),
            "oil_deg_std": 0.05,
            "bear_deg_mean": np.linspace(0.2, 1.1, n),
            "bear_deg_std": 0.05,
            "anomaly_score_mean": anomaly,
            "anomaly_score_max": anomaly + 0.1,
            "oil_deg_slope_7d": np.linspace(0.0, 0.05, n),
            "bear_deg_slope_7d": np.linspace(0.0, 0.04, n),
            "theta_oil_mean": 0.02,
            "theta_bear_mean": 0.03,
            "delta_bear_oil_mean": 1.0,
        }
    )
