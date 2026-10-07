"""Tests for LightGBM quantile RUL models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul.labels import DAILY_FEATURE_COLUMNS, RUL_MAX_DAYS, attach_rul_labels
from rul.plots import plot_rul_band
from rul.quantile import (
    NOMINAL_COVERAGE,
    evaluate_quantile_model,
    fit_quantile_models,
    monotone_constraint_vector,
    predict_quantiles,
    score_fold,
    warning_lead_time_days,
)

RUL_MAX = 10
FAST_LGBM = {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 2}


def test_monotone_constraints_only_on_anomaly_columns():
    vec = monotone_constraint_vector()
    assert len(vec) == len(DAILY_FEATURE_COLUMNS)
    for col, constraint in zip(DAILY_FEATURE_COLUMNS, vec, strict=True):
        if col in ("anomaly_score_mean", "anomaly_score_max"):
            assert constraint == -1
        else:
            assert constraint == 0


def test_lofo_quantile_never_trains_on_held_out_turbine():
    labeled = _labeled_panel()
    _, _, _, folds = evaluate_quantile_model(labeled, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    for result in folds:
        assert result.fold.turbine_id not in set(result.fold.train["turbine_id"])


def test_predict_quantiles_ordered_and_clipped():
    labeled = _labeled_panel()
    train = labeled.loc[labeled["turbine_id"] != "T_A"]
    bundle = fit_quantile_models(train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    test = labeled.loc[labeled["turbine_id"] == "T_A"].head(5)
    q10, q50, q90 = predict_quantiles(bundle, test)
    assert np.all(q10 <= q50 + 1e-9)
    assert np.all(q50 <= q90 + 1e-9)
    assert np.all(q10 >= 0.0)
    assert np.all(q90 <= float(RUL_MAX))


def test_pdp_anomaly_mean_is_non_increasing():
    labeled = _labeled_panel()
    train = labeled.loc[labeled["turbine_id"] != "T_A"]
    bundle = fit_quantile_models(train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    from rul.quantile import partial_dependence_anomaly_mean

    _, _, ok = partial_dependence_anomaly_mean(bundle, train, grid_size=10)
    if bundle.monotone_applied:
        assert ok is True


def test_warning_lead_time_from_p10_series():
    test = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=4, freq="D", tz="UTC"),
            "days_to_event": [40, 30, 12, 5],
        }
    )
    q10 = np.array([20.0, 16.0, 13.0, 8.0])
    assert warning_lead_time_days(test, q10, threshold=14) == pytest.approx(12.0)
    assert warning_lead_time_days(test, np.array([20.0, 16.0, 15.0, 14.5]), threshold=14) is None


def test_coverage_recorded_against_nominal_80_percent():
    labeled = _labeled_panel()
    _, coverage, _, _ = evaluate_quantile_model(labeled, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    assert not coverage.empty
    assert (coverage["nominal_coverage"] == NOMINAL_COVERAGE).all()
    assert "coverage_gap" in coverage.columns
    assert "pi_coverage" in coverage.columns


def test_band_figure_written(tmp_path: Path):
    labeled = _labeled_panel()
    fold = next(
        f
        for f in __import__("rul.evaluate", fromlist=["leave_one_failure_out"]).leave_one_failure_out(
            labeled, rul_max=RUL_MAX
        )
        if f.turbine_id == "T_A"
    )
    scored = score_fold(fold, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    assert scored is not None
    out = tmp_path / "band.png"
    plot_rul_band(scored, out, last_n_days=RUL_MAX)
    assert out.is_file()
    assert out.stat().st_size > 0


def _labeled_panel() -> pd.DataFrame:
    start = pd.Timestamp("2016-01-01", tz="UTC")
    event = pd.Timestamp("2016-02-15", tz="UTC")
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
