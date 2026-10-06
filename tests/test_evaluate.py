"""Unit and integration tests for leave-one-failure-out RUL scoring."""

import numpy as np
import pandas as pd
import pytest

from rul.evaluate import (
    CONSTANT_NOTE,
    constant_baseline,
    evaluate_baselines,
    fit_linear_anomaly_baseline,
    leave_one_failure_out,
    pinball_loss,
    pi_coverage,
    predict_linear_anomaly,
)
from rul.labels import attach_rul_labels, daily_from_scored_frame

RUL_MAX = 10


def test_pinball_and_coverage_match_hand_calculation():
    y_true = np.array([0.0, 10.0])
    y_hat = np.array([0.0, 0.0])
    assert pinball_loss(y_true, y_hat, 0.5) == pytest.approx(2.5)
    assert pinball_loss(y_true, y_hat, 0.1) == pytest.approx(0.5)

    coverage = pi_coverage(
        np.array([1.0, 5.0, 10.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 2.0, 2.0]),
    )
    assert coverage == pytest.approx(1.0 / 3.0)


def test_linear_baseline_fit_and_clip():
    intercept, slope = fit_linear_anomaly_baseline(
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 8.0, 6.0]),
    )
    assert intercept == pytest.approx(10.0)
    assert slope == pytest.approx(-2.0)
    clipped = predict_linear_anomaly(np.array([0.0, 100.0]), intercept, slope, rul_max=10)
    assert clipped[0] == pytest.approx(10.0)
    assert clipped[1] == pytest.approx(0.0)


def test_lofo_excludes_held_out_turbine_and_scores_only_the_window():
    labeled = _labeled_panel()
    folds = leave_one_failure_out(labeled, rul_max=RUL_MAX)
    by_turbine = {fold.turbine_id: fold for fold in folds}
    held = by_turbine["T_A"]

    assert "T_A" not in set(held.train["turbine_id"])
    assert set(held.test["turbine_id"]) == {"T_A"}
    assert held.test["days_to_event"].max() <= RUL_MAX
    assert held.test["days_to_event"].min() >= 1
    assert len(held.test) == RUL_MAX

    pre_window = labeled[
        (labeled["turbine_id"] == "T_A") & (labeled["days_to_event"] > RUL_MAX)
    ]
    assert not pre_window.empty
    assert not set(pre_window.index).intersection(held.train.index)
    assert not set(pre_window.index).intersection(held.test.index)

    healthy = labeled.loc[labeled["turbine_id"] == "T_C"]
    assert set(healthy.index).issubset(set(held.train.index))
    assert not set(healthy.index).intersection(held.test.index)


def test_constant_baseline_on_decreasing_window():
    labeled = _labeled_panel()
    result = evaluate_baselines(labeled, rul_max=RUL_MAX)
    constant = result[(result["baseline"] == "constant") & (result["turbine_id"] == "T_A")].iloc[0]
    assert constant["n_test"] == RUL_MAX
    assert constant["mae"] == pytest.approx(4.5)
    assert constant["pi_coverage"] == pytest.approx(0.1)
    assert bool(constant["interval_degenerate"])
    assert "Degenerate" in constant["note"]
    assert constant["note"] == CONSTANT_NOTE
    q10, q50, q90 = constant_baseline(3, rul_max=RUL_MAX)
    assert np.array_equal(q10, q50)
    assert np.array_equal(q50, q90)


def test_evaluate_baselines_integration_on_scored_frame():
    scored = _scored_panel()
    failures = pd.DataFrame(
        {
            "Turbine_ID": ["T_A", "T_B", "T_A"],
            "Timestamp": ["2016-01-21T00:00:00+00:00", "2016-01-21T00:00:00+00:00", "2016-01-21T00:00:00+00:00"],
            "Component": ["GEARBOX", "GEARBOX", "HYDRAULIC"],
            "Remarks": ["gearbox", "gearbox", "ignored"],
        }
    )
    daily = daily_from_scored_frame(scored)
    labeled = attach_rul_labels(daily, failures, rul_max=RUL_MAX)
    result = evaluate_baselines(labeled, rul_max=RUL_MAX)

    assert set(result["baseline"]) == {"constant", "linear_anomaly"}
    assert set(result["turbine_id"]) == {"T_A", "T_B"}
    assert result["mae"].notna().all()
    assert result["pinball_0.1"].notna().all()
    assert (result["n_train"] > 0).all()
    assert (result["n_test"] == RUL_MAX).all()
    linear = result.loc[result["baseline"] == "linear_anomaly"]
    assert linear["interval_degenerate"].all()


def _labeled_panel() -> pd.DataFrame:
    start = pd.Timestamp("2016-01-01", tz="UTC")
    event = pd.Timestamp("2016-02-15", tz="UTC")
    days = pd.date_range(start, event - pd.Timedelta(days=1), freq="D", tz="UTC")
    frames = []
    for turbine, anomaly in (("T_A", 1.0), ("T_B", 0.5)):
        frames.append(
            pd.DataFrame(
                {
                    "turbine_id": turbine,
                    "date": days,
                    "anomaly_score_mean": np.linspace(anomaly, anomaly + 1.0, len(days)),
                }
            )
        )
    healthy_days = pd.date_range("2016-01-01", periods=20, freq="D", tz="UTC")
    frames.append(
        pd.DataFrame(
            {
                "turbine_id": "T_C",
                "date": healthy_days,
                "anomaly_score_mean": 0.1,
            }
        )
    )
    daily = pd.concat(frames, ignore_index=True)
    failures = pd.DataFrame(
        {
            "turbine_id": ["T_A", "T_B"],
            "timestamp": [event, event],
            "remarks": ["a", "b"],
        }
    )
    return attach_rul_labels(daily, failures, rul_max=RUL_MAX)


def _scored_panel() -> pd.DataFrame:
    rows = []
    for turbine, level in (("T_A", 0.0), ("T_B", 1.0), ("T_C", -1.0)):
        for day in range(20):
            for minute in (0, 10):
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2016-01-01", tz="UTC")
                        + pd.Timedelta(days=day, minutes=minute),
                        "turbine_id": turbine,
                        "oil_residual": -(level + day / 20.0),
                        "bear_residual": -(level + day / 20.0),
                        "anomaly_score": level + day / 10.0,
                        "theta_oil": 0.01 + day / 1000.0,
                        "theta_bear": 0.02,
                        "delta_bear_oil": 1.0,
                    }
                )
    return pd.DataFrame(rows)
