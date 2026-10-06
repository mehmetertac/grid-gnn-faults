"""Unit tests for capped RUL labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rul.labels import (
    RUL_MAX_DAYS,
    attach_rul_labels,
    catalog_failure_frame,
    daily_from_scored_frame,
    piecewise_rul,
    summarize_events,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_piecewise_cap_matches_cmapss_rule():
    assert piecewise_rul(100) == RUL_MAX_DAYS
    assert piecewise_rul(30) == 30
    assert piecewise_rul(1) == 1
    assert piecewise_rul(80, rul_max=60) == 60
    with pytest.raises(ValueError):
        piecewise_rul(-1)


def test_catalog_summary_is_four_events_on_three_turbines():
    summary = summarize_events(catalog_failure_frame())
    assert summary["n_events"] == 4
    assert summary["n_turbines"] == 3
    turbines = {event["turbine_id"] for event in summary["events"]}
    assert turbines == {"T01", "T06", "T09"}


def test_failure_csv_filters_non_gearbox_components():
    log = pd.read_csv(FIXTURES / "catalog_gearbox_failures.csv")
    summary = summarize_events(log)
    assert summary["n_events"] == 4
    assert "T11" not in {event["turbine_id"] for event in summary["events"]}


def test_daily_aggregate_uses_degradation_sign_and_twin_columns():
    scored = pd.read_csv(FIXTURES / "synthetic_scored_10min.csv")
    daily = daily_from_scored_frame(scored)
    first = daily.iloc[0]
    assert first["oil_deg_mean"] == pytest.approx(3.0)
    assert first["oil_deg_std"] == pytest.approx(np.sqrt(2.0))
    assert first["anomaly_score_mean"] == pytest.approx(0.3)
    assert first["anomaly_score_max"] == pytest.approx(0.4)
    assert first["theta_oil_mean"] == pytest.approx(0.02)
    assert first["theta_bear_mean"] == pytest.approx(0.03)
    assert first["delta_bear_oil_mean"] == pytest.approx(1.1)
    assert pd.isna(first["bear_deg_slope_7d"])


def test_seven_day_slope_of_daily_degradation():
    rows = []
    for day in range(7):
        for minute in (0, 10):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2016-01-01", tz="UTC")
                    + pd.Timedelta(days=day, minutes=minute),
                    "turbine_id": "T01",
                    "oil_residual": -float(day),
                    "bear_residual": -float(day),
                    "anomaly_score": float(day),
                    "theta_oil": 0.01,
                    "theta_bear": 0.02,
                    "delta_bear_oil": 1.0,
                }
            )
    daily = daily_from_scored_frame(pd.DataFrame(rows))
    last = daily.iloc[-1]
    assert last["oil_deg_mean"] == pytest.approx(6.0)
    assert last["oil_deg_slope_7d"] == pytest.approx(1.0)
    assert last["bear_deg_slope_7d"] == pytest.approx(1.0)


def test_event_day_dropped_and_healthy_turbine_censored():
    daily = pd.DataFrame(
        {
            "turbine_id": ["T01", "T01", "T11"],
            "date": pd.to_datetime(
                ["2016-07-17", "2016-07-18", "2016-07-17"], utc=True
            ),
            "anomaly_score_mean": [0.2, 0.9, 0.0],
        }
    )
    labeled = attach_rul_labels(daily, catalog_failure_frame(), rul_max=90)
    t01 = labeled.loc[labeled["turbine_id"] == "T01"]
    assert list(t01["date"].dt.strftime("%Y-%m-%d")) == ["2016-07-17"]
    assert int(t01.iloc[0]["days_to_event"]) == 1
    assert t01.iloc[0]["rul"] == 1
    assert not bool(t01.iloc[0]["censored"])

    healthy = labeled.loc[labeled["turbine_id"] == "T11"].iloc[0]
    assert bool(healthy["censored"])
    assert healthy["rul"] == 90
    assert pd.isna(healthy["event_id"])


def test_clock_resets_toward_the_next_gearbox_event():
    dates = [
        "2016-10-10",
        "2016-10-11",
        "2016-10-12",
        "2017-10-17",
        "2017-10-18",
        "2017-10-19",
    ]
    daily = pd.DataFrame(
        {
            "turbine_id": "T09",
            "date": pd.to_datetime(dates, utc=True),
            "anomaly_score_mean": 0.5,
        }
    )
    labeled = attach_rul_labels(daily, catalog_failure_frame(), rul_max=90)
    kept = labeled["date"].dt.strftime("%Y-%m-%d").tolist()
    assert kept == ["2016-10-10", "2016-10-12", "2017-10-17", "2017-10-19"]

    by_day = labeled.set_index(labeled["date"].dt.strftime("%Y-%m-%d"))
    assert int(by_day.loc["2016-10-10", "days_to_event"]) == 1
    assert by_day.loc["2016-10-10", "rul"] == 1
    assert int(by_day.loc["2016-10-12", "days_to_event"]) == 371
    assert by_day.loc["2016-10-12", "rul"] == 90
    assert not bool(by_day.loc["2016-10-12", "censored"])
    assert int(by_day.loc["2017-10-17", "days_to_event"]) == 1
    assert bool(by_day.loc["2017-10-19", "censored"])
    assert by_day.loc["2017-10-19", "rul"] == 90
    assert by_day.loc["2016-10-10", "event_id"] != by_day.loc["2016-10-12", "event_id"]


def test_days_outside_the_window_stay_flat_at_the_cap():
    daily = pd.DataFrame(
        {
            "turbine_id": ["T06", "T06"],
            "date": pd.to_datetime(["2017-07-01", "2017-10-16"], utc=True),
            "anomaly_score_mean": [0.0, 1.0],
        }
    )
    labeled = attach_rul_labels(daily, catalog_failure_frame(), rul_max=90)
    far = labeled.iloc[0]
    near = labeled.iloc[1]
    assert int(far["days_to_event"]) > 90
    assert far["rul"] == 90
    assert not bool(far["censored"])
    assert int(near["days_to_event"]) == 1
    assert near["rul"] == 1


def test_missing_scored_columns_raise():
    with pytest.raises(ValueError, match="oil_residual"):
        daily_from_scored_frame(pd.DataFrame({"turbine_id": ["T01"], "timestamp": ["2016-01-01"]}))
