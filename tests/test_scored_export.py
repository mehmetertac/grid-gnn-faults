"""Fleet scored CSV helpers (no wind_digital_twin)."""

import pandas as pd

from rul.scored_export import concat_scored_frames


def test_concat_scored_frames_sorts_by_turbine_and_time():
    a = pd.DataFrame(
        {
            "turbine_id": ["T02", "T02"],
            "timestamp": ["2016-01-02", "2016-01-01"],
            "oil_residual": [0.1, 0.2],
            "bear_residual": [0.0, 0.0],
            "anomaly_score": [1.0, 1.0],
            "theta_oil": [0.5, 0.5],
            "theta_bear": [0.6, 0.6],
            "delta_bear_oil": [0.1, 0.1],
        }
    )
    b = pd.DataFrame(
        {
            "turbine_id": ["T01", "T01"],
            "timestamp": ["2016-01-01", "2016-01-02"],
            "oil_residual": [-0.1, -0.2],
            "bear_residual": [0.0, 0.0],
            "anomaly_score": [0.5, 0.5],
            "theta_oil": [0.4, 0.4],
            "theta_bear": [0.5, 0.5],
            "delta_bear_oil": [0.1, 0.1],
        }
    )
    out = concat_scored_frames([a, b])
    assert list(out["turbine_id"]) == ["T01", "T01", "T02", "T02"]
    assert out.iloc[0]["timestamp"] == "2016-01-01"


def test_concat_scored_frames_empty():
    out = concat_scored_frames([])
    assert list(out.columns) == [
        "turbine_id",
        "timestamp",
        "oil_residual",
        "bear_residual",
        "anomaly_score",
        "theta_oil",
        "theta_bear",
        "delta_bear_oil",
    ]
