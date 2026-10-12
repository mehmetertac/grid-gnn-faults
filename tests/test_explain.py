"""Tests for TreeSHAP sensor explanations."""

from pathlib import Path

import numpy as np
import pandas as pd

from rul.explain import (
    FEATURE_TO_SENSOR,
    SENSOR_ORDER,
    aggregate_shap_to_sensors,
    build_sensor_ranking_csv,
    field_intuition_paragraph,
    run_explain,
    select_explain_fold,
    tree_shap_values,
    validate_sensor_map,
)
from rul.labels import DAILY_FEATURE_COLUMNS, attach_rul_labels
from rul.quantile import (
    fit_quantile_models,
    load_quantile_bundle,
    predict_quantile_triple,
    save_quantile_bundle,
)

RUL_MAX = 10
FAST_LGBM = {"n_estimators": 8, "num_leaves": 5, "min_child_samples": 2}


def test_every_daily_feature_maps_to_one_sensor():
    validate_sensor_map()
    assert set(FEATURE_TO_SENSOR) == set(DAILY_FEATURE_COLUMNS)


def test_sensor_shap_sums_match_feature_shap():
    labeled = _labeled_panel()
    fold = select_explain_fold(labeled, rul_max=RUL_MAX, turbine_id="T_A")
    bundle = fit_quantile_models(fold.train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    feat = tree_shap_values(bundle, fold.train.head(12), 0.5)
    sensor = aggregate_shap_to_sensors(feat, bundle.feature_columns)
    assert np.allclose(sensor.sum(axis=1), feat.sum(axis=1), atol=1e-5)


def test_held_out_turbine_absent_from_training_fold():
    labeled = _labeled_panel()
    fold = select_explain_fold(labeled, rul_max=RUL_MAX, turbine_id="T_A")
    assert fold.turbine_id == "T_A"
    assert fold.turbine_id not in set(fold.train["turbine_id"])


def test_sensor_ranking_csv_schema_and_names(tmp_path: Path):
    labeled = _labeled_panel()
    fold = select_explain_fold(labeled, rul_max=RUL_MAX, turbine_id="T_A")
    bundle = fit_quantile_models(fold.train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    ranking = build_sensor_ranking_csv(bundle, fold.train)
    assert set(ranking["learner"]) == {"p10", "p50"}
    assert set(ranking["sensor"]).issubset(set(SENSOR_ORDER))
    assert len(ranking) == len(SENSOR_ORDER) * 2
    assert ranking["rank"].min() == 1
    assert "mean_abs_shap" in ranking.columns
    assert "features" in ranking.columns


def test_run_explain_writes_figures_and_csv(tmp_path: Path):
    labeled = _labeled_panel()
    paths = run_explain(
        labeled,
        tmp_path,
        rul_max=RUL_MAX,
        lgbm_overrides=FAST_LGBM,
        write_field_notes=None,
        turbine_id="T_A",
    )
    assert paths["ranking_csv"].is_file()
    assert paths["shap_global"].is_file() and paths["shap_global"].stat().st_size > 0
    assert paths["shap_local_timeline"].is_file() and paths["shap_local_timeline"].stat().st_size > 0


def test_bundle_save_load_round_trip_predictions(tmp_path: Path):
    labeled = _labeled_panel()
    fold = select_explain_fold(labeled, rul_max=RUL_MAX, turbine_id="T_B")
    bundle = fit_quantile_models(fold.train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    sample = fold.test.head(4)
    p10_a, p50_a, _ = predict_quantile_triple(bundle, sample, 0.1, 0.5, 0.9)
    path = tmp_path / "bundle.joblib"
    save_quantile_bundle(bundle, path)
    loaded = load_quantile_bundle(path)
    p10_b, p50_b, _ = predict_quantile_triple(loaded, sample, 0.1, 0.5, 0.9)
    np.testing.assert_allclose(p10_a, p10_b, rtol=0, atol=1e-9)
    np.testing.assert_allclose(p50_a, p50_b, rtol=0, atol=1e-9)


def test_field_intuition_paragraph_non_empty():
    labeled = _labeled_panel()
    fold = select_explain_fold(labeled, rul_max=RUL_MAX, turbine_id="T_A")
    bundle = fit_quantile_models(fold.train, rul_max=RUL_MAX, lgbm_overrides=FAST_LGBM)
    text = field_intuition_paragraph(fold.test, bundle)
    assert isinstance(text, str) and len(text) > 40


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
