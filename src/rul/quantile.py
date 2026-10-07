"""LightGBM quantile RUL models on Monday daily features.

Three boosters at q=0.1/0.5/0.9, leave-one-failure-out scoring, partial
dependence on anomaly_score_mean, and maintenance warning lead time from P10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm.basic import LightGBMError

from rul.evaluate import (
    METRIC_COLUMNS,
    FailureFold,
    leave_one_failure_out,
    score_predictions,
)
from rul.labels import DAILY_FEATURE_COLUMNS, RUL_MAX_DAYS

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
NOMINAL_COVERAGE = 0.8
DEFAULT_MAINTENANCE_THRESHOLD_DAYS = 14
PDP_TOLERANCE_DAYS = 1e-4

ANOMALY_MONOTONE_COLUMNS: frozenset[str] = frozenset(
    {"anomaly_score_mean", "anomaly_score_max"}
)

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "num_leaves": 7,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "random_state": 11,
    "verbosity": -1,
    "bagging_fraction": 1.0,
    "feature_fraction": 1.0,
    "bagging_freq": 0,
}


@dataclass(frozen=True)
class QuantileModelBundle:
    """One quantile booster per level, fit on the same feature matrix."""

    models: dict[float, LGBMRegressor]
    feature_columns: tuple[str, ...]
    monotone_applied: bool
    rul_max: int


@dataclass(frozen=True)
class FoldQuantileResult:
    """Predictions and diagnostics for one held-out gearbox event."""

    fold: FailureFold
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    metrics: dict[str, float]
    nominal_coverage: float
    coverage_gap: float
    mean_band_width: float
    early_mean_band_width: float
    late_mean_band_width: float
    early_coverage: float
    late_coverage: float
    pdp_grid: np.ndarray
    pdp_q50: np.ndarray
    pdp_monotone_ok: bool
    warning_lead_days: float | None
    monotone_applied: bool


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return daily feature columns, filling missing with NaN."""
    missing = [col for col in DAILY_FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Labeled frame is missing feature columns: {missing}")
    return frame.loc[:, DAILY_FEATURE_COLUMNS].copy()


def monotone_constraint_vector(
    feature_columns: tuple[str, ...] = DAILY_FEATURE_COLUMNS,
) -> list[int]:
    """``-1`` on anomaly score features (higher score → lower RUL), else ``0``."""
    return [
        -1 if col in ANOMALY_MONOTONE_COLUMNS else 0 for col in feature_columns
    ]


def _lgbm_params(
    quantile: float,
    overrides: dict[str, Any] | None = None,
    monotone: list[int] | None = None,
) -> dict[str, Any]:
    params = {**DEFAULT_LGBM_PARAMS, "objective": "quantile", "alpha": quantile}
    if monotone is not None:
        params["monotone_constraints"] = monotone
    if overrides:
        params.update(overrides)
    return params


def fit_quantile_models(
    train: pd.DataFrame,
    *,
    rul_max: int = RUL_MAX_DAYS,
    target_column: str = "rul",
    lgbm_overrides: dict[str, Any] | None = None,
) -> QuantileModelBundle:
    """Fit q10/q50/q90 boosters on training rows (includes censored at ``rul_max``)."""
    x = feature_matrix(train)
    y = train[target_column].astype(float).to_numpy()
    monotone = monotone_constraint_vector(tuple(x.columns))
    models: dict[float, LGBMRegressor] = {}
    monotone_applied = True

    def _fit_all(use_monotone: bool) -> dict[float, LGBMRegressor]:
        fitted: dict[float, LGBMRegressor] = {}
        mono = monotone if use_monotone else None
        for q in QUANTILES:
            model = LGBMRegressor(**_lgbm_params(q, lgbm_overrides, mono))
            model.fit(x, y)
            fitted[q] = model
        return fitted

    try:
        models = _fit_all(True)
    except (ValueError, TypeError, LightGBMError):
        monotone_applied = False
        models = _fit_all(False)
    return QuantileModelBundle(
        models=models,
        feature_columns=tuple(x.columns),
        monotone_applied=monotone_applied,
        rul_max=rul_max,
    )


def predict_quantiles(
    bundle: QuantileModelBundle,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict q10/q50/q90, clip to ``[0, rul_max]``, enforce q10 ≤ q50 ≤ q90."""
    x = feature_matrix(frame)
    raw = {
        q: bundle.models[q].predict(x) for q in QUANTILES
    }
    stacked = np.column_stack([raw[q] for q in QUANTILES])
    clipped = np.clip(stacked, 0.0, float(bundle.rul_max))
    ordered = np.sort(clipped, axis=1)
    return ordered[:, 0], ordered[:, 1], ordered[:, 2]


def partial_dependence_anomaly_mean(
    bundle: QuantileModelBundle,
    train: pd.DataFrame,
    *,
    grid_size: int = 25,
    quantile: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """PDP of ``anomaly_score_mean`` at training medians; q50 must be non-increasing."""
    x_base = feature_matrix(train)
    medians = x_base.median(numeric_only=True)
    col = "anomaly_score_mean"
    finite = x_base[col].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        grid = np.array([0.0])
    else:
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            grid = np.array([lo])
        else:
            grid = np.linspace(lo, hi, grid_size)

    model = bundle.models[quantile]
    rows: list[np.ndarray] = []
    for value in grid:
        row = medians.copy()
        row[col] = value
        rows.append(row.to_numpy(dtype=float))
    design = pd.DataFrame(rows, columns=x_base.columns)
    preds = model.predict(design)
    preds = np.clip(preds, 0.0, float(bundle.rul_max))
    if grid.size <= 1:
        monotone_ok = True
    else:
        diffs = np.diff(preds)
        monotone_ok = bool(np.all(diffs <= PDP_TOLERANCE_DAYS))
    return grid, preds, monotone_ok


def warning_lead_time_days(
    test: pd.DataFrame,
    q10: np.ndarray,
    *,
    threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
) -> float | None:
    """First chronological day with P10 below ``threshold``; return ``days_to_event``."""
    if test.empty or len(q10) == 0:
        return None
    work = test.copy()
    work["_q10"] = q10
    if "date" in work.columns:
        work = work.sort_values("date")
    below = work.loc[work["_q10"] < threshold]
    if below.empty:
        return None
    first = below.iloc[0]
    days = first.get("days_to_event")
    if pd.isna(days):
        return None
    return float(days)


def _half_window_metrics(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    days_to_event: np.ndarray,
    rul_max: int,
) -> tuple[float, float, float, float]:
    """Mean band width and coverage in early vs late halves of the test window."""
    split = rul_max / 2.0
    days = np.asarray(days_to_event, dtype=float)
    early = days > split
    late = ~early
    width = np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)

    def _mean_width(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(width[mask]))

    def _cov(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        yt = y_true[mask]
        lo = q10[mask]
        hi = q90[mask]
        inside = (yt >= lo) & (yt <= hi)
        return float(np.mean(inside))

    return (
        _mean_width(early),
        _mean_width(late),
        _cov(early),
        _cov(late),
    )


def score_fold(
    fold: FailureFold,
    *,
    rul_max: int = RUL_MAX_DAYS,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> FoldQuantileResult | None:
    """Train quantile models on ``fold.train`` and score ``fold.test``."""
    if not fold.trainable or fold.n_test == 0:
        return None
    bundle = fit_quantile_models(fold.train, rul_max=rul_max, lgbm_overrides=lgbm_overrides)
    q10, q50, q90 = predict_quantiles(bundle, fold.test)
    y_true = fold.test["rul"].to_numpy(dtype=float)
    metrics = score_predictions(y_true, q10, q50, q90)
    nominal = NOMINAL_COVERAGE
    coverage = metrics["pi_coverage"]
    gap = coverage - nominal if np.isfinite(coverage) else float("nan")
    mean_width = float(np.mean(q90 - q10))
    early_w, late_w, early_c, late_c = _half_window_metrics(
        y_true,
        q10,
        q50,
        q90,
        fold.test["days_to_event"].to_numpy(),
        rul_max,
    )
    grid, pdp_q50, pdp_ok = partial_dependence_anomaly_mean(bundle, fold.train)
    warn = warning_lead_time_days(
        fold.test,
        q10,
        threshold=maintenance_threshold,
    )
    return FoldQuantileResult(
        fold=fold,
        q10=q10,
        q50=q50,
        q90=q90,
        metrics=metrics,
        nominal_coverage=nominal,
        coverage_gap=gap,
        mean_band_width=mean_width,
        early_mean_band_width=early_w,
        late_mean_band_width=late_w,
        early_coverage=early_c,
        late_coverage=late_c,
        pdp_grid=grid,
        pdp_q50=pdp_q50,
        pdp_monotone_ok=pdp_ok,
        warning_lead_days=warn,
        monotone_applied=bundle.monotone_applied,
    )


def evaluate_quantile_model(
    frame: pd.DataFrame,
    *,
    rul_max: int = RUL_MAX_DAYS,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FoldQuantileResult]]:
    """LOFO quantile evaluation: metrics, coverage detail, warnings, fold results."""
    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    fold_results: list[FoldQuantileResult] = []

    for fold in leave_one_failure_out(frame, rul_max=rul_max):
        scored = score_fold(
            fold,
            rul_max=rul_max,
            maintenance_threshold=maintenance_threshold,
            lgbm_overrides=lgbm_overrides,
        )
        if scored is None:
            nan_metrics = {name: float("nan") for name in METRIC_COLUMNS}
            metric_rows.append(
                _metric_row(fold, nan_metrics, monotone_applied=False, trainable=fold.trainable)
            )
            continue
        fold_results.append(scored)
        metric_rows.append(
            _metric_row(
                fold,
                scored.metrics,
                monotone_applied=scored.monotone_applied,
                trainable=True,
                pdp_monotone_ok=scored.pdp_monotone_ok,
            )
        )
        coverage_rows.append(
            {
                "event_id": fold.event_id,
                "turbine_id": fold.turbine_id,
                "pi_coverage": scored.metrics["pi_coverage"],
                "nominal_coverage": scored.nominal_coverage,
                "coverage_gap": scored.coverage_gap,
                "mean_band_width": scored.mean_band_width,
                "early_mean_band_width": scored.early_mean_band_width,
                "late_mean_band_width": scored.late_mean_band_width,
                "early_coverage": scored.early_coverage,
                "late_coverage": scored.late_coverage,
                "pdp_monotone_ok": scored.pdp_monotone_ok,
            }
        )
        warning_rows.append(
            {
                "event_id": fold.event_id,
                "turbine_id": fold.turbine_id,
                "maintenance_threshold_days": maintenance_threshold,
                "warning_lead_days": scored.warning_lead_days,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    warnings_df = pd.DataFrame(warning_rows)
    return metrics_df, coverage_df, warnings_df, fold_results


def pdp_tables(fold_results: list[FoldQuantileResult]) -> pd.DataFrame:
    """Long table of partial-dependence grids per fold."""
    rows: list[dict[str, object]] = []
    for result in fold_results:
        for x_val, y_val in zip(result.pdp_grid, result.pdp_q50, strict=True):
            rows.append(
                {
                    "event_id": result.fold.event_id,
                    "turbine_id": result.fold.turbine_id,
                    "anomaly_score_mean": float(x_val),
                    "pdp_q50": float(y_val),
                    "pdp_monotone_ok": result.pdp_monotone_ok,
                }
            )
    return pd.DataFrame(rows)


def _metric_row(
    fold: FailureFold,
    metrics: dict[str, float],
    *,
    monotone_applied: bool,
    trainable: bool,
    pdp_monotone_ok: bool | None = None,
) -> dict[str, object]:
    return {
        "event_id": fold.event_id,
        "turbine_id": fold.turbine_id,
        "baseline": "quantile_lgbm",
        "n_train": fold.n_train,
        "n_test": fold.n_test,
        "trainable": trainable,
        **metrics,
        "interval_degenerate": False,
        "monotone_applied": monotone_applied,
        "pdp_monotone_ok": pdp_monotone_ok,
        "note": "LightGBM quantile at 0.1/0.5/0.9 on daily features.",
    }


__all__ = [
    "ANOMALY_MONOTONE_COLUMNS",
    "DEFAULT_MAINTENANCE_THRESHOLD_DAYS",
    "DEFAULT_LGBM_PARAMS",
    "FoldQuantileResult",
    "NOMINAL_COVERAGE",
    "QuantileModelBundle",
    "evaluate_quantile_model",
    "feature_matrix",
    "fit_quantile_models",
    "monotone_constraint_vector",
    "partial_dependence_anomaly_mean",
    "pdp_tables",
    "predict_quantiles",
    "score_fold",
    "warning_lead_time_days",
]
