"""Conformalized quantile regression (CQR) on gearbox RUL quantile models.

Calibration uses a time-ordered, failure-disjoint holdout from turbines other
than the LOFO test turbine. See docs/RUL_DESIGN.md for coverage caveats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rul.evaluate import FailureFold, leave_one_failure_out, pi_coverage
from rul.labels import RUL_MAX_DAYS
from rul.quantile import (
    QuantileModelBundle,
    feature_matrix,
    fit_quantile_models,
    predict_quantile_triple,
)

CONFIDENCE_80 = 0.80
CONFIDENCE_90 = 0.90
CAL_FRAC = 0.2
MIN_CAL_ROWS = 10
GAP_DAYS = 1
HALF_WINDOW_DAYS = 45

CONFORMAL_QUANTILES: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95)


def _cqr_quantile_levels(confidence_level: float) -> tuple[float, float, float]:
    lower = round((1.0 - confidence_level) / 2.0, 4)
    upper = round((1.0 + confidence_level) / 2.0, 4)
    return lower, upper, 0.5


def conformalize_quantile_triple(
    bundle: QuantileModelBundle,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    *,
    confidence_level: float,
    target_column: str = "rul",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply MAPIE CQR with prefit quantile models; return raw and conformal bounds.

    Returns ``q_lo, q_med, q_hi, cqr_lo, cqr_hi, y_pred`` on ``test``.
    Conformal bounds are unclipped for honest coverage scoring.
    """
    q_lo, q_hi, q_med = _cqr_quantile_levels(confidence_level)
    for q in (q_lo, q_hi, q_med):
        if q not in bundle.models:
            raise KeyError(f"bundle missing quantile {q} for confidence={confidence_level}")

    x_cal = feature_matrix(cal)
    y_cal = cal[target_column].astype(float).to_numpy()
    x_test = feature_matrix(test)

    raw_lo, raw_med, raw_hi = predict_quantile_triple(
        bundle, test, q_lo, q_med, q_hi, clip_and_order=False
    )

    from mapie.regression import ConformalizedQuantileRegressor

    estimators = [
        bundle.models[q_lo],
        bundle.models[q_hi],
        bundle.models[q_med],
    ]
    cqr = ConformalizedQuantileRegressor(
        estimator=estimators,
        confidence_level=confidence_level,
        prefit=True,
    )
    n_test = len(test)
    nan_out = (
        raw_lo,
        raw_med,
        raw_hi,
        np.full(n_test, np.nan),
        np.full(n_test, np.nan),
        raw_med,
    )
    try:
        cqr.conformalize(x_cal, y_cal)
        y_pred, y_intervals = cqr.predict_interval(x_test)
    except ValueError:
        return nan_out
    intervals = np.asarray(y_intervals)
    if intervals.ndim == 3:
        cqr_lo = intervals[:, 0, 0]
        cqr_hi = intervals[:, 1, 0]
    else:
        cqr_lo = intervals[:, 0]
        cqr_hi = intervals[:, 1]

    return raw_lo, raw_med, raw_hi, cqr_lo, cqr_hi, np.asarray(y_pred, dtype=float)


@dataclass(frozen=True)
class ConformalSplit:
    """Train and calibration subsets for one LOFO fold."""

    train_fit: pd.DataFrame
    calibration: pd.DataFrame
    n_eligible: int
    calibration_small: bool
    calibrated: bool


def split_conformal_calibration(
    labeled: pd.DataFrame,
    fold: FailureFold,
    *,
    cal_frac: float = CAL_FRAC,
    min_cal_rows: int = MIN_CAL_ROWS,
    gap_days: int = GAP_DAYS,
) -> ConformalSplit:
    """Time-ordered train/cal split on non-held-out turbines before the test window."""
    if fold.test.empty:
        empty = fold.test.iloc[0:0]
        return ConformalSplit(empty, empty, 0, True, False)

    test_start = pd.to_datetime(fold.test["date"], utc=True).min()
    cutoff = test_start - pd.Timedelta(days=gap_days)
    other = labeled.loc[labeled["turbine_id"] != fold.turbine_id].copy()
    eligible = other.loc[pd.to_datetime(other["date"], utc=True) < cutoff].copy()
    eligible = eligible.sort_values("date").reset_index(drop=True)

    if eligible.empty:
        empty = eligible.iloc[0:0]
        return ConformalSplit(empty, empty, 0, True, False)

    uncensored = eligible.loc[~eligible["censored"].astype(bool)].copy()
    if uncensored.empty:
        return ConformalSplit(eligible.iloc[0:0], uncensored, len(eligible), True, False)

    n_cal = min(len(uncensored), max(MIN_CAL_ROWS, int(len(uncensored) * cal_frac)))
    calibration = uncensored.iloc[-n_cal:].copy()
    cal_start = pd.to_datetime(calibration["date"], utc=True).min()
    train_fit = eligible.loc[pd.to_datetime(eligible["date"], utc=True) < cal_start].copy()

    if train_fit.empty:
        train_fit = eligible.loc[~eligible.index.isin(calibration.index)].copy()

    small = len(calibration) < min_cal_rows
    calibrated = len(train_fit) > 0 and len(calibration) > 0
    return ConformalSplit(
        train_fit=train_fit,
        calibration=calibration,
        n_eligible=len(eligible),
        calibration_small=small,
        calibrated=calibrated,
    )


@dataclass(frozen=True)
class FoldConformalResult:
    """Raw and conformal RUL intervals on one held-out failure window."""

    fold: FailureFold
    split: ConformalSplit
    days_to_event: np.ndarray
    y_true: np.ndarray
    raw_q10: np.ndarray
    raw_q50: np.ndarray
    raw_q90: np.ndarray
    cqr80_lo: np.ndarray
    cqr80_hi: np.ndarray
    cqr90_lo: np.ndarray
    cqr90_hi: np.ndarray
    cqr80_lo_disp: np.ndarray
    cqr80_hi_disp: np.ndarray
    cqr90_lo_disp: np.ndarray
    cqr90_hi_disp: np.ndarray
    width_narrows_near_failure: bool | None


def _clip_display(lo: np.ndarray, hi: np.ndarray, rul_max: int) -> tuple[np.ndarray, np.ndarray]:
    lo_c = np.clip(lo, 0.0, float(rul_max))
    hi_c = np.clip(hi, 0.0, float(rul_max))
    return lo_c, np.maximum(lo_c, hi_c)


def _width_narrows(days: np.ndarray, width: np.ndarray) -> bool | None:
    days = np.asarray(days, dtype=float)
    width = np.asarray(width, dtype=float)
    if days.size == 0:
        return None
    early = days > HALF_WINDOW_DAYS
    late = days <= HALF_WINDOW_DAYS
    if not np.any(early) or not np.any(late):
        return None
    return float(np.mean(width[late])) < float(np.mean(width[early]))


def score_fold_conformal(
    labeled: pd.DataFrame,
    fold: FailureFold,
    *,
    rul_max: int = RUL_MAX_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> FoldConformalResult | None:
    """Fit quantiles on a time-ordered train block, CQR-calibrate, score the test window."""
    if fold.n_test == 0:
        return None
    split = split_conformal_calibration(labeled, fold)
    nan = np.full(fold.n_test, np.nan)
    y_true = fold.test["rul"].to_numpy(dtype=float)
    days = fold.test["days_to_event"].to_numpy(dtype=float)

    if not split.calibrated or split.train_fit.empty:
        return FoldConformalResult(
            fold=fold,
            split=split,
            days_to_event=days,
            y_true=y_true,
            raw_q10=nan,
            raw_q50=nan,
            raw_q90=nan,
            cqr80_lo=nan,
            cqr80_hi=nan,
            cqr90_lo=nan,
            cqr90_hi=nan,
            cqr80_lo_disp=nan,
            cqr80_hi_disp=nan,
            cqr90_lo_disp=nan,
            cqr90_hi_disp=nan,
            width_narrows_near_failure=None,
        )

    bundle = fit_quantile_models(
        split.train_fit,
        quantiles=CONFORMAL_QUANTILES,
        rul_max=rul_max,
        lgbm_overrides=lgbm_overrides,
    )
    raw10, raw50, raw90 = predict_quantile_triple(
        bundle, fold.test, 0.1, 0.5, 0.9, clip_and_order=False
    )
    c80_lo = c80_hi = c90_lo = c90_hi = nan.copy()
    if len(split.calibration) >= MIN_CAL_ROWS:
        _, _, _, c80_lo, c80_hi, _ = conformalize_quantile_triple(
            bundle,
            split.calibration,
            fold.test,
            confidence_level=CONFIDENCE_80,
        )
        _, _, _, c90_lo, c90_hi, _ = conformalize_quantile_triple(
            bundle,
            split.calibration,
            fold.test,
            confidence_level=CONFIDENCE_90,
        )
    d80_lo, d80_hi = _clip_display(c80_lo, c80_hi, rul_max)
    d90_lo, d90_hi = _clip_display(c90_lo, c90_hi, rul_max)
    width80 = c80_hi - c80_lo
    return FoldConformalResult(
        fold=fold,
        split=split,
        days_to_event=days,
        y_true=y_true,
        raw_q10=raw10,
        raw_q50=raw50,
        raw_q90=raw90,
        cqr80_lo=c80_lo,
        cqr80_hi=c80_hi,
        cqr90_lo=c90_lo,
        cqr90_hi=c90_hi,
        cqr80_lo_disp=d80_lo,
        cqr80_hi_disp=d80_hi,
        cqr90_lo_disp=d90_lo,
        cqr90_hi_disp=d90_hi,
        width_narrows_near_failure=_width_narrows(days, width80),
    )


def _coverage_row(
    event_id: str,
    turbine_id: str,
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    nominal: float,
    *,
    prefix: str,
) -> dict[str, object]:
    cov = pi_coverage(y_true, lo, hi) if y_true.size else float("nan")
    width = float(np.mean(hi - lo)) if y_true.size else float("nan")
    return {
        "event_id": event_id,
        "turbine_id": turbine_id,
        f"{prefix}_coverage": cov,
        f"{prefix}_nominal": nominal,
        f"{prefix}_coverage_gap": cov - nominal if np.isfinite(cov) else float("nan"),
        f"{prefix}_mean_width": width,
    }


def _early_late_width(
    days: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rul_max: int,
) -> tuple[float, float]:
    split = rul_max / 2.0
    width = hi - lo
    early = days > split
    late = days <= split

    def _mean(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(width[mask]))

    return _mean(early), _mean(late)


def evaluate_conformal_rul(
    frame: pd.DataFrame,
    *,
    rul_max: int = RUL_MAX_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[FoldConformalResult]]:
    """LOFO conformal evaluation: per-event and pooled coverage, width vs horizon."""
    coverage_rows: list[dict[str, object]] = []
    width_rows: list[dict[str, object]] = []
    fold_results: list[FoldConformalResult] = []

    pool: dict[str, list[np.ndarray]] = {
        "y": [],
        "raw10": [],
        "raw90": [],
        "c80_lo": [],
        "c80_hi": [],
        "c90_lo": [],
        "c90_hi": [],
        "days": [],
    }

    for fold in leave_one_failure_out(frame, rul_max=rul_max):
        result = score_fold_conformal(
            frame, fold, rul_max=rul_max, lgbm_overrides=lgbm_overrides
        )
        if result is None:
            continue
        fold_results.append(result)
        if not np.isfinite(result.raw_q10).any():
            nan_row: dict[str, object] = {
                "event_id": fold.event_id,
                "turbine_id": fold.turbine_id,
                "n_cal": len(result.split.calibration),
                "n_train_fit": len(result.split.train_fit),
                "calibrated": result.split.calibrated,
                "calibration_small": result.split.calibration_small,
                "width_narrows_near_failure": result.width_narrows_near_failure,
            }
            for prefix, nominal in (("raw80", CONFIDENCE_80), ("cqr80", CONFIDENCE_80), ("cqr90", CONFIDENCE_90)):
                nan_row[f"{prefix}_coverage"] = float("nan")
                nan_row[f"{prefix}_nominal"] = nominal
                nan_row[f"{prefix}_coverage_gap"] = float("nan")
                nan_row[f"{prefix}_mean_width"] = float("nan")
            nan_row["cqr80_early_mean_width"] = float("nan")
            nan_row["cqr80_late_mean_width"] = float("nan")
            coverage_rows.append(nan_row)
            continue

        y = result.y_true
        row: dict[str, object] = {
            "event_id": fold.event_id,
            "turbine_id": fold.turbine_id,
            "n_cal": len(result.split.calibration),
            "n_train_fit": len(result.split.train_fit),
            "calibrated": result.split.calibrated,
            "calibration_small": result.split.calibration_small,
            "width_narrows_near_failure": result.width_narrows_near_failure,
        }
        row.update(
            _coverage_row(
                fold.event_id,
                fold.turbine_id,
                y,
                result.raw_q10,
                result.raw_q90,
                CONFIDENCE_80,
                prefix="raw80",
            )
        )
        row.update(
            _coverage_row(
                fold.event_id,
                fold.turbine_id,
                y,
                result.cqr80_lo,
                result.cqr80_hi,
                CONFIDENCE_80,
                prefix="cqr80",
            )
        )
        row.update(
            _coverage_row(
                fold.event_id,
                fold.turbine_id,
                y,
                result.cqr90_lo,
                result.cqr90_hi,
                CONFIDENCE_90,
                prefix="cqr90",
            )
        )
        early_w, late_w = _early_late_width(
            result.days_to_event, result.cqr80_lo, result.cqr80_hi, rul_max
        )
        row["cqr80_early_mean_width"] = early_w
        row["cqr80_late_mean_width"] = late_w
        coverage_rows.append(row)

        for d, w in zip(result.days_to_event, result.cqr80_hi - result.cqr80_lo, strict=True):
            width_rows.append(
                {
                    "event_id": fold.event_id,
                    "turbine_id": fold.turbine_id,
                    "days_to_event": float(d),
                    "cqr80_width": float(w),
                }
            )

        pool["y"].append(y)
        pool["raw10"].append(result.raw_q10)
        pool["raw90"].append(result.raw_q90)
        pool["c80_lo"].append(result.cqr80_lo)
        pool["c80_hi"].append(result.cqr80_hi)
        pool["c90_lo"].append(result.cqr90_lo)
        pool["c90_hi"].append(result.cqr90_hi)
        pool["days"].append(result.days_to_event)

    if pool["y"]:
        y_all = np.concatenate(pool["y"])
        pooled_row: dict[str, object] = {
            "event_id": "pooled",
            "turbine_id": "pooled",
            "n_cal": float("nan"),
            "n_train_fit": float("nan"),
            "calibrated": True,
            "calibration_small": False,
            "width_narrows_near_failure": _width_narrows(
                np.concatenate(pool["days"]),
                np.concatenate(pool["c80_hi"]) - np.concatenate(pool["c80_lo"]),
            ),
        }
        pooled_row.update(
            _coverage_row(
                "pooled",
                "pooled",
                y_all,
                np.concatenate(pool["raw10"]),
                np.concatenate(pool["raw90"]),
                CONFIDENCE_80,
                prefix="raw80",
            )
        )
        pooled_row.update(
            _coverage_row(
                "pooled",
                "pooled",
                y_all,
                np.concatenate(pool["c80_lo"]),
                np.concatenate(pool["c80_hi"]),
                CONFIDENCE_80,
                prefix="cqr80",
            )
        )
        pooled_row.update(
            _coverage_row(
                "pooled",
                "pooled",
                y_all,
                np.concatenate(pool["c90_lo"]),
                np.concatenate(pool["c90_hi"]),
                CONFIDENCE_90,
                prefix="cqr90",
            )
        )
        early_w, late_w = _early_late_width(
            np.concatenate(pool["days"]),
            np.concatenate(pool["c80_lo"]),
            np.concatenate(pool["c80_hi"]),
            rul_max,
        )
        pooled_row["cqr80_early_mean_width"] = early_w
        pooled_row["cqr80_late_mean_width"] = late_w
        coverage_rows.append(pooled_row)

    return pd.DataFrame(coverage_rows), pd.DataFrame(width_rows), fold_results


def predict_conformal_rul_for_frame(
    labeled: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    rul_max: int = RUL_MAX_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Production-style CQR on ``predict`` using all labeled rows before each day.

    Uses one global time-ordered split on non-censored tail for calibration and
    earlier rows (all turbines) for quantile fit. For deployment when LOFO is not
    run per row.
    """
    work = labeled.sort_values("date").reset_index(drop=True)
    uncensored = work.loc[~work["censored"].astype(bool)]
    if uncensored.empty or len(work) < 10:
        out = predict.copy()
        for col in (
            "rul_p10",
            "rul_p50",
            "rul_p90",
            "cqr80_lo",
            "cqr80_hi",
            "cqr90_lo",
            "cqr90_hi",
        ):
            out[col] = np.nan
        return out

    n_cal = max(MIN_CAL_ROWS, int(len(uncensored) * CAL_FRAC))
    cal = uncensored.iloc[-n_cal:]
    cal_start = cal["date"].min()
    train_fit = work.loc[work["date"] < cal_start]
    if train_fit.empty:
        train_fit = work.loc[~work.index.isin(cal.index)]

    bundle = fit_quantile_models(
        train_fit,
        quantiles=CONFORMAL_QUANTILES,
        rul_max=rul_max,
        lgbm_overrides=lgbm_overrides,
    )
    raw10, raw50, raw90 = predict_quantile_triple(
        bundle, predict, 0.1, 0.5, 0.9, clip_and_order=True
    )
    _, _, _, c80_lo, c80_hi, _ = conformalize_quantile_triple(
        bundle, cal, predict, confidence_level=CONFIDENCE_80
    )
    _, _, _, c90_lo, c90_hi, _ = conformalize_quantile_triple(
        bundle, cal, predict, confidence_level=CONFIDENCE_90
    )
    d80_lo, d80_hi = _clip_display(c80_lo, c80_hi, rul_max)
    d90_lo, d90_hi = _clip_display(c90_lo, c90_hi, rul_max)
    out = predict.copy()
    out["rul_p10"] = raw10
    out["rul_p50"] = raw50
    out["rul_p90"] = raw90
    out["cqr80_lo"] = d80_lo
    out["cqr80_hi"] = d80_hi
    out["cqr90_lo"] = d90_lo
    out["cqr90_hi"] = d90_hi
    return out


__all__ = [
    "CONFIDENCE_80",
    "CONFIDENCE_90",
    "FoldConformalResult",
    "ConformalSplit",
    "conformalize_quantile_triple",
    "evaluate_conformal_rul",
    "predict_conformal_rul_for_frame",
    "score_fold_conformal",
    "split_conformal_calibration",
]
