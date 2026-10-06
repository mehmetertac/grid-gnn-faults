"""Leave-one-failure-out scoring for capped gearbox RUL.

Tuesday and Wednesday call this module. Both baselines emit a point
prediction copied to q10, q50, and q90, so the interval is degenerate until
a quantile model exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rul.labels import RUL_MAX_DAYS

METRIC_COLUMNS: tuple[str, ...] = (
    "mae",
    "pinball_0.1",
    "pinball_0.5",
    "pinball_0.9",
    "pi_coverage",
)

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "turbine_id",
    "days_to_event",
    "rul",
    "censored",
    "anomaly_score_mean",
)

CONSTANT_NOTE = (
    "Degenerate interval at RUL_max. Coverage on a decreasing window counts "
    "only the days whose true RUL equals the cap."
)
LINEAR_NOTE = (
    "OLS point prediction from anomaly_score_mean, copied to q10, q50, and q90."
)


@dataclass(frozen=True)
class FailureFold:
    """One held-out gearbox event and the rows used to score it."""

    event_id: str
    turbine_id: str
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def n_train(self) -> int:
        return int(len(self.train))

    @property
    def n_test(self) -> int:
        return int(len(self.test))

    @property
    def trainable(self) -> bool:
        return self.n_train > 0


def leave_one_failure_out(
    frame: pd.DataFrame,
    rul_max: int = RUL_MAX_DAYS,
) -> list[FailureFold]:
    """Hold out each gearbox event's run-to-failure window.

    Test rows are that event on that turbine with ``0 < days_to_event <= rul_max``.
    Training rows are every other turbine. The held-out turbine is absent, so
    a second event on the same machine cannot leak.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Labeled frame is missing columns: {missing}")

    events = (
        frame.loc[frame["event_id"].notna(), ["event_id", "turbine_id"]]
        .drop_duplicates()
        .sort_values(["turbine_id", "event_id"])
    )
    folds: list[FailureFold] = []
    days = pd.to_numeric(frame["days_to_event"], errors="coerce")
    for event_id, turbine_id in events.itertuples(index=False):
        in_window = frame["event_id"].eq(event_id) & days.gt(0) & days.le(rul_max)
        test = frame.loc[in_window].copy()
        train = frame.loc[frame["turbine_id"] != turbine_id].copy()
        folds.append(
            FailureFold(
                event_id=str(event_id),
                turbine_id=str(turbine_id),
                train=train,
                test=test,
            )
        )
    return folds


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error of the median prediction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball loss at quantile ``q``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be between 0 and 1")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    error = y_true - y_pred
    loss = np.where(error >= 0.0, q * error, (1.0 - q) * (-error))
    return float(np.mean(loss))


def pi_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true RUL values inside [lower, upper]."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if y_true.size == 0:
        return float("nan")
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def score_predictions(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> dict[str, float]:
    """MAE on q50, pinball at 0.1/0.5/0.9, and interval coverage."""
    return {
        "mae": mae(y_true, q50),
        "pinball_0.1": pinball_loss(y_true, q10, 0.1),
        "pinball_0.5": pinball_loss(y_true, q50, 0.5),
        "pinball_0.9": pinball_loss(y_true, q90, 0.9),
        "pi_coverage": pi_coverage(y_true, q10, q90),
    }


def constant_baseline(
    n: int,
    rul_max: float = RUL_MAX_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict ``rul_max`` at every quantile. The interval has zero width."""
    value = np.full(int(n), float(rul_max), dtype=float)
    return value.copy(), value.copy(), value.copy()


def fit_linear_anomaly_baseline(
    anomaly: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """OLS ``rul = a + b * anomaly`` on finite training rows only."""
    x = np.asarray(anomaly, dtype=float)
    target = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(target)
    x = x[mask]
    target = target[mask]
    if x.size == 0:
        raise ValueError("linear baseline has no finite training rows")
    x_mean = float(x.mean())
    y_mean = float(target.mean())
    variance = float(np.sum((x - x_mean) ** 2))
    if variance <= 0.0:
        return y_mean, 0.0
    slope = float(np.sum((x - x_mean) * (target - y_mean)) / variance)
    intercept = y_mean - slope * x_mean
    return intercept, slope


def predict_linear_anomaly(
    anomaly: np.ndarray,
    intercept: float,
    slope: float,
    rul_max: float = RUL_MAX_DAYS,
) -> np.ndarray:
    """Clip the linear anomaly map to ``[0, rul_max]``."""
    raw = intercept + slope * np.asarray(anomaly, dtype=float)
    return np.clip(raw, 0.0, float(rul_max))


def evaluate_baselines(
    frame: pd.DataFrame,
    rul_max: int = RUL_MAX_DAYS,
) -> pd.DataFrame:
    """Score the constant and linear baselines on every leave-one-failure-out fold.

    Censored rows stay in training. Metrics use only the held-out window.
    """
    rows: list[dict[str, object]] = []
    for fold in leave_one_failure_out(frame, rul_max=rul_max):
        rows.append(_score_constant(fold, rul_max))
        rows.append(_score_linear(fold, rul_max))
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


_RESULT_COLUMNS: list[str] = [
    "event_id",
    "turbine_id",
    "baseline",
    "n_train",
    "n_test",
    "trainable",
    *METRIC_COLUMNS,
    "interval_degenerate",
    "note",
]


def _score_constant(fold: FailureFold, rul_max: int) -> dict[str, object]:
    q10, q50, q90 = constant_baseline(fold.n_test, rul_max=rul_max)
    metrics = _metrics(fold.test["rul"].to_numpy(), q10, q50, q90)
    return _row(fold, "constant", metrics, note=CONSTANT_NOTE, degenerate=_is_degenerate(q10, q90))


def _score_linear(fold: FailureFold, rul_max: int) -> dict[str, object]:
    if not fold.trainable or fold.n_test == 0:
        metrics = _nan_metrics()
        return _row(fold, "linear_anomaly", metrics, note=LINEAR_NOTE, degenerate=True)
    try:
        intercept, slope = fit_linear_anomaly_baseline(
            fold.train["anomaly_score_mean"].to_numpy(),
            fold.train["rul"].to_numpy(),
        )
    except ValueError:
        return _row(fold, "linear_anomaly", _nan_metrics(), note=LINEAR_NOTE, degenerate=True)
    point = predict_linear_anomaly(
        fold.test["anomaly_score_mean"].to_numpy(),
        intercept,
        slope,
        rul_max=rul_max,
    )
    metrics = _metrics(fold.test["rul"].to_numpy(), point, point, point)
    return _row(
        fold,
        "linear_anomaly",
        metrics,
        note=LINEAR_NOTE,
        degenerate=_is_degenerate(point, point),
    )


def _metrics(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> dict[str, float]:
    if np.asarray(y_true).size == 0:
        return _nan_metrics()
    return score_predictions(y_true, q10, q50, q90)


def _nan_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_COLUMNS}


def _is_degenerate(lower: np.ndarray, upper: np.ndarray) -> bool:
    if np.asarray(lower).size == 0:
        return True
    return bool(np.allclose(lower, upper))


def _row(
    fold: FailureFold,
    baseline: str,
    metrics: dict[str, float],
    note: str,
    degenerate: bool,
) -> dict[str, object]:
    return {
        "event_id": fold.event_id,
        "turbine_id": fold.turbine_id,
        "baseline": baseline,
        "n_train": fold.n_train,
        "n_test": fold.n_test,
        "trainable": fold.trainable,
        **metrics,
        "interval_degenerate": degenerate,
        "note": note,
    }


__all__ = [
    "CONSTANT_NOTE",
    "FailureFold",
    "LINEAR_NOTE",
    "constant_baseline",
    "evaluate_baselines",
    "fit_linear_anomaly_baseline",
    "leave_one_failure_out",
    "mae",
    "pi_coverage",
    "pinball_loss",
    "predict_linear_anomaly",
    "score_predictions",
]
