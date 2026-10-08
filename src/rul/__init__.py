"""Gearbox remaining-useful-life labels and leave-one-failure-out scoring."""

from rul.evaluate import (
    constant_baseline,
    evaluate_baselines,
    fit_linear_anomaly_baseline,
    leave_one_failure_out,
    score_predictions,
)
from rul.labels import (
    RUL_MAX_DAYS,
    attach_rul_labels,
    daily_from_scored_frame,
    summarize_events,
)

__all__ = [
    "RUL_MAX_DAYS",
    "attach_rul_labels",
    "constant_baseline",
    "daily_from_scored_frame",
    "evaluate_baselines",
    "fit_linear_anomaly_baseline",
    "leave_one_failure_out",
    "score_predictions",
    "summarize_events",
]
