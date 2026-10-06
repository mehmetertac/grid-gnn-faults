"""Daily RUL labels from Week 10 scored frames and EDP gearbox logs.

The thermal model, residual corrector, and isolation forest stay in
wind-digital-twin. This module only aggregates columns those pieces already
emit and joins them to gearbox events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RUL_MAX_DAYS = 90
SLOPE_WINDOW_DAYS = 7
NS_PER_DAY = 86_400 * 1_000_000_000

# Published EDP gearbox catalog from wind-digital-twin config.GEARBOX_FAILURES.
# summarize_events recounts whatever log is passed; a local CSV can differ.
CATALOGUED_GEARBOX_EVENTS: tuple[dict[str, str], ...] = (
    {
        "turbine_id": "T01",
        "timestamp": "2016-07-18T02:10:00+00:00",
        "remarks": "Gearbox pump damaged",
    },
    {
        "turbine_id": "T06",
        "timestamp": "2017-10-17T08:38:00+00:00",
        "remarks": "Gearbox bearings damaged",
    },
    {
        "turbine_id": "T09",
        "timestamp": "2016-10-11T08:06:00+00:00",
        "remarks": "Gearbox repaired",
    },
    {
        "turbine_id": "T09",
        "timestamp": "2017-10-18T08:32:00+00:00",
        "remarks": "Gearbox noise",
    },
)

SCORED_VALUE_COLUMNS: tuple[str, ...] = (
    "oil_residual",
    "bear_residual",
    "anomaly_score",
    "theta_oil",
    "theta_bear",
    "delta_bear_oil",
)

DAILY_FEATURE_COLUMNS: tuple[str, ...] = (
    "oil_deg_mean",
    "oil_deg_std",
    "bear_deg_mean",
    "bear_deg_std",
    "anomaly_score_mean",
    "anomaly_score_max",
    "oil_deg_slope_7d",
    "bear_deg_slope_7d",
    "theta_oil_mean",
    "theta_bear_mean",
    "delta_bear_oil_mean",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "event_id",
    "days_to_event",
    "rul",
    "censored",
)


def piecewise_rul(days_to_event: float, rul_max: int = RUL_MAX_DAYS) -> float:
    """C-MAPSS piecewise-linear cap: min(rul_max, days until the event)."""
    if days_to_event < 0:
        raise ValueError("days_to_event must be >= 0")
    return float(min(rul_max, days_to_event))


def catalog_failure_frame() -> pd.DataFrame:
    """Return the four gearbox events shipped in the Week 10 twin config."""
    return pd.DataFrame(list(CATALOGUED_GEARBOX_EVENTS))


def summarize_events(failures: pd.DataFrame) -> dict[str, object]:
    """Count gearbox log lines actually present in ``failures``."""
    events = normalize_failure_log(failures)
    records = events[["event_id", "turbine_id", "timestamp", "remarks"]].to_dict(
        orient="records"
    )
    return {
        "n_events": int(len(events)),
        "n_turbines": int(events["turbine_id"].nunique()) if len(events) else 0,
        "events": records,
    }


def normalize_failure_log(failures: pd.DataFrame) -> pd.DataFrame:
    """Keep ``Component == GEARBOX`` rows and assign a stable event id.

    Accepts Week 10 names (``Turbine_ID``, ``Timestamp``, ``Component``,
    ``Remarks``) or the lowercase equivalents. A frame with no component
    column is treated as already filtered.
    """
    if failures is None or failures.empty:
        return _empty_events()

    renamed = failures.rename(
        columns={
            "Turbine_ID": "turbine_id",
            "Timestamp": "timestamp",
            "Component": "component",
            "Remarks": "remarks",
        }
    )
    missing = [col for col in ("turbine_id", "timestamp") if col not in renamed.columns]
    if missing:
        raise ValueError(f"Failure log is missing columns: {missing}")

    events = renamed.copy()
    if "component" in events.columns:
        component = events["component"].astype(str).str.strip().str.upper()
        events = events.loc[component == "GEARBOX"].copy()
    if events.empty:
        return _empty_events()

    if "remarks" not in events.columns:
        events["remarks"] = ""
    events["turbine_id"] = events["turbine_id"].astype(str)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events["remarks"] = events["remarks"].fillna("").astype(str)
    events["event_date"] = events["timestamp"].dt.floor("D")
    events = events.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    events["event_id"] = [
        f"{row.turbine_id}|{row.event_date.strftime('%Y-%m-%d')}|{i}"
        for i, row in events.iterrows()
    ]
    return events[
        ["event_id", "turbine_id", "timestamp", "event_date", "remarks"]
    ].reset_index(drop=True)


def daily_from_scored_frame(
    scored: pd.DataFrame,
    turbine_id: str | None = None,
    slope_window: int = SLOPE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Aggregate a 10-minute scored frame to one row per turbine-day.

    Degradation is ``-residual``, matching Week 10 ``degradation_signal``
    (higher means hotter than the thermal expectation). The 7-day slope uses
    the trailing daily means. The first ``slope_window - 1`` days stay null.
    """
    frame = _prepare_scored_frame(scored, turbine_id=turbine_id)
    frame["oil_deg"] = -frame["oil_residual"].astype(float)
    frame["bear_deg"] = -frame["bear_residual"].astype(float)
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("D")

    daily = frame.groupby(["turbine_id", "date"], as_index=False).agg(
        oil_deg_mean=("oil_deg", "mean"),
        oil_deg_std=("oil_deg", "std"),
        bear_deg_mean=("bear_deg", "mean"),
        bear_deg_std=("bear_deg", "std"),
        anomaly_score_mean=("anomaly_score", "mean"),
        anomaly_score_max=("anomaly_score", "max"),
        theta_oil_mean=("theta_oil", "mean"),
        theta_bear_mean=("theta_bear", "mean"),
        delta_bear_oil_mean=("delta_bear_oil", "mean"),
    )
    return _add_slopes(daily, window=slope_window)


def attach_rul_labels(
    daily: pd.DataFrame,
    failures: pd.DataFrame,
    rul_max: int = RUL_MAX_DAYS,
) -> pd.DataFrame:
    """Attach capped RUL, censor flags, and the next gearbox event id.

    The calendar day of a log timestamp is dropped. After an event the clock
    resets toward the next gearbox event on that turbine. Days with no later
    gearbox event are right-censored at ``rul_max``.
    """
    if rul_max < 1:
        raise ValueError("rul_max must be >= 1")
    events = normalize_failure_log(failures)
    frame = _normalize_daily(daily)
    if frame.empty:
        return _empty_labeled(frame)

    pieces = [
        _label_one_turbine(group, events.loc[events["turbine_id"] == turbine], rul_max)
        for turbine, group in frame.groupby("turbine_id", sort=False)
    ]
    labeled = pd.concat(pieces, ignore_index=True)
    feature_cols = [col for col in labeled.columns if col not in ("turbine_id", "date", *LABEL_COLUMNS)]
    order = ["turbine_id", "date", *LABEL_COLUMNS, *feature_cols]
    return labeled[order]


def _prepare_scored_frame(
    scored: pd.DataFrame,
    turbine_id: str | None,
) -> pd.DataFrame:
    if isinstance(scored.index, pd.DatetimeIndex):
        frame = scored.reset_index()
        index_name = frame.columns[0]
        if index_name not in ("timestamp", "Timestamp"):
            frame = frame.rename(columns={index_name: "timestamp"})
    else:
        frame = scored.copy()

    frame = frame.rename(columns={"Turbine_ID": "turbine_id", "Timestamp": "timestamp"})
    if turbine_id is not None:
        frame["turbine_id"] = turbine_id
    missing = [col for col in ("turbine_id", "timestamp", *SCORED_VALUE_COLUMNS) if col not in frame.columns]
    if missing:
        raise ValueError(f"Scored frame is missing columns: {missing}")
    return frame


def _normalize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    if "date" not in frame.columns and isinstance(daily.index, pd.DatetimeIndex):
        frame = daily.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
    frame = frame.rename(columns={"Turbine_ID": "turbine_id", "Timestamp": "date"})
    missing = [col for col in ("turbine_id", "date") if col not in frame.columns]
    if missing:
        raise ValueError(f"Daily frame is missing columns: {missing}")
    frame["turbine_id"] = frame["turbine_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.floor("D")
    return frame.sort_values(["turbine_id", "date"]).reset_index(drop=True)


def _add_slopes(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in daily.groupby("turbine_id", sort=False):
        ordered = group.sort_values("date").copy()
        ordered["oil_deg_slope_7d"] = _rolling_slope(ordered["oil_deg_mean"], window)
        ordered["bear_deg_slope_7d"] = _rolling_slope(ordered["bear_deg_mean"], window)
        parts.append(ordered)
    if not parts:
        empty = daily.copy()
        empty["oil_deg_slope_7d"] = pd.Series(dtype=float)
        empty["bear_deg_slope_7d"] = pd.Series(dtype=float)
        return empty
    return pd.concat(parts, ignore_index=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def slope(values: np.ndarray) -> float:
        if values.size < 2 or not np.isfinite(values).all():
            return float("nan")
        x = np.arange(values.size, dtype=float)
        gradient = np.polyfit(x, values.astype(float), 1)[0]
        return float(gradient)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def _label_one_turbine(
    group: pd.DataFrame,
    events: pd.DataFrame,
    rul_max: int,
) -> pd.DataFrame:
    labeled = group.copy()
    if events.empty:
        return _mark_censored(labeled, rul_max)

    event_ns = _midnight_ns(events["event_date"])
    feature_ns = _midnight_ns(labeled["date"])
    keep = ~np.isin(feature_ns, event_ns)
    labeled = labeled.iloc[np.flatnonzero(keep)].copy()
    feature_ns = feature_ns[keep]
    if labeled.empty:
        return _mark_censored(labeled, rul_max)

    unique_ns = np.unique(event_ns)
    next_pos = np.searchsorted(unique_ns, feature_ns, side="right")
    has_next = next_pos < unique_ns.size
    next_ns = np.where(has_next, unique_ns[np.clip(next_pos, 0, unique_ns.size - 1)], -1)
    days = np.where(has_next, (next_ns - feature_ns) // NS_PER_DAY, 0)

    id_by_ns = {
        int(ns): event_id
        for ns, event_id in zip(event_ns, events["event_id"], strict=True)
    }
    event_ids: list[object] = []
    day_values: list[object] = []
    rul_values: list[float] = []
    censored: list[bool] = []
    for ok, day_count, stamp in zip(has_next, days, next_ns, strict=True):
        if not ok:
            event_ids.append(pd.NA)
            day_values.append(pd.NA)
            rul_values.append(float(rul_max))
            censored.append(True)
            continue
        event_ids.append(id_by_ns[int(stamp)])
        day_values.append(int(day_count))
        rul_values.append(piecewise_rul(int(day_count), rul_max=rul_max))
        censored.append(False)

    labeled["event_id"] = pd.Series(event_ids, index=labeled.index, dtype="object")
    labeled["days_to_event"] = pd.array(day_values, dtype="Int64")
    labeled["rul"] = rul_values
    labeled["censored"] = censored
    return labeled


def _mark_censored(group: pd.DataFrame, rul_max: int) -> pd.DataFrame:
    labeled = group.copy()
    labeled["event_id"] = pd.Series(pd.NA, index=labeled.index, dtype="object")
    labeled["days_to_event"] = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    labeled["rul"] = float(rul_max)
    labeled["censored"] = True
    return labeled


def _midnight_ns(values: pd.Series) -> np.ndarray:
    midnight = pd.to_datetime(values, utc=True).dt.floor("D")
    return midnight.astype("int64").to_numpy()


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["event_id", "turbine_id", "timestamp", "event_date", "remarks"]
    )


def _empty_labeled(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    for column in LABEL_COLUMNS:
        if column not in labeled.columns:
            labeled[column] = pd.Series(dtype="object")
    return labeled


__all__ = [
    "CATALOGUED_GEARBOX_EVENTS",
    "DAILY_FEATURE_COLUMNS",
    "RUL_MAX_DAYS",
    "attach_rul_labels",
    "catalog_failure_frame",
    "daily_from_scored_frame",
    "normalize_failure_log",
    "piecewise_rul",
    "summarize_events",
]
