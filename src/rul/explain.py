"""TreeSHAP explanations for gearbox RUL quantile models (P10 and P50).

Aggregates daily feature attributions to maintenance-meaningful sensor names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from rul.evaluate import FailureFold, leave_one_failure_out
from rul.labels import DAILY_FEATURE_COLUMNS, RUL_MAX_DAYS
from rul.quantile import (
    QuantileModelBundle,
    feature_matrix,
    fit_quantile_models,
    load_quantile_bundle,
    save_quantile_bundle,
)

DEFAULT_HELD_OUT_TURBINE = "T06"
EXPLAIN_QUANTILES: tuple[float, ...] = (0.1, 0.5)

SENSOR_GEARBOX_OIL = "Gearbox oil temperature"
SENSOR_GEARBOX_BEARING = "Gearbox bearing temperature"
SENSOR_THETA_OIL = "Oil temperature per unit power"
SENSOR_THETA_BEAR = "Bearing temperature per unit power"
SENSOR_DELTA_BEAR_OIL = "Bearing-oil temperature difference"
SENSOR_ANOMALY = "Hybrid anomaly score"

SENSOR_ORDER: tuple[str, ...] = (
    SENSOR_GEARBOX_OIL,
    SENSOR_GEARBOX_BEARING,
    SENSOR_THETA_OIL,
    SENSOR_THETA_BEAR,
    SENSOR_DELTA_BEAR_OIL,
    SENSOR_ANOMALY,
)

FEATURE_TO_SENSOR: dict[str, str] = {
    "oil_deg_mean": SENSOR_GEARBOX_OIL,
    "oil_deg_std": SENSOR_GEARBOX_OIL,
    "oil_deg_slope_7d": SENSOR_GEARBOX_OIL,
    "bear_deg_mean": SENSOR_GEARBOX_BEARING,
    "bear_deg_std": SENSOR_GEARBOX_BEARING,
    "bear_deg_slope_7d": SENSOR_GEARBOX_BEARING,
    "theta_oil_mean": SENSOR_THETA_OIL,
    "theta_bear_mean": SENSOR_THETA_BEAR,
    "delta_bear_oil_mean": SENSOR_DELTA_BEAR_OIL,
    "anomaly_score_mean": SENSOR_ANOMALY,
    "anomaly_score_max": SENSOR_ANOMALY,
}

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "oil_deg_mean": "Oil residual (daily mean)",
    "oil_deg_std": "Oil residual (daily std)",
    "oil_deg_slope_7d": "Oil residual (7-day slope)",
    "bear_deg_mean": "Bearing residual (daily mean)",
    "bear_deg_std": "Bearing residual (daily std)",
    "bear_deg_slope_7d": "Bearing residual (7-day slope)",
    "theta_oil_mean": "Oil temp per unit power",
    "theta_bear_mean": "Bearing temp per unit power",
    "delta_bear_oil_mean": "Bearing-oil delta",
    "anomaly_score_mean": "Anomaly score (daily mean)",
    "anomaly_score_max": "Anomaly score (daily max)",
}

THERMAL_SENSORS: frozenset[str] = frozenset(
    {
        SENSOR_GEARBOX_OIL,
        SENSOR_GEARBOX_BEARING,
        SENSOR_DELTA_BEAR_OIL,
    }
)
POWER_NORMALIZED_SENSORS: frozenset[str] = frozenset(
    {SENSOR_THETA_OIL, SENSOR_THETA_BEAR}
)


@dataclass(frozen=True)
class ExplainContext:
    """One LOFO fold, fitted bundle, and rows used for global vs local SHAP."""

    fold: FailureFold
    bundle: QuantileModelBundle
    global_frame: pd.DataFrame
    local_frame: pd.DataFrame


def validate_sensor_map() -> None:
    """Every daily feature maps to exactly one known sensor."""
    missing = [c for c in DAILY_FEATURE_COLUMNS if c not in FEATURE_TO_SENSOR]
    if missing:
        raise ValueError(f"FEATURE_TO_SENSOR missing columns: {missing}")
    for col in DAILY_FEATURE_COLUMNS:
        sensor = FEATURE_TO_SENSOR[col]
        if sensor not in SENSOR_ORDER:
            raise ValueError(f"Unknown sensor for {col}: {sensor}")


def features_for_sensor(sensor: str) -> tuple[str, ...]:
    return tuple(c for c in DAILY_FEATURE_COLUMNS if FEATURE_TO_SENSOR[c] == sensor)


def select_explain_fold(
    frame: pd.DataFrame,
    *,
    rul_max: int = RUL_MAX_DAYS,
    event_id: str | None = None,
    turbine_id: str | None = None,
) -> FailureFold:
    """Pick the LOFO fold to explain (default T06 bearing event when present)."""
    folds = leave_one_failure_out(frame, rul_max=rul_max)
    if event_id is not None:
        for fold in folds:
            if fold.event_id == event_id:
                return fold
        raise ValueError(f"No LOFO fold for event_id={event_id!r}")
    if turbine_id is not None:
        for fold in folds:
            if fold.turbine_id == turbine_id and fold.trainable and fold.n_test > 0:
                return fold
        raise ValueError(f"No trainable LOFO fold for turbine_id={turbine_id!r}")
    for fold in folds:
        if fold.turbine_id == DEFAULT_HELD_OUT_TURBINE and fold.trainable and fold.n_test > 0:
            return fold
    for fold in folds:
        if fold.trainable and fold.n_test > 0:
            return fold
    raise ValueError("No trainable LOFO fold with test rows in labeled frame")


def fit_bundle_for_fold(
    fold: FailureFold,
    *,
    rul_max: int = RUL_MAX_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
) -> QuantileModelBundle:
    if not fold.trainable:
        raise ValueError(f"Fold {fold.event_id} has no training rows")
    return fit_quantile_models(
        fold.train,
        quantiles=(*EXPLAIN_QUANTILES, 0.9),
        rul_max=rul_max,
        lgbm_overrides=lgbm_overrides,
    )


def build_explain_context(
    frame: pd.DataFrame,
    bundle: QuantileModelBundle,
    fold: FailureFold,
) -> ExplainContext:
    return ExplainContext(
        fold=fold,
        bundle=bundle,
        global_frame=fold.train.copy(),
        local_frame=fold.test.copy(),
    )


def tree_shap_values(
    bundle: QuantileModelBundle,
    frame: pd.DataFrame,
    quantile: float,
) -> np.ndarray:
    """Feature-level SHAP matrix (n_rows, n_features)."""
    if quantile not in bundle.models:
        raise KeyError(f"bundle has no model for quantile {quantile}")
    x = feature_matrix(frame)
    explainer = shap.TreeExplainer(bundle.models[quantile])
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values, dtype=float)


def aggregate_shap_to_sensors(
    feature_shap: np.ndarray,
    feature_columns: tuple[str, ...] = DAILY_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Sum feature SHAP into sensor columns (additive)."""
    validate_sensor_map()
    n = feature_shap.shape[0]
    out = pd.DataFrame({s: np.zeros(n, dtype=float) for s in SENSOR_ORDER})
    for j, col in enumerate(feature_columns):
        sensor = FEATURE_TO_SENSOR[col]
        out[sensor] = out[sensor].to_numpy() + feature_shap[:, j]
    return out


def sensor_ranking_table(
    sensor_shap: pd.DataFrame,
    *,
    learner: str,
) -> pd.DataFrame:
    """Rank sensors by mean absolute SHAP for one learner."""
    rows: list[dict[str, object]] = []
    n_rows = len(sensor_shap)
    ranked: list[tuple[str, float]] = []
    for sensor in SENSOR_ORDER:
        col = sensor_shap[sensor].to_numpy(dtype=float)
        mean_abs = float(np.mean(np.abs(col)))
        ranked.append((sensor, mean_abs))
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (sensor, mean_abs) in enumerate(ranked, start=1):
        col = sensor_shap[sensor].to_numpy(dtype=float)
        rows.append(
            {
                "sensor": sensor,
                "learner": learner,
                "mean_shap": float(np.mean(col)),
                "mean_abs_shap": mean_abs,
                "rank": rank,
                "n_rows": n_rows,
                "features": ";".join(features_for_sensor(sensor)),
            }
        )
    return pd.DataFrame(rows)


def build_sensor_ranking_csv(
    bundle: QuantileModelBundle,
    global_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Global SHAP on training rows for P10 and P50."""
    parts: list[pd.DataFrame] = []
    for q, learner in ((0.1, "p10"), (0.5, "p50")):
        feat = tree_shap_values(bundle, global_frame, q)
        sensors = aggregate_shap_to_sensors(feat, bundle.feature_columns)
        parts.append(sensor_ranking_table(sensors, learner=learner))
    return pd.concat(parts, ignore_index=True)


def _display_feature_names(feature_columns: tuple[str, ...]) -> list[str]:
    return [FEATURE_DISPLAY_NAMES.get(c, c) for c in feature_columns]


def plot_global_shap(
    bundle: QuantileModelBundle,
    global_frame: pd.DataFrame,
    ranking: pd.DataFrame,
    path: Path | str,
) -> Path:
    """P50 beeswarm plus P10/P50 sensor mean-|SHAP| bars."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = feature_matrix(global_frame)
    x_display = x.copy()
    x_display.columns = _display_feature_names(bundle.feature_columns)
    feat_p50 = tree_shap_values(bundle, global_frame, 0.5)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.35)
    ax_swarm = fig.add_subplot(gs[0])
    plt.sca(ax_swarm)
    shap.summary_plot(
        feat_p50,
        x_display,
        show=False,
        plot_type="dot",
        max_display=len(bundle.feature_columns),
    )
    ax_swarm.set_title("Global TreeSHAP (P50) — training rows, held-out turbine excluded")

    ax_bar = fig.add_subplot(gs[1])
    y_pos = np.arange(len(SENSOR_ORDER))
    width = 0.35
    p10 = ranking.loc[ranking["learner"] == "p10"].set_index("sensor")
    p50 = ranking.loc[ranking["learner"] == "p50"].set_index("sensor")
    p10_vals = [float(p10.loc[s, "mean_abs_shap"]) if s in p10.index else 0.0 for s in SENSOR_ORDER]
    p50_vals = [float(p50.loc[s, "mean_abs_shap"]) if s in p50.index else 0.0 for s in SENSOR_ORDER]
    ax_bar.barh(y_pos - width / 2, p10_vals, width, label="P10", color="C1", alpha=0.85)
    ax_bar.barh(y_pos + width / 2, p50_vals, width, label="P50", color="C0", alpha=0.85)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(SENSOR_ORDER, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean |SHAP| (days RUL)")
    ax_bar.set_title("Sensor-level importance (aggregated features)")
    ax_bar.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_local_sensor_timeline(
    bundle: QuantileModelBundle,
    local_frame: pd.DataFrame,
    *,
    event_id: str,
    turbine_id: str,
    path: Path | str,
    top_n: int = 6,
) -> Path:
    """Heatmap of signed sensor SHAP vs days to event on the held-out window (P50)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if local_frame.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(f"{turbine_id} — no local rows")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    feat = tree_shap_values(bundle, local_frame, 0.5)
    sensor_df = aggregate_shap_to_sensors(feat, bundle.feature_columns)
    days = pd.to_numeric(local_frame["days_to_event"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(days)
    days = days[order]
    sensor_df = sensor_df.iloc[order]

    mean_abs = sensor_df.abs().mean(axis=0)
    top_sensors = mean_abs.sort_values(ascending=False).head(top_n).index.tolist()
    matrix = sensor_df[top_sensors].to_numpy(dtype=float).T

    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[days.min(), days.max(), -0.5, len(top_sensors) - 0.5],
    )
    ax.set_yticks(range(len(top_sensors)))
    ax.set_yticklabels(top_sensors, fontsize=8)
    ax.set_xlabel("Days to event")
    ax.set_ylabel("Sensor")
    ax.set_title(f"Local P50 SHAP — {turbine_id} ({event_id})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("SHAP (days RUL; negative pulls prediction down)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def first_negative_shap_day(
    local_frame: pd.DataFrame,
    sensor_shap: pd.DataFrame,
    sensors: frozenset[str],
) -> float | None:
    """Earliest days_to_event where any sensor in the group has negative mean SHAP."""
    days = pd.to_numeric(local_frame["days_to_event"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(-days)
    cols = [s for s in sensor_shap.columns if s in sensors]
    if not cols:
        return None
    group = sensor_shap[cols].mean(axis=1).to_numpy(dtype=float)[order]
    days_sorted = days[order]
    for d, val in zip(days_sorted, group, strict=True):
        if np.isfinite(d) and np.isfinite(val) and val < 0:
            return float(d)
    return None


def field_intuition_paragraph(
    local_frame: pd.DataFrame,
    bundle: QuantileModelBundle,
) -> str:
    """Two or three sentences comparing thermal vs power-normalized SHAP timing."""
    if local_frame.empty:
        return (
            "No held-out run-to-failure rows were available for a local SHAP timeline, "
            "so thermal-vs-electrical timing could not be checked on this frame."
        )
    feat = tree_shap_values(bundle, local_frame, 0.5)
    sensor_df = aggregate_shap_to_sensors(feat, bundle.feature_columns)
    thermal_day = first_negative_shap_day(local_frame, sensor_df, THERMAL_SENSORS)
    power_day = first_negative_shap_day(local_frame, sensor_df, POWER_NORMALIZED_SENSORS)
    anomaly_day = first_negative_shap_day(
        local_frame, sensor_df, frozenset({SENSOR_ANOMALY})
    )

    def fmt(day: float | None) -> str:
        return f"{day:.0f} days before failure" if day is not None else "not in the 90-day window"

    sentences = [
        (
            f"On the held-out window, grouped thermal residuals and bearing-oil delta first "
            f"pull P50 RUL down at {fmt(thermal_day)}, while temperature-per-unit-power "
            f"features do so at {fmt(power_day)}."
        ),
    ]
    if thermal_day is not None and power_day is not None:
        if thermal_day > power_day:
            sentences.append(
                "That ordering disagrees with a simple gearbox story where bulk thermal "
                "rise should lead power-normalized coupling; the model may be leaning on "
                "operating-point normalization before raw residual drift."
            )
        elif thermal_day < power_day:
            sentences.append(
                "That matches field intuition for a gearbox: bulk thermal degradation "
                "registers before the power-normalized channels move the median RUL."
            )
        else:
            sentences.append(
                "Thermal and power-normalized groups turn negative on the same day, "
                "so the signature is simultaneous rather than clearly leading/lagging."
            )
    elif anomaly_day is not None and (thermal_day is None or anomaly_day > thermal_day):
        sentences.append(
            f"The hybrid anomaly score turns negative at {fmt(anomaly_day)}, which can "
            "dominate before or instead of individual thermocouple-derived features."
        )
    else:
        sentences.append(
            "Neither group shows a sustained negative P50 SHAP in this window; "
            "check that the labeled frame has enough pre-failure daily rows."
        )
    return " ".join(sentences[:3])


def run_explain(
    frame: pd.DataFrame,
    out_dir: Path | str,
    *,
    bundle: QuantileModelBundle | None = None,
    model_path: Path | str | None = None,
    event_id: str | None = None,
    turbine_id: str | None = None,
    rul_max: int = RUL_MAX_DAYS,
    lgbm_overrides: dict[str, Any] | None = None,
    write_field_notes: Path | str | None = None,
) -> dict[str, Path]:
    """Fit or load bundle, write figures and ``shap_sensor_ranking.csv``."""
    validate_sensor_map()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fold = select_explain_fold(
        frame,
        rul_max=rul_max,
        event_id=event_id,
        turbine_id=turbine_id,
    )
    if bundle is None:
        if model_path is not None:
            bundle = load_quantile_bundle(model_path)
        else:
            bundle = fit_bundle_for_fold(fold, rul_max=rul_max, lgbm_overrides=lgbm_overrides)

    ctx = build_explain_context(frame, bundle, fold)
    ranking = build_sensor_ranking_csv(ctx.bundle, ctx.global_frame)
    csv_path = root / "shap_sensor_ranking.csv"
    ranking.to_csv(csv_path, index=False)

    global_png = plot_global_shap(
        ctx.bundle,
        ctx.global_frame,
        ranking,
        figures / "shap_global.png",
    )
    local_png = plot_local_sensor_timeline(
        ctx.bundle,
        ctx.local_frame,
        event_id=fold.event_id,
        turbine_id=fold.turbine_id,
        path=figures / "shap_local_timeline.png",
    )

    if write_field_notes is not None:
        notes_path = Path(write_field_notes)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        paragraph = field_intuition_paragraph(ctx.local_frame, ctx.bundle)
        existing = ""
        if notes_path.is_file():
            existing = notes_path.read_text(encoding="utf-8")
        marker = "## Field intuition (held-out run)"
        block = f"{marker}\n\n{paragraph}\n"
        if marker in existing:
            head, _, _tail = existing.partition(marker)
            existing = head.rstrip() + "\n\n" + block
            notes_path.write_text(existing, encoding="utf-8")
        else:
            notes_path.write_text(existing.rstrip() + "\n\n" + block if existing else block, encoding="utf-8")

    return {
        "ranking_csv": csv_path,
        "shap_global": global_png,
        "shap_local_timeline": local_png,
        "model": Path(model_path) if model_path else root / "rul_quantile_bundle.joblib",
    }


__all__ = [
    "FEATURE_TO_SENSOR",
    "SENSOR_ORDER",
    "ExplainContext",
    "aggregate_shap_to_sensors",
    "build_explain_context",
    "build_sensor_ranking_csv",
    "field_intuition_paragraph",
    "fit_bundle_for_fold",
    "plot_global_shap",
    "plot_local_sensor_timeline",
    "run_explain",
    "select_explain_fold",
    "sensor_ranking_table",
    "tree_shap_values",
    "validate_sensor_map",
]
