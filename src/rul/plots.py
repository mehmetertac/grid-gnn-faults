"""Headline RUL figures: true RUL vs P50 with P10–P90 and conformal bands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.quantile import DEFAULT_MAINTENANCE_THRESHOLD_DAYS, FoldQuantileResult

if TYPE_CHECKING:
    from rul.conformal import FoldConformalResult


def plot_rul_band(
    result: FoldQuantileResult,
    path: Path | str,
    *,
    conformal: "FoldConformalResult | None" = None,
    last_n_days: int = 90,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
) -> Path:
    """One PNG per held-out event: raw quantile band plus optional CQR overlays."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    test = result.fold.test.copy()
    if test.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f"{result.fold.turbine_id} — no test rows")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    days = pd.to_numeric(test["days_to_event"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(days)
    days = days[order]
    y_true = test["rul"].to_numpy(dtype=float)[order]
    q10 = result.q10[order]
    q50 = result.q50[order]
    q90 = result.q90[order]

    c80_lo = c80_hi = c90_lo = c90_hi = None
    if conformal is not None and np.isfinite(conformal.raw_q50).any():
        c80_lo = conformal.cqr80_lo_disp[order]
        c80_hi = conformal.cqr80_hi_disp[order]
        c90_lo = conformal.cqr90_lo_disp[order]
        c90_hi = conformal.cqr90_hi_disp[order]

    mask = days <= float(last_n_days)
    if not np.any(mask):
        mask = np.ones_like(days, dtype=bool)
    x = days[mask]
    y_plot = y_true[mask]
    q10_plot = q10[mask]
    q50_plot = q50[mask]
    q90_plot = q90[mask]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, q10_plot, q90_plot, alpha=0.2, color="C0", label="Raw P10–P90")
    if c80_lo is not None:
        ax.fill_between(
            x,
            c80_lo[mask],
            c80_hi[mask],
            alpha=0.35,
            color="C2",
            label="CQR 80%",
        )
        ax.fill_between(
            x,
            c90_lo[mask],
            c90_hi[mask],
            alpha=0.15,
            color="C1",
            label="CQR 90%",
        )
    ax.plot(x, y_plot, "k-", linewidth=1.5, label="True RUL")
    ax.plot(x, q50_plot, color="C0", linewidth=1.5, label="P50")
    ax.axhline(
        maintenance_threshold,
        color="C3",
        linestyle="--",
        linewidth=1,
        label=f"Planning threshold ({maintenance_threshold:.0f} d)",
    )
    ax.set_xlabel("Days to event")
    ax.set_ylabel("RUL (days)")
    ax.set_title(f"{result.fold.turbine_id} — {result.fold.event_id}")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def write_all_band_figures(
    fold_results: list[FoldQuantileResult],
    figures_dir: Path | str,
    *,
    conformal_results: "list[FoldConformalResult] | None" = None,
    last_n_days: int = 90,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
) -> list[Path]:
    """Write one band PNG per scored fold."""
    root = Path(figures_dir)
    by_event = {r.fold.event_id: r for r in conformal_results or []}
    paths: list[Path] = []
    for result in fold_results:
        safe_id = result.fold.event_id.replace("|", "_").replace(":", "-")
        path = root / f"{result.fold.turbine_id}_{safe_id}_rul_band.png"
        paths.append(
            plot_rul_band(
                result,
                path,
                conformal=by_event.get(result.fold.event_id),
                last_n_days=last_n_days,
                maintenance_threshold=maintenance_threshold,
            )
        )
    return paths


def plot_headline_timeline(
    timeline: pd.DataFrame,
    path: Path | str,
    *,
    time_column: str = "date",
    failure_time: pd.Timestamp | None = None,
    lookback_days: int = 90,
) -> Path:
    """Four-panel timeline: physics, hybrid residual, anomaly, conformal RUL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = timeline.copy()
    if time_column not in frame.columns:
        raise ValueError(f"timeline missing {time_column!r}")
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    frame = frame.sort_values(time_column)

    if failure_time is not None:
        failure_time = pd.Timestamp(failure_time)
        if failure_time.tzinfo is None:
            failure_time = failure_time.tz_localize("UTC")
        else:
            failure_time = failure_time.tz_convert("UTC")
        start = failure_time - pd.Timedelta(days=lookback_days)
        frame = frame.loc[
            (frame[time_column] >= start) & (frame[time_column] <= failure_time)
        ]

    if frame.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("No timeline rows in window")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    t = frame[time_column]
    if failure_time is not None:
        x = (failure_time - t).dt.total_seconds() / 86400.0
        xlabel = "Days before failure"
    else:
        x = t
        xlabel = "Date"

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax0 = axes[0]
    if "oil_temp_actual" in frame.columns and "oil_temp_physics" in frame.columns:
        ax0.plot(x, frame["oil_temp_actual"], label="Oil actual", lw=1)
        ax0.plot(x, frame["oil_temp_physics"], label="Physics expected", lw=1, alpha=0.9)
    else:
        ax0.text(0.5, 0.5, "oil temps not provided", ha="center", transform=ax0.transAxes)
    ax0.set_ylabel("Oil temp (°C)")
    ax0.legend(loc="upper left", fontsize=7)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    if "oil_residual" in frame.columns:
        ax1.plot(x, frame["oil_residual"], color="tab:orange", lw=1, label="Hybrid oil residual")
        ax1.axhline(0.0, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("Residual (°C)")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[2]
    if "anomaly_score" in frame.columns:
        ax2.plot(x, frame["anomaly_score"], color="tab:purple", lw=1)
    ax2.set_ylabel("Anomaly score")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[3]
    has_rul = "rul_p50" in frame.columns and "cqr80_lo" in frame.columns
    if has_rul:
        ax3.fill_between(
            x,
            frame["cqr80_lo"],
            frame["cqr80_hi"],
            alpha=0.35,
            color="C2",
            label="CQR 80%",
        )
        if "cqr90_lo" in frame.columns:
            ax3.fill_between(
                x,
                frame["cqr90_lo"],
                frame["cqr90_hi"],
                alpha=0.15,
                color="C1",
                label="CQR 90%",
            )
        ax3.plot(x, frame["rul_p50"], color="C0", lw=1.2, label="P50 RUL")
    ax3.set_ylabel("RUL (days)")
    ax3.set_xlabel(xlabel)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    if failure_time is not None:
        for ax in axes:
            ax.axvline(0.0, color="red", ls="-", lw=0.8, alpha=0.7)

    fig.suptitle("Twin headline timeline")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = [
    "plot_headline_timeline",
    "plot_rul_band",
    "write_all_band_figures",
]
