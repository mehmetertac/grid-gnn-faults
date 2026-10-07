"""Headline RUL figures: true RUL vs P50 with P10–P90 band."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rul.quantile import DEFAULT_MAINTENANCE_THRESHOLD_DAYS, FoldQuantileResult


def plot_rul_band(
    result: FoldQuantileResult,
    path: Path | str,
    *,
    last_n_days: int = 90,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
) -> Path:
    """One PNG per held-out event: band over the last ``last_n_days`` before failure."""
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

    mask = days <= float(last_n_days)
    if not np.any(mask):
        mask = np.ones_like(days, dtype=bool)
    x = days[mask]
    y_plot = y_true[mask]
    q10_plot = q10[mask]
    q50_plot = q50[mask]
    q90_plot = q90[mask]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, q10_plot, q90_plot, alpha=0.25, label="P10–P90")
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
    ax.legend(loc="upper left", fontsize=8)
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
    last_n_days: int = 90,
    maintenance_threshold: float = DEFAULT_MAINTENANCE_THRESHOLD_DAYS,
) -> list[Path]:
    """Write one band PNG per scored fold."""
    root = Path(figures_dir)
    paths: list[Path] = []
    for result in fold_results:
        safe_id = result.fold.event_id.replace("|", "_").replace(":", "-")
        path = root / f"{result.fold.turbine_id}_{safe_id}_rul_band.png"
        paths.append(
            plot_rul_band(
                result,
                path,
                last_n_days=last_n_days,
                maintenance_threshold=maintenance_threshold,
            )
        )
    return paths


__all__ = ["plot_rul_band", "write_all_band_figures"]
