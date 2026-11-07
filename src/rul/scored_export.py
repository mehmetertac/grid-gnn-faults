"""Helpers for fleet scored CSV export (no wind_digital_twin import)."""

from __future__ import annotations

import pandas as pd

from rul.labels import SCORED_VALUE_COLUMNS


def concat_scored_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-turbine 10-minute scored frames in stable turbine order."""
    if not frames:
        columns = ["turbine_id", "timestamp", *SCORED_VALUE_COLUMNS]
        return pd.DataFrame(columns=columns)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)
    return combined


__all__ = ["concat_scored_frames"]
