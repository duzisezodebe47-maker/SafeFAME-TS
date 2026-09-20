"""Shared point-in-time data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


START_COLUMNS = ("start_date", "date", "Date", "MapDate", "Month")


def numerical_intervals(frame: pd.DataFrame, path: Path) -> tuple[pd.Series, pd.Series]:
    start_col = next((column for column in START_COLUMNS if column in frame), None)
    end_col = "end_date" if "end_date" in frame else start_col
    if start_col is None:
        raise ValueError(f"{path} has no recognized numerical date column")
    return (
        pd.to_datetime(frame[start_col], errors="coerce"),
        pd.to_datetime(frame[end_col], errors="coerce"),
    )


def load_time_ordered_frame(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    starts, ends = numerical_intervals(frame, path)
    if starts.isna().any() or ends.isna().any():
        raise ValueError(f"{path} contains invalid numerical intervals")
    if (ends < starts).any():
        raise ValueError(f"{path} contains intervals whose end precedes start")
    order = np.lexsort((np.arange(len(frame)), ends.astype("int64"), starts.astype("int64")))
    audit = {
        "original_time_order": "ascending" if starts.is_monotonic_increasing else "descending" if starts.is_monotonic_decreasing else "unsorted",
        "rows_moved_by_stable_sort": int(np.sum(order != np.arange(len(frame)))),
        "duplicate_start_dates": int(starts.duplicated().sum()),
    }
    ordered = frame.iloc[order].reset_index(drop=True)
    ordered_starts, _ = numerical_intervals(ordered, path)
    if not ordered_starts.is_monotonic_increasing:
        raise AssertionError(f"Failed to sort {path} chronologically")
    return ordered, audit

