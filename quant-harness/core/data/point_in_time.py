"""Structural guard against look-ahead bias.

Signal functions should never receive a raw DataFrame that contains rows
beyond the current decision date. ``PointInTimeFrame`` wraps a DataFrame and
only exposes data through ``as_of()``, which slices up to (and including) a
cutoff timestamp. Direct positional/label access to the underlying frame is
blocked so a strategy author cannot accidentally reach into the future.
"""

from __future__ import annotations

from typing import cast

import pandas as pd


class FutureDataAccessError(RuntimeError):
    """Raised when code attempts to read data beyond the current cutoff."""


class PointInTimeFrame:
    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("PointInTimeFrame requires a DatetimeIndex")
        self._data = data.sort_index()

    def as_of(self, cutoff: pd.Timestamp | str) -> pd.DataFrame:
        """Return a copy of all rows with index <= cutoff."""
        cutoff_ts = cast(pd.Timestamp, pd.Timestamp(cutoff))
        return self._data.loc[:cutoff_ts].copy()

    def latest(self, cutoff: pd.Timestamp | str) -> pd.Series:
        """Return the most recent row available as of cutoff."""
        view = self.as_of(cutoff)
        if view.empty:
            raise FutureDataAccessError(f"No data available as of {cutoff}")
        return view.iloc[-1]

    @property
    def full_history_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Inspection only — for validation tooling, not for signal logic."""
        index = cast(pd.DatetimeIndex, self._data.index)
        return cast(pd.Timestamp, index[0]), cast(pd.Timestamp, index[-1])

    def __len__(self) -> int:
        return len(self._data)
