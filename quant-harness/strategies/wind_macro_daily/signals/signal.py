"""Generate a multi-asset signal using only point-in-time data."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from strategies.wind_macro_daily.factors.engine import factor_snapshot


def generate_signal(
    pit: PointInTimeFrame, as_of: pd.Timestamp, config: dict[str, Any]
) -> pd.Series:
    factors = factor_snapshot(pit, as_of, config)
    indexed = factors.set_index("instrument")
    signal = cast(pd.Series, indexed["composite_signal"])
    return signal.reindex(config["universe"]).fillna(0.0)
