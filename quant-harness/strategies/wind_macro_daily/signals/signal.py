"""Generate a multi-asset signal using only point-in-time data."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from strategies.wind_macro_daily.factors.engine import factor_snapshot


def generate_signal(
    pit: PointInTimeFrame,
    as_of: pd.Timestamp,
    config: dict[str, Any],
    signal_variant: str = "composite_signal",
) -> pd.Series:
    factors = factor_snapshot(pit, as_of, config)
    indexed = factors.set_index("instrument")
    if signal_variant not in {
        "momentum",
        "carry_signal",
        "mean_reversion",
        "macro_signal",
        "composite_signal",
    }:
        raise ValueError(f"unsupported signal variant: {signal_variant}")
    signal = cast(pd.Series, indexed[signal_variant]).astype(float)
    if signal_variant != "composite_signal":
        signal = pd.Series(np.tanh(signal.to_numpy()), index=signal.index)
    return signal.reindex(config["universe"]).fillna(0.0)
