"""Signal generation for fx_correlation.

Deviates from `_template`'s `generate_signal(pit, as_of) -> float`: this
strategy trades a basket of FX pairs at once (each model's selected combos
imply positions across multiple underlying pairs simultaneously), so the
natural signal is a weight *vector* over `config.yaml`'s `pairs`, not a
single scalar. The look-ahead constraint is identical either way — every
read still goes through `PointInTimeFrame.as_of()`, never the raw panel.

`generate_weights` is a thin pass-through to
`backtest.run.blended_weights` for a single date — kept here, rather than
only in backtest/run.py, so it's discoverable as "the signal" the way
`_template/signals/signal.py` is.
"""

from __future__ import annotations

import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from strategies.fx_correlation.backtest.run import blended_weights


def generate_weights(
    pit: PointInTimeFrame, as_of: pd.Timestamp, train_df: pd.DataFrame, config: dict
) -> pd.Series:
    """Blended target weight for each pair in `config["pairs"]`, as of `as_of`."""
    dates = pd.DatetimeIndex([as_of])
    weights = blended_weights(pit, dates, config["pairs"], train_df, config)
    return weights.loc[as_of]
