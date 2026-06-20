"""Data loading for this strategy.

Implement `load_raw()` to pull and clean your source data (prices, yields,
fundamentals, ...) into a single DataFrame indexed by a DatetimeIndex.
Keep this layer dumb: cleaning and alignment only, no signal logic here.
"""

from __future__ import annotations

import pandas as pd


def load_raw() -> pd.DataFrame:
    raise NotImplementedError(
        "Implement load_raw() to return a DatetimeIndex-ed DataFrame of your raw inputs."
    )
