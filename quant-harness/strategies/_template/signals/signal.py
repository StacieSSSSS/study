"""Signal generation.

`generate_signal` must only read data through the `PointInTimeFrame` it is
given — never reach for a module-level raw DataFrame, and never call
`.shift()` with a negative period or `.bfill()` on history. Those are exactly
what `core.validation.bias_check` scans for.
"""

from __future__ import annotations

import pandas as pd

from core.data.point_in_time import PointInTimeFrame


def generate_signal(pit: PointInTimeFrame, as_of: pd.Timestamp) -> float:
    """Return a position weight (or score) for `as_of`, using only data <= as_of."""
    raise NotImplementedError("Implement generate_signal() using pit.as_of(as_of) / pit.latest(as_of).")
