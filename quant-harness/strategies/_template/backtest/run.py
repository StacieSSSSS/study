"""Backtest entry points for this strategy.

`load_data()` and `evaluate_window()` are called by
`core.validation.walk_forward` (via `make walk-forward STRATEGY=...`).
`main()` runs the full-history backtest and writes `reports/<name>/metrics.json`,
which `core.reporting.gate` checks against `config.yaml`'s `gate:` thresholds.

Expected shape: `load_data()` returns a DataFrame indexed by date with at
least a `return` column (the asset's periodic return). Adjust to taste once
you have a real strategy — this is plumbing, not policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from core.metrics.performance import summarize
from strategies._template.data.loader import load_raw
from strategies._template.signals.signal import generate_signal

STRATEGY_NAME = "_template"


def load_data() -> pd.DataFrame:
    return load_raw()


def _run_over_dates(
    pit: PointInTimeFrame, dates: pd.DatetimeIndex, returns: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    weights = pd.Series(index=dates, dtype=float)
    for as_of in dates:
        weights[as_of] = generate_signal(pit, as_of)
    # Position taken at the close on `as_of` earns the *next* period's return.
    strategy_returns = weights.shift(1).fillna(0.0) * returns.loc[dates]
    weights_df = weights.to_frame(name="weight")
    return weights_df, strategy_returns


def evaluate_window(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    history = pd.concat([train_df, test_df]).sort_index()
    pit = PointInTimeFrame(history)
    test_dates = cast(pd.DatetimeIndex, test_df.index)
    test_returns = cast(pd.Series, test_df["return"])
    weights_df, strategy_returns = _run_over_dates(pit, test_dates, test_returns)
    return summarize(strategy_returns, weights_df)


def main() -> None:
    data = load_data()
    pit = PointInTimeFrame(data)
    dates = cast(pd.DatetimeIndex, data.index)
    returns = cast(pd.Series, data["return"])
    weights_df, strategy_returns = _run_over_dates(pit, dates, returns)
    metrics = summarize(strategy_returns, weights_df)

    out_dir = Path("reports") / STRATEGY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_dir / 'metrics.json'}: {metrics}")


if __name__ == "__main__":
    main()
