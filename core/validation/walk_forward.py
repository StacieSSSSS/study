"""Rolling-window (walk-forward) out-of-sample validation.

A strategy is fit/calibrated on a trailing ``train_size`` window and then
evaluated, untouched, on the following ``test_size`` window. The window
slides forward by ``step`` and repeats. This is the harness's defense
against overfitting a single in-sample backtest.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import cast

import pandas as pd
import yaml


@dataclass(frozen=True)
class WalkForwardWindow:
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[WalkForwardWindow]:
    """Yield rolling (train, test) index windows over a sorted DatetimeIndex."""
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, and step must be positive")

    index = index.sort_values()
    n = len(index)
    start = 0
    while start + train_size + test_size <= n:
        train = index[start : start + train_size]
        test = index[start + train_size : start + train_size + test_size]
        yield WalkForwardWindow(train=train, test=test)
        start += step


def run_walk_forward(
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int,
    evaluate: Callable[[pd.DataFrame, pd.DataFrame], dict[str, float]],
) -> list[dict[str, float]]:
    """Run ``evaluate(train_df, test_df)`` over every rolling window.

    ``evaluate`` is supplied by the strategy: fit/calibrate on the train slice,
    then return out-of-sample metrics computed strictly on the test slice.
    """
    results: list[dict[str, float]] = []
    index = cast(pd.DatetimeIndex, data.index)
    for window in walk_forward_splits(index, train_size, test_size, step):
        train_df = data.loc[window.train]
        test_df = data.loc[window.test]
        results.append(evaluate(train_df, test_df))
    return results


def _load_strategy_config(strategy_dir: str) -> dict:
    with open(f"{strategy_dir}/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward validation for a strategy")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--strategy", required=True, help="strategy folder name under strategies/")
    args = parser.parse_args(argv)

    strategy_dir = f"strategies/{args.strategy}"
    config = _load_strategy_config(strategy_dir)
    wf_config = config.get("walk_forward", {})

    module = importlib.import_module(f"strategies.{args.strategy}.backtest.run")
    data = module.load_data()
    results = run_walk_forward(
        data,
        train_size=wf_config["train_size"],
        test_size=wf_config["test_size"],
        step=wf_config["step"],
        evaluate=module.evaluate_window,
    )

    if not results:
        print(
            f"walk-forward FAILED: no windows produced for {args.strategy} "
            "(check data length vs train/test/step)"
        )
        return 1

    avg_sharpe = sum(r.get("sharpe", 0.0) for r in results) / len(results)
    print(f"walk-forward OK: {len(results)} window(s), avg out-of-sample sharpe={avg_sharpe:.3f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
