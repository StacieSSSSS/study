"""Performance threshold gate.

Compares a strategy's latest backtest metrics against the thresholds
declared in its ``config.yaml``. A strategy that doesn't clear its own
declared bar fails ``make verify-full`` — the bar is part of the strategy's
definition, not an afterthought.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def check_thresholds(metrics: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    violations: list[str] = []

    min_sharpe = thresholds.get("min_sharpe")
    sharpe = metrics.get("sharpe", float("-inf"))
    if min_sharpe is not None and sharpe < min_sharpe:
        violations.append(f"sharpe {sharpe} < min_sharpe {min_sharpe}")

    drawdown_floor = thresholds.get("max_drawdown")
    drawdown = metrics.get("max_drawdown", float("-inf"))
    if drawdown_floor is not None and drawdown < drawdown_floor:
        violations.append(f"max_drawdown {drawdown} breaches floor {drawdown_floor}")

    max_turnover = thresholds.get("max_turnover")
    measured_turnover = metrics.get("turnover", float("inf"))
    if max_turnover is not None and measured_turnover > max_turnover:
        violations.append(f"turnover {measured_turnover} > max_turnover {max_turnover}")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a strategy's metrics against its declared thresholds")
    parser.add_argument("--strategy", required=True, help="strategy folder name under strategies/")
    parser.add_argument(
        "--metrics", default=None, help="path to metrics.json (default: reports/<strategy>/metrics.json)"
    )
    args = parser.parse_args(argv)

    strategy_dir = Path("strategies") / args.strategy
    with open(strategy_dir / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    thresholds = config.get("gate", {})

    metrics_path = Path(args.metrics) if args.metrics else Path("reports") / args.strategy / "metrics.json"
    if not metrics_path.exists():
        print(f"perf-gate FAILED: no metrics found at {metrics_path} — run the backtest first")
        return 1

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    violations = check_thresholds(metrics, thresholds)
    if violations:
        print(f"perf-gate FAILED for {args.strategy}:")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"perf-gate OK for {args.strategy}: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
