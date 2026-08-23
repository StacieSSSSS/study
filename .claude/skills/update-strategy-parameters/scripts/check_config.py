"""Validate strategy parameters without changing them."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import yaml


def _positive(mapping: dict[str, Any], keys: list[str], prefix: str, errors: list[str]) -> None:
    for key in keys:
        if float(mapping[key]) <= 0:
            errors.append(f"{prefix}.{key} must be positive")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()
    path = Path("strategies") / args.strategy / "config.yaml"
    with open(path, encoding="utf-8") as handle:
        config = cast(dict[str, Any], yaml.safe_load(handle))
    factors = cast(dict[str, Any], config["factors"])
    risk = cast(dict[str, Any], config["risk"])
    walk_forward = cast(dict[str, Any], config["walk_forward"])
    errors: list[str] = []

    fast = int(factors["fast_window"])
    medium = int(factors["medium_window"])
    zscore = int(factors["zscore_window"])
    if not 0 < fast < medium <= zscore:
        errors.append("require 0 < fast_window < medium_window <= zscore_window")
    _positive(factors, ["signal_clip"], "factors", errors)
    _positive(
        risk,
        [
            "annualization",
            "volatility_window",
            "target_volatility",
            "max_instrument_leverage",
            "max_portfolio_gross",
        ],
        "risk",
        errors,
    )
    _positive(walk_forward, ["train_size", "test_size", "step"], "walk_forward", errors)
    for asset_class, weights in factors["weights"].items():
        total = sum(float(value) for value in weights.values())
        if any(float(value) < 0 for value in weights.values()):
            errors.append(f"factors.weights.{asset_class} contains a negative weight")
        if abs(total - 1.0) > 1e-6:
            errors.append(f"factors.weights.{asset_class} sums to {total}, expected 1")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
