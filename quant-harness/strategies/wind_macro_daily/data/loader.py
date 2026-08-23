"""Point-in-time input preparation for the Wind macro daily strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return cast(dict[str, Any], yaml.safe_load(handle))


def _column(instrument: str, field: str) -> str:
    return f"{instrument}__{field}"


def _synthetic_data(config: dict[str, Any]) -> pd.DataFrame:
    """Deterministic correlated data for testing the full harness contract."""
    data_config = config["data"]
    dates = pd.bdate_range(data_config["start"], data_config["end"])
    seed = int(data_config["seed"])
    rng = np.random.default_rng(seed)
    count = len(dates)

    risk = rng.normal(0.0, 0.006, count)
    growth = np.zeros(count)
    inflation = np.zeros(count)
    usd = np.zeros(count)
    for index in range(1, count):
        growth[index] = 0.985 * growth[index - 1] + rng.normal(0.0, 0.08)
        inflation[index] = 0.990 * inflation[index - 1] + rng.normal(0.0, 0.06)
        usd[index] = 0.970 * usd[index - 1] + rng.normal(0.0, 0.10)

    output = pd.DataFrame(index=dates)
    output.index.name = "date"
    for number, instrument in enumerate(config["universe"]):
        spec = config["instruments"][instrument]
        local_rng = np.random.default_rng(seed + 101 * (number + 1))
        initial = float(spec["initial_level"])
        if spec["asset_class"] == "FX":
            beta = {"USDCNY": -0.20, "USDJPY": 0.05, "AUDUSD": 0.55, "EURUSD": 0.30}[
                instrument
            ]
            direction = -1.0 if instrument in {"USDCNY", "USDJPY"} else 1.0
            innovation = local_rng.normal(0.0, 0.0055, count) + beta * risk
            returns = np.zeros(count)
            for index in range(1, count):
                returns[index] = (
                    0.06 * returns[index - 1]
                    + innovation[index]
                    + direction * 0.00012 * usd[index - 1]
                    + 0.00008 * growth[index - 1]
                )
            close = initial * np.exp(np.cumsum(returns))
            carry = direction * (0.015 + 0.006 * inflation) + local_rng.normal(
                0.0, 0.002, count
            )
            macro = direction * usd + 0.35 * growth - 0.15 * inflation
        else:
            duration = float(spec["duration"])
            country = 1.0 if instrument.startswith("US_") else -0.65
            yield_change = np.zeros(count)
            shock = local_rng.normal(0.0, 0.00045, count)
            for index in range(1, count):
                yield_change[index] = (
                    0.08 * yield_change[index - 1]
                    + shock[index]
                    + country * 0.000012 * growth[index - 1]
                    + 0.000010 * inflation[index - 1]
                )
            close = initial + 100.0 * np.cumsum(yield_change)
            carry = 0.0025 + 0.0015 * np.tanh(inflation) + local_rng.normal(
                0.0, 0.0003, count
            )
            returns = -duration * yield_change + carry / 252.0
            macro = -country * growth - 0.40 * inflation

        output[_column(instrument, "close")] = close
        output[_column(instrument, "return")] = returns
        output[_column(instrument, "carry")] = carry
        output[_column(instrument, "macro")] = macro
    return output


def _wind_data(config: dict[str, Any]) -> pd.DataFrame:
    """Convert standardized Wind observations into the strategy's wide panel."""
    path = Path(config["data"]["wind"]["observations_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Wind observations not found at {path}; run data.wind_fetch and audit timestamps first"
        )
    observations = pd.read_csv(path, parse_dates=["reference_ts_utc"])
    if "backtest_safe" in observations.columns and not bool(
        cast(pd.Series, observations["backtest_safe"]).fillna(0).eq(1).all()
    ):
        raise ValueError("Wind input contains rows not marked backtest_safe=1")
    pivot = observations.pivot_table(
        index="reference_ts_utc", columns="dataset_id", values="value", aggfunc="last"
    ).sort_index()
    output = pd.DataFrame(index=pivot.index)
    for instrument in config["universe"]:
        spec = config["instruments"][instrument]
        dataset = spec["close_dataset"]
        if dataset not in pivot:
            raise ValueError(f"required Wind dataset missing: {dataset}")
        close = cast(pd.Series, pivot[dataset]).astype(float)
        if spec["asset_class"] == "FX":
            returns = np.log(close / close.shift(1))
        else:
            returns = -float(spec["duration"]) * close.diff() / 100.0
        output[_column(instrument, "close")] = close
        output[_column(instrument, "return")] = returns
        output[_column(instrument, "carry")] = 0.0
        output[_column(instrument, "macro")] = 0.0
    output.index.name = "date"
    return output.dropna(how="all")


def load_raw() -> pd.DataFrame:
    config = load_config()
    mode = config["data"]["mode"]
    if mode == "synthetic":
        return _synthetic_data(config)
    if mode == "wind":
        return _wind_data(config)
    raise ValueError(f"unsupported data.mode: {mode}")
