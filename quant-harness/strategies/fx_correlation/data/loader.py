"""Fetches and caches daily FX prices for the configured universe.

Designed to be re-run daily: `load_raw(refresh=True)` always re-downloads
from yfinance. The default `refresh=False` reuses a same-day cache so
walk-forward/backtest runs during the day don't re-hit the network every time.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _cache_path(config: dict) -> Path:
    strategy_dir = CONFIG_PATH.parent
    cache_dir = strategy_dir / config["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "prices.parquet"


def _fetch_prices(ticker_map: dict[str, str], history_years: int) -> pd.DataFrame:
    period = f"{history_years}y"
    series_by_pair: dict[str, pd.Series] = {}
    for pair, ticker in ticker_map.items():
        df: pd.DataFrame | None = None
        for _attempt in range(3):
            df = yf.download(ticker, period=period, progress=False)
            if df is not None and not df.empty:
                break
            time.sleep(1)
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {pair} ({ticker})")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        series_by_pair[pair] = close.rename(pair)

    prices = pd.concat(series_by_pair.values(), axis=1)
    prices = prices.sort_index()
    prices = prices.ffill()  # carries the last known *past* price forward, never a future one
    return prices


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Wide DataFrame of daily FX prices, columns = pair names, index = date."""
    config = _load_config()
    cache_path = _cache_path(config)

    if not refresh and cache_path.exists():
        cached_today = date.fromtimestamp(cache_path.stat().st_mtime) == date.today()
        if cached_today:
            return pd.read_parquet(cache_path)

    prices = _fetch_prices(config["ticker_map"], config["data"]["history_years"])
    prices.to_parquet(cache_path)
    return prices


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Wide DataFrame of daily log returns, columns = pair names, index = date.

    This is what every model and the backtest consume — returns, not raw
    prices, so correlation/cointegration/z-score math all operates on a
    stationary-ish series rather than a trending price level.
    """
    prices = load_prices(refresh=refresh)
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")
