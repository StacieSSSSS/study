"""Shared trade mechanic for all three models: a hedge-ratio spread, mean-reverted via
a rolling z-score. Models differ only in *which* combo they select — not in how the
selected combo is actually traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """OLS slope of y regressed on x: y ~= alpha + beta * x. Fit on whatever
    history is passed in — callers are responsible for only passing training
    data, so the ratio doesn't peek at the test window it will be applied to.
    """
    aligned = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    x_vals = aligned.iloc[:, 0].to_numpy()
    y_vals = aligned.iloc[:, 1].to_numpy()
    beta, _alpha = np.polyfit(x_vals, y_vals, deg=1)
    return float(beta)


def compute_spread(x: pd.Series, y: pd.Series, beta: float) -> pd.Series:
    """Residual series y - beta * x, aligned on the intersection of both indices."""
    aligned = pd.concat([x, y], axis=1, join="inner").dropna()
    return aligned.iloc[:, 1] - beta * aligned.iloc[:, 0]


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score using only trailing `lookback` observations at each point —
    safe to compute over a full history series since .rolling() never looks forward.
    """
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std()
    return (series - rolling_mean) / rolling_std


def latest_zscore(series: pd.Series, lookback: int) -> float:
    z = rolling_zscore(series, lookback)
    if z.empty or pd.isna(z.iloc[-1]):
        return float("nan")
    return float(z.iloc[-1])
