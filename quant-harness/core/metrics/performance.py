"""Shared performance metrics used by backtests and the perf-gate check."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of a periodic return series."""
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if pd.isna(std) or std < 1e-12:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, expressed as a negative fraction."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def turnover(weights: pd.DataFrame) -> float:
    """Average per-period absolute change in position weights across assets.

    The first row has no prior period to diff against and is dropped rather
    than counted as zero turnover.
    """
    return float(weights.diff().dropna(how="all").abs().sum(axis=1).mean())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    compounded = (1.0 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return float(compounded ** (periods_per_year / n_periods) - 1.0)


def summarize(
    returns: pd.Series, weights: pd.DataFrame | None = None, periods_per_year: int = 252
) -> dict[str, float]:
    equity_curve = (1.0 + returns).cumprod()
    result = {
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "annualized_return": annualized_return(returns, periods_per_year),
    }
    if weights is not None:
        result["turnover"] = turnover(weights)
    return result
