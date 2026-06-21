"""Cointegration test (ADF on the OLS residual spread) — used by Model C to
filter for combos whose spread is plausibly stationary, not just correlated.
Correlation alone doesn't justify a mean-reversion bet: two series can move
together and still drift apart permanently. The ADF p-value on the spread is
the check that the mean-reversion assumption actually holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from strategies.fx_correlation.lib.spread import compute_spread, hedge_ratio


@dataclass(frozen=True)
class AdfResult:
    statistic: float  # more negative = stronger evidence against "no cointegration"
    pvalue: float


def adf_test(spread: pd.Series, maxlag: int = 5) -> AdfResult:
    """Augmented Dickey-Fuller test for stationarity of `spread`.

    Use `pvalue` to decide *whether* a combo qualifies (it's the standard,
    interpretable threshold). Use `statistic` to rank *among* qualifying
    combos: statsmodels' p-value is a table lookup that saturates to 0.0 once
    the statistic is far enough into the tail, so for a strongly cointegrated
    universe several combos can tie at p=0.0 while their actual statistics
    still differ meaningfully (e.g. -20.1 vs -19.2) — the statistic is what
    actually orders them.

    `maxlag` bounds the AIC lag-order search. This is called per combo per
    backtest date (potentially thousands of times), so an unbounded search —
    statsmodels' default grows with series length — is the dominant cost of
    the whole backtest; 5 lags is more than enough for the daily FX data this
    is run on and cuts the per-call cost roughly 3x.
    """
    clean = spread.dropna()
    if len(clean) < 20:
        return AdfResult(float("nan"), float("nan"))
    try:
        result = adfuller(clean, maxlag=maxlag, autolag="AIC")
    except ValueError:
        return AdfResult(float("nan"), float("nan"))
    return AdfResult(float(result[0]), float(result[1]))


def adf_pvalue(spread: pd.Series, maxlag: int = 5) -> float:
    """p-value only — kept as a convenience wrapper around `adf_test`."""
    return adf_test(spread, maxlag).pvalue


def cointegration_pvalue(x: pd.Series, y: pd.Series) -> float:
    """Fit the hedge ratio on (x, y) and return the ADF p-value of the resulting spread."""
    beta = hedge_ratio(x, y)
    if pd.isna(beta):
        return float("nan")
    spread = compute_spread(x, y, beta)
    return adf_pvalue(spread)
