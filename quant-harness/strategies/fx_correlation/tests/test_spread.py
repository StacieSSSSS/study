import numpy as np
import pandas as pd
import pytest

from strategies.fx_correlation.lib.spread import compute_spread, hedge_ratio, latest_zscore, rolling_zscore


def test_hedge_ratio_recovers_known_slope():
    x = pd.Series(np.arange(100, dtype=float))
    y = 3.0 * x + 5.0
    assert hedge_ratio(x, y) == pytest.approx(3.0, abs=1e-9)


def test_hedge_ratio_nan_with_insufficient_data():
    x = pd.Series([1.0])
    y = pd.Series([2.0])
    assert np.isnan(hedge_ratio(x, y))


def test_compute_spread_zero_for_exact_relationship():
    x = pd.Series(np.arange(20, dtype=float))
    y = 2.0 * x
    spread = compute_spread(x, y, beta=2.0)
    assert (spread.abs() < 1e-9).all()


def test_rolling_zscore_flat_series_is_nan():
    # zero std -> 0/0, which pandas resolves to NaN rather than inf or a misleading 0.
    series = pd.Series([1.0] * 30)
    z = rolling_zscore(series, lookback=10)
    assert z.tail(5).isna().all()


def test_latest_zscore_matches_rolling_zscore_last_value():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=100))
    z = rolling_zscore(series, lookback=20)
    assert latest_zscore(series, lookback=20) == pytest.approx(z.iloc[-1])


def test_latest_zscore_nan_with_insufficient_history():
    series = pd.Series([1.0, 2.0, 3.0])
    assert np.isnan(latest_zscore(series, lookback=20))
