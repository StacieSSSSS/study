import numpy as np
import pandas as pd

from strategies.fx_correlation.lib.cointegration import adf_pvalue, adf_test, cointegration_pvalue


def test_adf_pvalue_low_for_stationary_series():
    rng = np.random.default_rng(0)
    stationary = pd.Series(rng.normal(size=300))
    assert adf_pvalue(stationary) < 0.05


def test_adf_pvalue_high_for_random_walk():
    rng = np.random.default_rng(1)
    random_walk = pd.Series(rng.normal(size=300)).cumsum()
    assert adf_pvalue(random_walk) > 0.10


def test_adf_pvalue_nan_with_too_little_data():
    short_series = pd.Series([1.0, 2.0, 3.0])
    assert np.isnan(adf_pvalue(short_series))


def test_cointegration_pvalue_low_for_cointegrated_pair():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(size=300)).cumsum() + 100
    noise = pd.Series(rng.normal(scale=0.1, size=300))
    y = 2.0 * x + noise  # y - 2*x is stationary noise -> cointegrated by construction
    assert cointegration_pvalue(x, y) < 0.05


def test_cointegration_pvalue_high_for_unrelated_random_walks():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=300)).cumsum()
    y = pd.Series(rng.normal(size=300)).cumsum()
    assert cointegration_pvalue(x, y) > 0.10


def _ar1(phi: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return pd.Series(x)


def test_adf_test_statistic_orders_by_mean_reversion_speed():
    # A fast-reverting AR(1) (phi=0.3) is much more stationary than a slow,
    # near-unit-root one (phi=0.95) — the statistic should reflect that even
    # in cases (like this one) where the p-values also still differ, because
    # the statistic is the finer-grained signal model_c ranks on once
    # p-values saturate to 0.0 for a batch of strongly cointegrated combos.
    fast = _ar1(0.3, 1000, seed=10)
    slow = _ar1(0.95, 1000, seed=10)
    assert adf_test(fast).statistic < adf_test(slow).statistic


def test_adf_test_nan_with_too_little_data():
    result = adf_test(pd.Series([1.0, 2.0, 3.0]))
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)
