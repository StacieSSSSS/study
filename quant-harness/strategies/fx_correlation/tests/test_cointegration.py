import numpy as np
import pandas as pd

from strategies.fx_correlation.lib.cointegration import adf_pvalue, cointegration_pvalue


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
