import numpy as np
import pandas as pd
import pytest

from strategies.fx_correlation.lib.correlation import (
    divergence_score,
    multi_window_correlation,
    rolling_corr_last,
    stability_score,
    strength_score,
)


def test_rolling_corr_last_perfectly_correlated():
    x = pd.Series(np.arange(50, dtype=float))
    y = pd.Series(np.arange(50, dtype=float) * 2 + 1)
    assert rolling_corr_last(x, y, window=20) == pytest.approx(1.0)


def test_rolling_corr_last_insufficient_history_returns_nan():
    x = pd.Series(np.arange(5, dtype=float))
    y = pd.Series(np.arange(5, dtype=float))
    assert np.isnan(rolling_corr_last(x, y, window=20))


def test_multi_window_correlation_returns_all_windows():
    x = pd.Series(np.arange(300, dtype=float))
    y = pd.Series(np.arange(300, dtype=float) + np.sin(np.arange(300)))
    windows = {"1m": 21, "12m": 252}
    result = multi_window_correlation(x, y, windows)
    assert set(result.keys()) == {"1m", "12m"}
    assert all(not np.isnan(v) for v in result.values())


def test_strength_score_averages_absolute_correlation():
    corrs = {"1m": 0.8, "3m": -0.6}
    assert strength_score(corrs) == pytest.approx(0.7)


def test_strength_score_nan_when_all_nan():
    assert np.isnan(strength_score({"1m": float("nan")}))


def test_stability_score_higher_for_consistent_correlation():
    stable = stability_score({"1m": 0.8, "3m": 0.81, "6m": 0.79})
    unstable = stability_score({"1m": 0.8, "3m": 0.1, "6m": -0.5})
    assert stable > unstable


def test_divergence_score_zero_when_equal():
    corrs = {"1m": 0.5, "12m": 0.5}
    assert divergence_score(corrs, "1m", "12m") == pytest.approx(0.0)


def test_divergence_score_positive_when_different():
    corrs = {"1m": 0.9, "12m": 0.3}
    assert divergence_score(corrs, "1m", "12m") == pytest.approx(0.6)
