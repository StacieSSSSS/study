import pandas as pd
import pytest

from core.metrics.performance import annualized_return, max_drawdown, sharpe_ratio, summarize, turnover


def test_sharpe_ratio_zero_for_zero_volatility():
    returns = pd.Series([0.001] * 10)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_positive_drift():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_is_negative_fraction():
    equity = pd.Series([1.0, 1.1, 0.9, 1.0, 1.2])
    dd = max_drawdown(equity)
    assert dd < 0
    assert dd == pytest.approx(0.9 / 1.1 - 1.0)


def test_max_drawdown_zero_for_monotonic_increase():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(equity) == 0.0


def test_turnover_averages_absolute_weight_changes():
    weights = pd.DataFrame({"a": [0.0, 0.5, 0.5], "b": [0.0, -0.5, 0.0]})
    assert turnover(weights) == pytest.approx((1.0 + 0.5) / 2)


def test_annualized_return_matches_simple_compounding():
    returns = pd.Series([0.1, 0.1])
    result = annualized_return(returns, periods_per_year=2)
    assert result == pytest.approx(1.1 * 1.1 - 1.0)


def test_summarize_includes_turnover_when_weights_given():
    returns = pd.Series([0.01, -0.005, 0.02])
    weights = pd.DataFrame({"a": [0.0, 0.5, 0.3]})
    result = summarize(returns, weights)
    assert "turnover" in result
    assert "sharpe" in result
    assert "max_drawdown" in result
