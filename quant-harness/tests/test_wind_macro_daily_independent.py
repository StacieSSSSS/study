from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from core.data.point_in_time import PointInTimeFrame
from strategies.wind_macro_daily.data.excel_loader import _read_price_panel
from strategies.wind_macro_daily.factors.engine import factor_snapshot
from strategies.wind_macro_daily.validation.run import _effectiveness_reason


def test_manual_treasury_yield_uses_duration_return() -> None:
    rows = [
        ("Wind", None),
        ("指标名称", "美国:国债收益率:5年"),
        ("单位", "%"),
        (pd.Timestamp("2024-01-02"), 2.00),
        (pd.Timestamp("2024-01-03"), 2.10),
    ]
    config = {
        "data": {
            "manual_excel": {
                "price_map": {"UST_5Y": "美国:国债收益率:5年"},
                "max_price_forward_fill_business_days": 5,
            }
        },
        "instruments": {
            "UST_5Y": {"return_model": "yield_duration", "duration": 4.6}
        },
    }

    panel, quality, active = _read_price_panel(rows, config)

    assert active == ["UST_5Y"]
    assert panel.loc[pd.Timestamp("2024-01-03"), "UST_5Y__return"] == pytest.approx(-0.0046)
    assert quality.loc[0, "return_model"] == "yield_duration"


def test_rate_and_fx_mean_reversion_have_opposite_position_conventions() -> None:
    dates = pd.bdate_range("2024-01-02", periods=100)
    rising = np.linspace(1.0, 2.0, len(dates))
    data = pd.DataFrame(index=dates)
    for instrument in ["UST_5Y", "EURUSD"]:
        data[f"{instrument}__close"] = rising
        data[f"{instrument}__return"] = 0.0
        data[f"{instrument}__carry"] = np.nan
        data[f"{instrument}__macro"] = np.nan
    config = {
        "universe": ["UST_5Y", "EURUSD"],
        "instruments": {
            "UST_5Y": {"asset_class": "UST"},
            "EURUSD": {"asset_class": "FX"},
        },
        "factors": {
            "fast_window": 20,
            "medium_window": 60,
            "zscore_window": 90,
            "signal_clip": 3.0,
            "weights": {
                "UST": {"momentum": 0.4, "carry": 0.2, "mean_reversion": 0.15, "macro": 0.25},
                "FX": {"momentum": 0.4, "carry": 0.25, "mean_reversion": 0.1, "macro": 0.25},
            },
        },
    }

    snapshot = factor_snapshot(
        PointInTimeFrame(data), cast(pd.Timestamp, dates[-1]), config
    ).set_index("instrument")

    assert snapshot.loc["UST_5Y", "mean_reversion"] > 0
    assert snapshot.loc["EURUSD", "mean_reversion"] < 0


def test_effectiveness_requires_every_configured_oos_threshold() -> None:
    thresholds = {
        "minimum_windows": 3,
        "minimum_active_position_days": 126,
        "minimum_average_oos_sharpe": 0.5,
        "minimum_worst_window_sharpe": -0.5,
        "minimum_positive_sharpe_fraction": 0.67,
        "minimum_average_oos_annualized_return": 0.0,
        "maximum_worst_oos_drawdown": -0.2,
    }
    passing = {
        "windows": 3,
        "active_position_days": 500,
        "average_oos_sharpe": 0.8,
        "worst_window_sharpe": -0.2,
        "positive_sharpe_fraction": 1.0,
        "average_oos_annualized_return": 0.05,
        "worst_oos_max_drawdown": -0.1,
    }
    reason, effective = _effectiveness_reason(passing, thresholds)
    assert effective is True
    assert reason == "all_oos_thresholds_passed"

    passing["average_oos_sharpe"] = 0.2
    reason, effective = _effectiveness_reason(passing, thresholds)
    assert effective is False
    assert "average_oos_sharpe" in reason
