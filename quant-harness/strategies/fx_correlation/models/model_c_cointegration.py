"""Model C — cointegration.

Correlation alone doesn't justify a mean-reversion bet — two series can be
highly correlated and still drift apart permanently. This model filters to
combos whose hedge-ratio spread passes an ADF stationarity test (low
p-value = real evidence of mean reversion, not just co-movement), then ranks
the survivors by the ADF *statistic* rather than the p-value itself: for a
strongly cointegrated universe, statsmodels' p-value saturates to 0.0 once
the statistic is far into the tail, so several combos can tie at p=0.0 while
their statistics (e.g. -20.1 vs -19.2) still differ meaningfully. The
statistic is what actually orders them.
"""

from __future__ import annotations

import math

from strategies.fx_correlation.models.base import ComboMetrics

NEED_ADF = True


def selection_score(metrics: ComboMetrics, cfg: dict) -> float:
    if math.isnan(metrics.adf_p) or metrics.adf_p > cfg["max_adf_pvalue"]:
        return float("-inf")
    if math.isnan(metrics.adf_stat):
        return float("-inf")
    return -metrics.adf_stat
