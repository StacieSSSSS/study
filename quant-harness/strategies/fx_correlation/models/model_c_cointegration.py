"""Model C — cointegration.

Correlation alone doesn't justify a mean-reversion bet — two series can be
highly correlated and still drift apart permanently. This model filters to
combos whose hedge-ratio spread passes an ADF stationarity test (low
p-value = real evidence of mean reversion, not just co-movement) and ranks
by how strong that evidence is.
"""

from __future__ import annotations

import math

from strategies.fx_correlation.models.base import ComboMetrics

NEED_ADF = True


def selection_score(metrics: ComboMetrics, cfg: dict) -> float:
    if math.isnan(metrics.adf_p) or metrics.adf_p > cfg["max_adf_pvalue"]:
        return float("-inf")
    return -metrics.adf_p
