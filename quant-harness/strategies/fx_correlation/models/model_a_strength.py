"""Model A — correlation strength & stability.

Selects the combos whose correlation is both strong (high average |corr|
across the 1/3/6/9/12-month windows) and stable (low dispersion across those
windows) — the read being "this relationship is reliable enough right now to
bet on mean reversion."
"""

from __future__ import annotations

import math

from strategies.fx_correlation.lib.correlation import stability_score, strength_score
from strategies.fx_correlation.models.base import ComboMetrics

NEED_ADF = False


def selection_score(metrics: ComboMetrics, cfg: dict) -> float:
    strength = strength_score(metrics.corrs)
    stability = stability_score(metrics.corrs)
    if math.isnan(strength) or math.isnan(stability):
        return float("-inf")
    return cfg["weight_strength"] * strength + cfg["weight_stability"] * stability
