"""Model B — correlation divergence / breakdown.

Selects combos whose short-window correlation has drifted furthest from its
own long-run baseline. Research consensus on correlation trading is that the
breakdown moment — not the steady-state correlation level — is where the
edge usually is: a sudden regime shift gets bet to normalize back.
"""

from __future__ import annotations

import math

from strategies.fx_correlation.lib.correlation import divergence_score
from strategies.fx_correlation.models.base import ComboMetrics

NEED_ADF = False


def selection_score(metrics: ComboMetrics, cfg: dict) -> float:
    divergence = divergence_score(metrics.corrs, cfg["short_window"], cfg["baseline_window"])
    if math.isnan(divergence) or divergence < cfg["min_divergence"]:
        return float("-inf")
    return divergence
