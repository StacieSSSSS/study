"""Z-score momentum and the position-management action it implies.

The mean-reversion models already use the current z-score level as a value
signal: how stretched is this spread right now. This module adds a timing
dimension on top of it: is that stretch already correcting (the z-score
moving back toward zero) or still building (moving further away)? Combining
"how stretched" with "which way is it moving" is what turns a ranking into a
position-management action.
"""

from __future__ import annotations

import math

# (zscore_bucket, momentum_bucket) -> action. All 9 cells are covered; some
# actions appear in more than one cell because the underlying reasoning
# converges (e.g. "moderate but still extending" and "neutral but flat" both
# mean "no edge yet" -> 观望).
ACTIONS: dict[tuple[str, str], str] = {
    ("extreme", "reverting"): "大力买入",
    ("extreme", "flat"): "买入",
    ("extreme", "extending"): "谨慎加仓",
    ("moderate", "reverting"): "买入",
    ("moderate", "flat"): "持有",
    ("moderate", "extending"): "观望",
    ("neutral", "reverting"): "获利了结",
    ("neutral", "flat"): "观望",
    ("neutral", "extending"): "减仓",
}


def reversion_momentum(zscore_now: float, zscore_lagged: float) -> float:
    """Positive = the z-score is moving back toward zero (reversion already
    underway). Negative = it's moving further away (still extending). Sign
    is relative to which side of zero `zscore_now` is on, not an absolute
    direction — a z-score becoming less positive and one becoming less
    negative are both "reverting".
    """
    if math.isnan(zscore_now) or math.isnan(zscore_lagged):
        return float("nan")
    if zscore_now == 0:
        return 0.0
    direction = -1.0 if zscore_now > 0 else 1.0
    return direction * (zscore_now - zscore_lagged)


def zscore_bucket(zscore: float, entry_z: float, exit_z: float) -> str:
    if math.isnan(zscore):
        return "neutral"
    magnitude = abs(zscore)
    if magnitude >= entry_z:
        return "extreme"
    if magnitude >= exit_z:
        return "moderate"
    return "neutral"


def momentum_bucket(momentum: float, momentum_threshold: float) -> str:
    if math.isnan(momentum):
        return "flat"
    if momentum > momentum_threshold:
        return "reverting"
    if momentum < -momentum_threshold:
        return "extending"
    return "flat"


def classify_action(
    zscore: float, momentum: float, entry_z: float, exit_z: float, momentum_threshold: float
) -> str:
    """The 7-level action implied by the current z-score and its momentum."""
    if math.isnan(zscore):
        return "观望"
    z_bucket = zscore_bucket(zscore, entry_z, exit_z)
    m_bucket = momentum_bucket(momentum, momentum_threshold)
    return ACTIONS[(z_bucket, m_bucket)]
