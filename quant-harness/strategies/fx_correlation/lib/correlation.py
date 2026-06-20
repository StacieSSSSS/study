"""Rolling multi-window correlation and the scores derived from it.

All functions here are pure: given two already-point-in-time-sliced return
series (the caller is responsible for not handing them anything beyond the
current `as_of` date — see `core.data.point_in_time.PointInTimeFrame`), they
compute correlation/strength/stability/divergence with no further notion of
"now". That separation is what keeps look-ahead bias out of this module
regardless of how it's called.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(x: pd.Series, y: pd.Series) -> np.ndarray:
    # Plain numpy from here on — this is the hot path (called for every combo
    # on every backtest date), and np.corrcoef on a raw array is roughly an
    # order of magnitude faster than repeated pd.Series.corr() calls.
    aligned = pd.concat([x, y], axis=1, join="inner").dropna()
    return aligned.to_numpy()


def _tail_corr(aligned: np.ndarray, window: int) -> float:
    if len(aligned) < window:
        return float("nan")
    tail = aligned[-window:]
    if np.std(tail[:, 0]) == 0 or np.std(tail[:, 1]) == 0:
        return float("nan")
    corr = np.corrcoef(tail[:, 0], tail[:, 1])[0, 1]
    return float(corr) if not np.isnan(corr) else float("nan")


def rolling_corr_last(x: pd.Series, y: pd.Series, window: int) -> float:
    """Pearson correlation of the trailing `window` observations of x and y.

    Returns NaN if fewer than `window` overlapping observations are available
    (e.g. early in a backtest before enough history has accumulated).
    """
    return _tail_corr(_align(x, y), window)


def multi_window_correlation(x: pd.Series, y: pd.Series, windows: dict[str, int]) -> dict[str, float]:
    """Correlation of x vs y at each named window, e.g. {"1m": 21, "12m": 252}.

    Aligns x and y once and reuses that for every window, rather than
    re-aligning per window — this is called for every combo on every
    backtest date, so the repeated align was the dominant cost of a full run.
    """
    aligned = _align(x, y)
    return {name: _tail_corr(aligned, w) for name, w in windows.items()}


def strength_score(corrs: dict[str, float]) -> float:
    """Average absolute correlation across windows — how strongly related, ignoring sign."""
    values = [abs(v) for v in corrs.values() if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def stability_score(corrs: dict[str, float]) -> float:
    """Higher is more stable: negative of the cross-window std of correlation."""
    values = [v for v in corrs.values() if not np.isnan(v)]
    if len(values) < 2:
        return float("nan")
    return float(-np.std(values))


def divergence_score(corrs: dict[str, float], short_key: str, baseline_key: str) -> float:
    """How far the short-window correlation has drifted from the long-run baseline."""
    short_corr = corrs.get(short_key, float("nan"))
    baseline_corr = corrs.get(baseline_key, float("nan"))
    if np.isnan(short_corr) or np.isnan(baseline_corr):
        return float("nan")
    return float(abs(short_corr - baseline_corr))
