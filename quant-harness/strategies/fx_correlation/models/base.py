"""Shared scaffolding for all three models.

A model is just a *selection score* function: given the current correlation/
z-score/cointegration metrics for a combo, how attractive is it right now?
Everything else — fitting the hedge ratio on train data, computing rolling
correlation and the z-scored spread as of a given date, sizing the position
once a combo is selected, and turning a set of selected combos into a weight
vector over the underlying FX pairs — is identical across models and lives
here so model_a/b/c.py only contain the ~10 lines that actually differ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from core.data.point_in_time import PointInTimeFrame
from strategies.fx_correlation.lib.cointegration import adf_test
from strategies.fx_correlation.lib.correlation import multi_window_correlation
from strategies.fx_correlation.lib.pairs import enumerate_combos
from strategies.fx_correlation.lib.spread import compute_spread, hedge_ratio, latest_zscore


@dataclass(frozen=True)
class ComboMetrics:
    pair1: str
    pair2: str
    beta: float
    corrs: dict[str, float]
    zscore: float
    adf_p: float
    adf_stat: float = float("nan")


SelectionScoreFn = Callable[[ComboMetrics, dict], float]


def fit_hedge_ratios(train_df: pd.DataFrame, combos: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Fit each combo's hedge ratio once on the training window — held fixed
    through the test window so the ratio itself can't leak test-period information.
    """
    return {
        (p1, p2): hedge_ratio(cast(pd.Series, train_df[p1]), cast(pd.Series, train_df[p2]))
        for p1, p2 in combos
    }


def compute_combo_metrics(
    view: pd.DataFrame,
    pair1: str,
    pair2: str,
    beta: float,
    windows: dict[str, int],
    zscore_lookback: int,
    need_adf: bool = True,
) -> ComboMetrics:
    """`view` is the panel already sliced to the current `as_of` date via
    `PointInTimeFrame.as_of()` — callers slice once per date and reuse the
    view across all combos rather than re-slicing per combo.

    Nothing below needs more than the longest configured correlation window:
    correlation itself only looks at trailing windows up to that length, the
    z-score only needs `zscore_lookback` trailing rows, and the ADF test is
    deliberately evaluated on the same bounded recent window rather than the
    ever-growing full history — recent cointegration evidence is what matters
    for a live trade, and it keeps the (otherwise expensive) ADF test's cost
    constant per call instead of growing with backtest length.
    """
    max_lookback = max(max(windows.values()), zscore_lookback)
    recent = view.tail(max_lookback)
    x, y = cast(pd.Series, recent[pair1]), cast(pd.Series, recent[pair2])
    corrs = multi_window_correlation(x, y, windows)
    if np.isnan(beta):
        return ComboMetrics(pair1, pair2, beta, corrs, float("nan"), float("nan"))
    spread = compute_spread(x, y, beta)
    z = latest_zscore(spread, zscore_lookback)
    if need_adf:
        adf_result = adf_test(spread)
        return ComboMetrics(pair1, pair2, beta, corrs, z, adf_result.pvalue, adf_result.statistic)
    return ComboMetrics(pair1, pair2, beta, corrs, z, float("nan"), float("nan"))


def position_size(z: float, entry_z: float, exit_z: float) -> float:
    """Stateless position sizing from a z-score, in [-1, 1].

    Flat inside `exit_z`, full size beyond `entry_z`, linear ramp between —
    a spread trading too far above its mean (z > 0) is bet to fall, so the
    sign is flipped relative to z.
    """
    if np.isnan(z) or abs(z) <= exit_z:
        return 0.0
    if abs(z) >= entry_z:
        return -float(np.sign(z))
    frac = (abs(z) - exit_z) / (entry_z - exit_z)
    return -float(np.sign(z)) * frac


def select_top_k(
    combos: list[tuple[str, str]],
    scores: dict[tuple[str, str], float],
    top_k: int,
) -> list[tuple[str, str]]:
    """The top_k combos by score, dropping any with a non-finite (filtered-out) score."""
    ranked = sorted(
        (c for c in combos if np.isfinite(scores.get(c, float("-inf")))),
        key=lambda c: scores[c],
        reverse=True,
    )
    return ranked[:top_k]


def combo_weights_to_pair_weights(
    selected: list[tuple[str, str]],
    metrics_by_combo: dict[tuple[str, str], ComboMetrics],
    universe: list[str],
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    """Turn the selected combos' positions into a weight vector over the
    underlying FX pairs, equal-weighted across however many combos are
    selected (so gross exposure is bounded regardless of top_k).
    """
    weights = pd.Series(0.0, index=universe)
    if not selected:
        return weights
    per_combo_budget = 1.0 / len(selected)
    for combo in selected:
        metrics = metrics_by_combo[combo]
        size = position_size(metrics.zscore, entry_z, exit_z)
        # spread = y - beta * x ; betting the spread falls (size<0) means short y, long beta*x.
        weights[metrics.pair2] += size * per_combo_budget
        weights[metrics.pair1] += -size * metrics.beta * per_combo_budget
    return weights


def all_combos(universe: list[str]) -> list[tuple[str, str]]:
    return enumerate_combos(universe)


def score_date(
    pit: PointInTimeFrame,
    as_of: pd.Timestamp,
    universe: list[str],
    combos: list[tuple[str, str]],
    betas: dict[tuple[str, str], float],
    windows: dict[str, int],
    zscore_lookback: int,
    selection_score_fn: SelectionScoreFn,
    model_cfg: dict,
    need_adf: bool = True,
) -> tuple[dict[tuple[str, str], ComboMetrics], dict[tuple[str, str], float]]:
    """Metrics and selection scores for every combo as of a single date."""
    view = pit.as_of(as_of)
    metrics_by_combo = {
        combo: compute_combo_metrics(
            view, combo[0], combo[1], betas[combo], windows, zscore_lookback, need_adf
        )
        for combo in combos
    }
    scores = {combo: selection_score_fn(metrics_by_combo[combo], model_cfg) for combo in combos}
    return metrics_by_combo, scores


def run_over_dates(
    pit: PointInTimeFrame,
    dates: pd.DatetimeIndex,
    universe: list[str],
    train_df: pd.DataFrame,
    windows: dict[str, int],
    zscore_lookback: int,
    top_k: int,
    entry_z: float,
    exit_z: float,
    selection_score_fn: SelectionScoreFn,
    model_cfg: dict,
    need_adf: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], ComboMetrics]]:
    """Daily target weights over `dates`, plus the most recent date's combo
    metrics (handy for callers that want a snapshot without a second pass).
    """
    combos = all_combos(universe)
    betas = fit_hedge_ratios(train_df, combos)

    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    last_metrics: dict[tuple[str, str], ComboMetrics] = {}
    for as_of in dates:
        metrics_by_combo, scores = score_date(
            pit, as_of, universe, combos, betas, windows, zscore_lookback,
            selection_score_fn, model_cfg, need_adf,
        )
        selected = select_top_k(combos, scores, top_k)
        weight_rows[as_of] = combo_weights_to_pair_weights(
            selected, metrics_by_combo, universe, entry_z, exit_z
        )
        last_metrics = metrics_by_combo

    weights_df = pd.DataFrame(weight_rows).T
    weights_df.index.name = None
    return weights_df, last_metrics


def rank_table(
    combos: list[tuple[str, str]],
    metrics_by_combo: dict[tuple[str, str], ComboMetrics],
    scores: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """One row per combo, sorted by score descending, rank 1 = most attractive.
    Used by the conviction report to compare rankings across models.
    """
    rows = []
    for combo in combos:
        metrics = metrics_by_combo[combo]
        row = {
            "pair1": metrics.pair1,
            "pair2": metrics.pair2,
            "score": scores.get(combo, float("nan")),
            "zscore": metrics.zscore,
            "adf_p": metrics.adf_p,
            "adf_stat": metrics.adf_stat,
        }
        row.update({f"corr_{name}": value for name, value in metrics.corrs.items()})
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")
    table = table.reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.loc[table["score"].isna() | ~np.isfinite(table["score"]), "rank"] = np.nan
    return table
