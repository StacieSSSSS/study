import numpy as np
import pytest

from strategies.fx_correlation.models.base import (
    ComboMetrics,
    combo_weights_to_pair_weights,
    position_size,
    rank_table,
    select_top_k,
)


def test_position_size_flat_inside_exit_band():
    assert position_size(z=0.3, entry_z=1.5, exit_z=0.5) == 0.0


def test_position_size_full_size_beyond_entry():
    assert position_size(z=2.0, entry_z=1.5, exit_z=0.5) == pytest.approx(-1.0)
    assert position_size(z=-2.0, entry_z=1.5, exit_z=0.5) == pytest.approx(1.0)


def test_position_size_linear_ramp_between_bands():
    midpoint_z = (1.5 + 0.5) / 2
    size = position_size(z=midpoint_z, entry_z=1.5, exit_z=0.5)
    assert size == pytest.approx(-0.5)


def test_position_size_nan_z_is_flat():
    assert position_size(z=float("nan"), entry_z=1.5, exit_z=0.5) == 0.0


def test_select_top_k_drops_filtered_combos_and_sorts_descending():
    combos = [("A", "B"), ("B", "C"), ("A", "C")]
    scores = {("A", "B"): 0.5, ("B", "C"): float("-inf"), ("A", "C"): 0.9}
    assert select_top_k(combos, scores, top_k=2) == [("A", "C"), ("A", "B")]


def test_select_top_k_respects_top_k_limit():
    combos = [("A", "B"), ("B", "C"), ("A", "C")]
    scores = {c: 1.0 for c in combos}
    assert len(select_top_k(combos, scores, top_k=1)) == 1


def _metrics(pair1, pair2, beta, zscore):
    return ComboMetrics(pair1=pair1, pair2=pair2, beta=beta, corrs={}, zscore=zscore, adf_p=float("nan"))


def test_combo_weights_to_pair_weights_empty_selection_is_all_zero():
    weights = combo_weights_to_pair_weights([], {}, universe=["A", "B"], entry_z=1.5, exit_z=0.5)
    assert (weights == 0.0).all()


def test_combo_weights_to_pair_weights_single_combo_matches_hedge_ratio():
    combo: tuple[str, str] = ("A", "B")
    # z=2.0 is beyond entry_z=1.5 -> full size, sign flipped (mean-reversion bet)
    metrics_by_combo: dict[tuple[str, str], ComboMetrics] = {combo: _metrics("A", "B", beta=2.0, zscore=2.0)}
    weights = combo_weights_to_pair_weights(
        [combo], metrics_by_combo, universe=["A", "B"], entry_z=1.5, exit_z=0.5
    )
    # size = -1.0 (z=2.0 > entry_z); pair2 gets size, pair1 gets -size*beta
    assert weights["B"] == pytest.approx(-1.0)
    assert weights["A"] == pytest.approx(2.0)


def test_combo_weights_to_pair_weights_splits_budget_across_selections():
    combos = [("A", "B"), ("A", "C")]
    metrics_by_combo = {
        ("A", "B"): _metrics("A", "B", beta=1.0, zscore=2.0),
        ("A", "C"): _metrics("A", "C", beta=1.0, zscore=2.0),
    }
    weights = combo_weights_to_pair_weights(
        combos, metrics_by_combo, universe=["A", "B", "C"], entry_z=1.5, exit_z=0.5
    )
    assert weights.abs().sum() == pytest.approx(2.0)  # 2 combos x (|size|+|size*beta|)/2 budget each


def test_rank_table_assigns_rank_1_to_highest_score_and_nan_rank_to_filtered():
    combos = [("A", "B"), ("B", "C")]
    metrics_by_combo = {
        ("A", "B"): _metrics("A", "B", beta=1.0, zscore=1.0),
        ("B", "C"): _metrics("B", "C", beta=1.0, zscore=0.5),
    }
    scores = {("A", "B"): 0.8, ("B", "C"): float("-inf")}
    table = rank_table(combos, metrics_by_combo, scores)
    best_row = table.loc[table["pair1"] == "A"].iloc[0]
    filtered_row = table.loc[table["pair1"] == "B"].iloc[0]
    assert best_row["rank"] == 1
    assert np.isnan(filtered_row["rank"])
