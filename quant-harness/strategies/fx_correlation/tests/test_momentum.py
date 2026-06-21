import math

import pytest

from strategies.fx_correlation.lib.momentum import (
    classify_action,
    momentum_bucket,
    reversion_momentum,
    zscore_bucket,
)


def test_reversion_momentum_positive_when_positive_zscore_shrinking():
    # z was 2.0, now 1.2 -> moving toward zero -> reverting -> positive
    assert reversion_momentum(zscore_now=1.2, zscore_lagged=2.0) == pytest.approx(0.8)


def test_reversion_momentum_positive_when_negative_zscore_rising_toward_zero():
    # z was -2.0, now -1.2 -> also moving toward zero -> reverting -> positive
    assert reversion_momentum(zscore_now=-1.2, zscore_lagged=-2.0) == pytest.approx(0.8)


def test_reversion_momentum_negative_when_extending_further():
    # z was 1.2, now 2.0 -> moving away from zero -> extending -> negative
    assert reversion_momentum(zscore_now=2.0, zscore_lagged=1.2) == pytest.approx(-0.8)


def test_reversion_momentum_nan_propagates():
    assert math.isnan(reversion_momentum(float("nan"), 1.0))
    assert math.isnan(reversion_momentum(1.0, float("nan")))


def test_reversion_momentum_zero_at_zero():
    assert reversion_momentum(0.0, 5.0) == 0.0


@pytest.mark.parametrize(
    ("zscore", "expected"),
    [(2.0, "extreme"), (-2.0, "extreme"), (1.0, "moderate"), (-1.0, "moderate"), (0.2, "neutral")],
)
def test_zscore_bucket(zscore, expected):
    assert zscore_bucket(zscore, entry_z=1.5, exit_z=0.5) == expected


def test_zscore_bucket_nan_is_neutral():
    assert zscore_bucket(float("nan"), entry_z=1.5, exit_z=0.5) == "neutral"


@pytest.mark.parametrize(
    ("momentum", "expected"),
    [(0.5, "reverting"), (-0.5, "extending"), (0.1, "flat"), (-0.1, "flat")],
)
def test_momentum_bucket(momentum, expected):
    assert momentum_bucket(momentum, momentum_threshold=0.3) == expected


def test_momentum_bucket_nan_is_flat():
    assert momentum_bucket(float("nan"), momentum_threshold=0.3) == "flat"


@pytest.mark.parametrize(
    ("zscore", "momentum", "expected"),
    [
        (2.0, 0.5, "大力买入"),
        (2.0, 0.0, "买入"),
        (2.0, -0.5, "谨慎加仓"),
        (1.0, 0.5, "买入"),
        (1.0, 0.0, "持有"),
        (1.0, -0.5, "观望"),
        (0.2, 0.5, "获利了结"),
        (0.2, 0.0, "观望"),
        (0.2, -0.5, "减仓"),
    ],
)
def test_classify_action_covers_all_nine_cells(zscore, momentum, expected):
    assert classify_action(zscore, momentum, entry_z=1.5, exit_z=0.5, momentum_threshold=0.3) == expected


def test_classify_action_nan_zscore_is_wait():
    assert classify_action(float("nan"), 0.0, entry_z=1.5, exit_z=0.5, momentum_threshold=0.3) == "观望"
