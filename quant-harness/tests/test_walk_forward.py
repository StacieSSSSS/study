import pandas as pd
import pytest

from core.validation.walk_forward import run_walk_forward, walk_forward_splits


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_splits_cover_expected_number_of_windows():
    splits = list(walk_forward_splits(_index(10), train_size=4, test_size=2, step=2))
    # start=0 -> train[0:4] test[4:6]; start=2 -> train[2:6] test[6:8]; start=4 -> train[4:8] test[8:10]
    assert len(splits) == 3


def test_train_and_test_windows_do_not_overlap_and_are_contiguous():
    splits = list(walk_forward_splits(_index(10), train_size=4, test_size=2, step=2))
    first = splits[0]
    assert len(first.train) == 4
    assert len(first.test) == 2
    assert first.train[-1] < first.test[0]


def test_rejects_non_positive_parameters():
    with pytest.raises(ValueError):
        list(walk_forward_splits(_index(10), train_size=0, test_size=2, step=2))


def test_run_walk_forward_calls_evaluate_per_window():
    data = pd.DataFrame({"value": range(10)}, index=_index(10))
    calls = []

    def evaluate(train_df, test_df):
        calls.append((len(train_df), len(test_df)))
        return {"sharpe": 1.0}

    results = run_walk_forward(data, train_size=4, test_size=2, step=2, evaluate=evaluate)
    assert len(results) == len(calls) == 3
    assert all(r == {"sharpe": 1.0} for r in results)
