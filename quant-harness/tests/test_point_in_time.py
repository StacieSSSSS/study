import pandas as pd
import pytest

from core.data.point_in_time import FutureDataAccessError, PointInTimeFrame


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)


def test_as_of_excludes_future_rows():
    pit = PointInTimeFrame(_frame())
    view = pit.as_of("2024-01-03")
    assert list(view["value"]) == [1, 2, 3]


def test_latest_returns_most_recent_row_at_cutoff():
    pit = PointInTimeFrame(_frame())
    row = pit.latest("2024-01-03")
    assert row["value"] == 3


def test_latest_raises_when_no_data_available_yet():
    pit = PointInTimeFrame(_frame())
    with pytest.raises(FutureDataAccessError):
        pit.latest("2023-12-31")


def test_requires_datetime_index():
    with pytest.raises(TypeError):
        PointInTimeFrame(pd.DataFrame({"value": [1, 2, 3]}))


def test_len_reflects_full_history():
    pit = PointInTimeFrame(_frame())
    assert len(pit) == 5
