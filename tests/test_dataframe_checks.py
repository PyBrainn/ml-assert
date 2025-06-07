import pandas as pd
import pytest

from ml_assert.data.checks import (
    assert_column_in_range,
    assert_no_nulls,
    assert_unique,
    assert_values_in_set,
)


def test_assert_no_nulls_success():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    # Should not raise
    assert_no_nulls(df)


def test_assert_no_nulls_failure():
    df = pd.DataFrame({"a": [1, None, 2]})
    with pytest.raises(AssertionError) as exc:
        assert_no_nulls(df)
    assert "Column a contains 1 null values" in str(exc.value)


def test_assert_unique_success():
    df = pd.DataFrame({"id": [1, 2, 3]})
    # Should not raise
    assert_unique(df, "id")


def test_assert_unique_failure():
    df = pd.DataFrame({"id": [1, 2, 2, 3]})
    with pytest.raises(AssertionError) as exc:
        assert_unique(df, "id")
    assert "Column id has duplicate values: [2]" in str(exc.value)


def test_assert_column_in_range_success():
    df = pd.DataFrame({"val": [0.5, 1.0, 2.5]})
    # Should not raise
    assert_column_in_range(df, "val", min_value=0.0, max_value=3.0)


def test_assert_column_in_range_below():
    df = pd.DataFrame({"val": [1.0, -1.0, 2.0]})
    with pytest.raises(AssertionError) as exc:
        assert_column_in_range(df, "val", min_value=0.0)
    assert "Column val has values below 0.0: [-1.0]" in str(exc.value)


def test_assert_column_in_range_above():
    df = pd.DataFrame({"val": [1.0, 5.0, 2.0]})
    with pytest.raises(AssertionError) as exc:
        assert_column_in_range(df, "val", max_value=4.0)
    assert "Column val has values above 4.0: [5.0]" in str(exc.value)


def test_assert_values_in_set_success():
    df = pd.DataFrame({"col": ["a", "b", "a"]})
    allowed = {"a", "b", "c"}
    # Should not raise
    assert_values_in_set(df, "col", allowed)


def test_assert_values_in_set_failure():
    df = pd.DataFrame({"col": ["a", "x", "b"]})
    with pytest.raises(AssertionError) as exc:
        assert_values_in_set(df, "col", {"a", "b"})
    assert "Column col has values not in allowed set: ['x']" in str(exc.value)
