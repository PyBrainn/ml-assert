import pandas as pd
import pytest

from ml_assert.core.dsl import DataFrameAssertion


def test_dsl_success():
    df = pd.DataFrame(
        {"id": [1, 2, 3], "score": [0.5, 0.8, 1.0], "category": ["a", "b", "a"]}
    )
    # Should not raise
    DataFrameAssertion(df).schema(
        {"id": "int64", "score": "float64", "category": "object"}
    ).no_nulls().unique("id").in_range("score", 0.0, 1.0).values_in_set(
        "category", ["a", "b"]
    ).validate()


def test_dsl_schema_failure():
    df = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).schema({"id": "int64", "score": "float64"}).validate()
    assert "Missing column: score" in str(exc.value)


def test_dsl_no_nulls_failure():
    df = pd.DataFrame({"col": [1, None]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).no_nulls().validate()
    assert "Column col contains 1 null values" in str(exc.value)


def test_dsl_unique_failure():
    df = pd.DataFrame({"id": [1, 2, 2]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).unique("id").validate()
    assert "duplicate values" in str(exc.value)


def test_dsl_in_range_failure():
    df = pd.DataFrame({"val": [1, 5, 3]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).in_range("val", 0.0, 4.0).validate()
    assert "values above 4.0" in str(exc.value)


def test_dsl_values_in_set_failure():
    df = pd.DataFrame({"col": ["a", "x"]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).values_in_set("col", ["a", "b"]).validate()
    assert "not in allowed set" in str(exc.value)
