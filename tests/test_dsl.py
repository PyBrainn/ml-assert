import pandas as pd
import pytest

from ml_assert.core.dsl import DataFrameAssertion
from ml_assert.schema import schema


def test_dsl_success():
    df = pd.DataFrame(
        {"id": [1, 2, 3], "score": [0.5, 0.8, 1.0], "category": ["a", "b", "a"]}
    )
    # Should not raise
    s = schema()
    s.col("id").is_type("int64").is_unique()
    s.col("score").is_type("float64").in_range(0.0, 1.0)
    s.col("category").is_type("object")

    DataFrameAssertion(df).satisfies(s).no_nulls().values_in_set(
        "category", ["a", "b"]
    ).validate()


def test_dsl_schema_failure():
    df = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(AssertionError, match="Missing column: score"):
        s = schema()
        s.col("id").is_type("int64")
        s.col("score").is_type("float64")
        DataFrameAssertion(df).satisfies(s).validate()


def test_dsl_no_nulls_failure():
    df = pd.DataFrame({"id": [1, None, 3]})
    with pytest.raises(AssertionError) as exc:
        DataFrameAssertion(df).no_nulls().validate()
    assert "Column id contains 1 null values" in str(exc.value)


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
