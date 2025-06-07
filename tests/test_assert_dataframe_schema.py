import pandas as pd
import pytest

from ml_assert.data.schema import assert_dataframe_schema


def test_assert_dataframe_schema_success():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    schema = {"a": "int64", "b": "float64"}
    # Should not raise
    assert_dataframe_schema(df, schema)


def test_assert_dataframe_schema_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    schema = {"a": "int64", "b": "float64"}
    with pytest.raises(AssertionError) as exc:
        assert_dataframe_schema(df, schema)
    assert "Missing column: b" in str(exc.value)


def test_assert_dataframe_schema_dtype_mismatch():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    schema = {"a": "int64", "b": "float64"}
    with pytest.raises(AssertionError) as exc:
        assert_dataframe_schema(df, schema)
    assert "Column b has dtype object, expected float64" in str(exc.value)
