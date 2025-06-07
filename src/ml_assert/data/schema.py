"""
Data schema assertions for pandas DataFrames.
"""

from pathlib import Path

import pandas as pd
import yaml


def validate_schema(file_path: Path, schema_path: Path) -> None:
    """Read a CSV and a schema YAML and validate the DataFrame."""
    df = pd.read_csv(file_path)
    schema_def = yaml.safe_load(schema_path.read_text())
    assert_dataframe_schema(df, schema_def)
    print("Schema validation passed.")


def assert_dataframe_schema(df: pd.DataFrame, schema: dict[str, str]) -> None:
    """
    Assert that DataFrame 'df' has exactly the columns and dtypes defined in 'schema'.

    schema: mapping of column name -> expected pandas dtype (e.g. "int64", "float64").
    Raises AssertionError if any column is missing or has a mismatched dtype.
    """
    for col, dtype in schema.items():
        if col not in df.columns:
            raise AssertionError(f"Missing column: {col}")
        actual = str(df[col].dtype)
        if actual != dtype:
            raise AssertionError(f"Column {col} has dtype {actual}, expected {dtype}")
