"""
Data validation checks for pandas DataFrames.
"""

from collections.abc import Iterable

import pandas as pd


def assert_no_nulls(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Assert that specified columns have no null values. If columns is None, checks all columns.
    Raises AssertionError with count of nulls per column.
    """
    cols = columns or list(df.columns)
    for col in cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise AssertionError(f"Column {col} contains {null_count} null values")


def assert_unique(df: pd.DataFrame, column: str) -> None:
    """
    Assert that the values in the specified column are unique.
    Raises AssertionError listing duplicate values.
    """
    duplicated = df[column][df[column].duplicated()]
    if not duplicated.empty:
        dupes = duplicated.unique().tolist()
        raise AssertionError(f"Column {column} has duplicate values: {dupes}")


def assert_column_in_range(
    df: pd.DataFrame,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    """
    Assert that values in column fall within [min_value, max_value].
    Raises AssertionError listing out-of-range values.
    """
    series = df[column]
    if min_value is not None:
        below = series[series < min_value]
        if not below.empty:
            raise AssertionError(
                f"Column {column} has values below {min_value}: {below.tolist()}"
            )
    if max_value is not None:
        above = series[series > max_value]
        if not above.empty:
            raise AssertionError(
                f"Column {column} has values above {max_value}: {above.tolist()}"
            )


def assert_values_in_set(df: pd.DataFrame, column: str, allowed_set: Iterable) -> None:
    """
    Assert that all values in column are within allowed_set.
    Raises AssertionError listing disallowed values.
    """
    unique_vals = set(df[column].unique())
    allowed = set(allowed_set)
    invalid = unique_vals - allowed
    if invalid:
        raise AssertionError(
            f"Column {column} has values not in allowed set: {list(invalid)}"
        )
