# Data Assertions

The `ml_assert.data` module provides a suite of functions for validating the integrity and structure of your `pandas` DataFrames.

## `assert_dataframe_schema`

Validates that a DataFrame conforms to a specified schema.

- **Usage**: `assert_dataframe_schema(df, schema)`
- **Parameters**:
    - `df` (`pd.DataFrame`): The DataFrame to validate.
    - `schema` (`dict[str, str]`): A dictionary mapping column names to expected `dtype` strings (e.g., `'int64'`, `'float64'`, `'object'`).
- **Raises**: `AssertionError` if a column is missing, an extra column is present, or a column's `dtype` does not match.

## `assert_no_nulls`

Asserts that specified columns (or all columns) in a DataFrame contain no `NaN` or `None` values.

- **Usage**: `assert_no_nulls(df, columns=None)`
- **Parameters**:
    - `df` (`pd.DataFrame`): The DataFrame to check.
    - `columns` (`Optional[list[str]]`): A list of column names to check. If `None` (default), all columns are checked.
- **Raises**: `AssertionError` if any null values are found.

## `assert_unique`

Asserts that all values in a specified column are unique.

- **Usage**: `assert_unique(df, column)`
- **Parameters**:
    - `df` (`pd.DataFrame`): The DataFrame containing the column.
    - `column` (`str`): The name of the column to check for uniqueness.
- **Raises**: `AssertionError` if any duplicate values are found.

## `assert_column_in_range`

Asserts that all values in a numeric column fall within a specified inclusive range.

- **Usage**: `assert_column_in_range(df, column, min_value=None, max_value=None)`
- **Parameters**:
    - `df` (`pd.DataFrame`): The DataFrame containing the column.
    - `column` (`str`): The name of the numeric column.
    - `min_value` (`Optional[float]`): The minimum allowed value (inclusive).
    - `max_value` (`Optional[float]`): The maximum allowed value (inclusive).
- **Raises**: `AssertionError` if any value is outside the `[min_value, max_value]` range.

## `assert_values_in_set`

Asserts that all values in a column are present in a given set of allowed values.

- **Usage**: `assert_values_in_set(df, column, allowed_set)`
- **Parameters**:
    - `df` (`pd.DataFrame`): The DataFrame containing the column.
    - `column` (`str`): The name of the column to check.
    - `allowed_set` (`Iterable`): A set or list of allowed values.
- **Raises**: `AssertionError` if any value is found that is not in `allowed_set`.
