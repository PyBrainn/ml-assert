from typing import Any

import pandas as pd


class Column:
    def __init__(self, name: str):
        self.name = name
        self.tests = []

    def is_type(self, dtype: Any):
        def test(series: pd.Series):
            if not pd.api.types.is_dtype_equal(series.dtype, dtype):
                raise AssertionError(
                    f"Column '{self.name}' has type {series.dtype}, expected {dtype}."
                )

        self.tests.append(test)
        return self

    def is_unique(self):
        def test(series: pd.Series):
            if not series.is_unique:
                raise AssertionError(f"Column '{self.name}' is not unique.")

        self.tests.append(test)
        return self

    def in_range(self, min_value: Any = None, max_value: Any = None):
        def test(series: pd.Series):
            if min_value is not None and (series < min_value).any():
                raise AssertionError(
                    f"Column '{self.name}' has values less than {min_value}."
                )
            if max_value is not None and (series > max_value).any():
                raise AssertionError(
                    f"Column '{self.name}' has values greater than {max_value}."
                )

        self.tests.append(test)
        return self

    def run_tests(self, series: pd.Series):
        for test_fn in self.tests:
            test_fn(series)


class Schema:
    def __init__(self):
        self.columns = {}

    def col(self, name: str) -> Column:
        column = Column(name)
        self.columns[name] = column
        return column

    def validate(self, df: pd.DataFrame):
        for name, column in self.columns.items():
            if name not in df.columns:
                raise AssertionError(f"Missing column: {name}")
            column.run_tests(df[name])


def schema() -> Schema:
    return Schema()
