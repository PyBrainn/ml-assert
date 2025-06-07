"""
Chainable assertion DSL for pandas DataFrames.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ml_assert.core.base import Assertion
from ml_assert.data.checks import (
    assert_column_in_range,
    assert_no_nulls,
    assert_unique,
    assert_values_in_set,
)
from ml_assert.model.performance import (
    assert_accuracy_score,
    assert_f1_score,
    assert_precision_score,
    assert_recall_score,
    assert_roc_auc_score,
)
from ml_assert.schema import Schema


class DataFrameAssertion(Assertion):
    """
    A chainable assertion builder for pandas DataFrames.

    Usage:
        DataFrameAssertion(df) \
            .schema({"id": "int64", "score": "float64"}) \
            .no_nulls() \
            .unique("id") \
            .in_range("score", 0.0, 1.0) \
            .validate()
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._assertions: list = []

    def satisfies(self, schema: Schema) -> "DataFrameAssertion":
        """
        Assert that the DataFrame satisfies the given schema.
        """
        self._assertions.append(lambda: schema.validate(self._df))
        return self

    def no_nulls(self, columns: list[str] | None = None) -> "DataFrameAssertion":
        """
        Assert specified columns (or all) contain no null values.
        """
        self._assertions.append(lambda: assert_no_nulls(self._df, columns))
        return self

    def unique(self, column: str) -> "DataFrameAssertion":
        """
        Assert values in 'column' are unique.
        """
        self._assertions.append(lambda: assert_unique(self._df, column))
        return self

    def in_range(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> "DataFrameAssertion":
        """
        Assert values in 'column' fall within [min_value, max_value].
        """
        self._assertions.append(
            lambda: assert_column_in_range(self._df, column, min_value, max_value)
        )
        return self

    def values_in_set(self, column: str, allowed_set: Iterable) -> "DataFrameAssertion":
        """
        Assert all values in 'column' are in allowed_set.
        """
        self._assertions.append(
            lambda: assert_values_in_set(self._df, column, allowed_set)
        )
        return self

    def validate(self) -> None:
        """
        Execute all chained assertions. Raises on first failure.
        """
        for assert_fn in self._assertions:
            assert_fn()

    __call__ = validate  # allow instance() syntax


class ModelAssertion(Assertion):
    """
    A chainable assertion builder for model performance.

    Usage:
        assert_model(y_true, y_pred, y_scores) \
            .accuracy(min_score=0.8) \
            .precision(min_score=0.75) \
            .recall(min_score=0.85) \
            .f1(min_score=0.79) \
            .roc_auc(min_score=0.9) \
            .validate()
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray | None = None,
    ):
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_scores = y_scores
        self._assertions: list = []

    def accuracy(self, min_score: float) -> "ModelAssertion":
        """Asserts that the accuracy score is above a minimum value."""
        self._assertions.append(
            lambda: assert_accuracy_score(self._y_true, self._y_pred, min_score)
        )
        return self

    def precision(self, min_score: float) -> "ModelAssertion":
        """Asserts that the precision score is above a minimum value."""
        self._assertions.append(
            lambda: assert_precision_score(self._y_true, self._y_pred, min_score)
        )
        return self

    def recall(self, min_score: float) -> "ModelAssertion":
        """Asserts that the recall score is above a minimum value."""
        self._assertions.append(
            lambda: assert_recall_score(self._y_true, self._y_pred, min_score)
        )
        return self

    def f1(self, min_score: float) -> "ModelAssertion":
        """Asserts that the F1 score is above a minimum value."""
        self._assertions.append(
            lambda: assert_f1_score(self._y_true, self._y_pred, min_score)
        )
        return self

    def roc_auc(self, min_score: float) -> "ModelAssertion":
        """Asserts that the ROC AUC score is above a minimum value."""
        if self._y_scores is None:
            raise ValueError("y_scores must be provided for ROC AUC assertion.")
        self._assertions.append(
            lambda: assert_roc_auc_score(self._y_true, self._y_scores, min_score)
        )
        return self

    def validate(self) -> None:
        """
        Execute all chained assertions. Raises on first failure.
        """
        for assert_fn in self._assertions:
            assert_fn()

    __call__ = validate


def assert_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> ModelAssertion:
    """
    Entry point for chainable model performance assertions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: Target scores, can be probability estimates of the positive class,
                  confidence values, or non-thresholded measure of decisions.

    Returns:
        A ModelAssertion instance.
    """
    return ModelAssertion(y_true, y_pred, y_scores)
