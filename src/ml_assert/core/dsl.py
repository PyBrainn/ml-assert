"""
Chainable assertion DSL for pandas DataFrames.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml_assert.core.base import Assertion, AssertionResult
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
        """Initialize the DataFrame assertion."""
        super().__init__()
        self._df = df
        self._assertions: list[dict[str, Any]] = []

    def satisfies(self, schema: Schema) -> "DataFrameAssertion":
        """
        Assert that the DataFrame satisfies the given schema.

        Args:
            schema: The schema to validate against.

        Returns:
            self for method chaining.
        """
        self._assertions.append(
            {
                "name": "schema",
                "fn": lambda: schema.validate(self._df),
                "args": {"schema": str(schema)},
            }
        )
        return self

    def no_nulls(self, columns: list[str] | None = None) -> "DataFrameAssertion":
        """
        Assert specified columns (or all) contain no null values.

        Args:
            columns: List of columns to check, or None for all columns.

        Returns:
            self for method chaining.
        """
        self._assertions.append(
            {
                "name": "no_nulls",
                "fn": lambda: assert_no_nulls(self._df, columns),
                "args": {"columns": columns},
            }
        )
        return self

    def unique(self, column: str) -> "DataFrameAssertion":
        """
        Assert values in 'column' are unique.

        Args:
            column: Name of the column to check.

        Returns:
            self for method chaining.
        """
        self._assertions.append(
            {
                "name": "unique",
                "fn": lambda: assert_unique(self._df, column),
                "args": {"column": column},
            }
        )
        return self

    def in_range(
        self,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> "DataFrameAssertion":
        """
        Assert values in 'column' fall within [min_value, max_value].

        Args:
            column: Name of the column to check.
            min_value: Minimum allowed value (inclusive).
            max_value: Maximum allowed value (inclusive).

        Returns:
            self for method chaining.
        """
        self._assertions.append(
            {
                "name": "in_range",
                "fn": lambda: assert_column_in_range(
                    self._df, column, min_value, max_value
                ),
                "args": {
                    "column": column,
                    "min_value": min_value,
                    "max_value": max_value,
                },
            }
        )
        return self

    def values_in_set(self, column: str, allowed_set: Iterable) -> "DataFrameAssertion":
        """
        Assert all values in 'column' are in allowed_set.

        Args:
            column: Name of the column to check.
            allowed_set: Set of allowed values.

        Returns:
            self for method chaining.
        """
        self._assertions.append(
            {
                "name": "values_in_set",
                "fn": lambda: assert_values_in_set(self._df, column, allowed_set),
                "args": {"column": column, "allowed_set": list(allowed_set)},
            }
        )
        return self

    def validate(self) -> AssertionResult:
        """
        Execute all chained assertions.

        Returns:
            AssertionResult containing the results of all assertions.

        Raises:
            AssertionError: If any assertion fails.
        """
        results = []
        for assertion in self._assertions:
            try:
                assertion["fn"]()
                results.append(
                    {
                        "name": assertion["name"],
                        "success": True,
                        "args": assertion["args"],
                    }
                )
            except AssertionError as e:
                results.append(
                    {
                        "name": assertion["name"],
                        "success": False,
                        "args": assertion["args"],
                        "error": str(e),
                    }
                )
                raise

        return AssertionResult(
            success=all(r["success"] for r in results),
            message="All DataFrame assertions passed"
            if all(r["success"] for r in results)
            else "Some DataFrame assertions failed",
            timestamp=datetime.now(),
            metadata={"results": results},
        )

    __call__ = validate


class ModelAssertion(Assertion):
    """
    A chainable assertion builder for model performance metrics.

    Usage:
        ModelAssertion(y_true, y_pred) \
            .accuracy(0.8) \
            .precision(0.7) \
            .recall(0.6) \
            .validate()
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Initialize the model assertion."""
        super().__init__()
        self._y_true = y_true
        self._y_pred = y_pred
        self._assertions: list[dict[str, Any]] = []

    def accuracy(
        self, threshold: float = None, min_score: float = None
    ) -> "ModelAssertion":
        """
        Assert accuracy score is above threshold.

        Args:
            threshold: Minimum acceptable accuracy score.
            min_score: (deprecated) Minimum acceptable accuracy score.

        Returns:
            self for method chaining.
        """
        if threshold is None and min_score is not None:
            threshold = min_score
        self._assertions.append(
            {
                "name": "accuracy",
                "fn": lambda: assert_accuracy_score(
                    self._y_true, self._y_pred, threshold
                ),
                "args": {"threshold": threshold},
            }
        )
        return self

    def precision(
        self, threshold: float = None, min_score: float = None
    ) -> "ModelAssertion":
        """
        Assert precision score is above threshold.

        Args:
            threshold: Minimum acceptable precision score.
            min_score: (deprecated) Minimum acceptable precision score.

        Returns:
            self for method chaining.
        """
        if threshold is None and min_score is not None:
            threshold = min_score
        self._assertions.append(
            {
                "name": "precision",
                "fn": lambda: assert_precision_score(
                    self._y_true, self._y_pred, threshold
                ),
                "args": {"threshold": threshold},
            }
        )
        return self

    def recall(
        self, threshold: float = None, min_score: float = None
    ) -> "ModelAssertion":
        """
        Assert recall score is above threshold.

        Args:
            threshold: Minimum acceptable recall score.
            min_score: (deprecated) Minimum acceptable recall score.

        Returns:
            self for method chaining.
        """
        if threshold is None and min_score is not None:
            threshold = min_score
        self._assertions.append(
            {
                "name": "recall",
                "fn": lambda: assert_recall_score(
                    self._y_true, self._y_pred, threshold
                ),
                "args": {"threshold": threshold},
            }
        )
        return self

    def f1(self, threshold: float = None, min_score: float = None) -> "ModelAssertion":
        """
        Assert F1 score is above threshold.

        Args:
            threshold: Minimum acceptable F1 score.
            min_score: (deprecated) Minimum acceptable F1 score.

        Returns:
            self for method chaining.
        """
        if threshold is None and min_score is not None:
            threshold = min_score
        self._assertions.append(
            {
                "name": "f1",
                "fn": lambda: assert_f1_score(self._y_true, self._y_pred, threshold),
                "args": {"threshold": threshold},
            }
        )
        return self

    def roc_auc(
        self, threshold: float = None, min_score: float = None
    ) -> "ModelAssertion":
        """
        Assert ROC AUC score is above threshold.

        Args:
            threshold: Minimum acceptable ROC AUC score.
            min_score: (deprecated) Minimum acceptable ROC AUC score.

        Returns:
            self for method chaining.
        """
        if threshold is None and min_score is not None:
            threshold = min_score
        if not hasattr(self, "_y_scores"):
            raise ValueError("y_scores must be provided for ROC AUC assertion")
        self._assertions.append(
            {
                "name": "roc_auc",
                "fn": lambda: assert_roc_auc_score(
                    self._y_true, self._y_scores, threshold
                ),
                "args": {"threshold": threshold},
            }
        )
        return self

    def validate(self) -> AssertionResult:
        """
        Execute all chained assertions.

        Returns:
            AssertionResult containing the results of all assertions.

        Raises:
            AssertionError: If any assertion fails.
        """
        results = []
        for assertion in self._assertions:
            try:
                assertion["fn"]()
                results.append(
                    {
                        "name": assertion["name"],
                        "success": True,
                        "args": assertion["args"],
                    }
                )
            except AssertionError as e:
                results.append(
                    {
                        "name": assertion["name"],
                        "success": False,
                        "args": assertion["args"],
                        "error": str(e),
                    }
                )
                raise

        return AssertionResult(
            success=all(r["success"] for r in results),
            message="All model assertions passed"
            if all(r["success"] for r in results)
            else "Some model assertions failed",
            timestamp=datetime.now(),
            metadata={"results": results},
        )

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
    return ModelAssertion(y_true, y_pred)
