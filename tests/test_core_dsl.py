"""Tests for the chainable assertion DSL."""

import numpy as np
import pandas as pd
import pytest

from ml_assert.core.dsl import DataFrameAssertion, ModelAssertion
from ml_assert.schema import schema


def test_dataframe_assertion_chain():
    """Test chaining DataFrame assertions."""
    df = pd.DataFrame(
        {"id": [1, 2, 3], "score": [0.5, 0.7, 0.9], "category": ["A", "B", "A"]}
    )

    # Test successful assertions
    result = (
        DataFrameAssertion(df)
        .no_nulls()
        .unique("id")
        .in_range("score", 0.0, 1.0)
        .values_in_set("category", {"A", "B"})
        .validate()
    )

    assert result.success
    assert "All DataFrame assertions passed" in result.message
    assert len(result.metadata["results"]) == 4
    assert all(r["success"] for r in result.metadata["results"])

    # Test failed assertion
    with pytest.raises(AssertionError):
        DataFrameAssertion(df).in_range("score", 0.0, 0.8).validate()

    # Test schema validation
    s = schema()
    s.col("id").is_type("int64")
    s.col("score").is_type("float64")
    s.col("category").is_type("object")
    result = DataFrameAssertion(df).satisfies(s).validate()

    assert result.success
    assert "All DataFrame assertions passed" in result.message
    assert len(result.metadata["results"]) == 1
    assert result.metadata["results"][0]["name"] == "schema"


def test_model_assertion_chain():
    """Test chaining model performance assertions."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 1, 0])

    # Test successful assertions
    result = (
        ModelAssertion(y_true, y_pred)
        .accuracy(0.6)
        .precision(0.5)
        .recall(0.5)
        .f1(0.5)
        .validate()
    )

    assert result.success
    assert "All model assertions passed" in result.message
    assert len(result.metadata["results"]) == 4
    assert all(r["success"] for r in result.metadata["results"])

    # Test failed assertion
    with pytest.raises(AssertionError):
        ModelAssertion(y_true, y_pred).accuracy(0.8).validate()

    # Test ROC AUC
    y_scores = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
    ma = ModelAssertion(y_true, y_pred)
    ma._y_scores = y_scores
    result = ma.roc_auc(0.7).validate()
    assert result.success
    assert "All model assertions passed" in result.message
    assert len(result.metadata["results"]) == 1
    assert result.metadata["results"][0]["name"] == "roc_auc"
