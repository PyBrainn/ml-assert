import numpy as np
import pytest

from ml_assert.core.dsl import ModelAssertion
from ml_assert.model.performance import (
    assert_accuracy_score,
    assert_f1_score,
    assert_precision_score,
    assert_recall_score,
    assert_roc_auc_score,
)

Y_TRUE = np.array([0, 1, 1, 0, 1, 0])
Y_PRED = np.array([0, 1, 0, 0, 1, 1])
Y_SCORES = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])


def test_accuracy_score_pass():
    """Test that assert_accuracy_score passes when the score is high enough."""
    assert_accuracy_score(Y_TRUE, Y_PRED, min_score=0.6)


def test_accuracy_score_fail():
    """Test that assert_accuracy_score fails when the score is too low."""
    with pytest.raises(AssertionError):
        assert_accuracy_score(Y_TRUE, Y_PRED, min_score=0.7)


def test_precision_score_pass():
    """Test that assert_precision_score passes when the score is high enough."""
    assert_precision_score(Y_TRUE, Y_PRED, min_score=0.6)


def test_precision_score_fail():
    """Test that assert_precision_score fails when the score is too low."""
    with pytest.raises(AssertionError):
        assert_precision_score(Y_TRUE, Y_PRED, min_score=0.7)


def test_recall_score_pass():
    """Test that assert_recall_score passes when the score is high enough."""
    assert_recall_score(Y_TRUE, Y_PRED, min_score=0.6)


def test_recall_score_fail():
    """Test that assert_recall_score fails when the score is too low."""
    with pytest.raises(AssertionError):
        assert_recall_score(Y_TRUE, Y_PRED, min_score=0.7)


def test_f1_score_pass():
    """Test that assert_f1_score passes when the score is high enough."""
    assert_f1_score(Y_TRUE, Y_PRED, min_score=0.6)


def test_f1_score_fail():
    """Test that assert_f1_score fails when the score is too low."""
    with pytest.raises(AssertionError):
        assert_f1_score(Y_TRUE, Y_PRED, min_score=0.7)


def test_roc_auc_score_pass():
    """Test that assert_roc_auc_score passes when the score is high enough."""
    assert_roc_auc_score(Y_TRUE, Y_SCORES, min_score=0.8)


def test_roc_auc_score_fail():
    """Test that assert_roc_auc_score fails when the score is too low."""
    with pytest.raises(AssertionError):
        assert_roc_auc_score(Y_TRUE, Y_SCORES, min_score=0.9)


def test_model_assertion_chain():
    """Test that the ModelAssertion DSL chain works as expected."""
    y_true = Y_TRUE
    y_pred = Y_PRED
    y_scores = Y_SCORES
    # Test with min_score
    ma = ModelAssertion(y_true, y_pred)
    ma._y_scores = y_scores
    ma.accuracy(min_score=0.6).precision(min_score=0.6).recall(min_score=0.6).f1(
        min_score=0.6
    ).roc_auc(min_score=0.8).validate()
    # Test with threshold
    ma2 = ModelAssertion(y_true, y_pred)
    ma2._y_scores = y_scores
    ma2.accuracy(threshold=0.6).precision(threshold=0.6).recall(threshold=0.6).f1(
        threshold=0.6
    ).roc_auc(threshold=0.8).validate()


def test_model_assertion_chain_fail():
    """Test that the ModelAssertion DSL chain fails on any failing assertion."""
    with pytest.raises(AssertionError):
        ModelAssertion(Y_TRUE, Y_PRED).accuracy(min_score=0.7).validate()
    with pytest.raises(AssertionError):
        ModelAssertion(Y_TRUE, Y_PRED).accuracy(threshold=0.7).validate()


def test_roc_auc_missing_scores_fail():
    """Test that ROC AUC assertion fails if y_scores is not provided."""
    with pytest.raises(ValueError, match="y_scores must be provided"):
        ModelAssertion(Y_TRUE, Y_PRED).roc_auc(min_score=0.8).validate()
    with pytest.raises(ValueError, match="y_scores must be provided"):
        ModelAssertion(Y_TRUE, Y_PRED).roc_auc(threshold=0.8).validate()
