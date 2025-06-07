import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def assert_accuracy_score(
    y_true: np.ndarray, y_pred: np.ndarray, min_score: float
) -> None:
    """Asserts that the accuracy score is above a minimum value."""
    score = accuracy_score(y_true, y_pred)
    if score < min_score:
        raise AssertionError(
            f"Accuracy score {score:.4f} is below the minimum threshold {min_score:.4f}"
        )


def assert_precision_score(
    y_true: np.ndarray, y_pred: np.ndarray, min_score: float
) -> None:
    """Asserts that the precision score is above a minimum value."""
    score = precision_score(y_true, y_pred)
    if score < min_score:
        raise AssertionError(
            f"Precision score {score:.4f} is below the minimum threshold {min_score:.4f}"
        )


def assert_recall_score(
    y_true: np.ndarray, y_pred: np.ndarray, min_score: float
) -> None:
    """Asserts that the recall score is above a minimum value."""
    score = recall_score(y_true, y_pred)
    if score < min_score:
        raise AssertionError(
            f"Recall score {score:.4f} is below the minimum threshold {min_score:.4f}"
        )


def assert_f1_score(y_true: np.ndarray, y_pred: np.ndarray, min_score: float) -> None:
    """Asserts that the F1 score is above a minimum value."""
    score = f1_score(y_true, y_pred)
    if score < min_score:
        raise AssertionError(
            f"F1 score {score:.4f} is below the minimum threshold {min_score:.4f}"
        )


def assert_roc_auc_score(
    y_true: np.ndarray, y_scores: np.ndarray, min_score: float
) -> None:
    """Asserts that the ROC AUC score is above a minimum value."""
    score = roc_auc_score(y_true, y_scores)
    if score < min_score:
        raise AssertionError(
            f"ROC AUC score {score:.4f} is below the minimum threshold {min_score:.4f}"
        )
