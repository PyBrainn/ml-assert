import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score


class CrossValidationError(Exception):
    """Custom exception for cross-validation related errors."""

    pass


def _validate_inputs(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str,
    n_splits: int,
) -> None:
    """Validate input parameters for cross-validation."""
    if not isinstance(model, BaseEstimator):
        raise CrossValidationError("Model must be a scikit-learn estimator")

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise CrossValidationError("X and y must be numpy arrays")

    if X.shape[0] != y.shape[0]:
        raise CrossValidationError("X and y must have the same number of samples")

    if cv_type not in ["kfold", "stratified", "loo"]:
        raise CrossValidationError(
            "cv_type must be one of: 'kfold', 'stratified', 'loo'"
        )

    if cv_type != "loo" and n_splits < 2:
        raise CrossValidationError(
            "n_splits must be at least 2 for k-fold cross-validation"
        )


def _get_cv_splitter(
    cv_type: str, n_splits: int
) -> KFold | StratifiedKFold | LeaveOneOut:
    """Get the appropriate cross-validation splitter."""
    if cv_type == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    elif cv_type == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:  # loo
        return LeaveOneOut()


def _compute_cv_scores(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str,
    n_splits: int,
    scoring: str | Callable,
) -> dict[str, Any]:
    """Compute cross-validation scores for a given metric."""
    _validate_inputs(model, X, y, cv_type, n_splits)

    cv = _get_cv_splitter(cv_type, n_splits)

    try:
        scores = cross_val_score(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # Use all available CPU cores
        )

        return {
            "scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
        }
    except Exception as e:
        raise CrossValidationError(f"Error during cross-validation: {str(e)}") from e


def assert_cv_accuracy_score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    min_score: float,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> None:
    """Assert that the mean accuracy score across cross-validation folds is above a minimum value."""
    cv_results = _compute_cv_scores(
        model=model, X=X, y=y, cv_type=cv_type, n_splits=n_splits, scoring="accuracy"
    )

    if cv_results["mean_score"] < min_score:
        raise AssertionError(
            f"Mean accuracy score {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f}) "
            f"is below the minimum threshold {min_score:.4f}"
        )


def assert_cv_precision_score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    min_score: float,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> None:
    """Assert that the mean precision score across cross-validation folds is above a minimum value."""
    cv_results = _compute_cv_scores(
        model=model, X=X, y=y, cv_type=cv_type, n_splits=n_splits, scoring="precision"
    )

    if cv_results["mean_score"] < min_score:
        raise AssertionError(
            f"Mean precision score {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f}) "
            f"is below the minimum threshold {min_score:.4f}"
        )


def assert_cv_recall_score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    min_score: float,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> None:
    """Assert that the mean recall score across cross-validation folds is above a minimum value."""
    cv_results = _compute_cv_scores(
        model=model, X=X, y=y, cv_type=cv_type, n_splits=n_splits, scoring="recall"
    )

    if cv_results["mean_score"] < min_score:
        raise AssertionError(
            f"Mean recall score {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f}) "
            f"is below the minimum threshold {min_score:.4f}"
        )


def assert_cv_f1_score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    min_score: float,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> None:
    """Assert that the mean F1 score across cross-validation folds is above a minimum value."""
    cv_results = _compute_cv_scores(
        model=model, X=X, y=y, cv_type=cv_type, n_splits=n_splits, scoring="f1"
    )

    if cv_results["mean_score"] < min_score:
        raise AssertionError(
            f"Mean F1 score {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f}) "
            f"is below the minimum threshold {min_score:.4f}"
        )


def assert_cv_roc_auc_score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    min_score: float,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> None:
    """Assert that the mean ROC AUC score across cross-validation folds is above a minimum value."""
    cv_results = _compute_cv_scores(
        model=model, X=X, y=y, cv_type=cv_type, n_splits=n_splits, scoring="roc_auc"
    )

    if cv_results["mean_score"] < min_score:
        raise AssertionError(
            f"Mean ROC AUC score {cv_results['mean_score']:.4f} (±{cv_results['std_score']:.4f}) "
            f"is below the minimum threshold {min_score:.4f}"
        )


def get_cv_summary(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str = "kfold",
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """Get a summary of all cross-validation metrics."""
    metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    summary = {}
    for metric_name, scoring in metrics.items():
        try:
            cv_results = _compute_cv_scores(
                model=model,
                X=X,
                y=y,
                cv_type=cv_type,
                n_splits=n_splits,
                scoring=scoring,
            )
            summary[metric_name] = {
                "mean": cv_results["mean_score"],
                "std": cv_results["std_score"],
                "min": cv_results["min_score"],
                "max": cv_results["max_score"],
            }
        except Exception as e:
            warnings.warn(f"Could not compute {metric_name}: {str(e)}", stacklevel=2)
            summary[metric_name] = None

    return summary
