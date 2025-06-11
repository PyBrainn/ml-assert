import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_assert.model.cross_validation import (
    CrossValidationError,
    assert_cv_accuracy_score,
    assert_cv_f1_score,
    assert_cv_precision_score,
    assert_cv_recall_score,
    assert_cv_roc_auc_score,
    get_cv_summary,
)


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    return X, y


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return LogisticRegression(random_state=42)


def test_cv_accuracy_score(sample_data, sample_model):
    """Test cross-validation accuracy score assertion."""
    X, y = sample_data
    # Should pass with a reasonable threshold
    assert_cv_accuracy_score(sample_model, X, y, min_score=0.5)

    # Should fail with an unreasonable threshold
    with pytest.raises(AssertionError):
        assert_cv_accuracy_score(sample_model, X, y, min_score=1.0)


def test_cv_precision_score(sample_data, sample_model):
    """Test cross-validation precision score assertion."""
    X, y = sample_data
    # Should pass with a reasonable threshold
    assert_cv_precision_score(sample_model, X, y, min_score=0.5)

    # Should fail with an unreasonable threshold
    with pytest.raises(AssertionError):
        assert_cv_precision_score(sample_model, X, y, min_score=1.0)


def test_cv_recall_score(sample_data, sample_model):
    """Test cross-validation recall score assertion."""
    X, y = sample_data
    # Should pass with a reasonable threshold
    assert_cv_recall_score(sample_model, X, y, min_score=0.5)

    # Should fail with an unreasonable threshold
    with pytest.raises(AssertionError):
        assert_cv_recall_score(sample_model, X, y, min_score=1.0)


def test_cv_f1_score(sample_data, sample_model):
    """Test cross-validation F1 score assertion."""
    X, y = sample_data
    # Should pass with a reasonable threshold
    assert_cv_f1_score(sample_model, X, y, min_score=0.5)

    # Should fail with an unreasonable threshold
    with pytest.raises(AssertionError):
        assert_cv_f1_score(sample_model, X, y, min_score=1.0)


def test_cv_roc_auc_score(sample_data, sample_model):
    """Test cross-validation ROC AUC score assertion."""
    X, y = sample_data
    # Should pass with a reasonable threshold
    assert_cv_roc_auc_score(sample_model, X, y, min_score=0.5)

    # Should fail with an unreasonable threshold
    with pytest.raises(AssertionError):
        assert_cv_roc_auc_score(sample_model, X, y, min_score=1.0)


def test_different_cv_types(sample_data, sample_model):
    """Test different cross-validation types."""
    X, y = sample_data

    # Test K-Fold
    assert_cv_accuracy_score(
        sample_model, X, y, min_score=0.5, cv_type="kfold", n_splits=5
    )

    # Test Stratified K-Fold
    assert_cv_accuracy_score(
        sample_model, X, y, min_score=0.5, cv_type="stratified", n_splits=5
    )

    # Test Leave-One-Out
    assert_cv_accuracy_score(sample_model, X, y, min_score=0.5, cv_type="loo")


def test_invalid_inputs(sample_data):
    """Test invalid input handling."""
    X, y = sample_data

    # Test invalid model
    with pytest.raises(CrossValidationError):
        assert_cv_accuracy_score("not_a_model", X, y, min_score=0.5)

    # Test invalid data types
    with pytest.raises(CrossValidationError):
        assert_cv_accuracy_score(LogisticRegression(), "not_an_array", y, min_score=0.5)

    # Test invalid CV type
    with pytest.raises(CrossValidationError):
        assert_cv_accuracy_score(
            LogisticRegression(), X, y, min_score=0.5, cv_type="invalid"
        )

    # Test invalid n_splits
    with pytest.raises(CrossValidationError):
        assert_cv_accuracy_score(LogisticRegression(), X, y, min_score=0.5, n_splits=1)


def test_cv_summary(sample_data, sample_model):
    """Test cross-validation summary function."""
    X, y = sample_data
    summary = get_cv_summary(sample_model, X, y)

    # Check that all metrics are present
    assert all(
        metric in summary
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    # Check that each metric has the required statistics
    for metric_stats in summary.values():
        if metric_stats is not None:  # Some metrics might be None for certain datasets
            assert all(stat in metric_stats for stat in ["mean", "std", "min", "max"])


def test_different_models(sample_data):
    """Test cross-validation with different model types."""
    X, y = sample_data

    # Test with Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    assert_cv_accuracy_score(rf_model, X, y, min_score=0.5)

    # Test with Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    assert_cv_accuracy_score(lr_model, X, y, min_score=0.5)


def test_parallel_processing(sample_data, sample_model):
    """Test that parallel processing works correctly."""
    X, y = sample_data

    # Test with different numbers of splits to ensure parallel processing works
    for n_splits in [2, 5, 10]:
        assert_cv_accuracy_score(sample_model, X, y, min_score=0.5, n_splits=n_splits)
