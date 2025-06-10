import numpy as np

from ml_assert.fairness.fairness import FairnessMetrics


def test_fairness_metrics():
    """Test that the FairnessMetrics computes fairness metrics correctly."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    sensitive_attribute = np.array([0, 0, 1, 1, 0, 1])
    metrics = FairnessMetrics(y_true, y_pred, sensitive_attribute)
    assert metrics.demographic_parity() == 0.3333333333333333
    assert metrics.equal_opportunity() == 0.5
