import numpy as np
import pytest

from ml_assert.stats.distribution import (
    assert_chi2_test,
    assert_ks_test,
    assert_wasserstein_distance,
)


def test_assert_ks_test_success():
    a = np.random.normal(0, 1, size=1000)
    b = a.copy()
    # should not raise
    assert_ks_test(a, b, alpha=0.01)


def test_assert_ks_test_failure():
    a = np.random.normal(0, 1, size=1000)
    b = np.random.normal(5, 1, size=1000)
    with pytest.raises(AssertionError) as exc:
        assert_ks_test(a, b, alpha=0.01)
    msg = str(exc.value)
    assert "KS test failed" in msg


def test_assert_chi2_test_success():
    observed = np.array([10, 20, 30])
    expected = observed.copy()
    # should not raise
    assert_chi2_test(observed, expected, alpha=0.01)


def test_assert_chi2_test_failure():
    observed = np.array([10, 0, 0])
    expected = np.array([10, 10, 10])
    with pytest.raises(AssertionError) as exc:
        assert_chi2_test(observed, expected, alpha=0.01)
    msg = str(exc.value)
    assert "Chi-square test" in msg


def test_assert_wasserstein_distance_success():
    a = np.linspace(0, 1, 100)
    b = np.linspace(0, 1, 100)
    # threshold high enough
    assert_wasserstein_distance(a, b, max_distance=0.1)


def test_assert_wasserstein_distance_failure():
    a = np.linspace(0, 1, 100)
    b = np.linspace(5, 6, 100)
    with pytest.raises(AssertionError) as exc:
        assert_wasserstein_distance(a, b, max_distance=1.0)
    msg = str(exc.value)
    assert "Wasserstein distance" in msg
