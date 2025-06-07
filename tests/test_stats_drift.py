import pandas as pd
import pytest

from ml_assert.stats import (
    assert_chi2_test,
    assert_ks_test,
    assert_no_drift,
    assert_wasserstein_distance,
)


# KS tests
def test_ks_pass():
    s1 = [1, 2, 3]
    s2 = [1, 2, 3]
    assert_ks_test(s1, s2, alpha=0.05)


def test_ks_fail():
    # disjoint distributions: large samples to trigger failure
    s1 = [0] * 100
    s2 = [1] * 100
    with pytest.raises(AssertionError) as exc:
        assert_ks_test(s1, s2, alpha=0.05)
    assert "KS test failed" in str(exc.value)


# Chi-square tests
def test_chi2_pass():
    obs = [10, 20]
    exp = [10, 20]
    assert_chi2_test(obs, exp, alpha=0.05)


def test_chi2_fail():
    obs = [100, 0]
    exp = [50, 50]
    with pytest.raises(AssertionError) as exc:
        assert_chi2_test(obs, exp, alpha=0.05)
    assert "Chi-square test" in str(exc.value)


# Wasserstein tests
def test_wasserstein_pass():
    s1 = [0, 0, 0]
    s2 = [1, 1, 1]
    assert_wasserstein_distance(s1, s2, max_distance=2.0)


def test_wasserstein_fail():
    s1 = [0, 0, 0]
    s2 = [10, 10, 10]
    with pytest.raises(AssertionError) as exc:
        assert_wasserstein_distance(s1, s2, max_distance=1.0)
    assert "Wasserstein distance" in str(exc.value)


# No drift tests
def test_no_drift_pass():
    df1 = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "a"]})
    df2 = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "a"]})
    assert_no_drift(df1, df2, alpha=0.05)


def test_no_drift_fail_numeric():
    # disjoint distributions: large samples to trigger failure
    df1 = pd.DataFrame({"num": [0] * 100})
    df2 = pd.DataFrame({"num": [1] * 100})
    with pytest.raises(AssertionError):
        assert_no_drift(df1, df2, alpha=0.05)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
def test_no_drift_fail_categorical():
    df1 = pd.DataFrame({"cat": ["a", "a", "b"]})
    df2 = pd.DataFrame({"cat": ["x", "x", "y"]})
    with pytest.raises(AssertionError):
        assert_no_drift(df1, df2, alpha=0.05)


def test_wasserstein_distance_fails():
    """Test that assert_wasserstein_distance fails when distance exceeds max."""
    series1 = [1, 2, 3]
    series2 = [11, 12, 13]
    with pytest.raises(AssertionError, match="exceeds max"):
        assert_wasserstein_distance(series1, series2, max_distance=5.0)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
def test_no_drift_autodetect_columns():
    """Test that assert_no_drift correctly auto-detects column types and fails."""
    df_train = pd.DataFrame({"numeric": [1, 2, 3], "categorical": ["a", "b", "c"]})
    df_test = pd.DataFrame({"numeric": [10, 20, 30], "categorical": ["d", "e", "f"]})
    with pytest.raises(AssertionError):
        # This should fail on the numeric column first
        assert_no_drift(df_train, df_test)
