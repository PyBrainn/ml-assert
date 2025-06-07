"""
Distribution tests for pandas Series.
"""

import numpy as np
from scipy import stats


def assert_ks_test(
    sample1: np.ndarray, sample2: np.ndarray, alpha: float = 0.05
) -> None:
    """
    Perform a two-sample Kolmogorov-Smirnov test between sample1 and sample2.
    Raises AssertionError if p-value < alpha.
    """
    stat, pvalue = stats.ks_2samp(sample1, sample2)
    if pvalue < alpha:
        raise AssertionError(
            f"KS test failed (statistic={stat:.4f}, p-value={pvalue:.4f} < {alpha})"
        )


def assert_chi2_test(
    observed: np.ndarray, expected: np.ndarray, alpha: float = 0.05
) -> None:
    """
    Perform a Chi-squared goodness-of-fit test.
    observed and expected should be counts of same shape.
    Raises AssertionError if p-value < alpha.
    """
    try:
        stat, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
    except ValueError as e:
        raise AssertionError(f"Chi-square test invalid: {e}") from e
    if pvalue < alpha:
        raise AssertionError(
            f"Chi-square test failed: p-value {pvalue} < alpha {alpha}"
        )


def assert_wasserstein_distance(
    sample1: np.ndarray, sample2: np.ndarray, max_distance: float
) -> None:
    """
    Compute the Wasserstein (Earth Mover's) distance between sample1 and sample2.
    Raises AssertionError if distance > max_distance.
    """
    dist = stats.wasserstein_distance(sample1, sample2)
    if dist > max_distance:
        raise AssertionError(
            f"Wasserstein distance {dist:.4f} exceeds max {max_distance:.4f}"
        )
