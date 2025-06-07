"""
Statistical tests and drift detection for pandas Series/DataFrames.
"""

from collections.abc import Iterable

import pandas as pd
from scipy.stats import chisquare, ks_2samp, wasserstein_distance


def assert_ks_test(
    series1: Iterable,
    series2: Iterable,
    alpha: float = 0.05,
) -> None:
    """
    Perform Kolmogorov-Smirnov two-sample test between two distributions.
    Raises AssertionError if p-value < alpha.
    """
    stat, p_value = ks_2samp(series1, series2)
    if p_value < alpha:
        raise AssertionError(
            f"KS test failed for series: p-value {p_value:.4f} < alpha {alpha}"
        )


def assert_chi2_test(
    observed: Iterable[int],
    expected: Iterable[int],
    alpha: float = 0.05,
) -> None:
    """
    Perform Chi-square test between observed and expected frequencies.
    Raises AssertionError if p-value < alpha.
    """
    try:
        stat, p_value = chisquare(observed, expected)
    except ValueError as e:
        raise AssertionError(f"Chi-square test invalid: {e}") from e
    if p_value < alpha:
        raise AssertionError(
            f"Chi-square test failed: p-value {p_value:.4f} < alpha {alpha}"
        )


def assert_wasserstein_distance(
    series1: Iterable,
    series2: Iterable,
    max_distance: float,
) -> None:
    """
    Compute Wasserstein distance between two distributions and assert it <= max_distance.
    Raises AssertionError if distance > max_distance.
    """
    dist = wasserstein_distance(series1, series2)
    if dist > max_distance:
        raise AssertionError(
            f"Wasserstein distance {dist:.4f} > max_distance {max_distance}"
        )


def assert_no_drift(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    alpha: float = 0.05,
) -> None:
    """
    Assert no distribution drift between df1 and df2 for specified columns.

    Numeric columns: KS test; categorical columns: Chi-square test on value counts.
    """
    # Numeric drift
    num_cols = numeric_columns or df1.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        assert_ks_test(df1[col], df2[col], alpha)

    # Categorical drift
    cat_cols = (
        categorical_columns
        or df1.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    for col in cat_cols:
        counts1 = df1[col].value_counts().sort_index()
        counts2 = (
            df2[col].value_counts().reindex(counts1.index, fill_value=0).sort_index()
        )
        assert_chi2_test(counts1.tolist(), counts2.tolist(), alpha)
