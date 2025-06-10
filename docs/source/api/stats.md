# Statistical API Reference

## assert_no_drift

High-level function to detect distributional drift between datasets.

```python
from ml_assert.stats.drift import assert_no_drift

# Check for drift between reference and current datasets
assert_no_drift(df_ref, df_cur, alpha=0.05)
```

### Parameters

- `df_ref`: Reference DataFrame (e.g., training data)
- `df_cur`: Current DataFrame (e.g., inference data)
- `alpha`: Significance level for statistical tests (default: 0.05)

### Raises

- `AssertionError` if drift is detected in any column

## Low-level Statistical Tests

### ks_test

Kolmogorov-Smirnov test for continuous variables.

```python
from ml_assert.stats.drift import ks_test

# Perform KS test
statistic, p_value = ks_test(ref_data, cur_data)
```

### chi2_test

Chi-squared test for categorical variables.

```python
from ml_assert.stats.drift import chi2_test

# Perform Chi-squared test
statistic, p_value = chi2_test(ref_data, cur_data)
```

### wasserstein_distance

Wasserstein distance (Earth Mover's Distance) for continuous variables.

```python
from ml_assert.stats.drift import wasserstein_distance

# Calculate Wasserstein distance
distance = wasserstein_distance(ref_data, cur_data)
```

## Distribution Assertions

### assert_distribution

Assert that a dataset follows a specific distribution.

```python
from ml_assert.stats.distribution import assert_distribution

# Assert normal distribution
assert_distribution(data, distribution="normal", alpha=0.05)
```

### Parameters

- `data`: Array-like data to test
- `distribution`: Name of the distribution to test against
- `alpha`: Significance level for the test (default: 0.05)

### Supported Distributions

- "normal"
- "uniform"
- "exponential"
- "poisson"
