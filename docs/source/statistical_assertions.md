# Statistical Assertions

The `ml_assert.stats` module provides functions for statistical comparison and drift detection between datasets.

---

## Quick Start

```python
from ml_assert.stats.drift import assert_no_drift

# Check for drift between reference and current datasets
assert_no_drift(df_ref, df_cur, alpha=0.05)
```

---

## Drift Detection

### High-Level Drift Detection

The `assert_no_drift` function automatically detects drift in all columns of a DataFrame.

```python
from ml_assert.stats.drift import assert_no_drift

# Check for drift between training and test sets
assert_no_drift(df_train, df_test, alpha=0.05)
```

**Parameters:**
- `df_ref`: Reference DataFrame (e.g., training data)
- `df_cur`: Current DataFrame (e.g., inference data)
- `alpha`: Significance level for statistical tests (default: 0.05)

### Low-Level Statistical Tests

#### Kolmogorov-Smirnov Test

For comparing continuous distributions.

```python
from ml_assert.stats.drift import ks_test

# Compare two numeric samples
statistic, p_value = ks_test(sample1, sample2, alpha=0.05)
```

#### Chi-Squared Test

For comparing categorical distributions.

```python
from ml_assert.stats.drift import chi2_test

# Compare two categorical samples
statistic, p_value = chi2_test(sample1, sample2, alpha=0.05)
```

#### Wasserstein Distance

For measuring the distance between distributions.

```python
from ml_assert.stats.drift import wasserstein_distance

# Calculate Wasserstein distance
distance = wasserstein_distance(sample1, sample2)
```

---

## Distribution Assertions

### Distribution Testing

Assert that a dataset follows a specific distribution.

```python
from ml_assert.stats.distribution import assert_distribution

# Assert normal distribution
assert_distribution(data, distribution="normal", alpha=0.05)
```

**Supported Distributions:**
- "normal"
- "uniform"
- "exponential"
- "poisson"

---

## Examples

### Basic Drift Detection

```python
from ml_assert.stats.drift import assert_no_drift

# Check for drift in all columns
assert_no_drift(df_train, df_test, alpha=0.05)
```

### Custom Statistical Tests

```python
from ml_assert.stats.drift import ks_test, chi2_test

# Compare numeric columns
statistic, p_value = ks_test(df_train["age"], df_test["age"], alpha=0.05)

# Compare categorical columns
statistic, p_value = chi2_test(df_train["category"], df_test["category"], alpha=0.05)
```

### Distribution Testing

```python
from ml_assert.stats.distribution import assert_distribution

# Test for normal distribution
assert_distribution(data, distribution="normal", alpha=0.05)

# Test for uniform distribution
assert_distribution(data, distribution="uniform", alpha=0.05)
```

For more detailed API reference, see [Stats API](api/stats.md).
