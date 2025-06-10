# ml-assert

<p align="center">
  <img src="https://raw.githubusercontent.com/HeyShinde/ml-assert/main/docs/source/assets/logo.svg" alt="ml-assert logo" width="200"/>
</p>

<p align="center">
  <strong>A lightweight, chainable assertion toolkit for validating data and models in ML workflows.</strong>
</p>

---

`ml-assert` is a Python library that provides a fluent, expressive API to act as a guardrail in your automated ML pipelines. It doesn't just calculate metrics; it **asserts** that your data and models meet specific, mission-critical criteria. If an assertion fails, it fails loudly and immediately, stopping the pipeline to prevent bad models or corrupt data from moving downstream.

This is crucial for building robust, production-ready ML systems where data quality, model performance, and artifact integrity are non-negotiable.

## Core Features

-   **DataFrame Assertions**: Validate `pandas` DataFrame properties like schema, null values, column uniqueness, value ranges, and set membership.
-   **Statistical Drift Detection**: Use low-level statistical tests (Kolmogorov-Smirnov, Chi-Squared, Wasserstein) or a high-level `assert_no_drift` function to detect changes between datasets.
-   **Model Performance Assertions**: Chain assertions for key classification metrics (Accuracy, Precision, Recall, F1, ROC AUC) to ensure your model meets performance targets.
-   **Extensible Plugin System**: Leverage built-in plugins (`file_exists`, `dvc_check`) or create your own to add custom checks.
-   **Declarative CLI**: Define your assertion suite in a single `config.yaml` and run it from the command line, generating JSON and HTML reports.

## Installation

```bash
pip install ml-assert
```

## How It Works: Assertion vs. Calculation

A typical metrics library might *calculate* an accuracy of 75% and let the pipeline continue. `ml-assert` *asserts* that accuracy must be `>= 80%`. If it's 75%, it raises an `AssertionError`, halting execution.

This paradigm shift from passive calculation to active assertion is what makes `ml-assert` a powerful tool for ML Ops.

## Usage Examples

### 1. DataFrameAssertion DSL

Chain assertions to validate a `pandas` DataFrame. The chain stops at the first failure.

```python
import pandas as pd
import numpy as np
from ml_assert import Assertion, schema

# DataFrame with a column full of nulls and an out-of-range value
data = {
    'user_id': list(range(100, 110)),
    'age': [25, 30, 99, 45, 30, 50, 60, 22, 33, 41], # 99 is out of range
    'plan_type': ['basic', 'premium', 'basic', 'premium', 'premium', 'basic', 'free', 'free', 'premium', 'basic'],
    'empty_col': [np.nan] * 10
}
df = pd.DataFrame(data)

# This check will FAIL because `age` has a value > 70
try:
    s = schema()
    s.col("user_id").is_unique()
    s.col("age").in_range(18, 70)
    s.col("plan_type").is_type("object")

    Assertion(df).satisfies(s).no_nulls().validate()
except AssertionError as e:
    print(f"As expected, validation failed: {e}")

# This check will PASS because we only check specific columns
s2 = schema()
s2.col("user_id").is_unique()
Assertion(df).satisfies(s2).no_nulls(['user_id', 'age', 'plan_type']).validate()

print("Partial validation passed!")
```

### 2. High-Level Drift Detection

Detect distributional drift between a reference (training) and current (inference) dataset. `assert_no_drift` intelligently applies KS tests to numeric columns and Chi-Squared tests to categorical columns.

```python
import pandas as pd
import numpy as np
from ml_assert.stats.drift import assert_no_drift

# Reference dataset
df_ref = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 500),
    'city': np.random.choice(['NY', 'LA', 'SF'], 500, p=[0.5, 0.3, 0.2])
})

# Current dataset with a deliberate drift
df_cur = pd.DataFrame({
    'temperature': np.random.normal(30, 5, 500), # Mean shifted by +10
    'city': np.random.choice(['NY', 'LA', 'SF'], 500, p=[0.2, 0.3, 0.5]) # Proportions changed
})

# This will FAIL and identify the drifting column ('temperature').
try:
    assert_no_drift(df_ref, df_cur, alpha=0.05)
except AssertionError as e:
    print(f"As expected, drift was detected: {e}")

# This will PASS because the data is identical.
assert_no_drift(df_ref, df_ref.copy(), alpha=0.05)
print("No drift detected in identical datasets.")

```

### 3. Model Performance Assertions

Ensure your model's predictions meet your minimum quality bar.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml_assert import assert_model

# Generate data and train a simple model
X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# Chain assertions for key metrics
# This will PASS if all metrics meet their thresholds.
assert_model(y_test, y_pred, y_scores) \
    .accuracy(min_score=0.80) \
    .precision(min_score=0.80) \
    .recall(min_score=0.80) \
    .f1(min_score=0.80) \
    .roc_auc(min_score=0.90) \
    .validate()

print("All model performance metrics passed!")
```

### 4. CLI for Automated Runs

Define a suite of checks in a YAML file and execute it with the `ml_assert` CLI. This is perfect for CI/CD pipelines.

**`config.yaml`**
```yaml
steps:
  - type: drift
    train: 'ref.csv'
    test: 'cur.csv'
    alpha: 0.05
    # The CLI run will fail on this step due to drift

  - type: model_performance
    y_true: 'y_true.csv'
    y_pred: 'y_pred.csv'
    y_scores: 'y_scores.csv'
    assertions:
      accuracy: 0.75
      roc_auc: 0.80

  - type: file_exists
    path: 'my_model.pkl'

  - type: dvc_check
    path: 'model_data.csv'
```

**Run from your terminal:**
```bash
# poetry run ml_assert run config.yaml
# The command will fail because of the drift, and generate reports.
ml_assert run config.yaml
```

This command generates two reports:
-   `config.report.json`: A machine-readable summary.
-   `config.report.html`: A human-friendly HTML report.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.
