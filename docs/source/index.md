# ml-assert

A lightweight, chainable assertion toolkit for validating data and models in ML workflows.

## Features

- **Data Validation**: Schema, nulls, uniqueness, ranges, and value sets.
- **Statistical Checks**: Distribution drift (KS, Chi-square) and drift detection.
- **Model Performance**: Accuracy, precision, recall, F1-score, and ROC AUC.
- **Fluent Interface**: Chain assertions for clean, readable code.
- **CLI Runner**: Run checks from a YAML configuration.

## Installation

```bash
pip install ml-assert
```

## Usage

### Data Assertions

Create a `DataFrame` and use `DataFrameAssertion` to chain validation checks.

```python
import pandas as pd
from ml_assert import DataFrameAssertion

df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["A", "B", "C"],
    "score": [0.9, 0.8, 0.7]
})

DataFrameAssertion(df) \
    .schema({"id": "int64", "name": "object", "score": "float64"}) \
    .no_nulls() \
    .unique("id") \
    .in_range("score", 0.0, 1.0) \
    .validate()
```

### Model Performance Assertions

Validate model predictions and scores using `assert_model`.

```python
import numpy as np
from ml_assert import assert_model

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])
y_scores = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])

assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.6) \
    .precision(min_score=0.65) \
    .recall(min_score=0.65) \
    .f1(min_score=0.65) \
    .roc_auc(min_score=0.8) \
    .validate()
```

### Statistical Drift Detection

Check for statistical drift between two `DataFrames`.

```python
from ml_assert.stats import assert_no_drift

df_train = pd.DataFrame({'num': [1, 2, 3], 'cat': ['a', 'b', 'a']})
df_test = pd.DataFrame({'num': [1, 2, 4], 'cat': ['a', 'b', 'x']})

assert_no_drift(df_train, df_test, alpha=0.05)
```

## CLI Usage

Run checks from a YAML file.

```bash
ml_assert run --config /path/to/config.yaml
```

See the `examples` directory for more details.
