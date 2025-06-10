# Model Performance Assertions

The `ml_assert` module provides a fluent interface for validating model performance metrics.

---

## Quick Start

```python
from ml_assert import assert_model

# Assert model performance
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.8) \
    .precision(min_score=0.8) \
    .recall(min_score=0.8) \
    .f1(min_score=0.8) \
    .roc_auc(min_score=0.9) \
    .validate()
```

---

## Available Metrics

### Accuracy

Measures the proportion of correct predictions.

```python
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.8) \
    .validate()
```

### Precision

Measures the ability of the classifier not to label as positive a sample that is negative.

```python
assert_model(y_true, y_pred, y_scores) \
    .precision(min_score=0.8) \
    .validate()
```

### Recall

Measures the ability of the classifier to find all the positive samples.

```python
assert_model(y_true, y_pred, y_scores) \
    .recall(min_score=0.8) \
    .validate()
```

### F1 Score

The harmonic mean of precision and recall.

```python
assert_model(y_true, y_pred, y_scores) \
    .f1(min_score=0.8) \
    .validate()
```

### ROC AUC

Area Under the Receiver Operating Characteristic Curve, measuring the ability to distinguish between classes.

```python
assert_model(y_true, y_pred, y_scores) \
    .roc_auc(min_score=0.9) \
    .validate()
```

---

## Error Handling & Result Reporting

- All assertion methods raise `AssertionError` if a check fails during chaining, unless `.validate()` is called.
- `.validate()` returns an `AssertionResult` object:
    - `success` (bool): True if all assertions passed.
    - `message` (str): Summary message.
    - `timestamp` (datetime): When the check was run.
    - `metadata` (dict): Details of each assertion (name, args, success, error if any).

---

## Examples

### Basic Performance Check

```python
from ml_assert import assert_model

# Check basic performance metrics
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.8) \
    .precision(min_score=0.8) \
    .recall(min_score=0.8) \
    .validate()
```

### Comprehensive Performance Check

```python
from ml_assert import assert_model

# Check all performance metrics
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.8) \
    .precision(min_score=0.8) \
    .recall(min_score=0.8) \
    .f1(min_score=0.8) \
    .roc_auc(min_score=0.9) \
    .validate()
```

### Custom Thresholds

```python
from ml_assert import assert_model

# Use different thresholds for different metrics
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.75) \
    .precision(min_score=0.85) \
    .recall(min_score=0.70) \
    .f1(min_score=0.80) \
    .roc_auc(min_score=0.90) \
    .validate()
```

For more detailed API reference, see [Model API](api/model.md).
