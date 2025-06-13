# Cross-Validation

ML-Assert provides robust cross-validation support for evaluating machine learning models. This feature allows you to perform comprehensive model evaluation using various cross-validation strategies and metrics.

## Overview

Cross-validation is a crucial technique for assessing model performance and generalization ability. ML-Assert implements several cross-validation strategies and provides easy-to-use assertion functions for various metrics.

## Available Cross-Validation Strategies

### K-Fold Cross-Validation
The standard k-fold cross-validation splits the data into k equal parts and uses each part as a validation set while training on the remaining k-1 parts.

```python
from ml_assert.model.cross_validation import assert_cv_accuracy_score

# Using 5-fold cross-validation
assert_cv_accuracy_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.85,
    cv_type='kfold',
    n_splits=5
)
```

### Stratified K-Fold Cross-Validation
Stratified k-fold cross-validation maintains the class distribution in each fold, which is particularly useful for imbalanced datasets.

```python
# Using stratified 5-fold cross-validation
assert_cv_accuracy_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.85,
    cv_type='stratified',
    n_splits=5
)
```

### Leave-One-Out Cross-Validation
Leave-One-Out (LOO) cross-validation uses each sample as a validation set while training on all other samples. This is useful for small datasets.

```python
# Using leave-one-out cross-validation
assert_cv_accuracy_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.85,
    cv_type='loo'
)
```

## Available Metrics

ML-Assert supports various metrics for cross-validation evaluation:

### Accuracy Score
```python
from ml_assert.model.cross_validation import assert_cv_accuracy_score

assert_cv_accuracy_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.85
)
```

### Precision Score
```python
from ml_assert.model.cross_validation import assert_cv_precision_score

assert_cv_precision_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.80
)
```

### Recall Score
```python
from ml_assert.model.cross_validation import assert_cv_recall_score

assert_cv_recall_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.80
)
```

### F1 Score
```python
from ml_assert.model.cross_validation import assert_cv_f1_score

assert_cv_f1_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.80
)
```

### ROC AUC Score
```python
from ml_assert.model.cross_validation import assert_cv_roc_auc_score

assert_cv_roc_auc_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.80
)
```

## Getting Cross-Validation Summary

You can get a comprehensive summary of all metrics across cross-validation folds:

```python
from ml_assert.model.cross_validation import get_cv_summary

summary = get_cv_summary(
    model=my_model,
    X=X,
    y=y,
    cv_type='kfold',
    n_splits=5
)

print(summary)
```

The summary includes:
- Mean score for each metric
- Standard deviation of scores
- Minimum score across folds
- Maximum score across folds

## Advanced Usage

### Parallel Processing
Cross-validation computations are automatically parallelized using all available CPU cores for faster evaluation.

### Custom Cross-Validation
You can use any scikit-learn compatible cross-validation splitter by passing it directly to the assertion functions.

### Error Handling
The cross-validation module includes comprehensive error handling for:
- Invalid model types
- Incompatible data types
- Invalid cross-validation parameters
- Computation errors

## Best Practices

1. **Choose the Right Strategy**:
   - Use k-fold for balanced datasets
   - Use stratified k-fold for imbalanced datasets
   - Use leave-one-out for small datasets

2. **Set Appropriate Thresholds**:
   - Consider the problem domain
   - Account for class imbalance
   - Use multiple metrics for comprehensive evaluation

3. **Monitor Performance**:
   - Check standard deviation across folds
   - Look for consistent performance
   - Investigate high variance in scores

## Examples

### Complete Model Evaluation
```python
from ml_assert.model.cross_validation import (
    assert_cv_accuracy_score,
    assert_cv_precision_score,
    assert_cv_recall_score,
    assert_cv_f1_score,
    assert_cv_roc_auc_score,
    get_cv_summary
)

# Evaluate model with multiple metrics
assert_cv_accuracy_score(model, X, y, min_score=0.85)
assert_cv_precision_score(model, X, y, min_score=0.80)
assert_cv_recall_score(model, X, y, min_score=0.80)
assert_cv_f1_score(model, X, y, min_score=0.80)
assert_cv_roc_auc_score(model, X, y, min_score=0.80)

# Get detailed summary
summary = get_cv_summary(model, X, y)
print("Model Performance Summary:")
for metric, stats in summary.items():
    if stats is not None:
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
```

### Handling Imbalanced Data
```python
# Using stratified cross-validation for imbalanced data
assert_cv_f1_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.75,
    cv_type='stratified',
    n_splits=5
)
```

### Small Dataset Evaluation
```python
# Using leave-one-out cross-validation for small dataset
assert_cv_accuracy_score(
    model=my_model,
    X=X,
    y=y,
    min_score=0.80,
    cv_type='loo'
)
```
