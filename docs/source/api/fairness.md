# Fairness API Reference

## Fairness Metrics

### assert_fairness

High-level function to assert fairness metrics for a model.

```python
from ml_assert.fairness import assert_fairness

# Assert fairness metrics
assert_fairness(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
    metrics=["demographic_parity", "equal_opportunity"],
    threshold=0.1
)
```

### Parameters

- `y_true`: True labels
- `y_pred`: Predicted labels
- `sensitive_features`: Array of sensitive feature values
- `metrics`: List of fairness metrics to check
- `threshold`: Maximum allowed difference between groups (default: 0.1)

### Supported Metrics

- `demographic_parity`: Equal positive prediction rates across groups
- `equal_opportunity`: Equal true positive rates across groups
- `equalized_odds`: Equal true positive and false positive rates
- `treatment_equality`: Equal ratio of false negatives to false positives

## Explainability

### assert_feature_importance

Assert minimum feature importance scores.

```python
from ml_assert.fairness import assert_feature_importance

# Assert feature importance
assert_feature_importance(
    model=model,
    X=X,
    min_importance=0.1,
    features=["feature1", "feature2"]
)
```

### Parameters

- `model`: Trained model with feature_importances_ attribute
- `X`: Feature matrix
- `min_importance`: Minimum importance score (0.0 to 1.0)
- `features`: List of features to check (optional)

### assert_shap_values

Assert SHAP values for feature importance.

```python
from ml_assert.fairness import assert_shap_values

# Assert SHAP values
assert_shap_values(
    model=model,
    X=X,
    min_importance=0.1,
    features=["feature1", "feature2"]
)
```

### Parameters

- `model`: Trained model
- `X`: Feature matrix
- `min_importance`: Minimum absolute SHAP value
- `features`: List of features to check (optional)
