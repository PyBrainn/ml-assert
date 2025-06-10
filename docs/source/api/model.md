# Model API Reference

## assert_model

A builder for model performance assertions.

```python
from ml_assert import assert_model

# Create model assertions
assert_model(y_true, y_pred, y_scores) \
    .accuracy(min_score=0.80) \
    .precision(min_score=0.80) \
    .recall(min_score=0.80) \
    .f1(min_score=0.80) \
    .roc_auc(min_score=0.90) \
    .validate()
```

### Methods

#### `accuracy(min_score)`
Asserts minimum accuracy score.

**Parameters:**
- `min_score`: Minimum acceptable accuracy score (0.0 to 1.0)

**Returns:**
- `self` for method chaining

#### `precision(min_score)`
Asserts minimum precision score.

**Parameters:**
- `min_score`: Minimum acceptable precision score (0.0 to 1.0)

**Returns:**
- `self` for method chaining

#### `recall(min_score)`
Asserts minimum recall score.

**Parameters:**
- `min_score`: Minimum acceptable recall score (0.0 to 1.0)

**Returns:**
- `self` for method chaining

#### `f1(min_score)`
Asserts minimum F1 score.

**Parameters:**
- `min_score`: Minimum acceptable F1 score (0.0 to 1.0)

**Returns:**
- `self` for method chaining

#### `roc_auc(min_score)`
Asserts minimum ROC AUC score.

**Parameters:**
- `min_score`: Minimum acceptable ROC AUC score (0.0 to 1.0)

**Returns:**
- `self` for method chaining

#### `validate()`
Executes all chained assertions.

**Raises:**
- `AssertionError` if any assertion fails
