# Model Performance Assertions

The `ml_assert.model` module provides functions to validate the performance of classification models. These are typically used via the `assert_model` fluent interface.

## `assert_accuracy_score`

Asserts that the model's accuracy is above a minimum threshold. Accuracy is the proportion of correct predictions among the total number of cases processed.

- **Usage**: `assert_accuracy_score(y_true, y_pred, min_score)`
- **Parameters**:
    - `y_true` (`np.ndarray`): Ground truth (correct) labels.
    - `y_pred` (`np.ndarray`): Predicted labels, as returned by a classifier.
    - `min_score` (`float`): The minimum acceptable accuracy score.
- **Raises**: `AssertionError` if `accuracy_score(y_true, y_pred) < min_score`.

## `assert_precision_score`

Asserts that the model's precision is above a minimum threshold. Precision is the ability of the classifier not to label as positive a sample that is negative.

- **Usage**: `assert_precision_score(y_true, y_pred, min_score)`
- **Parameters**:
    - `y_true`, `y_pred` (`np.ndarray`): Ground truth and predicted labels.
    - `min_score` (`float`): The minimum acceptable precision score.
- **Raises**: `AssertionError` if `precision_score(y_true, y_pred) < min_score`.

## `assert_recall_score`

Asserts that the model's recall is above a minimum threshold. Recall is the ability of the classifier to find all the positive samples.

- **Usage**: `assert_recall_score(y_true, y_pred, min_score)`
- **Parameters**:
    - `y_true`, `y_pred` (`np.ndarray`): Ground truth and predicted labels.
    - `min_score` (`float`): The minimum acceptable recall score.
- **Raises**: `AssertionError` if `recall_score(y_true, y_pred) < min_score`.

## `assert_f1_score`

Asserts that the model's F1 score is above a minimum threshold. The F1 score is the harmonic mean of precision and recall.

- **Usage**: `assert_f1_score(y_true, y_pred, min_score)`
- **Parameters**:
    - `y_true`, `y_pred` (`np.ndarray`): Ground truth and predicted labels.
    - `min_score` (`float`): The minimum acceptable F1 score.
- **Raises**: `AssertionError` if `f1_score(y_true, y_pred) < min_score`.

## `assert_roc_auc_score`

Asserts that the model's ROC AUC score is above a minimum threshold. Area Under the Receiver Operating Characteristic Curve (ROC AUC) measures the ability of a classifier to distinguish between classes.

- **Usage**: `assert_roc_auc_score(y_true, y_scores, min_score)`
- **Parameters**:
    - `y_true` (`np.ndarray`): Ground truth labels.
    - `y_scores` (`np.ndarray`): Target scores, can be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    - `min_score` (`float`): The minimum acceptable ROC AUC score.
- **Raises**: `AssertionError` if `roc_auc_score(y_true, y_scores) < min_score`.
