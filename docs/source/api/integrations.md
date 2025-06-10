# Integrations API Reference

## MLflow Integration

### assert_mlflow_model

Assert MLflow model properties and metrics.

```python
from ml_assert.integrations.mlflow import assert_mlflow_model

# Assert MLflow model
assert_mlflow_model(
    model_uri="runs:/run_id/model",
    metrics={
        "accuracy": 0.8,
        "precision": 0.8,
        "recall": 0.8
    }
)
```

### Parameters

- `model_uri`: MLflow model URI
- `metrics`: Dictionary of metric names and minimum values
- `params`: Dictionary of parameter names and expected values (optional)

## Prometheus Integration

### assert_prometheus_metrics

Assert Prometheus metrics for model monitoring.

```python
from ml_assert.integrations.prometheus import assert_prometheus_metrics

# Assert Prometheus metrics
assert_prometheus_metrics(
    metrics={
        "model_accuracy": 0.8,
        "prediction_latency": 100
    },
    labels={"model": "my_model"}
)
```

### Parameters

- `metrics`: Dictionary of metric names and thresholds
- `labels`: Dictionary of label names and values
- `timeout`: Maximum time to wait for metrics (default: 30 seconds)

## Slack Integration

### assert_slack_notification

Assert Slack notification delivery.

```python
from ml_assert.integrations.slack import assert_slack_notification

# Assert Slack notification
assert_slack_notification(
    channel="#ml-monitoring",
    message="Model drift detected",
    timeout=30
)
```

### Parameters

- `channel`: Slack channel name
- `message`: Expected message content
- `timeout`: Maximum time to wait for notification (default: 30 seconds)

## DVC Integration

### assert_dvc_artifact

Assert DVC artifact properties.

```python
from ml_assert.integrations.dvc import assert_dvc_artifact

# Assert DVC artifact
assert_dvc_artifact(
    path="models/model.pkl",
    exists=True,
    size_mb=10
)
```

### Parameters

- `path`: Path to DVC artifact
- `exists`: Whether artifact should exist
- `size_mb`: Expected size in megabytes (optional)
- `md5`: Expected MD5 hash (optional)
