"""Tests for combined integration scenarios."""

import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from ml_assert.core.base import AssertionResult
from ml_assert.integrations.mlflow import MLflowLogger
from ml_assert.integrations.prometheus import PrometheusExporter
from ml_assert.integrations.slack import SlackAlerter


@pytest.fixture
def temp_mlflow_dir():
    """Create a temporary directory for MLflow tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_slack_response():
    """Mock successful Slack API response."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    return mock


@pytest.fixture
def mock_prometheus_server():
    """Mock Prometheus HTTP server."""
    with patch("ml_assert.integrations.prometheus.start_http_server") as mock:
        yield mock


def test_integration_workflow(
    mock_slack_response, mock_prometheus_server, temp_mlflow_dir
):
    """Test a complete workflow using all integrations together."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    slack = SlackAlerter(webhook_url="https://hooks.slack.com/test")
    prometheus = PrometheusExporter(port=8000)
    mlflow_logger = MLflowLogger(
        experiment_name="test-experiment",
        run_name="test-run",
        tracking_uri=f"file://{temp_mlflow_dir}",
    )

    # Start Prometheus server
    prometheus.start()
    mock_prometheus_server.assert_called_once_with(8000, registry=prometheus.registry)

    # Create test data and results
    test_data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    assert len(test_data) == 100  # Verify test data creation

    # Create multiple assertion results
    results = [
        AssertionResult(
            success=True,
            message="Accuracy check passed",
            timestamp=datetime.now(),
            metadata={"accuracy": 0.95},
        ),
        AssertionResult(
            success=False,
            message="Fairness check failed",
            timestamp=datetime.now(),
            metadata={"demographic_parity": 0.15},
        ),
    ]

    # Record results in Prometheus
    for result in results:
        prometheus.record_assertion(result)

    # Send alerts to Slack (should only send for failed assertions)
    with patch("requests.post", return_value=mock_slack_response) as mock_post:
        for result in results:
            if not result.success:
                slack.send_alert(result)
        assert mock_post.call_count == 1  # Only one failed assertion

    # Log to MLflow
    mlflow_logger.start_run()
    mlflow_logger.log_assertion_result_mlassert(
        results[0], step_name="accuracy"
    )  # Log successful assertion
    mlflow_logger.log_assertion_result_mlassert(
        results[1], step_name="fairness"
    )  # Log failed assertion
    mlflow_logger.end_run()

    # Verify Prometheus metrics
    counter_samples = list(prometheus.assertion_counter.collect())[0].samples
    passed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "passed"
    )
    failed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "failed"
    )
    assert passed_count == 1.0
    assert failed_count == 1.0


def test_integration_error_handling(mock_prometheus_server, temp_mlflow_dir):
    """Test error handling across all integrations."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    slack = SlackAlerter(webhook_url="https://hooks.slack.com/test")
    prometheus = PrometheusExporter(port=8000)
    mlflow_logger = MLflowLogger(
        experiment_name="test-experiment",
        run_name="test-run",
        tracking_uri=f"file://{temp_mlflow_dir}",
    )

    # Start Prometheus server
    prometheus.start()

    # Test Slack API failure
    with patch("requests.post", side_effect=Exception("Slack API error")):
        result = AssertionResult(
            success=False,
            message="Test failure",
            timestamp=datetime.now(),
            metadata={"error": "test"},
        )
        with pytest.raises(Exception) as exc_info:
            slack.send_alert(result)
        assert "Slack API error" in str(exc_info.value)

    # Test MLflow logging failure
    result = AssertionResult(
        success=True,
        message="Test success",
        timestamp=datetime.now(),
        metadata={"metric": 0.95},
    )
    mlflow_logger.start_run()
    mlflow_logger.log_assertion_result_mlassert(result, step_name="test")
    mlflow_logger.end_run()

    # Test Prometheus recording (should not fail)
    result = AssertionResult(
        success=False,
        message="Test failure",
        timestamp=datetime.now(),
        metadata={"error": "test"},
    )
    prometheus.record_assertion(result)  # Should not raise

    # Verify Prometheus metrics after error handling
    counter_samples = list(prometheus.assertion_counter.collect())[0].samples
    failed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "failed"
    )
    assert failed_count == 1.0


def test_integration_concurrent_usage(mock_prometheus_server, temp_mlflow_dir):
    """Test concurrent usage of integrations."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    exporters = [PrometheusExporter(port=8000 + i) for i in range(3)]
    slack_alerters = [
        SlackAlerter(webhook_url=f"https://hooks.slack.com/test{i}") for i in range(3)
    ]
    mlflow_loggers = [
        MLflowLogger(
            experiment_name=f"test-experiment-{i}",
            run_name=f"test-run-{i}",
            tracking_uri=f"file://{temp_mlflow_dir}",
        )
        for i in range(3)
    ]

    # Start all Prometheus servers
    for exporter in exporters:
        exporter.start()

    # Create test result
    result = AssertionResult(
        success=True,
        message="Concurrent test",
        timestamp=datetime.now(),
        metadata={"test": "concurrent"},
    )

    # Record in all exporters
    for exporter in exporters:
        exporter.record_assertion(result)

    # Send to all Slack instances
    with patch("requests.post", return_value=MagicMock()) as mock_post:
        for alerter in slack_alerters:
            alerter.send_alert(result)
        assert mock_post.call_count == 3

    # Log to all MLflow instances
    for logger in mlflow_loggers:
        logger.start_run()
        logger.log_assertion_result_mlassert(result, step_name="test")
        logger.end_run()


def test_integration_metadata_handling(mock_prometheus_server, temp_mlflow_dir):
    """Test handling of different metadata types across integrations."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    prometheus = PrometheusExporter(port=8000)
    mlflow_logger = MLflowLogger(
        experiment_name="test-experiment",
        run_name="test-run",
        tracking_uri=f"file://{temp_mlflow_dir}",
    )

    # Start Prometheus server
    prometheus.start()

    # Test with various metadata types
    test_cases = [
        {
            "success": True,
            "message": "Numeric metadata",
            "metadata": {"accuracy": 0.95, "precision": 0.92, "recall": 0.88},
        },
        {
            "success": False,
            "message": "String metadata",
            "metadata": {
                "error_type": "validation_error",
                "error_message": "Invalid input",
            },
        },
        {
            "success": True,
            "message": "Mixed metadata",
            "metadata": {
                "score": 0.85,
                "status": "passed",
                "details": {"subtest1": True, "subtest2": False},
            },
        },
    ]

    # Process each test case
    for i, case in enumerate(test_cases):
        mlflow_logger.start_run()
        result = AssertionResult(
            success=case["success"],
            message=case["message"],
            timestamp=datetime.now(),
            metadata=case["metadata"],
        )

        # Record in Prometheus
        prometheus.record_assertion(result)

        # Log to MLflow with unique step_name
        mlflow_logger.log_assertion_result_mlassert(result, step_name=f"test_{i}")
        mlflow_logger.end_run()

    # Verify Prometheus metrics
    counter_samples = list(prometheus.assertion_counter.collect())[0].samples
    passed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "passed"
    )
    failed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "failed"
    )
    assert passed_count == 2.0  # Two successful assertions
    assert failed_count == 1.0  # One failed assertion
