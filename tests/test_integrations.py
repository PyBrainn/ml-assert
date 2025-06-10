from datetime import datetime
from unittest.mock import MagicMock, patch

from ml_assert.core.base import AssertionResult
from ml_assert.integrations.mlflow import MLflowLogger
from ml_assert.integrations.prometheus import PrometheusExporter
from ml_assert.integrations.slack import SlackAlerter


def test_slack_alerter():
    with patch("requests.post") as mock_post:
        alerter = SlackAlerter(webhook_url="https://hooks.slack.com/services/...")
        result = AssertionResult(
            success=False,
            message="Test failure",
            timestamp=datetime.now(),
            metadata={"key": "value"},
        )
        alerter.send_alert(result)
        mock_post.assert_called_once()
        # Verify the request was made with correct data
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["text"] == "Assertion FAILED: Test failure"
        assert kwargs["json"]["attachments"][0]["color"] == "danger"
        assert kwargs["headers"] == {"Content-Type": "application/json"}


def get_counter_value(counter, label):
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.labels.get("status") == label:
                return sample.value
    return 0


def get_gauge_value(gauge):
    for metric in gauge.collect():
        for sample in metric.samples:
            return sample.value
    return 0


def test_prometheus_exporter():
    with patch("ml_assert.integrations.prometheus.start_http_server") as mock_start:
        exporter = PrometheusExporter(port=8000)
        exporter.start()
        args, kwargs = mock_start.call_args
        assert args[0] == 8000
        assert "registry" in kwargs

        # Test recording successful assertion
        result = AssertionResult(
            success=True, message="Test", timestamp=datetime.now(), metadata={}
        )
        exporter.record_assertion(result)

        # Test recording failed assertion
        result2 = AssertionResult(
            success=False, message="Fail", timestamp=datetime.now(), metadata={}
        )
        exporter.record_assertion(result2)

        # Check counter values
        counter_samples = list(exporter.assertion_counter.collect())[0].samples
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

        # Check gauge value (should be 0 since we had one pass and one fail)
        gauge_value = list(exporter.assertion_gauge.collect())[0].samples[0].value
        assert gauge_value == 0.0


def test_mlflow_logger():
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"

    with (
        patch("mlflow.start_run", return_value=mock_run) as mock_start_run,
        patch("mlflow.end_run") as mock_end_run,
        patch("mlflow.get_experiment_by_name", return_value=None),
        patch("mlflow.create_experiment", return_value="test-exp-id"),
        patch("mlflow.tracking.MlflowClient.log_metric") as mock_log_metric,
        patch("mlflow.tracking.MlflowClient.log_param") as mock_log_param,
    ):
        logger = MLflowLogger(experiment_name="test-exp")
        logger.start_run()
        mock_start_run.assert_called_once()

        # Test logging a successful assertion
        result = AssertionResult(
            success=True,
            message="Test passed",
            timestamp=datetime.now(),
            metadata={"key": "value"},
        )
        logger.log_assertion_result_mlassert(result, step_name="test_step")

        # Verify metrics and params were logged correctly
        mock_log_metric.assert_called_with("test-run-id", "test_step_passed", 1.0)
        mock_log_param.assert_any_call(
            "test-run-id", "test_step_message", "Test passed"
        )
        mock_log_param.assert_any_call("test-run-id", "test_step_key", "value")

        # Test logging a failed assertion
        result2 = AssertionResult(
            success=False,
            message="Test failed",
            timestamp=datetime.now(),
            metadata={"error": "test error"},
        )
        logger.log_assertion_result_mlassert(result2, step_name="test_step2")

        # Verify metrics and params were logged correctly
        mock_log_metric.assert_called_with("test-run-id", "test_step2_passed", 0.0)
        mock_log_param.assert_any_call(
            "test-run-id", "test_step2_message", "Test failed"
        )
        mock_log_param.assert_any_call("test-run-id", "test_step2_error", "test error")

        logger.end_run()
        mock_end_run.assert_called_once()
