import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_assert.core.base import AssertionResult
from ml_assert.fairness.explainability import ModelExplainer
from ml_assert.fairness.fairness import FairnessMetrics
from ml_assert.integrations.mlflow import MLflowLogger
from ml_assert.integrations.prometheus import PrometheusExporter
from ml_assert.integrations.slack import SlackAlerter


def get_free_port():
    # Use a random high port to avoid conflicts
    return random.randint(20000, 30000)


def create_test_model(tmp_path):
    """Create and save a test model."""
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "gender": np.random.choice(["M", "F"], 100),
        }
    )
    y = (X["feature1"] + X["feature2"] > 0).astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X.drop("gender", axis=1), y)

    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)

    return model_path, X, y


def test_end_to_end_integration(tmp_path, monkeypatch):
    """Test end-to-end integration of multiple components."""
    # Create test model and data
    model_path, X, y = create_test_model(tmp_path)

    # Mock Slack API
    class MockResponse:
        def raise_for_status(self):
            pass

    def mock_slack_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("requests.post", mock_slack_post)

    # Mock MLflow
    def mock_mlflow_log_metric(*args, **kwargs):
        pass

    def mock_mlflow_log_param(*args, **kwargs):
        pass

    monkeypatch.setattr("mlflow.log_metric", mock_mlflow_log_metric)
    monkeypatch.setattr("mlflow.log_param", mock_mlflow_log_param)

    # Initialize components
    prometheus = PrometheusExporter(port=get_free_port())
    slack = SlackAlerter(webhook_url="https://hooks.slack.com/test")
    mlflow = MLflowLogger(experiment_name="test-experiment", run_name="test-run")

    # Start Prometheus server
    prometheus.start()

    # Run model predictions
    model = joblib.load(model_path)
    y_pred = model.predict(X.drop("gender", axis=1))

    # Calculate fairness metrics (fix argument order)
    sensitive_attr = (X["gender"] == "M").astype(int)
    fairness = FairnessMetrics(y, y_pred, sensitive_attr)

    # Create assertion results
    accuracy_result = AssertionResult(
        True, "Accuracy check", datetime.now(), {"accuracy": sum(y == y_pred) / len(y)}
    )

    fairness_result = AssertionResult(
        fairness.demographic_parity() < 0.1,
        "Fairness check",
        datetime.now(),
        {"demographic_parity": fairness.demographic_parity()},
    )

    # Record results in Prometheus
    prometheus.record_assertion(accuracy_result)
    prometheus.record_assertion(fairness_result)

    # Send alerts to Slack
    if not accuracy_result.success:
        slack.send_alert(accuracy_result)
    if not fairness_result.success:
        slack.send_alert(fairness_result)

    # Start MLflow run
    mlflow.start_run()
    # Log to MLflow
    mlflow.log_assertion_result_mlassert(accuracy_result, step_name="accuracy")
    mlflow.log_assertion_result_mlassert(fairness_result, step_name="fairness")
    mlflow.end_run()

    # Generate model explanations
    model_loaded = joblib.load(model_path)
    explainer = ModelExplainer(
        model_loaded, feature_names=X.drop("gender", axis=1).columns
    )
    # Generate and verify explanation
    shap_values = explainer.explain(X.drop("gender", axis=1))
    assert shap_values is not None

    # Save explanation report
    output_dir = tmp_path / "explanation_report"
    output_dir.mkdir()
    explainer.save_explanation_report(X.drop("gender", axis=1), str(output_dir))

    # Verify outputs
    assert (output_dir / "summary_plot.png").exists()
    assert (output_dir / "dependence_feature1.png").exists()
    assert (output_dir / "dependence_feature2.png").exists()

    # Check Prometheus metrics
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
    assert passed_count + failed_count == 2.0  # Two assertions recorded


def test_integration_error_handling(tmp_path, monkeypatch):
    """Test error handling in integration components."""
    # Create test model and data
    model_path, X, y = create_test_model(tmp_path)

    # Mock Slack API to fail
    def mock_slack_post(*args, **kwargs):
        raise ValueError("Invalid webhook URL")

    monkeypatch.setattr("requests.post", mock_slack_post)

    # Initialize components
    prometheus = PrometheusExporter(port=get_free_port())
    slack = SlackAlerter(webhook_url="https://hooks.slack.com/test")

    # Start Prometheus server
    prometheus.start()

    # Create assertion result
    result = AssertionResult(
        False, "Test error handling", datetime.now(), {"error": "test"}
    )

    # Test error handling in Prometheus
    prometheus.record_assertion(result)

    # Test error handling in Slack
    with pytest.raises(ValueError, match="Invalid webhook URL"):
        slack.send_alert(result)

    # Verify Prometheus metrics
    counter_samples = list(prometheus.assertion_counter.collect())[0].samples
    failed_count = next(
        sample.value
        for sample in counter_samples
        if sample.labels["status"] == "failed"
    )
    assert failed_count == 1.0


def test_integration_concurrent_usage(tmp_path):
    """Test concurrent usage of integration components."""
    # Create test model and data
    model_path, X, y = create_test_model(tmp_path)

    # Initialize multiple exporters
    exporters = [PrometheusExporter(port=get_free_port()) for _ in range(3)]

    # Start all exporters
    for exporter in exporters:
        exporter.start()

    # Create assertion result
    result = AssertionResult(
        True, "Test concurrent usage", datetime.now(), {"test": "concurrent"}
    )

    # Record in all exporters
    for exporter in exporters:
        exporter.record_assertion(result)

    # Verify metrics in all exporters
    for exporter in exporters:
        counter_samples = list(exporter.assertion_counter.collect())[0].samples
        passed_count = next(
            sample.value
            for sample in counter_samples
            if sample.labels["status"] == "passed"
        )
        assert passed_count == 1.0
    # No cleanup needed for exporters
