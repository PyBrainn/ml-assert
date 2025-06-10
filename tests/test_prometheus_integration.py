from datetime import datetime
from unittest.mock import patch

import pytest

from ml_assert.core.base import AssertionResult
from ml_assert.integrations.prometheus import PrometheusExporter


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


def test_prometheus_exporter_record_assertion():
    """Test that the PrometheusExporter records assertion results correctly."""
    with patch(
        "ml_assert.integrations.prometheus.start_http_server"
    ) as mock_start_server:
        exporter = PrometheusExporter(port=8000)
        exporter.start()
        # Accept any call with port and registry
        args, kwargs = mock_start_server.call_args
        assert args[0] == 8000
        assert "registry" in kwargs
        # Test recording
        result = AssertionResult(
            success=True, message="Test", timestamp=datetime.now(), metadata={}
        )
        exporter.record_assertion(result)
        result2 = AssertionResult(
            success=False, message="Fail", timestamp=datetime.now(), metadata={}
        )
        exporter.record_assertion(result2)
        # Instead of accessing _value, use collect() to get the value
        counter_value = list(exporter.assertion_counter.collect())[0].samples[0].value
        gauge_value = list(exporter.assertion_gauge.collect())[0].samples[0].value
        assert counter_value == 1.0
        assert gauge_value == 0.0


def test_prometheus_exporter_port_conflict():
    """Test handling of port conflicts when starting the server."""
    with patch(
        "ml_assert.integrations.prometheus.start_http_server",
        side_effect=OSError("Address already in use"),
    ):
        exporter = PrometheusExporter(port=8000)
        with pytest.raises(OSError) as exc_info:
            exporter.start()
        assert "Address already in use" in str(exc_info.value)


def test_prometheus_exporter_metric_collection():
    """Test comprehensive metric collection and aggregation."""
    with patch("ml_assert.integrations.prometheus.start_http_server"):
        exporter = PrometheusExporter(port=8000)
        exporter.start()

        # Record multiple assertions with different outcomes
        results = [
            AssertionResult(
                success=True, message="Test 1", timestamp=datetime.now(), metadata={}
            ),
            AssertionResult(
                success=True, message="Test 2", timestamp=datetime.now(), metadata={}
            ),
            AssertionResult(
                success=False, message="Test 3", timestamp=datetime.now(), metadata={}
            ),
            AssertionResult(
                success=False, message="Test 4", timestamp=datetime.now(), metadata={}
            ),
            AssertionResult(
                success=True, message="Test 5", timestamp=datetime.now(), metadata={}
            ),
        ]

        for result in results:
            exporter.record_assertion(result)

        # Verify counter metrics
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
        assert passed_count == 3.0  # Three passed assertions
        assert failed_count == 2.0  # Two failed assertions

        # Verify gauge value (should be 1 since we had 3 passes and 2 fails)
        gauge_value = list(exporter.assertion_gauge.collect())[0].samples[0].value
        assert gauge_value == 1.0


def test_prometheus_exporter_metadata_handling():
    """Test handling of metadata in assertion results."""
    with patch("ml_assert.integrations.prometheus.start_http_server"):
        exporter = PrometheusExporter(port=8000)
        exporter.start()

        # Test with metadata containing numeric values
        result = AssertionResult(
            success=True,
            message="Test with metadata",
            timestamp=datetime.now(),
            metadata={"accuracy": 0.95, "precision": 0.92},
        )
        exporter.record_assertion(result)

        # Verify metrics are recorded correctly
        counter_samples = list(exporter.assertion_counter.collect())[0].samples
        passed_count = next(
            sample.value
            for sample in counter_samples
            if sample.labels["status"] == "passed"
        )
        assert passed_count == 1.0


def test_prometheus_exporter_multiple_instances():
    """Test behavior with multiple exporter instances."""
    with patch("ml_assert.integrations.prometheus.start_http_server") as mock_start:
        # Create multiple exporters with different ports
        exporters = [PrometheusExporter(port=8000 + i) for i in range(3)]

        # Start all exporters
        for exporter in exporters:
            exporter.start()

        # Verify each exporter was started with correct port
        assert mock_start.call_count == 3
        for i, call in enumerate(mock_start.call_args_list):
            args, _ = call
            assert args[0] == 8000 + i

        # Record assertions in each exporter
        result = AssertionResult(
            success=True, message="Test", timestamp=datetime.now(), metadata={}
        )
        for exporter in exporters:
            exporter.record_assertion(result)

        # Verify metrics in each exporter
        for exporter in exporters:
            counter_samples = list(exporter.assertion_counter.collect())[0].samples
            passed_count = next(
                sample.value
                for sample in counter_samples
                if sample.labels["status"] == "passed"
            )
            assert passed_count == 1.0


def test_prometheus_exporter_error_handling():
    """Test error handling in the Prometheus exporter."""
    with patch("ml_assert.integrations.prometheus.start_http_server"):
        exporter = PrometheusExporter(port=8000)
        exporter.start()

        # Test with invalid assertion result
        with pytest.raises(AttributeError):
            exporter.record_assertion(None)

        # Test with assertion result missing required attributes
        class InvalidResult:
            pass

        with pytest.raises(AttributeError):
            exporter.record_assertion(InvalidResult())
