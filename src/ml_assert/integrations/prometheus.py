from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from ml_assert.core.base import AssertionResult


class PrometheusExporter:
    """Exposes assertion results as Prometheus metrics."""

    def __init__(self, port: int = 8000):
        """
        Initialize the Prometheus exporter.

        Args:
            port: The port on which to serve metrics.
        """
        self.port = port
        self.registry = CollectorRegistry()
        self.assertion_counter = Counter(
            "ml_assert_assertions_total",
            "Total number of assertions",
            ["status"],
            registry=self.registry,
        )
        self.assertion_gauge = Gauge(
            "ml_assert_assertions_passed",
            "Number of passed assertions",
            registry=self.registry,
        )
        self.started = False

    def start(self) -> None:
        """Start the HTTP server to serve Prometheus metrics."""
        if not self.started:
            start_http_server(self.port, registry=self.registry)
            self.started = True

    def record_assertion(self, result: AssertionResult):
        """
        Record an assertion result as Prometheus metrics.

        Args:
            result: The AssertionResult to record.
        """
        status = "passed" if result.success else "failed"
        self.assertion_counter.labels(status=status).inc()
        if result.success:
            self.assertion_gauge.inc()
        else:
            self.assertion_gauge.dec()
        # Optionally, record metadata as labels if needed (not implemented here)
