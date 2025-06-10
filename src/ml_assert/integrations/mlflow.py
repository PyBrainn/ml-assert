"""MLflow integration for ml-assert."""

import contextlib
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from ml_assert.core.base import Assertion, AssertionResult


class MLflowLogger:
    """Logs assertion results to MLflow."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow logger.

        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Optional name for this run
            tracking_uri: Optional MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run_id = None
        self._active_run = None

    @contextlib.contextmanager
    def run(self):
        """Context manager for MLflow runs."""
        try:
            self.start_run()
            yield self
        finally:
            self.end_run()

    def start_run(self) -> None:
        """Start a new MLflow run."""
        if self._active_run is not None:
            self.end_run()

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id

        self._active_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.run_name,
        )
        self._run_id = self._active_run.info.run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run with a given status."""
        if self._active_run is not None:
            mlflow.end_run(status=status)
            self._active_run = None
            self._run_id = None

    def log_assertion_result(
        self,
        assertion: Assertion,
        result: bool,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log an assertion result to MLflow.

        Args:
            assertion: The assertion that was run
            result: Whether the assertion passed
            metrics: Optional metrics to log
            params: Optional parameters to log
        """
        if not self._run_id:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log assertion result
        self.client.log_metric(
            self._run_id,
            f"assertion_{assertion.__class__.__name__}_passed",
            1.0 if result else 0.0,
        )

        # Log additional metrics if provided
        if metrics:
            for name, value in metrics.items():
                self.client.log_metric(self._run_id, name, value)

        # Log parameters if provided
        if params:
            for name, value in params.items():
                self.client.log_param(self._run_id, name, str(value))

    def log_dataframe_assertion(
        self,
        df: pd.DataFrame,
        assertion_name: str,
        result: bool,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log a DataFrame assertion result to MLflow.

        Args:
            df: The DataFrame that was checked
            assertion_name: Name of the assertion
            result: Whether the assertion passed
            metrics: Optional metrics to log
        """
        if not self._run_id:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log basic DataFrame info
        self.client.log_param(self._run_id, f"{assertion_name}_rows", len(df))
        self.client.log_param(
            self._run_id, f"{assertion_name}_columns", len(df.columns)
        )

        # Log assertion result
        self.client.log_metric(
            self._run_id,
            f"{assertion_name}_passed",
            1.0 if result else 0.0,
        )

        # Log additional metrics if provided
        if metrics:
            for name, value in metrics.items():
                self.client.log_metric(self._run_id, name, value)

    def log_model_assertion(
        self,
        model_name: str,
        assertion_name: str,
        result: bool,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log a model assertion result to MLflow.

        Args:
            model_name: Name of the model
            assertion_name: Name of the assertion
            result: Whether the assertion passed
            metrics: Optional metrics to log
        """
        if not self._run_id:
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log model info
        self.client.log_param(self._run_id, "model_name", model_name)
        self.client.log_param(self._run_id, "assertion_name", assertion_name)

        # Log assertion result
        self.client.log_metric(
            self._run_id,
            f"{assertion_name}_passed",
            1.0 if result else 0.0,
        )

        # Log additional metrics if provided
        if metrics:
            for name, value in metrics.items():
                self.client.log_metric(self._run_id, name, value)

    def log_assertion_result_mlassert(
        self, result: AssertionResult, step_name: str | None = None
    ) -> None:
        """
        Log an AssertionResult to MLflow.

        Args:
            result: The AssertionResult to log.
            step_name: Optional name for the step/assertion.
        """
        if not self._run_id:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        # Log main result
        self.client.log_metric(
            self._run_id,
            f"{step_name or 'assertion'}_passed",
            1.0 if result.success else 0.0,
        )
        self.client.log_param(
            self._run_id, f"{step_name or 'assertion'}_message", result.message
        )
        self.client.log_param(
            self._run_id,
            f"{step_name or 'assertion'}_timestamp",
            result.timestamp.isoformat(),
        )
        # Log metadata as params
        for k, v in (result.metadata or {}).items():
            self.client.log_param(
                self._run_id, f"{step_name or 'assertion'}_{k}", str(v)
            )
