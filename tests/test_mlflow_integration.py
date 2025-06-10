"""Tests for MLflow integration."""

import tempfile

import mlflow
import numpy as np
import pandas as pd
import pytest

from ml_assert.integrations.mlflow import MLflowLogger


@pytest.fixture
def temp_mlflow_dir():
    """Create a temporary directory for MLflow tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mlflow_logger(temp_mlflow_dir):
    """Create an MLflowLogger instance with temporary tracking."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    logger = MLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
    )
    return logger


def test_mlflow_logger_initialization(temp_mlflow_dir):
    """Test MLflowLogger initialization."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    logger = MLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
    )
    assert logger.experiment_name == "test_experiment"
    assert logger.run_name == "test_run"
    assert logger._run_id is None
    assert logger._active_run is None


def test_mlflow_logger_start_end_run(temp_mlflow_dir):
    """Test starting and ending MLflow runs."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    logger = MLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run_start_end",
    )

    # Start run
    logger.start_run()
    assert logger._run_id is not None
    assert logger._active_run is not None
    run_id = logger._run_id

    # End run
    logger.end_run()
    assert logger._run_id is None
    assert logger._active_run is None

    # Verify run status
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"


def test_mlflow_logger_context_manager(temp_mlflow_dir):
    """Test MLflowLogger context manager."""
    mlflow.set_tracking_uri(f"file://{temp_mlflow_dir}")
    logger = MLflowLogger(
        experiment_name="test_experiment",
        run_name="test_run_context_manager",
    )

    # Ensure no run is active
    mlflow.end_run()
    with logger.run() as active_logger:
        assert active_logger._run_id is not None
        assert active_logger._active_run is not None
        run_id = active_logger._run_id

    assert logger._run_id is None
    assert logger._active_run is None

    # Verify run status
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"


def test_log_dataframe_assertion(mlflow_logger):
    """Test logging DataFrame assertion results."""
    df = pd.DataFrame(
        {
            "numeric": np.random.normal(0, 1, 100),
            "categorical": ["A", "B", "C"] * 33 + ["A"],
        }
    )

    mlflow_logger.start_run()
    # Log DataFrame assertion
    mlflow_logger.log_dataframe_assertion(
        df=df,
        assertion_name="test_assertion_df",
        result=True,
        metrics={"accuracy": 0.95},
    )

    # Verify logged data
    run = mlflow.get_run(mlflow_logger._run_id)
    metrics = run.data.metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.95
    mlflow_logger.end_run()


def test_log_model_assertion(mlflow_logger):
    """Test logging model assertion results."""
    mlflow_logger.start_run()
    # Log model assertion
    mlflow_logger.log_model_assertion(
        model_name="test_model",
        assertion_name="test_assertion_model",
        result=True,
        metrics={"precision": 0.85, "recall": 0.90},
    )

    # Verify logged data
    run = mlflow.get_run(mlflow_logger._run_id)
    metrics = run.data.metrics
    assert "precision" in metrics
    assert metrics["precision"] == 0.85
    assert "recall" in metrics
    assert metrics["recall"] == 0.90
    mlflow_logger.end_run()


def test_log_assertion_result(mlflow_logger):
    """Test logging generic assertion results."""

    class TestAssertion:
        def __init__(self):
            self.name = "TestAssertion"

        def validate(self):
            pass

    assertion = TestAssertion()
    mlflow_logger.start_run()
    # Log assertion result
    mlflow_logger.log_assertion_result(
        assertion=assertion,
        result=True,
        metrics={"f1": 0.88},
    )

    # Verify logged data
    run = mlflow.get_run(mlflow_logger._run_id)
    metrics = run.data.metrics
    assert "f1" in metrics
    assert metrics["f1"] == 0.88
    mlflow_logger.end_run()


def test_no_active_run_error(mlflow_logger):
    """Test that logging without an active run raises an error."""

    class TestAssertion:
        def __init__(self):
            self.name = "TestAssertion"

        def validate(self):
            pass

    with pytest.raises(RuntimeError, match="No active MLflow run"):
        mlflow_logger.log_assertion_result(
            assertion=TestAssertion(),
            result=True,
        )
