import json
import subprocess

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from ml_assert.cli import app

runner = CliRunner()


def write_csv(path, df):
    df.to_csv(path, index=False)


def test_run_success(tmp_path):
    # Prepare data files
    df = pd.DataFrame({"id": [1, 2, 3], "val": [0.1, 0.2, 0.3]})
    csv1 = tmp_path / "train.csv"
    csv2 = tmp_path / "test.csv"
    write_csv(csv1, df)
    write_csv(csv2, df)
    # Prepare YAML schema
    schema = {"id": "int64", "val": "float64"}
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(json.dumps(schema))  # cheat: JSON is also YAML

    # Prepare config
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
steps:
  - type: schema
    file: {csv1}
    schema_file: {schema_file}
  - type: drift
    train: {csv1}
    test: {csv2}
    alpha: 0.05
"""
    )

    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 0
    # Check report
    report = tmp_path / "config.report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert len(data) == 2
    assert all(step["status"] == "passed" for step in data)


def test_run_failure(tmp_path):
    # Prepare data files with drift
    df1 = pd.DataFrame({"num": [0] * 50})
    df2 = pd.DataFrame({"num": [1] * 50})
    csv1 = tmp_path / "train.csv"
    csv2 = tmp_path / "test.csv"
    write_csv(csv1, df1)
    write_csv(csv2, df2)
    # Schema same for pass
    schema = {"num": "int64"}
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(json.dumps(schema))
    # Config
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
steps:
  - type: schema
    file: {csv1}
    schema_file: {schema_file}
  - type: drift
    train: {csv1}
    test: {csv2}
    alpha: 0.01
"""
    )

    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code != 0
    report = tmp_path / "config.report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    # Schema passes, drift fails
    assert data[0]["status"] == "passed"
    assert data[1]["status"] == "failed"


def test_schema_command_pass(tmp_path):
    schema = {"col1": "int64", "col2": "object"}
    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(json.dumps(schema))  # cheat: JSON is also YAML

    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    result = runner.invoke(
        app, ["schema", str(csv_file), "--schema-file", str(schema_file)]
    )
    assert result.exit_code == 0
    assert "Schema validation passed." in result.stdout


def test_run_unknown_step_type(tmp_path):
    """Test that the run command fails with an unknown step type."""
    config = tmp_path / "config.yaml"
    config.write_text(
        """
steps:
  - type: unknown_step
"""
    )
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code != 0
    report = json.loads((tmp_path / "config.report.json").read_text())
    assert report[0]["status"] == "failed"
    assert "Unknown step type" in report[0]["message"]


def test_drift_command_pass(tmp_path):
    """Test the drift command directly."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    df.to_csv(train_file, index=False)
    df.to_csv(test_file, index=False)

    result = runner.invoke(app, ["drift", str(train_file), str(test_file)])
    assert result.exit_code == 0


def test_run_all_step_types(tmp_path):
    """Test the run command with model performance and plugin steps."""
    # --- Model Performance Data ---
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.4, 0.6])
    np.savetxt(tmp_path / "y_true.csv", y_true)
    np.savetxt(tmp_path / "y_pred.csv", y_pred)
    np.savetxt(tmp_path / "y_scores.csv", y_scores)

    # --- Plugin Data ---
    plugin_file = tmp_path / "plugin_data.txt"
    plugin_file.touch()

    # --- Config File ---
    config = tmp_path / "config.yaml"
    config.write_text(f"""
steps:
  - type: model_performance
    y_true: {tmp_path / "y_true.csv"}
    y_pred: {tmp_path / "y_pred.csv"}
    y_scores: {tmp_path / "y_scores.csv"}
    assertions:
      accuracy: 0.5
      f1: 0.5
  - type: file_exists
    path: {plugin_file}
""")

    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 0

    report = json.loads((tmp_path / "config.report.json").read_text())
    assert len(report) == 2
    assert all(step["status"] == "passed" for step in report)


def test_cli_as_main():
    """Test running the CLI script directly."""
    result = subprocess.run(
        ["python", "-m", "src.ml_assert.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ml-assert CLI" in result.stdout
