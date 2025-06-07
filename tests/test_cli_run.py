import json

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
