"""
Command-line interface for ml_assert.
"""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import yaml

from ml_assert.core.dsl import assert_model
from ml_assert.data.schema import validate_schema
from ml_assert.plugins.base import get_plugins
from ml_assert.stats.drift import assert_no_drift

app = typer.Typer(help="ml-assert CLI")


@app.command()
def schema(
    file: Annotated[Path, typer.Argument(help="Path to CSV file")],
    schema_file: Annotated[Path, typer.Option(help="Path to YAML schema file")],
):
    """
    Validate a CSV file against a schema.
    """
    validate_schema(file, schema_file)


@app.command()
def drift(
    train: Annotated[Path, typer.Argument(help="Path to training CSV file")],
    test: Annotated[Path, typer.Argument(help="Path to test CSV file")],
    alpha: Annotated[float, typer.Option(help="Significance level for tests")] = 0.05,
):
    """
    Check for drift between two datasets.
    """
    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test)
    assert_no_drift(df_train, df_test, alpha=alpha)


@app.command()
def run(
    config_file: Annotated[Path, typer.Argument(help="Path to YAML config file")],
):
    """
    Run a full suite of assertions from a config file.
    """
    config = yaml.safe_load(config_file.read_text())
    steps = config.get("steps", [])
    results = []
    plugins = get_plugins()

    for step in steps:
        stype = step.get("type")
        try:
            if stype == "schema":
                validate_schema(Path(step["file"]), Path(step["schema_file"]))
            elif stype == "drift":
                df_train = pd.read_csv(step["train"])
                df_test = pd.read_csv(step["test"])
                assert_no_drift(df_train, df_test, alpha=step.get("alpha", 0.05))
            elif stype == "model_performance":
                y_true = np.loadtxt(step["y_true"])
                y_pred = np.loadtxt(step["y_pred"])
                y_scores = np.loadtxt(step["y_scores"]) if "y_scores" in step else None
                model_asserter = assert_model(y_true, y_pred, y_scores)
                for metric, threshold in step.get("assertions", {}).items():
                    getattr(model_asserter, metric)(threshold)
                model_asserter.validate()
            elif stype in plugins:
                plugins[stype]().run(step)
            else:
                raise ValueError(f"Unknown step type or plugin: {stype}")
            results.append({"type": stype, "status": "passed", "message": ""})
        except Exception as e:
            results.append({"type": stype, "status": "failed", "message": str(e)})

    report_path = config_file.with_suffix(".report.json")
    report_path.write_text(json.dumps(results, indent=2))
    typer.echo(f"Wrote JSON report to {report_path}")

    # Generate and write HTML report
    html_report = [
        "<html><body><h1>ml-assert Report</h1><table border='1'><tr><th>Step</th><th>Status</th><th>Message</th></tr>"
    ]
    for r in results:
        html_report.append(
            f"<tr><td>{r['type']}</td><td>{r['status']}</td><td>{r['message']}</td></tr>"
        )
    html_report.append("</table></body></html>")
    html_path = config_file.with_suffix(".report.html")
    html_path.write_text("\n".join(html_report))
    typer.echo(f"Wrote HTML report to {html_path}")

    if any(r["status"] == "failed" for r in results):
        typer.echo("Some steps failed.")
        raise typer.Exit(code=1)
    typer.echo("All steps passed.")


if __name__ == "__main__":
    app()
