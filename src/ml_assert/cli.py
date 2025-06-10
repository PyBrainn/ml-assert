"""
Command-line interface for ml_assert.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
import yaml

from ml_assert.core.base import AssertionResult
from ml_assert.core.dsl import DataFrameAssertion, ModelAssertion, assert_model
from ml_assert.fairness.explainability import ModelExplainer
from ml_assert.fairness.fairness import FairnessMetrics
from ml_assert.integrations.mlflow import MLflowLogger
from ml_assert.integrations.prometheus import PrometheusExporter
from ml_assert.integrations.slack import SlackAlerter
from ml_assert.plugins.base import get_plugins
from ml_assert.schema import Schema
from ml_assert.stats.drift import assert_no_drift

app = typer.Typer(help="ml-assert CLI")


def _build_schema_from_yaml(schema_def: dict) -> Schema:
    """Build a Schema object from a YAML definition."""
    s = Schema()
    for col_name, rules in schema_def.items():
        col_builder = s.col(col_name)
        if not isinstance(rules, dict):
            # Handles simple case: { "col": "int64" }
            col_builder.is_type(rules)
            continue
        if "type" in rules:
            col_builder.is_type(rules["type"])
        if rules.get("unique"):
            col_builder.is_unique()
        if "range" in rules:
            col_builder.in_range(rules["range"].get("min"), rules["range"].get("max"))
    return s


@app.command()
def schema(
    file: Annotated[Path, typer.Argument(help="Path to CSV file")],
    schema_file: Annotated[Path, typer.Option(help="Path to YAML schema file")],
):
    """
    Validate a CSV file against a schema.
    """
    df = pd.read_csv(file)
    schema_def = yaml.safe_load(schema_file.read_text())
    schema_obj = _build_schema_from_yaml(schema_def)
    schema_obj.validate(df)
    print("Schema validation passed.")


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


def run_assertion(config: dict[str, Any]) -> AssertionResult:
    """
    Run an assertion based on configuration.

    Args:
        config: Assertion configuration dictionary.

    Returns:
        AssertionResult containing the assertion outcome.
    """
    try:
        assertion_type = config.get("type")
        if assertion_type == "dataframe":
            return run_dataframe_assertion(config)
        elif assertion_type == "model":
            return run_model_assertion(config)
        else:
            raise ValueError(f"Unknown assertion type: {assertion_type}")
    except Exception as e:
        return AssertionResult(
            success=False,
            message=f"Error running assertion: {str(e)}",
            timestamp=datetime.now(),
            metadata={"error": str(e)},
        )


def run_dataframe_assertion(config: dict[str, Any]) -> AssertionResult:
    """
    Run a DataFrame assertion based on configuration.

    Args:
        config: DataFrame assertion configuration.

    Returns:
        AssertionResult containing the assertion outcome.
    """
    try:
        df = pd.read_csv(config["data_path"])
        assertion = DataFrameAssertion(df)

        for check in config.get("checks", []):
            check_type = check.get("type")
            if check_type == "schema":
                assertion.satisfies(check["schema"])
            elif check_type == "nulls":
                assertion.no_nulls(check.get("columns", df.columns))
            elif check_type == "duplicates":
                assertion.no_duplicates(check.get("columns", df.columns))
            elif check_type == "range":
                assertion.in_range(check["column"], check["min"], check["max"])
            else:
                raise ValueError(f"Unknown check type: {check_type}")

        return assertion.validate()
    except Exception as e:
        return AssertionResult(
            success=False,
            message=f"Error in DataFrame assertion: {str(e)}",
            timestamp=datetime.now(),
            metadata={"error": str(e)},
        )


def run_model_assertion(config: dict[str, Any]) -> AssertionResult:
    """
    Run a model assertion based on configuration.

    Args:
        config: Model assertion configuration.

    Returns:
        AssertionResult containing the assertion outcome.
    """
    try:
        y_true = np.load(config["y_true_path"])
        y_pred = np.load(config["y_pred_path"])
        assertion = ModelAssertion(y_true, y_pred)

        for metric in config.get("metrics", []):
            metric_type = metric.get("type")
            threshold = metric.get("threshold")
            if metric_type == "accuracy":
                assertion.accuracy(threshold)
            elif metric_type == "precision":
                assertion.precision(threshold)
            elif metric_type == "recall":
                assertion.recall(threshold)
            elif metric_type == "f1":
                assertion.f1(threshold)
            elif metric_type == "roc_auc":
                y_scores = np.load(metric["y_scores_path"])
                assertion._y_scores = y_scores
                assertion.roc_auc(threshold)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

        return assertion.validate()
    except Exception as e:
        return AssertionResult(
            success=False,
            message=f"Error in model assertion: {str(e)}",
            timestamp=datetime.now(),
            metadata={"error": str(e)},
        )


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

    # Optional integrations
    slack_webhook = config.get("slack_webhook")
    prometheus_port = config.get("prometheus_port", 8000)
    slack_alerter = SlackAlerter(slack_webhook) if slack_webhook else None
    prometheus_exporter = None
    try:
        prometheus_exporter = (
            PrometheusExporter(port=prometheus_port) if prometheus_port else None
        )
        if prometheus_exporter:
            try:
                prometheus_exporter.start()
            except OSError as e:
                results.append(
                    {
                        "type": "prometheus_exporter",
                        "status": "failed",
                        "message": f"PrometheusExporter failed to start: {e}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                prometheus_exporter = None
        for step in steps:
            stype = step.get("type")
            try:
                if stype == "schema":
                    df = pd.read_csv(step["file"])
                    schema_def = yaml.safe_load(Path(step["schema_file"]).read_text())
                    schema_obj = _build_schema_from_yaml(schema_def)
                    schema_obj.validate(df)
                elif stype == "drift":
                    df_train = pd.read_csv(step["train"])
                    df_test = pd.read_csv(step["test"])
                    assert_no_drift(df_train, df_test, alpha=step.get("alpha", 0.05))
                elif stype == "model_performance":
                    y_true = np.loadtxt(step["y_true"])
                    y_pred = np.loadtxt(step["y_pred"])
                    y_scores = (
                        np.loadtxt(step["y_scores"]) if "y_scores" in step else None
                    )
                    model_asserter = assert_model(y_true, y_pred, y_scores)
                    for metric, threshold in step.get("assertions", {}).items():
                        getattr(model_asserter, metric)(threshold)
                    model_asserter.validate()
                elif stype == "fairness":
                    y_true = np.loadtxt(step["y_true"])
                    y_pred = np.loadtxt(step["y_pred"])
                    sensitive_attr = np.loadtxt(step["sensitive_attr"])
                    metrics = FairnessMetrics(y_true, y_pred, sensitive_attr)
                    if "demographic_parity" in step:
                        dp = metrics.demographic_parity()
                        if dp > step["demographic_parity"]:
                            raise AssertionError(
                                f"Demographic parity {dp:.4f} exceeds threshold {step['demographic_parity']:.4f}"
                            )
                    if "equal_opportunity" in step:
                        eo = metrics.equal_opportunity()
                        if eo > step["equal_opportunity"]:
                            raise AssertionError(
                                f"Equal opportunity {eo:.4f} exceeds threshold {step['equal_opportunity']:.4f}"
                            )
                elif stype == "explainability":
                    model = step.get("model")
                    X = pd.read_csv(step["features"])
                    explainer = ModelExplainer(model, feature_names=X.columns)
                    if "output_dir" in step:
                        explainer.save_explanation_report(
                            X,
                            step["output_dir"],
                            include_plots=step.get("include_plots", True),
                        )
                    else:
                        shap_values = explainer.explain(X)
                        output_path = Path(step.get("output", "shap_values.npy"))
                        np.save(output_path, shap_values["shap_values"])
                    if "plots" in step:
                        plots_config = step["plots"]
                        if "summary" in plots_config:
                            explainer.plot_summary(
                                X, output_path=plots_config["summary"].get("output")
                            )
                        if "dependence" in plots_config:
                            for dep_config in plots_config["dependence"]:
                                explainer.plot_dependence(
                                    X,
                                    dep_config["feature"],
                                    interaction_index=dep_config.get(
                                        "interaction_index"
                                    ),
                                    output_path=dep_config.get("output"),
                                )
                elif stype in plugins:
                    plugin_result = plugins[stype]().run(step)
                    results.append(
                        {
                            "type": stype,
                            "status": "passed" if plugin_result.success else "failed",
                            "message": plugin_result.message,
                            "metadata": plugin_result.metadata,
                            "timestamp": plugin_result.timestamp.isoformat(),
                        }
                    )
                    if prometheus_exporter:
                        prometheus_exporter.record_assertion(plugin_result)
                    if not plugin_result.success:
                        if slack_alerter:
                            slack_alerter.send_alert(plugin_result)
                        raise AssertionError(plugin_result.message)
                    continue
                else:
                    raise ValueError(f"Unknown step type or plugin: {stype}")
                results.append({"type": stype, "status": "passed", "message": ""})
                if prometheus_exporter:
                    # For built-in steps, create a dummy AssertionResult for Prometheus
                    prometheus_exporter.record_assertion(
                        AssertionResult(True, "", datetime.now(), {})
                    )
            except Exception as e:
                results.append({"type": stype, "status": "failed", "message": str(e)})
                if prometheus_exporter:
                    prometheus_exporter.record_assertion(
                        AssertionResult(False, str(e), datetime.now(), {})
                    )
                if slack_alerter:
                    slack_alerter.send_alert(
                        AssertionResult(
                            False,
                            f"Assertion failed in step '{stype}': {e}",
                            datetime.now(),
                            {},
                        )
                    )
    finally:
        report_path = config_file.with_suffix(".report.json")
        report_path.write_text(json.dumps(results, indent=2))
        typer.echo(f"Wrote JSON report to {report_path}")
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
        raise typer.Exit(code=0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ml-assert CLI: Run ml-assert assertions"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)

        result = run_assertion(config)

        # Write report file
        report_path = args.config.replace(".yaml", ".report.json")
        with open(report_path, "w") as f:
            json.dump(
                {
                    "success": result.success,
                    "message": result.message,
                    "timestamp": result.timestamp.isoformat(),
                    "metadata": result.metadata,
                },
                f,
                indent=2,
            )

        # Handle integrations
        if "integrations" in config:
            for integration in config["integrations"]:
                integration_type = integration.get("type")
                if integration_type == "slack":
                    alerter = SlackAlerter(integration["webhook_url"])
                    alerter.send_alert(result)
                elif integration_type == "prometheus":
                    exporter = PrometheusExporter(integration.get("port", 8000))
                    exporter.start()
                    exporter.record_assertion(result)
                elif integration_type == "mlflow":
                    logger = MLflowLogger(
                        tracking_uri=integration.get("tracking_uri"),
                        experiment_name=integration.get("experiment_name"),
                    )
                    logger.start_run()
                    logger.log_assertion_result(result)
                    logger.end_run()

        if not result.success:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
