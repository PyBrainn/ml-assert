# Fairness and Explainability

The `ml_assert.fairness` module provides tools for assessing model fairness and explainability.

## Fairness Metrics

The `FairnessMetrics` class provides methods to compute various fairness metrics:

### Demographic Parity

Demographic parity measures the difference in positive prediction rates across different groups defined by a sensitive attribute.

```python
from ml_assert.fairness.fairness import FairnessMetrics

metrics = FairnessMetrics(y_true, y_pred, sensitive_attr)
dp = metrics.demographic_parity()
```

### Equal Opportunity

Equal opportunity measures the difference in true positive rates across different groups.

```python
eo = metrics.equal_opportunity()
```

## Model Explainability

The `ModelExplainer` class uses SHAP (SHapley Additive exPlanations) to explain model predictions:

```python
from ml_assert.fairness.explainability import ModelExplainer

explainer = ModelExplainer(model, feature_names=X.columns)
```

### Basic Usage

Generate SHAP values for your model:

```python
shap_values = explainer.explain(X)
```

### Visualization

Generate various plots to understand model behavior:

```python
# Summary plot of feature importance
explainer.plot_summary(X, output_path="summary_plot.png")

# Dependence plot for a specific feature
explainer.plot_dependence(X, feature="age", output_path="dependence_age.png")

# Dependence plot with interaction
explainer.plot_dependence(
    X,
    feature="income",
    interaction_index="education",
    output_path="dependence_income_education.png"
)
```

### Feature Analysis

Analyze feature importance and interactions:

```python
# Get feature importance scores
importance_df = explainer.get_feature_importance(X)

# Analyze feature interactions
interactions = explainer.analyze_interactions(X, top_n=5)
```

### Comprehensive Reports

Generate a complete explanation report:

```python
explainer.save_explanation_report(
    X,
    output_dir="explanation_report",
    include_plots=True
)
```

This will generate:
- Raw SHAP values (`shap_values.npy`)
- Feature importance CSV (`feature_importance.csv`)
- Feature interactions CSV (`feature_interactions.csv`)
- Summary plot (`summary_plot.png`)
- Dependence plots for top features

## Using in YAML Config

You can include fairness and explainability checks in your YAML configuration:

```yaml
steps:
  # Fairness checks
  - type: fairness
    y_true: "data/y_true.csv"
    y_pred: "data/y_pred.csv"
    sensitive_attr: "data/sensitive_attr.csv"
    demographic_parity: 0.1  # Maximum allowed difference
    equal_opportunity: 0.1   # Maximum allowed difference

  # Model explainability with comprehensive report
  - type: explainability
    model: "models/trained_model.pkl"
    features: "data/features.csv"
    output_dir: "results/explanation_report"
    include_plots: true
    plots:
      summary:
        output: "results/explanation_report/summary_plot.png"
      dependence:
        - feature: "age"
          output: "results/explanation_report/dependence_age.png"
        - feature: "income"
          interaction_index: "education"
          output: "results/explanation_report/dependence_income_education.png"
```

### Fairness Step

The fairness step checks if your model's predictions satisfy fairness criteria:

- `y_true`: Path to ground truth labels
- `y_pred`: Path to model predictions
- `sensitive_attr`: Path to sensitive attribute values
- `demographic_parity`: Maximum allowed difference in positive prediction rates
- `equal_opportunity`: Maximum allowed difference in true positive rates

### Explainability Step

The explainability step provides comprehensive model explanations:

- `model`: Path to the trained model
- `features`: Path to feature data
- `output_dir`: Directory to save the explanation report (optional)
- `include_plots`: Whether to generate visualization plots (default: true)
- `plots`: Configuration for specific plots:
  - `summary`: Generate a summary plot of feature importance
  - `dependence`: Generate dependence plots for specific features
    - `feature`: Name of the feature to plot
    - `interaction_index`: Optional feature to show interaction with
    - `output`: Path to save the plot
