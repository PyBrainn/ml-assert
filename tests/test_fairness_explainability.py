import numpy as np
import pandas as pd

from ml_assert.fairness.explainability import ModelExplainer


class MockModel:
    def predict(self, X):
        return np.zeros(len(X))

    def __call__(self, X):
        return self.predict(X)


def test_model_explainer():
    """Test that the ModelExplainer generates SHAP values correctly."""
    model = MockModel()
    X = pd.DataFrame(np.random.rand(10, 5), columns=["A", "B", "C", "D", "E"])
    explainer = ModelExplainer(model, feature_names=X.columns)
    result = explainer.explain(X)
    assert "shap_values" in result
    assert result["shap_values"].shape == (10, 5)
