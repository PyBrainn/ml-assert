from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ModelExplainer:
    """A class to provide model explainability using SHAP values."""

    def __init__(self, model: Any, feature_names: list | None = None):
        """
        Initialize the ModelExplainer.

        Args:
            model: The trained model to explain.
            feature_names: Optional list of feature names.
        """
        self.model = model
        self.feature_names = feature_names
        # Use a default masker (the input data) if none is provided
        self.explainer = shap.Explainer(
            model,
            masker=pd.DataFrame(
                np.zeros((1, len(feature_names))), columns=feature_names
            ),
        )

    def explain(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Generate SHAP values for the given input data.

        Args:
            X: Input data as a pandas DataFrame.

        Returns:
            A dictionary containing SHAP values.
        """
        shap_values = self.explainer(X)
        return {"shap_values": shap_values.values}

    def plot_summary(self, X: pd.DataFrame, output_path: str | None = None) -> None:
        """
        Generate and optionally save a summary plot of SHAP values.

        Args:
            X: Input data as a pandas DataFrame.
            output_path: Optional path to save the plot.
        """
        shap_values = self.explainer(X)
        # Handle multiclass: select first class if 3D
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 0]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, X, show=False)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        interaction_index: str | None = None,
        output_path: str | None = None,
    ) -> None:
        """
        Generate and optionally save a dependence plot for a specific feature.

        Args:
            X: Input data as a pandas DataFrame.
            feature: Name of the feature to plot.
            interaction_index: Optional feature to show interaction with.
            output_path: Optional path to save the plot.
        """
        shap_values = self.explainer(X)
        # Handle multiclass: select first class if 3D
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 0]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, values, X, interaction_index=interaction_index, show=False
        )
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature importance based on mean absolute SHAP values.

        Args:
            X: Input data as a pandas DataFrame.

        Returns:
            DataFrame with feature names and their importance scores.
        """
        shap_values = self.explainer(X)
        importance = np.abs(shap_values.values).mean(axis=0)
        n_features = len(self.feature_names)
        if importance.size != n_features:
            # Handle multiclass: importance shape (n_classes, n_features)
            if importance.size % n_features == 0:
                importance = importance.reshape(-1, n_features).mean(axis=0)
            else:
                raise ValueError(
                    f"Mismatch between number of features ({n_features}) and importance shape {importance.shape}."
                )
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def analyze_interactions(
        self, X: pd.DataFrame, top_n: int = 5
    ) -> list[tuple[str, str, float]]:
        """
        Analyze feature interactions using SHAP interaction values.

        Args:
            X: Input data as a pandas DataFrame.
            top_n: Number of top interactions to return.

        Returns:
            List of tuples containing (feature1, feature2, interaction_strength).
        """
        shap_values = self.explainer(X)
        interaction_values = shap_values.values

        # Calculate interaction strengths
        interactions = []
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                interaction_strength = np.abs(
                    interaction_values[:, i] * interaction_values[:, j]
                ).mean()
                interactions.append(
                    (self.feature_names[i], self.feature_names[j], interaction_strength)
                )

        # Sort by interaction strength and return top N
        return sorted(interactions, key=lambda x: x[2], reverse=True)[:top_n]

    def save_explanation_report(
        self, X: pd.DataFrame, output_dir: str, include_plots: bool = True
    ) -> None:
        """
        Generate a comprehensive explanation report.

        Args:
            X: Input data as a pandas DataFrame.
            output_dir: Directory to save the report and plots.
            include_plots: Whether to include visualization plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate SHAP values
        shap_values = self.explainer(X)

        # Save raw SHAP values
        np.save(output_dir / "shap_values.npy", shap_values.values)

        # Generate and save feature importance
        importance_df = self.get_feature_importance(X)
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

        # Generate and save interaction analysis
        interactions = self.analyze_interactions(X)
        pd.DataFrame(interactions, columns=["feature1", "feature2", "strength"]).to_csv(
            output_dir / "feature_interactions.csv", index=False
        )

        if include_plots:
            # Generate summary plot
            self.plot_summary(X, str(output_dir / "summary_plot.png"))

            # Generate dependence plots for top features
            for feature in importance_df["feature"].head(3):
                self.plot_dependence(
                    X,
                    feature,
                    output_path=str(output_dir / f"dependence_{feature}.png"),
                )
