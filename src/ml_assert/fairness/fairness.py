import numpy as np


class FairnessMetrics:
    """A class to compute fairness metrics for a model."""

    def __init__(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attribute: np.ndarray
    ):
        """
        Initialize the FairnessMetrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_attribute: Sensitive attribute (e.g., gender, race).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_attribute = sensitive_attribute

    def demographic_parity(self) -> float:
        """
        Compute the demographic parity metric.

        Returns:
            The demographic parity score.
        """
        # Compute the probability of positive prediction for each sensitive group
        groups = np.unique(self.sensitive_attribute)
        probs = [
            np.mean(self.y_pred[self.sensitive_attribute == group]) for group in groups
        ]
        # Return the maximum difference in probabilities
        return max(probs) - min(probs)

    def equal_opportunity(self) -> float:
        """
        Compute the equal opportunity metric.

        Returns:
            The equal opportunity score.
        """
        # Compute the true positive rate for each sensitive group
        groups = np.unique(self.sensitive_attribute)
        tprs = [
            np.mean(
                self.y_pred[(self.sensitive_attribute == group) & (self.y_true == 1)]
            )
            for group in groups
        ]
        # Return the maximum difference in true positive rates
        return max(tprs) - min(tprs)
