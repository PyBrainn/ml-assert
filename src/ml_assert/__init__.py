"""
ml_assert: A comprehensive assertion-and-validation toolkit for ML workflows.
"""

__version__ = "0.1.0"

from .core.dsl import DataFrameAssertion, assert_model

__all__ = ["DataFrameAssertion", "assert_model"]
