"""Top-level package for ml-assert."""

__author__ = """Shinde"""
__email__ = "aditya@heyshinde.com"
__version__ = "1.0.4"

from .core.dsl import DataFrameAssertion as Assertion
from .core.dsl import assert_model
from .schema import schema
from .stats.drift import assert_no_drift

__all__ = ["Assertion", "schema", "assert_no_drift", "assert_model"]
