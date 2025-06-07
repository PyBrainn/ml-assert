"""Top-level package for ml-assert."""

__author__ = """Shinde"""
__email__ = "data@shinde.io"
__version__ = "1.0.3"

from .core.dsl import DataFrameAssertion as Assertion
from .core.dsl import assert_model
from .schema import schema
from .stats.drift import assert_no_drift

__all__ = ["Assertion", "schema", "assert_no_drift", "assert_model"]
