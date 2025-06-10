"""
Base assertion classes for ml_assert.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AssertionResult:
    """Result of an assertion execution."""

    success: bool
    message: str
    timestamp: datetime
    metadata: dict[str, Any]


class Assertion(ABC):
    """Base class for all assertions in ml_assert."""

    def __init__(self):
        """Initialize the assertion."""
        self._last_result: AssertionResult | None = None

    @abstractmethod
    def validate(self) -> AssertionResult:
        """
        Run the assertion.

        Returns:
            AssertionResult containing the result of the assertion.

        Raises:
            AssertionError: If the assertion fails and should stop execution.
        """
        raise NotImplementedError  # pragma: no cover

    def __call__(self) -> AssertionResult:
        """
        Execute the assertion and return the result.

        Returns:
            AssertionResult containing the result of the assertion.
        """
        self._last_result = self.validate()
        return self._last_result

    @property
    def last_result(self) -> AssertionResult | None:
        """
        Get the result of the last assertion execution.

        Returns:
            The last AssertionResult, or None if no assertion has been run.
        """
        return self._last_result
