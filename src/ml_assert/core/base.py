"""
Base assertion classes for ml_assert.
"""

from abc import ABC, abstractmethod


class Assertion(ABC):
    """Base class for all assertions in ml_assert."""

    @abstractmethod
    def validate(self):
        """Run the assertion, raising AssertionError on failure."""
        raise NotImplementedError  # pragma: no cover

    def __call__(self):
        return self.validate()
