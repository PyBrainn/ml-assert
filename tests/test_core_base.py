import pytest

from ml_assert.core.base import Assertion


class MockAssertion(Assertion):
    def validate(self):
        pass


def test_base_assertion_not_implemented():
    """Test that the base Assertion class raises NotImplementedError."""

    class BadAssertion(Assertion):
        pass

    with pytest.raises(TypeError):
        BadAssertion()


def test_base_assertion_call_dunder():
    """Test that the __call__ dunder method works."""
    assertion = MockAssertion()
    # This should not raise an error
    assertion()
