import importlib.metadata
from abc import ABC, abstractmethod

from ml_assert.core.base import AssertionResult


class Plugin(ABC):
    """
    Abstract base class for all ml-assert plugins.

    A plugin must implement the `run` method, which contains the core assertion logic.
    Plugins must return an AssertionResult indicating success, message, timestamp, and metadata.
    """

    @abstractmethod
    def run(self, config: dict) -> AssertionResult:
        """
        Execute the plugin's logic.

        Args:
            config: The dictionary for this step from the main config YAML file.
                    It contains the 'type' and any other parameters the plugin needs.

        Returns:
            AssertionResult: The result of the plugin execution.
        """
        pass  # pragma: no cover


def get_plugins() -> dict[str, type[Plugin]]:
    """Discover and load plugins from entry points."""
    plugins = {}
    entry_points = importlib.metadata.entry_points(group="ml_assert.plugins")
    for entry_point in entry_points:
        plugins[entry_point.name] = entry_point.load()
    return plugins
