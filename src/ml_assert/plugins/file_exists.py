from pathlib import Path

from .base import Plugin


class FileExistsPlugin(Plugin):
    """A simple plugin to assert that a file exists."""

    def run(self, config: dict) -> None:
        """
        Asserts that the file specified in the 'path' key of the config exists.

        Raises:
            AssertionError: If the file does not exist.
            KeyError: If the 'path' key is missing from the config.
        """
        file_path = Path(config["path"])
        if not file_path.exists():
            raise AssertionError(f"File not found: {file_path}")
