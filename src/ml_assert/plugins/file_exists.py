from datetime import datetime
from pathlib import Path

from ml_assert.core.base import AssertionResult

from .base import Plugin


class FileExistsPlugin(Plugin):
    """A simple plugin to assert that a file exists."""

    def run(self, config: dict) -> AssertionResult:
        """
        Asserts that the file specified in the 'path' key of the config exists.

        Returns:
            AssertionResult: The result of the check.
        """
        file_path = Path(config["path"])
        if file_path.exists():
            return AssertionResult(
                success=True,
                message=f"File exists: {file_path}",
                timestamp=datetime.now(),
                metadata={"path": str(file_path)},
            )
        else:
            return AssertionResult(
                success=False,
                message=f"File not found: {file_path}",
                timestamp=datetime.now(),
                metadata={"path": str(file_path)},
            )
