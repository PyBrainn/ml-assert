import json
import subprocess
from datetime import datetime

from ml_assert.core.base import AssertionResult

from .base import Plugin


class DVCArtifactCheckPlugin(Plugin):
    """
    A plugin to assert that a DVC artifact is in sync using the DVC CLI.
    """

    def run(self, config: dict) -> AssertionResult:
        """
        Asserts that the DVC-tracked file has not changed since 'dvc add'.

        Returns:
            AssertionResult: The result of the check.
        """
        path = config["path"]
        try:
            result = subprocess.run(
                ["dvc", "data", "status", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            return AssertionResult(
                success=False,
                message="The 'dvc' command was not found. Please ensure DVC is installed and in your PATH.",
                timestamp=datetime.now(),
                metadata={"path": path, "error": str(e)},
            )
        if result.returncode != 0 and result.stderr:
            return AssertionResult(
                success=False,
                message=f"DVC command failed: {result.stderr}",
                timestamp=datetime.now(),
                metadata={"path": path, "stderr": result.stderr},
            )
        if not result.stdout.strip():
            return AssertionResult(
                success=True,
                message=f"DVC artifact '{path}' is in sync.",
                timestamp=datetime.now(),
                metadata={"path": path},
            )
        status_dict = json.loads(result.stdout)
        path_status = status_dict.get(path)
        if path_status:
            return AssertionResult(
                success=False,
                message=f"DVC artifact '{path}' is not in sync. Status: {path_status}",
                timestamp=datetime.now(),
                metadata={"path": path, "status": path_status},
            )
        return AssertionResult(
            success=True,
            message=f"DVC artifact '{path}' is in sync.",
            timestamp=datetime.now(),
            metadata={"path": path},
        )
