import json
import subprocess

from .base import Plugin


class DVCArtifactCheckPlugin(Plugin):
    """
    A plugin to assert that a DVC artifact is in sync using the DVC CLI.
    """

    def run(self, config: dict) -> None:
        """
        Asserts that the DVC-tracked file has not changed since 'dvc add'.

        It uses the 'dvc data status --json' command to check the hash of
        the file against the hash stored in the corresponding .dvc file.
        An empty status or a status indicating the file is unmodified means
        the file is in sync.

        Raises:
            AssertionError: If the artifact has been modified.
            KeyError: If the 'path' key is missing from the config.
            FileNotFoundError: If the 'dvc' command is not found.
        """
        path = config["path"]

        try:
            # Check the status of all DVC-tracked files
            result = subprocess.run(
                ["dvc", "data", "status", "--json"],
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit codes
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "The 'dvc' command was not found. "
                "Please ensure DVC is installed and in your PATH."
            ) from e

        if result.returncode != 0 and result.stderr:
            # Handle cases where DVC itself throws an error
            raise RuntimeError(f"DVC command failed: {result.stderr}")

        # If there's no output, the file is in sync
        if not result.stdout.strip():
            return

        status_dict = json.loads(result.stdout)

        # Check the status for the specific path from the config
        path_status = status_dict.get(path)

        # If the path is not in the status dict, or its status is empty,
        # it's considered in sync. A non-empty status indicates a change.
        if path_status:
            raise AssertionError(
                f"DVC artifact '{path}' is not in sync. Status: {path_status}"
            )
