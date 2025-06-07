from unittest.mock import MagicMock, patch

import pytest

from ml_assert.plugins.base import Plugin
from ml_assert.plugins.dvc_check import DVCArtifactCheckPlugin
from ml_assert.plugins.file_exists import FileExistsPlugin


def test_file_exists_plugin_success(tmp_path):
    """Test that FileExistsPlugin passes when the file exists."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("content")

    plugin = FileExistsPlugin()
    plugin.run({"path": str(p)})  # Should not raise


def test_file_exists_plugin_failure(tmp_path):
    """Test that FileExistsPlugin fails when the file does not exist."""
    plugin = FileExistsPlugin()
    with pytest.raises(AssertionError, match="File not found"):
        plugin.run({"path": str(tmp_path / "nonexistent.txt")})


@patch("subprocess.run")
def test_dvc_check_plugin_success_clean(mock_subprocess_run):
    """Test DVC plugin passes when status is clean (empty JSON)."""
    mock_result = MagicMock()
    mock_result.stdout = "{}"
    mock_result.stderr = ""
    mock_result.returncode = 0
    mock_subprocess_run.return_value = mock_result

    plugin = DVCArtifactCheckPlugin()
    plugin.run({"path": "data/data.csv"})  # Should not raise


@patch("subprocess.run")
def test_dvc_check_plugin_success_no_output(mock_subprocess_run):
    """Test DVC plugin passes when there is no stdout output."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.returncode = 0
    mock_subprocess_run.return_value = mock_result

    plugin = DVCArtifactCheckPlugin()
    plugin.run({"path": "data/data.csv"})  # Should not raise


@patch("subprocess.run")
def test_dvc_check_plugin_failure_modified(mock_subprocess_run):
    """Test DVC plugin fails when artifact is modified."""
    mock_result = MagicMock()
    mock_result.stdout = '{"data/data.csv": [{"status": "modified"}]}'
    mock_result.stderr = ""
    mock_result.returncode = 0
    mock_subprocess_run.return_value = mock_result

    plugin = DVCArtifactCheckPlugin()
    with pytest.raises(AssertionError, match="not in sync"):
        plugin.run({"path": "data/data.csv"})


@patch("subprocess.run")
def test_dvc_check_dvc_command_fails(mock_subprocess_run):
    """Test DVC plugin raises error if DVC command itself fails."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "DVC error: repo not found"
    mock_result.returncode = 1
    mock_subprocess_run.return_value = mock_result

    plugin = DVCArtifactCheckPlugin()
    with pytest.raises(RuntimeError, match="DVC command failed"):
        plugin.run({"path": "data/data.csv"})


def test_dvc_check_plugin_dvc_not_found():
    """Test DVC plugin raises error if dvc command is not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError):
        plugin = DVCArtifactCheckPlugin()
        with pytest.raises(FileNotFoundError):
            plugin.run({"path": "data/data.csv"})


def test_base_plugin_not_implemented():
    """Test that the base Plugin class raises NotImplementedError."""

    class BadPlugin(Plugin):
        pass

    with pytest.raises(TypeError):
        BadPlugin()
