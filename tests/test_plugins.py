from unittest.mock import MagicMock, patch

import pytest

from ml_assert.core.base import AssertionResult
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
    result = plugin.run({"path": str(p)})
    assert isinstance(result, AssertionResult)
    assert result.success is True
    assert "File exists" in result.message
    assert result.metadata["path"] == str(p)


def test_file_exists_plugin_failure(tmp_path):
    """Test that FileExistsPlugin fails when the file does not exist."""
    plugin = FileExistsPlugin()
    result = plugin.run({"path": str(tmp_path / "nonexistent.txt")})
    assert isinstance(result, AssertionResult)
    assert result.success is False
    assert "File not found" in result.message
    assert result.metadata["path"] == str(tmp_path / "nonexistent.txt")


@patch("subprocess.run")
def test_dvc_check_plugin_success_clean(mock_subprocess_run):
    """Test DVC plugin passes when status is clean (empty JSON)."""
    mock_result = MagicMock()
    mock_result.stdout = "{}"
    mock_result.stderr = ""
    mock_result.returncode = 0
    mock_subprocess_run.return_value = mock_result

    plugin = DVCArtifactCheckPlugin()
    result = plugin.run({"path": "data/data.csv"})
    assert isinstance(result, AssertionResult)
    assert result.success is True
    assert "in sync" in result.message
    assert result.metadata["path"] == "data/data.csv"


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
    result = plugin.run({"path": "data/data.csv"})
    assert isinstance(result, AssertionResult)
    assert result.success is False
    assert "not in sync" in result.message
    assert result.metadata["path"] == "data/data.csv"
    assert "status" in result.metadata


@patch("subprocess.run")
def test_dvc_check_dvc_command_fails(mock_subprocess_run):
    """Test DVC plugin returns AssertionResult if DVC command itself fails."""
    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = "DVC error: repo not found"
    mock_result.returncode = 1
    mock_subprocess_run.return_value = mock_result
    plugin = DVCArtifactCheckPlugin()
    result = plugin.run({"path": "data/data.csv"})
    assert isinstance(result, AssertionResult)
    assert result.success is False
    assert "DVC command failed" in result.message
    assert result.metadata["path"] == "data/data.csv"
    assert "stderr" in result.metadata


@patch("subprocess.run")
def test_dvc_check_plugin_dvc_not_found(mock_subprocess_run):
    """Test DVC plugin raises error if dvc command is not found."""
    mock_subprocess_run.side_effect = FileNotFoundError("dvc not found")
    plugin = DVCArtifactCheckPlugin()
    result = plugin.run({"path": "data/data.csv"})
    assert isinstance(result, AssertionResult)
    assert result.success is False
    assert "dvc" in result.message.lower()
    assert result.metadata["path"] == "data/data.csv"
    assert "error" in result.metadata


def test_base_plugin_not_implemented():
    """Test that the base Plugin class raises NotImplementedError."""

    class BadPlugin(Plugin):
        pass

    with pytest.raises(TypeError):
        BadPlugin()
