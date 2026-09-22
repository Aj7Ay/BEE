"""Tests for bee ollama command."""

from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_ollama_vet_with_no_model_fails():
    """bee ollama vet with no model name must exit non-zero with clear message."""
    result = runner.invoke(app, ["ollama", "vet"])

    assert result.exit_code != 0
    assert "required" in result.output.lower() or "model" in result.output.lower()


def test_ollama_vet_with_model_when_ollama_unavailable():
    """bee ollama vet exits 1 when Ollama is not running."""
    # This test verifies graceful error handling when Ollama is unreachable
    result = runner.invoke(app, ["ollama", "vet", "qwen3:8b"])

    # Should exit with error since Ollama is not running
    assert result.exit_code != 0


def test_ollama_unknown_action_fails():
    """bee ollama with unknown action must fail clearly."""
    result = runner.invoke(app, ["ollama", "list"])

    assert result.exit_code != 0
    assert "unknown" in result.output.lower()
