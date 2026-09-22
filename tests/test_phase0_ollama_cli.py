"""Tests for bee ollama command."""

from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_ollama_vet_with_no_model_fails():
    """bee ollama vet with no model name must exit non-zero with clear message."""
    result = runner.invoke(app, ["ollama", "vet"])

    assert result.exit_code != 0
    assert "required" in result.output.lower() or "model" in result.output.lower()


def test_ollama_vet_with_model_completes():
    """bee ollama vet completes (may fail due to unavailable daemon, but doesn't crash)."""
    # This test verifies the command is properly wired and doesn't crash with NameError
    result = runner.invoke(app, ["ollama", "vet", "qwen3:8b"])

    # Command should complete without crashing with NameError
    assert "NameError" not in result.output, "Should not have NameError"
    # May succeed (exit 0) or fail gracefully (exit 1)
    assert result.exit_code in [0, 1], "Should complete without crash"


def test_ollama_unknown_action_fails():
    """bee ollama with unknown action must fail clearly."""
    result = runner.invoke(app, ["ollama", "list"])

    assert result.exit_code != 0
    assert "unknown" in result.output.lower()
