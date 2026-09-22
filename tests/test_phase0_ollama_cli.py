"""Phase 0: test bee ollama command does not crash (placeholder stage)."""

from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_ollama_vet_with_no_model_fails():
    """bee ollama vet with no model name must exit non-zero with clear message."""
    result = runner.invoke(app, ["ollama", "vet"])

    assert result.exit_code != 0
    assert "required" in result.output.lower() or "model" in result.output.lower()


def test_ollama_vet_with_model_shows_placeholder():
    """bee ollama vet with a model name must exit 0 and show not-implemented."""
    result = runner.invoke(app, ["ollama", "vet", "qwen3:8b"])

    assert result.exit_code == 0
    assert "not yet implemented" in result.output.lower()


def test_ollama_unknown_action_fails():
    """bee ollama with unknown action must fail clearly."""
    result = runner.invoke(app, ["ollama", "list"])

    assert result.exit_code != 0
    assert "unknown" in result.output.lower()
