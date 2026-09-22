"""End-to-end tests for bee ollama vet command."""

from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_ollama_vet_command_exists():
    """Test that bee ollama vet command is registered and accepts arguments."""
    # Test help text
    result = runner.invoke(app, ["ollama", "vet", "--help"])
    assert result.exit_code == 0
    assert "Model name" in result.output or "model" in result.output.lower()


def test_ollama_vet_requires_model_name():
    """Test that bee ollama vet requires a model name."""
    result = runner.invoke(app, ["ollama", "vet"])
    # Should fail because model is required
    assert result.exit_code != 0
    assert "required" in result.output.lower() or "model" in result.output.lower()
