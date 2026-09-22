"""Phase 0: test bee vet command basic functionality (placeholder stage)."""

from pathlib import Path

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def test_vet_command_with_gguf_file(tmp_path):
    """bee vet on a real GGUF file must not crash (Phase 0: placeholder)."""
    gguf_file = tmp_path / "model.gguf"
    builders.write_gguf(gguf_file)

    result = runner.invoke(app, ["vet", str(gguf_file)])

    # Phase 0: vet command exists and runs. Doesn't need to be fully functional yet.
    # Main thing is it doesn't crash with a Python traceback.
    assert "Traceback" not in result.output
    assert result.exit_code in (0, 1)  # May exit 1 if dependencies are missing, that's OK


def test_vet_command_no_args_fails(tmp_path):
    """bee vet with no arguments must exit non-zero."""
    result = runner.invoke(app, ["vet"])

    assert result.exit_code != 0
    assert "missing" in result.output.lower() or "required" in result.output.lower()
