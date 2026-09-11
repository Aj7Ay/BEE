from typer.main import get_command
from typer.testing import CliRunner

from bee import __version__
from bee.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "bee" in result.output
    assert __version__ in result.output


def test_help_runs_without_error():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_version_option_is_registered():
    # Checked against the Click command definition directly, rather than
    # Rich-rendered --help text: Rich's terminal-width/height detection
    # differs between environments (observed: passes locally, wraps and
    # drops content on GitHub Actions' Linux runners), so asserting on the
    # rendered layout is inherently flaky.
    command = get_command(app)
    option_names = {opt for param in command.params for opt in param.opts}
    assert "--version" in option_names
