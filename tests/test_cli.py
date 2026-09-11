from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "bee" in result.output
    assert "0.1.0" in result.output


def test_help_runs_without_error():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--version" in result.output
