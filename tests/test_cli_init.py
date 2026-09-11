from typer.testing import CliRunner

from bee.cli.main import app

runner = CliRunner()


def test_init_creates_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / ".bee" / "config.yaml").exists()
    assert (tmp_path / ".bee" / "bee.db").exists()
    assert "Initialized" in result.output


def test_init_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "already exists" in result.output


def test_init_respects_custom_db_path(tmp_path, monkeypatch):
    # Regression test: config.yaml used to always land in ./.bee/ even when
    # --db pointed the database somewhere else, splitting the workspace.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["--db", "other/nested/bee.db", "init"])

    assert result.exit_code == 0
    assert (tmp_path / "other" / "nested" / "bee.db").exists()
    assert (tmp_path / "other" / "nested" / "config.yaml").exists()
    assert not (tmp_path / ".bee").exists()
