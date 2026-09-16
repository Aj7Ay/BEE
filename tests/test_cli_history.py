import json

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def test_history_reports_no_runs_initially(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["history"])
    assert result.exit_code == 0
    assert "No runs recorded yet." in result.output


def test_history_lists_a_completed_scan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    runner.invoke(app, ["scan", str(file_path)])

    result = runner.invoke(app, ["history"])

    assert result.exit_code == 0
    assert str(file_path) in result.output


def test_show_reprints_a_stored_run_by_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    scan_result = runner.invoke(app, ["--format", "json", "scan", str(file_path)])
    run_id = json.loads(scan_result.output)["id"]

    result = runner.invoke(app, ["--format", "json", "show", run_id])

    assert result.exit_code == 0
    assert json.loads(result.output)["id"] == run_id


def test_show_reports_error_for_unknown_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["show", "does-not-exist"])

    assert result.exit_code == 1
    assert "No run found" in result.output


def test_show_fail_on_exits_nonzero_when_threshold_met(tmp_path, monkeypatch):
    import pickle

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    (target / "evil.safetensors").write_bytes(pickle.dumps({"x": 1}, protocol=4))
    scan_result = runner.invoke(app, ["--format", "json", "scan", str(target)])
    run_id = json.loads(scan_result.output)["id"]

    result = runner.invoke(app, ["show", run_id, "--fail-on", "high"])

    assert result.exit_code == 1


def test_show_fail_on_exits_zero_when_threshold_not_met(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    scan_result = runner.invoke(app, ["--format", "json", "scan", str(file_path)])
    run_id = json.loads(scan_result.output)["id"]

    result = runner.invoke(app, ["show", run_id, "--fail-on", "info"])

    assert result.exit_code == 0


def test_show_fail_on_rejects_invalid_severity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    scan_result = runner.invoke(app, ["--format", "json", "scan", str(file_path)])
    run_id = json.loads(scan_result.output)["id"]

    result = runner.invoke(app, ["show", run_id, "--fail-on", "extreme"])

    assert result.exit_code != 0
    assert "must be one of" in result.output
