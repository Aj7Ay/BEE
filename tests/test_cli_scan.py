import json

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def test_scan_reports_mismatch_in_text_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_safetensors(target / "model.pt")  # declared pytorch, detected safetensors

    result = runner.invoke(app, ["scan", str(target)])

    assert result.exit_code == 0
    assert "BEE SCAN" in result.output
    assert "1 medium" in result.output


def test_scan_reports_mismatch_in_json_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_safetensors(target / "model.pt")

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["findings"][0]["id"] == "BEE-FMT-001"


def test_scan_single_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)

    result = runner.invoke(app, ["scan", str(file_path)])

    assert result.exit_code == 0
    assert "model.gguf" in result.output


def test_scan_persists_run_to_db(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)

    runner.invoke(app, ["scan", str(file_path)])

    assert (tmp_path / ".bee" / "bee.db").exists()
