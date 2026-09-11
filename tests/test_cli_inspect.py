import json

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def test_inspect_text_output_shows_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "weights.pt"
    builders.write_safetensors(file_path)

    result = runner.invoke(app, ["inspect", str(file_path)])

    assert result.exit_code == 0
    assert "Declared format:  pytorch" in result.output
    assert "Detected format:  safetensors" in result.output
    assert "BEE-FMT-001" in result.output


def test_inspect_json_output_shows_no_finding_when_matching(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)

    result = runner.invoke(app, ["--format", "json", "inspect", str(file_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifact"]["detected_format"] == "gguf"
    assert payload["findings"] == []


def test_inspect_does_not_create_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)

    runner.invoke(app, ["inspect", str(file_path)])

    assert not (tmp_path / ".bee").exists()
