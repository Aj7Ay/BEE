import json

from typer.testing import CliRunner

from bee.cli.main import app
from bee.core.artifact import Artifact
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
    assert "1 low" in result.output


def test_scan_surfaces_critical_finding_in_text_mode(tmp_path, monkeypatch):
    import pickle

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    (target / "evil.safetensors").write_bytes(pickle.dumps({"x": 1}, protocol=4))

    result = runner.invoke(app, ["scan", str(target)])

    assert result.exit_code == 0
    assert "Critical/High findings:" in result.output
    assert "BEE-FMT-001" in result.output
    assert "critical" in result.output


def test_scan_continues_after_unreadable_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_gguf(target / "good.gguf")
    (target / "broken.bin").write_bytes(b"data")

    real_from_file = Artifact.from_file.__func__

    def _maybe_raise(cls, path):
        if path.name == "broken.bin":
            raise OSError("Permission denied")
        return real_from_file(cls, path)

    monkeypatch.setattr(Artifact, "from_file", classmethod(_maybe_raise))

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["artifacts"]) == 1
    assert payload["artifacts"][0]["path"].endswith("good.gguf")
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-IO-001" in finding_ids
    io_finding = next(f for f in payload["findings"] if f["id"] == "BEE-IO-001")
    assert io_finding["severity"] == "high"
    assert io_finding["artifact_path"].endswith("broken.bin")


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
