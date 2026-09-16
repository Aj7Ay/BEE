import json

from typer.testing import CliRunner

from bee.cli.main import app
from bee.core.artifact import Artifact
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


def test_inspect_reports_unreadable_file_without_crashing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "broken.bin"
    file_path.write_bytes(b"data")

    def _raise(cls, path):
        raise OSError("Permission denied")

    monkeypatch.setattr(Artifact, "from_file", classmethod(_raise))

    result = runner.invoke(app, ["inspect", str(file_path)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "could not read" in result.output


def test_inspect_refuses_escaping_symlink_by_default(tmp_path, monkeypatch):
    # Regression test: `bee inspect` used to have no symlink handling at
    # all -- it would follow, hash, and print the magic bytes of whatever
    # an escaping symlink pointed to, silently. This must match `bee scan`
    # exactly: refuse by default, flag it, read nothing.
    monkeypatch.chdir(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"top-secret-content")
    real_hash_prefix = secret.read_bytes()[:16].hex()

    sym_dir = tmp_path / "sym"
    sym_dir.mkdir()
    link = sym_dir / "weights.pt"
    link.symlink_to(secret)

    result = runner.invoke(app, ["--format", "json", "inspect", str(link)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifact"]["sha256"] == ""
    assert payload["artifact"]["magic_bytes_hex"] == ""
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-SYM-001" in finding_ids
    # The content must never appear anywhere in the output, in any form.
    assert real_hash_prefix not in result.output


def test_inspect_follow_symlinks_opts_into_reading_and_still_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    sym_dir = tmp_path / "sym"
    sym_dir.mkdir()
    link = sym_dir / "weights.pt"
    link.symlink_to(outside)

    result = runner.invoke(app, ["--format", "json", "inspect", str(link), "--follow-symlinks"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifact"]["sha256"] != ""
    assert payload["artifact"]["detected_format"] == "gguf"
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-SYM-001" in finding_ids


def test_inspect_still_reads_symlink_inside_its_own_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.gguf"
    builders.write_gguf(real)
    link = tmp_path / "link.gguf"
    link.symlink_to(real)

    result = runner.invoke(app, ["--format", "json", "inspect", str(link)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifact"]["sha256"] != ""
    assert payload["artifact"]["detected_format"] == "gguf"
    assert payload["findings"] == []
