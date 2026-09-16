import json
import sqlite3

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def _run_id_from_scan(target: str) -> str:
    result = runner.invoke(app, ["--format", "json", "scan", target])
    return json.loads(result.output)["id"]


def test_verify_unknown_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["verify", "does-not-exist"])

    assert result.exit_code == 1
    assert "No run found" in result.output


def test_verify_clean_run_is_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["verify", run_id])

    assert result.exit_code == 0
    assert "Evidence record:  OK" in result.output
    assert "OK" in result.output
    assert "CHANGED" not in result.output
    assert "MISSING" not in result.output


def test_verify_json_output_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["run_id"] == run_id
    assert payload["evidence_ok"] is True
    assert payload["ok"] is True
    assert payload["artifacts"][0]["status"] == "ok"


def test_verify_detects_modified_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    # The file changes after it was vetted.
    file_path.write_bytes(b"GGUF" + b"\x99\x99\x99\x99" + b"\xff" * 8)

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["ok"] is False
    assert payload["artifacts"][0]["status"] == "changed"
    assert payload["artifacts"][0]["recorded_sha256"] != payload["artifacts"][0]["current_sha256"]


def test_verify_detects_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    file_path.unlink()

    result = runner.invoke(app, ["verify", run_id])

    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_verify_detects_evidence_tampering(tmp_path, monkeypatch):
    # Directly corrupt the stored run's findings in the database, the way
    # someone editing the sqlite file by hand (or a bug elsewhere) would --
    # bee verify must not just trust whatever the row currently says.
    # Uses a file that genuinely produces a finding, so clearing it below
    # is an actual alteration, not a no-op on an already-empty list.
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.pt"
    builders.write_safetensors(file_path)  # declared pytorch, detected safetensors -> BEE-FMT-001
    run_id = _run_id_from_scan(str(file_path))
    assert json.loads(runner.invoke(app, ["--format", "json", "show", run_id]).output)["findings"]

    db_path = tmp_path / ".bee" / "bee.db"
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
    tampered = json.loads(row[0])
    tampered["findings"] = []  # pretend nothing was found
    for key in tampered["summary"]["findings_by_severity"]:
        tampered["summary"]["findings_by_severity"][key] = 0
    conn.execute("UPDATE runs SET data = ? WHERE id = ?", (json.dumps(tampered), run_id))
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["verify", run_id])

    assert result.exit_code == 1
    assert "TAMPERED" in result.output


def test_verify_skips_artifact_with_no_recorded_hash(tmp_path, monkeypatch):
    # An artifact with no original hash (e.g. an unresolved escaping
    # symlink at scan time) has nothing to compare against -- skipped,
    # not treated as a failure on its own.
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    target = tmp_path / "models"
    target.mkdir()
    (target / "link.gguf").symlink_to(outside)
    run_id = _run_id_from_scan(str(target))

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    assert payload["ok"] is True
    assert payload["artifacts"][0]["status"] == "skipped"


def test_verify_detects_symlink_swapped_in_after_vetting(tmp_path, monkeypatch):
    # The TOCTOU case: a file that was a normal, in-root file (or a safe
    # in-root symlink) at scan time is replaced with an escaping symlink
    # before verify runs. Re-hashing it naively would silently follow the
    # new target -- the same class of bug fixed in scan/inspect for the
    # initial vetting must also hold up on re-verification.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    real_path = target / "weights.gguf"
    builders.write_gguf(real_path)
    run_id = _run_id_from_scan(str(target))

    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"sensitive-content")
    real_path.unlink()
    real_path.symlink_to(secret)

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["ok"] is False
    assert payload["artifacts"][0]["status"] == "escaped"
