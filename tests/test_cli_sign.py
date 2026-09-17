import json
import sqlite3

from typer.testing import CliRunner

from bee.cli.main import app
from tests.fixtures import builders

runner = CliRunner()


def _run_id_from_scan(target: str) -> str:
    result = runner.invoke(app, ["--format", "json", "scan", target])
    return json.loads(result.output)["id"]


def test_keygen_creates_keypair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"

    result = runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    assert result.exit_code == 0
    assert (key_dir / "bee_ed25519").exists()
    assert (key_dir / "bee_ed25519.pub").exists()
    assert "fingerprint" in result.output


def test_keygen_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])
    original = (key_dir / "bee_ed25519").read_bytes()

    result = runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    assert result.exit_code == 1
    assert (key_dir / "bee_ed25519").read_bytes() == original


def test_keygen_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])
    original = (key_dir / "bee_ed25519").read_bytes()

    result = runner.invoke(app, ["keygen", "--key-dir", str(key_dir), "--force"])

    assert result.exit_code == 0
    assert (key_dir / "bee_ed25519").read_bytes() != original


def test_sign_without_a_key_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["sign", run_id, "--key", str(tmp_path / "no-such-key")])

    assert result.exit_code == 1
    assert "Generate one with" in result.output


def test_sign_then_verify_reports_signed_and_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    sign_result = runner.invoke(app, ["sign", run_id, "--key", str(key_dir / "bee_ed25519")])
    assert sign_result.exit_code == 0

    verify_result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(verify_result.output)

    assert verify_result.exit_code == 0
    assert payload["signed"] is True
    assert payload["evidence_ok"] is True
    assert payload["signer_fingerprint"] is not None


def test_signed_run_survives_the_evidence_tampering_that_defeats_unsigned_ones(tmp_path, monkeypatch):
    # The actual attack this feature exists to defeat: edit the stored
    # findings AND recompute evidence_sha256 to match, exactly as in the
    # 0.5.0 review that demonstrated an unsigned record's plain hash
    # comparison can be defeated this way. A signature must not be
    # forgeable the same way -- the attacker doesn't have the private key.
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    file_path = tmp_path / "model.pt"
    builders.write_safetensors(file_path)  # produces a real finding (BEE-FMT-001)
    run_id = _run_id_from_scan(str(file_path))
    runner.invoke(app, ["sign", run_id, "--key", str(key_dir / "bee_ed25519")])

    # Attacker: clear the findings, then recompute evidence_sha256 the
    # same way BEE does (they have the source, the algorithm is public).
    from bee.core.run import compute_evidence_hash
    from bee.storage.db import load_run

    db_path = tmp_path / ".bee" / "bee.db"
    run = load_run(db_path, run_id)
    run.findings = []
    run.summary.findings_by_severity = {k: 0 for k in run.summary.findings_by_severity}
    forged_hash = compute_evidence_hash(run.target_path, run.artifacts, run.findings, run.scanner_version)
    run.evidence_sha256 = forged_hash  # attacker updates the stored hash to match

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE runs SET data = ? WHERE id = ?", (run.model_dump_json(), run_id))
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    # The unkeyed hash "matches" (the attacker made sure of that) -- but
    # the signature, made over the ORIGINAL content, does not verify
    # against this new hash, because the attacker cannot forge it.
    assert result.exit_code == 1
    assert payload["signed"] is True
    assert payload["evidence_ok"] is False
    assert payload["ok"] is False


def test_unsigned_run_still_uses_plain_hash_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    assert payload["signed"] is False
    assert payload["evidence_ok"] is True
