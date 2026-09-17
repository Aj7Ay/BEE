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


# ---------------------------------------------------------------------------
# --signer / --pubkey pinning: without a pin, "VALID" only means "signed
# by *some* key embedded in the record" -- an attacker can re-sign under
# their own key. These tests exercise that exact attack and confirm
# pinning closes it.
# ---------------------------------------------------------------------------


def test_unpinned_verify_accepts_a_key_substitution_attack(tmp_path, monkeypatch):
    # The attack the pin exists to stop: attacker downgrades a critical
    # finding, then re-signs the forged record under their OWN key
    # (generated fresh, not the original signer's). An unpinned verify
    # has no way to tell -- and correctly says so, since it never claimed
    # to check *whose* key.
    monkeypatch.chdir(tmp_path)
    owner_keys = tmp_path / "owner-keys"
    runner.invoke(app, ["keygen", "--key-dir", str(owner_keys)])

    file_path = tmp_path / "model.pt"
    builders.write_safetensors(file_path)  # real finding: BEE-FMT-001
    run_id = _run_id_from_scan(str(file_path))
    runner.invoke(app, ["sign", run_id, "--key", str(owner_keys / "bee_ed25519")])

    original = json.loads(runner.invoke(app, ["--format", "json", "verify", run_id]).output)
    original_signer = original["signer_fingerprint"]

    # Attacker: own keypair, own database write access.
    attacker_keys = tmp_path / "attacker-keys"
    runner.invoke(app, ["keygen", "--key-dir", str(attacker_keys)])

    from bee.core.run import compute_evidence_hash
    from bee.core.signing import load_private_key, sign_hash
    from bee.storage.db import load_run, save_run

    db_path = tmp_path / ".bee" / "bee.db"
    run = load_run(db_path, run_id)
    run.findings = []
    run.summary.findings_by_severity = {k: 0 for k in run.summary.findings_by_severity}
    forged_hash = compute_evidence_hash(run.target_path, run.artifacts, run.findings, run.scanner_version)
    attacker_private_key = load_private_key(attacker_keys / "bee_ed25519")
    signature_hex, public_key_hex = sign_hash(attacker_private_key, forged_hash)
    run.evidence_sha256 = forged_hash
    run.signature = signature_hex
    run.public_key = public_key_hex
    save_run(db_path, run)

    result = runner.invoke(app, ["--format", "json", "verify", run_id])
    payload = json.loads(result.output)

    # Unpinned: the forged-and-re-signed record is internally consistent,
    # so verify correctly reports VALID -- it never checked the signer.
    assert result.exit_code == 0
    assert payload["signed"] is True
    assert payload["evidence_ok"] is True
    assert payload["signer_fingerprint"] != original_signer  # the tell, if anyone looks

    # Pinned to the real owner's key: the substitution is caught.
    pinned_result = runner.invoke(
        app, ["--format", "json", "verify", run_id, "--signer", original_signer]
    )
    pinned_payload = json.loads(pinned_result.output)

    assert pinned_result.exit_code == 1
    assert pinned_payload["signer_matches"] is False
    assert pinned_payload["ok"] is False


def test_verify_signer_pin_accepts_the_correct_signer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))
    runner.invoke(app, ["sign", run_id, "--key", str(key_dir / "bee_ed25519")])

    fingerprint = json.loads(
        runner.invoke(app, ["--format", "json", "verify", run_id]).output
    )["signer_fingerprint"]

    result = runner.invoke(app, ["--format", "json", "verify", run_id, "--signer", fingerprint])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["signer_matches"] is True
    assert payload["ok"] is True


def test_verify_pubkey_pin_accepts_the_correct_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))
    runner.invoke(app, ["sign", run_id, "--key", str(key_dir / "bee_ed25519")])

    result = runner.invoke(
        app, ["--format", "json", "verify", run_id, "--pubkey", str(key_dir / "bee_ed25519.pub")]
    )
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["signer_matches"] is True


def test_verify_pin_rejects_an_unsigned_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["--format", "json", "verify", run_id, "--signer", "deadbeefdeadbeef"])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["signed"] is False
    assert payload["ok"] is False


# ---------------------------------------------------------------------------
# Pin input-validation edge cases: the pin mechanism itself is correct,
# but the human typing it can get the case, path, or presence wrong.
# ---------------------------------------------------------------------------


def test_verify_signer_pin_is_case_insensitive(tmp_path, monkeypatch):
    # A fingerprint pasted in uppercase (password managers, some
    # terminals) is the same fingerprint, not a different key -- it must
    # not read as a substitution attack.
    monkeypatch.chdir(tmp_path)
    key_dir = tmp_path / "keys"
    runner.invoke(app, ["keygen", "--key-dir", str(key_dir)])

    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))
    runner.invoke(app, ["sign", run_id, "--key", str(key_dir / "bee_ed25519")])

    fp = json.loads(runner.invoke(app, ["--format", "json", "verify", run_id]).output)["signer_fingerprint"]

    result = runner.invoke(app, ["--format", "json", "verify", run_id, "--signer", fp.upper()])
    payload = json.loads(result.output)

    assert result.exit_code == 0
    assert payload["signer_matches"] is True


def test_verify_pubkey_missing_file_fails_cleanly_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["verify", run_id, "--pubkey", str(tmp_path / "does" / "not" / "exist.pub")])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "could not read" in result.output


def test_verify_rejects_empty_signer_instead_of_silently_unpinning(tmp_path, monkeypatch):
    # The dangerous shape in CI: `--signer "$EXPECTED"` with $EXPECTED
    # unset expands to `--signer ""`. A gate meant to *require* a signer
    # must fail loud on that, not silently fall back to unpinned
    # verification and report VALID on an unrelated key.
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["verify", run_id, "--signer", ""])

    assert result.exit_code == 1
    assert "requires a non-empty value" in result.output


def test_verify_rejects_empty_pubkey_instead_of_silently_unpinning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)
    run_id = _run_id_from_scan(str(file_path))

    result = runner.invoke(app, ["verify", run_id, "--pubkey", ""])

    assert result.exit_code == 1
    assert "requires a non-empty value" in result.output
