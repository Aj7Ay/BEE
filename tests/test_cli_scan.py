import json

from typer.testing import CliRunner

from bee.cli.main import app
from bee.core.artifact import Artifact
from tests.fixtures import builders
from tests.test_gguf import _GGML_TYPE_F32, _build_gguf

runner = CliRunner()


def test_scan_reports_gguf_bounds_finding_for_truncated_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    (target / "truncated.gguf").write_bytes(
        _build_gguf(
            tensors=[("weight", [4], _GGML_TYPE_F32, 0)],
            tensor_data=b"\x00" * 4,  # declares 16 bytes, file only has 4
        )
    )

    result = runner.invoke(app, ["scan", str(target)])

    assert result.exit_code == 0
    assert "BEE-GGUF-001" in result.output


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

    real_from_file_with_content = Artifact.from_file_with_content.__func__

    def _maybe_raise(cls, path):
        if path.name == "broken.bin":
            raise OSError("Permission denied")
        return real_from_file_with_content(cls, path)

    monkeypatch.setattr(Artifact, "from_file_with_content", classmethod(_maybe_raise))

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


def test_scan_fail_on_exits_nonzero_when_threshold_met(tmp_path, monkeypatch):
    import pickle

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    (target / "evil.safetensors").write_bytes(pickle.dumps({"x": 1}, protocol=4))  # -> critical

    result = runner.invoke(app, ["scan", str(target), "--fail-on", "high"])

    assert result.exit_code == 1


def test_scan_fail_on_exits_zero_when_threshold_not_met(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_safetensors(target / "model.pt")  # -> low severity

    result = runner.invoke(app, ["scan", str(target), "--fail-on", "critical"])

    assert result.exit_code == 0


def test_scan_fail_on_rejects_invalid_severity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file_path = tmp_path / "model.gguf"
    builders.write_gguf(file_path)

    result = runner.invoke(app, ["scan", str(file_path), "--fail-on", "extreme"])

    assert result.exit_code != 0
    assert "must be one of" in result.output


def test_scan_deterministic_produces_identical_output_across_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_gguf(target / "model.gguf")

    first = runner.invoke(app, ["--format", "json", "scan", str(target), "--deterministic"])
    second = runner.invoke(app, ["--format", "json", "scan", str(target), "--deterministic"])

    assert first.exit_code == 0
    assert second.exit_code == 0
    first_obj = json.loads(first.output)
    second_obj = json.loads(second.output)
    # Clear timestamp fields for comparison (microsecond drift is expected)
    first_obj["provenance"]["acquisition"]["timestamp"] = ""
    second_obj["provenance"]["acquisition"]["timestamp"] = ""
    assert first_obj == second_obj


def test_scan_without_deterministic_flag_varies_run_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_gguf(target / "model.gguf")

    first = json.loads(runner.invoke(app, ["--format", "json", "scan", str(target)]).output)
    second = json.loads(runner.invoke(app, ["--format", "json", "scan", str(target)]).output)

    assert first["id"] != second["id"]


def test_scan_ignores_directory_named_bee_that_is_not_the_workspace(tmp_path, monkeypatch):
    # Regression test: exclusion must be by *location* of the actual .bee
    # workspace, not by matching the name ".bee" anywhere in the tree —
    # a real, unrelated directory happening to be named ".bee" must still
    # be scanned.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    nested_bee = target / "data" / ".bee"
    nested_bee.mkdir(parents=True)
    builders.write_gguf(nested_bee / "not_our_workspace.gguf")

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    paths = [a["path"] for a in payload["artifacts"]]
    assert any("not_our_workspace.gguf" in p for p in paths)


def test_scan_reports_result_even_when_db_cannot_be_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    builders.write_gguf(target / "model.gguf")

    # Make the database's parent directory impossible to create: a plain
    # file already sits where a directory would need to go.
    (tmp_path / "blocked").write_text("not a directory")
    bad_db = tmp_path / "blocked" / "bee.db"

    result = runner.invoke(app, ["--db", str(bad_db), "scan", str(target)])

    assert result.exit_code == 0
    assert "BEE SCAN" in result.output
    assert "Artifacts scanned: 1" in result.output
    assert "Warning: could not save run" in result.output


def test_scan_symlink_escaping_root_is_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    target = tmp_path / "models"
    target.mkdir()
    (target / "link.gguf").symlink_to(outside)

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-SYM-001" in finding_ids
    artifact = payload["artifacts"][0]
    assert artifact["is_symlink"] is True
    assert artifact["symlink_target"] == str(outside.resolve())
    # The whole point of the fix: content is never read for an escaping
    # symlink by default, so there is nothing to hash.
    assert artifact["sha256"] == ""
    assert artifact["sha512"] == ""
    assert artifact["detected_format"] == "unknown"


def test_scan_does_not_leak_escaping_symlink_target_content(tmp_path, monkeypatch):
    # The regression this guards against: an escaping symlink's target
    # used to be hashed *before* the escape was flagged, so its SHA-256
    # ended up in the run output (and the database) regardless — enough to
    # confirm a suspected file's contents from a directory an attacker
    # controls, even without exposing the file itself.
    import hashlib

    monkeypatch.chdir(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"sensitive-secret-content")
    real_hash = hashlib.sha256(secret.read_bytes()).hexdigest()

    target = tmp_path / "models"
    target.mkdir()
    (target / "link.gguf").symlink_to(secret)

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    assert real_hash not in result.output
    payload = json.loads(result.output)
    assert payload["artifacts"][0]["sha256"] == ""


def test_scan_follow_symlinks_opts_into_reading_escaping_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    target = tmp_path / "models"
    target.mkdir()
    (target / "link.gguf").symlink_to(outside)

    result = runner.invoke(app, ["--format", "json", "scan", str(target), "--follow-symlinks"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    artifact = payload["artifacts"][0]
    assert artifact["sha256"] != ""
    assert artifact["detected_format"] == "gguf"
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-SYM-001" in finding_ids  # still flagged, even though followed


def test_scan_still_reads_symlink_content_when_target_is_inside_root(tmp_path, monkeypatch):
    # Only *escaping* symlinks skip hashing by default -- a symlink whose
    # target is inside the scanned directory is normal filesystem
    # structure, not a host-escape risk, and must keep working as before.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    real = target / "real.gguf"
    builders.write_gguf(real)
    (target / "link.gguf").symlink_to(real)

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    link_artifact = next(a for a in payload["artifacts"] if a["path"].endswith("link.gguf"))
    assert link_artifact["sha256"] != ""
    assert link_artifact["detected_format"] == "gguf"
    finding_ids = [f["id"] for f in payload["findings"]]
    assert "BEE-SYM-001" not in finding_ids


def test_scan_does_not_leak_unclassified_file_content_anywhere(tmp_path, monkeypatch):
    # The redaction test suggested directly: plant a canary in a
    # non-model file inside the scan directory and confirm it appears in
    # neither the CLI output nor the persisted run database.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "models"
    target.mkdir()
    canary = b"CANARY_SECRET_DO_NOT_LEAK_0123456789"
    (target / ".env").write_bytes(canary)
    builders.write_gguf(target / "model.gguf")
    canary_hex_prefix = canary[:16].hex()

    result = runner.invoke(app, ["--format", "json", "scan", str(target)])

    assert result.exit_code == 0
    assert canary_hex_prefix not in result.output
    payload = json.loads(result.output)
    env_artifact = next(a for a in payload["artifacts"] if a["path"].endswith(".env"))
    assert env_artifact["detected_format"] == "unknown"
    assert env_artifact["magic_bytes_hex"] == ""

    db_path = tmp_path / ".bee" / "bee.db"
    assert canary_hex_prefix not in db_path.read_bytes().decode("utf-8", errors="ignore")
