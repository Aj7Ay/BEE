from bee import __version__
from bee.core.artifact import Artifact
from bee.core.run import Run, build_summary, compute_evidence_hash
from bee.evidence.finding import Confidence, Evidence, Finding, Severity


def _artifact() -> Artifact:
    return Artifact(
        path="model.bin", size=10, sha256="a" * 64, sha512="b" * 128,
        declared_format="pytorch", detected_format="safetensors",
        format_confidence=Confidence.SUPPORTED, magic_bytes_hex="00",
    )


def _finding() -> Finding:
    return Finding(
        id="BEE-FMT-001", severity=Severity.MEDIUM, title="t", description="d",
        artifact_path="model.bin",
        evidence=[Evidence(type="x", value="y", source="local_filesystem",
                             confidence=Confidence.SUPPORTED)],
    )


def test_build_summary_counts_all_severities():
    summary = build_summary([_artifact()], [_finding()])
    assert summary.artifact_count == 1
    assert summary.findings_by_severity[Severity.MEDIUM] == 1
    assert summary.findings_by_severity[Severity.CRITICAL] == 0


def test_run_from_scan_assigns_id_and_summary():
    run = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    assert run.target_path == "./models"
    assert len(run.id) > 0
    assert run.summary.artifact_count == 1


def test_deterministic_run_has_stable_id_and_timestamp_for_identical_input():
    run_a = Run.from_scan(
        target_path="./models", artifacts=[_artifact()], findings=[_finding()], deterministic=True
    )
    run_b = Run.from_scan(
        target_path="./models", artifacts=[_artifact()], findings=[_finding()], deterministic=True
    )
    assert run_a.id == run_b.id
    assert run_a.created_at == run_b.created_at


def test_deterministic_run_id_changes_with_content():
    run_a = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[], deterministic=True)
    run_b = Run.from_scan(target_path="./other", artifacts=[_artifact()], findings=[], deterministic=True)
    assert run_a.id != run_b.id


def test_non_deterministic_runs_have_distinct_ids():
    run_a = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[])
    run_b = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[])
    assert run_a.id != run_b.id


def test_run_records_scanner_version_and_evidence_hash():
    run = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    assert run.scanner_version == __version__
    assert len(run.evidence_sha256) == 64  # a real sha256 hex digest, not left blank


def test_evidence_hash_is_stable_for_identical_content_ids_and_timestamps_aside():
    # Non-deterministic runs still get identical evidence hashes for
    # identical scan content -- the hash identifies *what* was found,
    # the run id identifies *which recorded run* found it.
    run_a = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    run_b = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    assert run_a.id != run_b.id
    assert run_a.evidence_sha256 == run_b.evidence_sha256


def test_evidence_hash_changes_when_findings_differ():
    run_a = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    run_b = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[])
    assert run_a.evidence_sha256 != run_b.evidence_sha256


def test_compute_evidence_hash_matches_what_from_scan_stores():
    run = Run.from_scan(target_path="./models", artifacts=[_artifact()], findings=[_finding()])
    recomputed = compute_evidence_hash(run.target_path, run.artifacts, run.findings, run.scanner_version)
    assert recomputed == run.evidence_sha256
