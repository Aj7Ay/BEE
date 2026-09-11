from bee.core.artifact import Artifact
from bee.core.run import Run, build_summary
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
