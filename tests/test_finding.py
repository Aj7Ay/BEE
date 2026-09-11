import pytest
from pydantic import ValidationError

from bee.evidence.finding import Confidence, Evidence, Finding, Severity


def test_confidence_values():
    assert Confidence.VERIFIED.value == "verified"
    assert Confidence.UNKNOWN.value == "unknown"


def test_severity_values():
    assert Severity.CRITICAL.value == "critical"
    assert Severity.MEDIUM.value == "medium"


def test_evidence_round_trips_through_json():
    evidence = Evidence(
        type="magic_bytes", value="47474655", source="local_filesystem",
        confidence=Confidence.VERIFIED,
    )
    restored = Evidence.model_validate_json(evidence.model_dump_json())
    assert restored == evidence


def test_finding_round_trips_through_json():
    finding = Finding(
        id="BEE-FMT-001",
        severity=Severity.MEDIUM,
        title="Declared format does not match detected format",
        description="mismatch description",
        artifact_path="model.bin",
        evidence=[
            Evidence(type="file_extension_format", value="pytorch",
                      source="local_filesystem", confidence=Confidence.SUPPORTED),
        ],
    )
    restored = Finding.model_validate_json(finding.model_dump_json())
    assert restored == finding


def test_finding_requires_evidence_field():
    with pytest.raises(ValidationError):
        Finding(
            id="BEE-FMT-001", severity=Severity.LOW, title="t",
            description="d", artifact_path="p",
        )
