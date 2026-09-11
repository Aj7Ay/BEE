from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence
from bee.evidence.mismatch import check_mismatch


def _artifact(declared: str, detected: str) -> Artifact:
    return Artifact(
        path="model.bin",
        size=10,
        sha256="a" * 64,
        sha512="b" * 128,
        declared_format=declared,
        detected_format=detected,
        format_confidence=Confidence.SUPPORTED,
        magic_bytes_hex="00",
    )


def test_mismatch_detected_when_formats_differ():
    finding = check_mismatch(_artifact("pytorch", "safetensors"))
    assert finding is not None
    assert finding.id == "BEE-FMT-001"
    assert finding.severity.value == "medium"
    assert finding.artifact_path == "model.bin"
    assert len(finding.evidence) == 2


def test_no_finding_when_formats_match():
    assert check_mismatch(_artifact("gguf", "gguf")) is None


def test_no_finding_when_declared_unknown():
    assert check_mismatch(_artifact("unknown", "safetensors")) is None


def test_no_finding_when_detected_unknown():
    assert check_mismatch(_artifact("pytorch", "unknown")) is None
