import pytest

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
    finding = check_mismatch(_artifact("safetensors", "pickle"))
    assert finding is not None
    assert finding.id == "BEE-FMT-001"
    assert finding.artifact_path == "model.bin"
    assert len(finding.evidence) == 2


@pytest.mark.parametrize(
    ("declared", "detected", "expected_severity"),
    [
        # Inert format claimed, executable format found: the critical case
        # this finding exists for — code executes when the victim expects data.
        ("safetensors", "pickle", "critical"),
        ("gguf", "pytorch", "critical"),
        # Both sides execute code on load regardless: mislabeled, not a
        # change in risk (e.g. a legacy torch.save raw-pickle .pt file).
        ("pytorch", "pickle", "info"),
        ("pickle", "pytorch", "info"),
        # Claimed executable, actually inert data: no exec risk either way.
        ("pytorch", "safetensors", "low"),
        # Mislabeled between two inert formats: no exec risk either way.
        ("gguf", "numpy", "low"),
        # Anything touching an unclassified format (archive, onnx, ...):
        # genuine ambiguity, stays at the conservative default.
        ("pytorch", "onnx", "medium"),
        ("archive", "pytorch", "medium"),
    ],
)
def test_severity_matrix(declared, detected, expected_severity):
    finding = check_mismatch(_artifact(declared, detected))
    assert finding is not None
    assert finding.severity.value == expected_severity


def test_no_finding_when_formats_match():
    assert check_mismatch(_artifact("gguf", "gguf")) is None


def test_no_finding_when_declared_unknown():
    assert check_mismatch(_artifact("unknown", "safetensors")) is None


def test_no_finding_when_detected_unknown():
    assert check_mismatch(_artifact("pytorch", "unknown")) is None
