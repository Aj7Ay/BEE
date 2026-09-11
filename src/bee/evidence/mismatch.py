from __future__ import annotations

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity

# Formats whose loader can execute arbitrary code as a side effect of
# deserialization (pickle opcodes, including PyTorch's zip-embedded pickle).
_EXECUTABLE_FORMATS = frozenset({"pickle", "pytorch"})

# Formats with no such capability by construction (pure data layouts).
_SAFE_FORMATS = frozenset({"safetensors", "gguf", "numpy", "hdf5"})


def _severity_for_mismatch(declared: str, detected: str) -> Severity:
    declared_exec = declared in _EXECUTABLE_FORMATS
    detected_exec = detected in _EXECUTABLE_FORMATS
    declared_safe = declared in _SAFE_FORMATS
    detected_safe = detected in _SAFE_FORMATS

    if detected_exec and declared_safe:
        # A file claiming to be inert data but actually executable on load
        # (e.g. a pickle disguised as .safetensors) is the whole point of
        # this finding — arbitrary code execution on load.
        return Severity.CRITICAL
    if detected_exec and declared_exec:
        # Both sides execute code on load either way (e.g. a legacy raw-pickle
        # .pt file) — mislabeled, but not a change in what happens on load.
        return Severity.INFO
    if detected_safe and declared_exec:
        # Claims to be executable but is actually inert data — no exec risk.
        return Severity.LOW
    if detected_safe and declared_safe:
        # Mislabeled between two inert formats — no exec risk either way.
        return Severity.LOW
    # Anything involving an archive/onnx/other not classified above:
    # genuine ambiguity, keep the conservative default.
    return Severity.MEDIUM


def check_mismatch(artifact: Artifact) -> Finding | None:
    if artifact.declared_format == "unknown" or artifact.detected_format == "unknown":
        return None
    if artifact.declared_format == artifact.detected_format:
        return None
    return Finding(
        id="BEE-FMT-001",
        severity=_severity_for_mismatch(artifact.declared_format, artifact.detected_format),
        title="Declared format does not match detected format",
        description=(
            f"The file extension implies '{artifact.declared_format}', but "
            f"structural analysis detected '{artifact.detected_format}'."
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(
                type="file_extension_format",
                value=artifact.declared_format,
                source="local_filesystem",
                confidence=Confidence.SUPPORTED,
            ),
            Evidence(
                type="detected_format",
                value=artifact.detected_format,
                source="local_filesystem",
                confidence=artifact.format_confidence,
            ),
        ],
    )
