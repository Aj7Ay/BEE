from __future__ import annotations

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity


def check_mismatch(artifact: Artifact) -> Finding | None:
    if artifact.declared_format == "unknown" or artifact.detected_format == "unknown":
        return None
    if artifact.declared_format == artifact.detected_format:
        return None
    return Finding(
        id="BEE-FMT-001",
        severity=Severity.MEDIUM,
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
