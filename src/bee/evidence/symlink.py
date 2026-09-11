from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity


def check_symlink_escape(artifact: Artifact, scan_root: Path) -> Finding | None:
    """Flag a symlink whose resolved target lies outside the directory
    being scanned — an attacker-placed symlink in a model directory can
    otherwise turn a scan into a hash-harvesting primitive against
    arbitrary files on the host.
    """
    if not artifact.is_symlink or artifact.symlink_target is None:
        return None

    root = scan_root.resolve(strict=False)
    target = Path(artifact.symlink_target)
    if target == root or target.is_relative_to(root):
        return None

    return Finding(
        id="BEE-SYM-001",
        severity=Severity.HIGH,
        title="Symlink target escapes scan root",
        description=(
            f"'{artifact.path}' is a symlink resolving to '{artifact.symlink_target}', "
            f"which is outside the scanned directory '{root}'."
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(
                type="symlink_target",
                value=artifact.symlink_target,
                source="local_filesystem",
                confidence=Confidence.VERIFIED,
            ),
        ],
    )
