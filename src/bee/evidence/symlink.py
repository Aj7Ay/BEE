from __future__ import annotations

from pathlib import Path

from bee.evidence.finding import Confidence, Evidence, Finding, Severity


def is_escaping_symlink(path: Path, scan_root: Path) -> str | None:
    """If `path` is a symlink whose resolved target lies outside
    `scan_root`, return that resolved target (as a string). Otherwise
    (not a symlink, or the target is inside the root) return None.

    This resolves the target and makes the decision from the raw path
    alone, before any content is opened — so a caller can refuse to read
    (and therefore hash) an escaping symlink's target at all, rather than
    hashing it first and only flagging it afterwards. Even a hash of a
    file this scan was never asked to touch is information leakage: it's
    enough to confirm a suspected file's contents from a directory an
    attacker controls.
    """
    if not path.is_symlink():
        return None
    target = path.resolve(strict=False)
    root = scan_root.resolve(strict=False)
    if target == root or target.is_relative_to(root):
        return None
    return str(target)


def build_symlink_escape_finding(artifact_path: str, symlink_target: str, *, content_read: bool) -> Finding:
    read_note = (
        "Its target was read anyway because --follow-symlinks was passed."
        if content_read
        else "Its target was not read."
    )
    return Finding(
        id="BEE-SYM-001",
        severity=Severity.HIGH,
        title="Symlink target escapes scan root",
        description=(
            f"'{artifact_path}' is a symlink resolving to '{symlink_target}', "
            f"which is outside the scanned directory. {read_note}"
        ),
        artifact_path=artifact_path,
        evidence=[
            Evidence(
                type="symlink_target",
                value=symlink_target,
                source="local_filesystem",
                confidence=Confidence.VERIFIED,
            ),
        ],
    )
