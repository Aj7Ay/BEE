from __future__ import annotations

import hashlib
import json as jsonlib
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from bee import __version__
from bee.core.artifact import Artifact
from bee.evidence.finding import Finding, Severity


class RunSummary(BaseModel):
    artifact_count: int
    findings_by_severity: dict[Severity, int]


def build_summary(artifacts: list[Artifact], findings: list[Finding]) -> RunSummary:
    counts = Counter(finding.severity for finding in findings)
    findings_by_severity = {severity: counts.get(severity, 0) for severity in Severity}
    return RunSummary(artifact_count=len(artifacts), findings_by_severity=findings_by_severity)


def _deterministic_run_id(target_path: str, artifacts: list[Artifact]) -> str:
    hasher = hashlib.sha256()
    hasher.update(target_path.encode("utf-8"))
    for artifact in sorted(artifacts, key=lambda a: a.path):
        hasher.update(artifact.path.encode("utf-8"))
        hasher.update(artifact.sha256.encode("utf-8"))
    return hasher.hexdigest()[:32]


def compute_evidence_hash(
    target_path: str,
    artifacts: list[Artifact],
    findings: list[Finding],
    scanner_version: str,
) -> str:
    """A hash of the evidence itself -- what was scanned, what was found,
    and by what version of BEE -- independent of the run's id or
    timestamp. Two scans of identical, unchanged input produce the same
    evidence hash regardless of when they ran or whether --deterministic
    was passed.

    This is a plain, UNKEYED sha256 -- the algorithm is public and the
    hash is stored right next to the data it covers. That makes it good
    at what an integrity hash without a secret can be good at: catching
    accidental corruption and naive edits (a manual database UPDATE that
    forgets to also update this field, a bug that silently drops a
    finding). It does **not** resist a capable attacker who can write to
    the database: such an attacker can edit the record and recompute a
    matching hash the same way this function does, and `bee verify` will
    report it as consistent. Detecting that requires a keyed hash (HMAC,
    with a key not stored beside the database) or a real signature --
    planned, not yet built. A clean `bee verify` is evidence the record
    is internally self-consistent, not proof no one who understands this
    format has touched it.
    """
    payload = {
        "target_path": target_path,
        "scanner_version": scanner_version,
        "artifacts": [a.model_dump(mode="json") for a in artifacts],
        "findings": [f.model_dump(mode="json") for f in findings],
    }
    canonical = jsonlib.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Run(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_path: str
    # Absolute form of target_path, resolved at scan time (same process,
    # same cwd the scan itself ran in) -- `bee verify` anchors artifact
    # paths to this so it works when run from a different working
    # directory than the original scan used. Deliberately NOT part of
    # the evidence hash below: it's a machine-specific filesystem
    # location, not evidence about the artifact, and including it would
    # make identical content hash differently depending on where it
    # happens to be checked out.
    resolved_target_path: str = ""
    artifacts: list[Artifact]
    findings: list[Finding]
    summary: RunSummary
    scanner_version: str = __version__
    evidence_sha256: str = ""
    # Populated by `bee sign`, empty until then. signature is an Ed25519
    # signature (hex) over the evidence hash *at signing time*;
    # public_key (hex, raw 32 bytes) travels with it so verification
    # never needs the signer's key files, only the record itself. Unlike
    # evidence_sha256 alone, a valid signature cannot be reproduced by
    # someone who edits the record and recomputes the hash -- that
    # requires the private key, which is never stored here.
    signature: str = ""
    public_key: str = ""

    @classmethod
    def from_scan(
        cls,
        target_path: str,
        artifacts: list[Artifact],
        findings: list[Finding],
        deterministic: bool = False,
    ) -> "Run":
        summary = build_summary(artifacts, findings)
        evidence_sha256 = compute_evidence_hash(target_path, artifacts, findings, __version__)
        resolved_target_path = str(Path(target_path).resolve())
        if deterministic:
            return cls(
                id=_deterministic_run_id(target_path, artifacts),
                created_at=datetime.fromtimestamp(0, tz=timezone.utc),
                target_path=target_path,
                resolved_target_path=resolved_target_path,
                artifacts=artifacts,
                findings=findings,
                summary=summary,
                scanner_version=__version__,
                evidence_sha256=evidence_sha256,
            )
        return cls(
            target_path=target_path,
            resolved_target_path=resolved_target_path,
            artifacts=artifacts,
            findings=findings,
            summary=summary,
            scanner_version=__version__,
            evidence_sha256=evidence_sha256,
        )
