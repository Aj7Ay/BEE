from __future__ import annotations

import hashlib
import json as jsonlib
import uuid
from collections import Counter
from datetime import datetime, timezone

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
    was passed; a stored run's evidence hash changing (recomputed later
    against its own stored artifacts/findings) means the record itself
    was altered after the fact, not that the underlying model changed --
    that's what re-hashing each artifact's current file content, done
    separately by `bee verify`, is for.
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
    artifacts: list[Artifact]
    findings: list[Finding]
    summary: RunSummary
    scanner_version: str = __version__
    evidence_sha256: str = ""

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
        if deterministic:
            return cls(
                id=_deterministic_run_id(target_path, artifacts),
                created_at=datetime.fromtimestamp(0, tz=timezone.utc),
                target_path=target_path,
                artifacts=artifacts,
                findings=findings,
                summary=summary,
                scanner_version=__version__,
                evidence_sha256=evidence_sha256,
            )
        return cls(
            target_path=target_path,
            artifacts=artifacts,
            findings=findings,
            summary=summary,
            scanner_version=__version__,
            evidence_sha256=evidence_sha256,
        )
