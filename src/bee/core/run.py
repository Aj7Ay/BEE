from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

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


class Run(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_path: str
    artifacts: list[Artifact]
    findings: list[Finding]
    summary: RunSummary

    @classmethod
    def from_scan(
        cls,
        target_path: str,
        artifacts: list[Artifact],
        findings: list[Finding],
        deterministic: bool = False,
    ) -> "Run":
        summary = build_summary(artifacts, findings)
        if deterministic:
            return cls(
                id=_deterministic_run_id(target_path, artifacts),
                created_at=datetime.fromtimestamp(0, tz=timezone.utc),
                target_path=target_path,
                artifacts=artifacts,
                findings=findings,
                summary=summary,
            )
        return cls(
            target_path=target_path,
            artifacts=artifacts,
            findings=findings,
            summary=summary,
        )
