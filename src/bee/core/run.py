from __future__ import annotations

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


class Run(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_path: str
    artifacts: list[Artifact]
    findings: list[Finding]
    summary: RunSummary

    @classmethod
    def from_scan(cls, target_path: str, artifacts: list[Artifact], findings: list[Finding]) -> "Run":
        return cls(
            target_path=target_path,
            artifacts=artifacts,
            findings=findings,
            summary=build_summary(artifacts, findings),
        )
