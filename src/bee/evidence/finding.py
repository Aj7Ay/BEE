from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Confidence(str, Enum):
    VERIFIED = "verified"
    SUPPORTED = "supported"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Evidence(BaseModel):
    type: str
    value: str
    source: str
    confidence: Confidence


class Finding(BaseModel):
    id: str
    severity: Severity
    title: str
    description: str
    artifact_path: str
    evidence: list[Evidence]
    category: str = "format"
    line_number: int | None = None
    code_pattern: str = ""
    recommendation: str = ""
