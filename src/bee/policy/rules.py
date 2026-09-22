from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Action(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class FindingPolicy(BaseModel):
    """How to handle findings by severity level."""
    critical: Action = Action.BLOCK
    high: Action = Action.BLOCK
    medium: Action = Action.REVIEW
    low: Action = Action.ALLOW
    info: Action = Action.ALLOW


class FormatPolicy(BaseModel):
    """Format restrictions."""
    blocked: list[str] = Field(default_factory=lambda: ["pickle"])


class CodePolicy(BaseModel):
    """Custom code policy."""
    allowed: bool = True


class LicensePolicy(BaseModel):
    """License restrictions."""
    allowed: list[str] = []  # empty = any license allowed


class VulnerabilityPolicy(BaseModel):
    """Vulnerability severity thresholds."""
    critical: Action = Action.BLOCK
    high: Action = Action.BLOCK
    medium: Action = Action.ALLOW
    low: Action = Action.ALLOW


class IntegrityPolicy(BaseModel):
    """Integrity requirements."""
    require_sha256: bool = True


class ProvenancePolicy(BaseModel):
    """Provenance requirements."""
    require_publisher: bool = False
    require_repository: bool = False
    require_revision: bool = False


class SignaturePolicy(BaseModel):
    """Signature requirements."""
    required: bool = False


class Policy(BaseModel):
    """Complete BEE security policy."""
    name: str = "unnamed-policy"
    schema_version: str = "1.0"

    integrity: IntegrityPolicy = Field(default_factory=IntegrityPolicy)
    provenance: ProvenancePolicy = Field(default_factory=ProvenancePolicy)
    signature: SignaturePolicy = Field(default_factory=SignaturePolicy)
    formats: FormatPolicy = Field(default_factory=FormatPolicy)
    findings: FindingPolicy = Field(default_factory=FindingPolicy)
    custom_code: CodePolicy = Field(default_factory=CodePolicy)
    licenses: LicensePolicy = Field(default_factory=LicensePolicy)
    vulnerabilities: VulnerabilityPolicy = Field(default_factory=VulnerabilityPolicy)


def validate_policy_yaml(yaml_text: str) -> dict:
    """Validate a YAML policy string without loading it fully.

    Returns the parsed dict if valid, raises ValueError if not.
    Only top-level keys are validated; sub-structure validation happens
    in the Policy model.
    """
    import yaml

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}")

    if not isinstance(data, dict):
        raise ValueError("Policy must be a YAML mapping at the top level")

    # Optional top-level "policy" wrapper key
    if "policy" in data:
        data = data["policy"]

    valid_keys = {
        "name", "schema_version", "integrity", "provenance", "signature",
        "formats", "findings", "custom_code", "licenses", "vulnerabilities",
    }
    for key in data:
        if key not in valid_keys:
            raise ValueError(f"Unknown policy key: {key!r}. Valid keys: {valid_keys}")

    return data


def load_policy_from_dict(data: dict) -> Policy:
    """Parse a dict into a Policy model, filling defaults for missing fields."""
    if "policy" in data:
        data = data["policy"]

    return Policy(**data)
