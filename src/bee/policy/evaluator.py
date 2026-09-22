from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from bee.core.run import Provenance
from bee.evidence.finding import Finding, Severity
from bee.policy.rules import (
    Policy, Action,
    FindingPolicy, FormatPolicy, CodePolicy, LicensePolicy, VulnerabilityPolicy,
)


class Decision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class PolicyViolation(BaseModel):
    policy_rule: str
    description: str
    action: Action


class PolicyEvaluator:
    """Evaluates findings against a policy and produces a decision."""

    _SEVERITY_RANK = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
        Severity.INFO: 4,
    }

    _ACTION_TO_RANK = {
        Action.BLOCK: 1,
        Action.REVIEW: 2,
        Action.ALLOW: 4,
    }

    def evaluate(
        self,
        policy: Policy,
        findings: list[Finding],
        provenance: Provenance | None,
        license_info: dict | None = None,
        vulnerabilities: list[dict] | None = None,
    ) -> tuple[Decision, list[PolicyViolation]]:
        """Evaluate findings against policy. Returns (decision, violations)."""
        violations: list[PolicyViolation] = []

        violations.extend(self._check_formats(findings, policy.formats))
        violations.extend(self._check_findings(findings, policy.findings))
        violations.extend(self._check_code(findings, policy.custom_code))
        violations.extend(self._check_provenance(provenance, policy))
        violations.extend(self._check_licenses(license_info, policy.licenses))
        violations.extend(self._check_vulnerabilities(vulnerabilities, policy.vulnerabilities))

        decision = self._determine_decision(violations)

        return decision, violations

    def _check_formats(
        self, findings: list[Finding], fmt_policy: FormatPolicy
    ) -> list[PolicyViolation]:
        blocked_findings = [f for f in findings if f.id == "BEE-FMT-001"]
        if blocked_findings:
            return [PolicyViolation(
                policy_rule="formats.blocked",
                description="Format mismatch detected — file may be mislabeled or tampered",
                action=Action.BLOCK,
            )]
        return []

    def _check_findings(
        self, findings: list[Finding], policy: FindingPolicy
    ) -> list[PolicyViolation]:
        violations = []

        for severity, action in [
            (Severity.CRITICAL, policy.critical),
            (Severity.HIGH, policy.high),
            (Severity.MEDIUM, policy.medium),
            (Severity.LOW, policy.low),
            (Severity.INFO, policy.info),
        ]:
            severe_findings = [f for f in findings if f.severity == severity]
            if severe_findings and action == Action.BLOCK:
                violations.append(PolicyViolation(
                    policy_rule=f"findings.{severity.value}",
                    description=f"{len(severe_findings)} finding(s) at {severity.value} severity — policy requires block",
                    action=Action.BLOCK,
                ))
            elif severe_findings and action == Action.REVIEW:
                violations.append(PolicyViolation(
                    policy_rule=f"findings.{severity.value}",
                    description=f"{len(severe_findings)} finding(s) at {severity.value} severity — review recommended",
                    action=Action.REVIEW,
                ))

        return violations

    def _check_code(
        self, findings: list[Finding], code_policy: CodePolicy
    ) -> list[PolicyViolation]:
        if code_policy.allowed:
            return []

        code_findings = [f for f in findings if f.category == "code"]
        if code_findings:
            return [PolicyViolation(
                policy_rule="custom_code.allowed",
                description=f"{len(code_findings)} code finding(s) detected — policy disallows custom code",
                action=Action.BLOCK,
            )]
        return []

    def _check_provenance(
        self, provenance: Provenance | None, policy: Policy
    ) -> list[PolicyViolation]:
        violations = []
        pp = policy.provenance

        if pp.require_publisher and (not provenance or not provenance.publisher.name):
            violations.append(PolicyViolation(
                policy_rule="provenance.require_publisher",
                description="Publisher information required but not available",
                action=Action.REVIEW,
            ))

        if pp.require_repository and (not provenance or not provenance.source.repository):
            violations.append(PolicyViolation(
                policy_rule="provenance.require_repository",
                description="Repository information required but not available",
                action=Action.REVIEW,
            ))

        if pp.require_revision and (not provenance or not provenance.source.revision):
            violations.append(PolicyViolation(
                policy_rule="provenance.require_revision",
                description="Revision information required but not available",
                action=Action.REVIEW,
            ))

        return violations

    def _check_licenses(
        self, license_info: dict | None, lic_policy: LicensePolicy
    ) -> list[PolicyViolation]:
        if not lic_policy.allowed:
            return []

        if not license_info:
            return []

        licenses = license_info
        if isinstance(license_info, dict) and "spdx" in license_info:
            licenses = [license_info]
        elif isinstance(license_info, list):
            licenses = license_info

        for lic in licenses:
            spdx = lic.get("spdx", "") if isinstance(lic, dict) else str(lic)
            if spdx and spdx not in lic_policy.allowed:
                return [PolicyViolation(
                    policy_rule="licenses.allowed",
                    description=f"License {spdx!r} not in allowed list",
                    action=Action.BLOCK,
                )]

        return []

    def _check_vulnerabilities(
        self, vulnerabilities: list[dict] | None, vuln_policy: VulnerabilityPolicy
    ) -> list[PolicyViolation]:
        if not vulnerabilities:
            return []

        severity_map = {
            "critical": vuln_policy.critical,
            "high": vuln_policy.high,
            "medium": vuln_policy.medium,
            "low": vuln_policy.low,
        }

        for severity, action in severity_map.items():
            sev_vulns = [v for v in vulnerabilities if v.get("severity", "").lower() == severity]
            if sev_vulns and action == Action.BLOCK:
                return [PolicyViolation(
                    policy_rule=f"vulnerabilities.{severity}",
                    description=f"{len(sev_vulns)} vulnerability(ies) at {severity} severity — policy requires block",
                    action=Action.BLOCK,
                )]
            elif sev_vulns and action == Action.REVIEW:
                return [PolicyViolation(
                    policy_rule=f"vulnerabilities.{severity}",
                    description=f"{len(sev_vulns)} vulnerability(ies) at {severity} severity — review recommended",
                    action=Action.REVIEW,
                )]

        return []

    def _determine_decision(self, violations: list[PolicyViolation]) -> Decision:
        if any(v.action == Action.BLOCK for v in violations):
            return Decision.BLOCK
        if any(v.action == Action.REVIEW for v in violations):
            return Decision.REVIEW
        return Decision.ALLOW
