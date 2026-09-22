"""Unit tests for policy evaluator verdict logic."""

from dataclasses import dataclass
from bee.evidence.finding import Finding, Severity, Evidence, Confidence
from bee.policy.rules import Policy, FindingPolicy, IntegrityPolicy, FormatPolicy, VulnerabilityPolicy, Action
from bee.policy.evaluator import PolicyEvaluator


@dataclass
class MockArtifact:
    """Mock artifact for testing."""
    detected_format: str
    path: str


def test_evaluator_blocks_on_critical_finding():
    """Policy should block when critical finding present."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )
    findings = [Finding(
        id="BEE-TEST-001",
        severity=Severity.CRITICAL,
        title="Critical issue",
        description="Test",
        artifact_path="test.bin",
        evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
    )]

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, findings, None, None, [])

    assert len(violations) > 0, "Should have violations"
    assert any(v.action == "block" for v in violations), "Should block on critical"


def test_evaluator_allows_low_findings_with_allow_policy():
    """Policy should allow when only low findings present."""
    policy = Policy(
        findings=FindingPolicy(
            critical=Action.BLOCK,
            high=Action.BLOCK,
            medium=Action.REVIEW,
            low=Action.ALLOW
        ),
        integrity=IntegrityPolicy(require_sha256=False)
    )
    findings = [Finding(
        id="BEE-TEST-001",
        severity=Severity.LOW,
        title="Low issue",
        description="Test",
        artifact_path="test.bin",
        evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
    )]

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, findings, None, None, [])

    assert len(violations) == 0, "Low severity with allow policy should not violate"


def test_evaluator_reviews_on_medium_finding():
    """Policy should review when medium finding present."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW),
        integrity=IntegrityPolicy(require_sha256=False)
    )
    findings = [Finding(
        id="BEE-TEST-001",
        severity=Severity.MEDIUM,
        title="Medium issue",
        description="Test",
        artifact_path="test.bin",
        evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
    )]

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, findings, None, None, [])

    assert len(violations) > 0, "Should have violations"
    assert any(v.action == "review" for v in violations), "Should review on medium"


def test_evaluator_allows_clean_scan():
    """Policy should allow when no findings present."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, [])

    assert len(violations) == 0, "Clean scan should have no violations"


def test_evaluator_multiple_findings_takes_worst():
    """When multiple findings, worst action takes precedence."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.REVIEW, low=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )
    findings = [
        Finding(
            id="BEE-LOW-001",
            severity=Severity.LOW,
            title="Low",
            description="Test",
            artifact_path="test.bin",
            evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
        ),
        Finding(
            id="BEE-CRITICAL-001",
            severity=Severity.CRITICAL,
            title="Critical",
            description="Test",
            artifact_path="test.bin",
            evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
        ),
    ]

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, findings, None, None, [])

    assert len(violations) > 0, "Should have violations"
    assert any(v.action == "block" for v in violations), "Critical should cause block"


def test_evaluator_empty_policy_defaults_to_fail_closed():
    """Empty policy should still enforce fail-closed on critical."""
    policy = Policy(
        integrity=IntegrityPolicy(require_sha256=False)
    )
    findings = [Finding(
        id="BEE-CRITICAL-001",
        severity=Severity.CRITICAL,
        title="Critical",
        description="Test",
        artifact_path="test.bin",
        evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
    )]

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, findings, None, None, [])

    assert len(violations) > 0, "Default policy should block critical"


def test_evaluator_all_severity_levels():
    """Test all severity levels produce expected actions."""
    policy = Policy(
        findings=FindingPolicy(
            critical=Action.BLOCK,
            high=Action.BLOCK,
            medium=Action.REVIEW,
            low=Action.ALLOW,
            info=Action.ALLOW,
        ),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    severities = [
        (Severity.CRITICAL, "block"),
        (Severity.HIGH, "block"),
        (Severity.MEDIUM, "review"),
        (Severity.LOW, "allow"),
        (Severity.INFO, "allow"),
    ]

    for severity, expected_action in severities:
        findings = [Finding(
            id=f"BEE-{severity.value.upper()}-001",
            severity=severity,
            title=f"{severity.value} issue",
            description="Test",
            artifact_path="test.bin",
            evidence=[Evidence(type="test", value="test", source="test", confidence=Confidence.VERIFIED)],
        )]

        evaluator = PolicyEvaluator()
        _, violations = evaluator.evaluate(policy, findings, None, None, [])

        if expected_action == "allow":
            assert len(violations) == 0, f"{severity.value} with allow should not violate"
        else:
            assert len(violations) > 0, f"{severity.value} should violate"
            assert any(v.action == expected_action for v in violations), f"{severity.value} should {expected_action}"


def test_issue_4_formats_blocked_checks_detected_format():
    """Issue #4: formats.blocked should check detected_format against the list."""
    policy = Policy(
        formats=FormatPolicy(blocked=["pickle"]),
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW, low=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    # Finding with safetensors detected (not in blocked list)
    finding_safe = Finding(
        id="BEE-FMT-001",
        severity=Severity.CRITICAL,
        title="Format mismatch",
        description="Test",
        artifact_path="test.pt",
        evidence=[
            Evidence(type="file_extension_format", value="pytorch", source="test", confidence=Confidence.VERIFIED),
            Evidence(type="detected_format", value="safetensors", source="test", confidence=Confidence.VERIFIED),
        ],
    )

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [finding_safe], None, None, [])

    # Should not block because safetensors is not in the blocked list
    assert not any(v.policy_rule == "formats.blocked" for v in violations), "Safetensors should not be blocked"

    # Finding with pickle detected (in blocked list) at CRITICAL severity
    finding_pickle = Finding(
        id="BEE-FMT-001",
        severity=Severity.CRITICAL,
        title="Format mismatch",
        description="Test",
        artifact_path="test.safetensors",
        evidence=[
            Evidence(type="file_extension_format", value="safetensors", source="test", confidence=Confidence.VERIFIED),
            Evidence(type="detected_format", value="pickle", source="test", confidence=Confidence.VERIFIED),
        ],
    )

    _, violations = evaluator.evaluate(policy, [finding_pickle], None, None, [])

    # Should block because pickle is in the blocked list and critical severity triggers block
    assert any(v.policy_rule == "formats.blocked" and v.action == Action.BLOCK for v in violations), "Pickle should be blocked"

    # Finding with pickle at LOW severity should be allowed (respects findings policy)
    finding_pickle_low = Finding(
        id="BEE-FMT-001",
        severity=Severity.LOW,
        title="Format mismatch",
        description="Test",
        artifact_path="test.safetensors",
        evidence=[
            Evidence(type="file_extension_format", value="safetensors", source="test", confidence=Confidence.VERIFIED),
            Evidence(type="detected_format", value="pickle", source="test", confidence=Confidence.VERIFIED),
        ],
    )

    _, violations = evaluator.evaluate(policy, [finding_pickle_low], None, None, [])

    # Should be allowed because LOW severity allows this format (respects findings policy)
    assert not any(v.policy_rule == "formats.blocked" for v in violations), "Low severity pickle should be allowed per findings policy"


def test_issue_5_vulnerabilities_unknown_severity():
    """Issue #5: Vulnerabilities with unknown severity should be handled."""
    policy = Policy(
        vulnerabilities=VulnerabilityPolicy(unknown=Action.REVIEW),
        integrity=IntegrityPolicy(require_sha256=False)
    )
    
    # Vulnerability with unknown severity
    vulns = [{"severity": "unknown"}]
    
    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, vulns)
    
    # Should review because unknown severity defaults to review
    assert any(v.policy_rule == "vulnerabilities.unknown" and v.action == Action.REVIEW for v in violations), "Unknown severity should review"
    
    # Missing severity should also map to unknown
    vulns_missing = [{"name": "test"}]  # No severity field
    
    _, violations = evaluator.evaluate(policy, [], None, None, vulns_missing)
    
    # Should review because missing severity maps to unknown
    assert any(v.policy_rule == "vulnerabilities.unknown" and v.action == Action.REVIEW for v in violations), "Missing severity should review"


def test_issue_6_integrity_require_sha256_fail_closed():
    """Issue #6: integrity.require_sha256 should fail closed when provenance absent."""
    policy = Policy(
        integrity=IntegrityPolicy(require_sha256=True)
    )
    
    evaluator = PolicyEvaluator()
    
    # Missing provenance should block when sha256 is required
    _, violations = evaluator.evaluate(policy, [], None, None, [])
    
    assert any(v.policy_rule == "integrity.require_sha256" and v.action == Action.BLOCK for v in violations), "Missing provenance should block when sha256 required"


def test_problem_a_correctly_labeled_blocked_format():
    """Problem A: formats.blocked should catch correctly-labeled blocked formats (no BEE-FMT-001).

    A pickle file with correct extension (pickle.pkl) produces no BEE-FMT-001 finding,
    but detected_format is still "pickle". The artifact check should catch this.
    """
    policy = Policy(
        formats=FormatPolicy(blocked=["pickle"]),
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW, low=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    # No BEE-FMT-001 finding (correctly labeled), but artifact has detected_format="pickle"
    artifact = MockArtifact(detected_format="pickle", path="weights.pkl")

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, [], artifacts=[artifact])

    # Should block due to detected_format being in blocked list
    assert any(v.policy_rule == "formats.blocked" and v.action == Action.BLOCK for v in violations), \
        "Correctly-labeled pickle should be blocked via artifact detection"


def test_problem_b_blocked_format_not_overridden_by_findings_policy():
    """Problem B: formats.blocked should not be overridden by findings_policy (e.g., low: allow).

    Even if findings.low: allow is set, blocked formats should always block.
    """
    policy = Policy(
        formats=FormatPolicy(blocked=["pickle"]),
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW, low=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    # Correctly-labeled pickle artifact
    artifact = MockArtifact(detected_format="pickle", path="weights.pkl")

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, [], artifacts=[artifact])

    # Should block despite findings.low: allow, because blocked format is a standalone decision
    assert any(v.policy_rule == "formats.blocked" and v.action == Action.BLOCK for v in violations), \
        "Blocked format should not be overridden by findings policy"


def test_non_blocked_formats_still_allowed():
    """Non-blocked formats should not trigger violations."""
    policy = Policy(
        formats=FormatPolicy(blocked=["pickle"]),
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW, low=Action.ALLOW),
        integrity=IntegrityPolicy(require_sha256=False)
    )

    # Safetensors is not in blocked list
    artifact = MockArtifact(detected_format="safetensors", path="weights.safetensors")

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, [], artifacts=[artifact])

    # Should not block
    assert not any(v.policy_rule == "formats.blocked" for v in violations), \
        "Non-blocked format should be allowed"


