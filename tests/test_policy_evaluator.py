"""Unit tests for policy evaluator verdict logic."""

from bee.evidence.finding import Finding, Severity, Evidence, Confidence
from bee.policy.rules import Policy, FindingPolicy, Action
from bee.policy.evaluator import PolicyEvaluator


def test_evaluator_blocks_on_critical_finding():
    """Policy should block when critical finding present."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.ALLOW)
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
        )
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
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW)
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
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.BLOCK, medium=Action.REVIEW)
    )

    evaluator = PolicyEvaluator()
    _, violations = evaluator.evaluate(policy, [], None, None, [])

    assert len(violations) == 0, "Clean scan should have no violations"


def test_evaluator_multiple_findings_takes_worst():
    """When multiple findings, worst action takes precedence."""
    policy = Policy(
        findings=FindingPolicy(critical=Action.BLOCK, high=Action.REVIEW, low=Action.ALLOW)
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
    policy = Policy()
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
        )
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
