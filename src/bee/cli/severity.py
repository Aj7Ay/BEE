from __future__ import annotations

import typer

from bee.evidence.finding import Finding, Severity

# Most severe first — matches declaration order in Severity itself, spelled
# out here so "meets or exceeds" reads as "rank <= threshold".
SEVERITY_ORDER = list(Severity)

FAIL_ON_HELP = (
    "Exit non-zero if any finding's severity meets or exceeds this "
    "level (critical, high, medium, low, info)."
)


def parse_severity_option(value: str) -> Severity:
    try:
        return Severity(value.lower())
    except ValueError as exc:
        valid = ", ".join(s.value for s in Severity)
        raise typer.BadParameter(f"must be one of: {valid}") from exc


def exit_if_threshold_met(findings: list[Finding], threshold: Severity | None) -> None:
    """Shared by every command that supports --fail-on: raise typer.Exit(1)
    if any finding's severity meets or exceeds `threshold`. One
    implementation, so a command added later (verify, diff, ...) reuses
    the gate instead of reintroducing the gap of enforcing on one command
    but not another."""
    if threshold is None:
        return
    threshold_rank = SEVERITY_ORDER.index(threshold)
    if any(SEVERITY_ORDER.index(f.severity) <= threshold_rank for f in findings):
        raise typer.Exit(code=1)
