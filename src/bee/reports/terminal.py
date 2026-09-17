from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from bee.cli.sanitize import sanitize_for_terminal
from bee.core.run import Run
from bee.evidence.finding import Severity

console = Console()


def _safe_text(value: str) -> Text:
    # Text() never interprets its content as Rich markup ("[bold red]...")
    # and sanitize_for_terminal() neutralizes raw control bytes (ESC and
    # friends) -- together closing both the markup-injection and the
    # ANSI-escape-injection route a filename in a scanned directory could
    # otherwise use to recolor, relocate, or overwrite real output on the
    # operator's terminal.
    return Text(sanitize_for_terminal(value))


def render_run(run: Run) -> None:
    console.print("[bold]BEE SCAN[/bold]")
    console.print(Text("Target: ") + _safe_text(run.target_path))
    console.print(f"Artifacts scanned: {len(run.artifacts)}")

    findings_by_path: dict[str, int] = {}
    for finding in run.findings:
        findings_by_path[finding.artifact_path] = findings_by_path.get(finding.artifact_path, 0) + 1

    table = Table()
    table.add_column("PATH")
    table.add_column("FORMAT")
    table.add_column("SIZE")
    table.add_column("FINDINGS")
    for artifact in run.artifacts:
        finding_count = findings_by_path.get(artifact.path, 0)
        table.add_row(
            _safe_text(artifact.path),
            artifact.detected_format,
            str(artifact.size),
            str(finding_count) if finding_count else "-",
        )
    console.print(table)

    severity_parts = [
        f"{count} {severity.value}" for severity, count in run.summary.findings_by_severity.items()
    ]
    console.print("Findings: " + ", ".join(severity_parts))

    # Critical/high findings are surfaced explicitly, not just folded into
    # the aggregate count above — a finding whose artifact isn't in the
    # table (e.g. an unreadable file) would otherwise be invisible.
    urgent = [f for f in run.findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
    if urgent:
        console.print()
        console.print("[bold red]Critical/High findings:[/bold red]")
        for finding in urgent:
            line = Text(f"  {finding.id} [{finding.severity.value}] ")
            line += _safe_text(finding.artifact_path)
            line += Text(f": {finding.title}")
            console.print(line)
