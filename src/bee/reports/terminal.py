from __future__ import annotations

from rich.console import Console
from rich.table import Table

from bee.core.run import Run

console = Console()


def render_run(run: Run) -> None:
    console.print("[bold]BEE SCAN[/bold]")
    console.print(f"Target: {run.target_path}")
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
            artifact.path,
            artifact.detected_format,
            str(artifact.size),
            str(finding_count) if finding_count else "-",
        )
    console.print(table)

    severity_parts = [
        f"{count} {severity.value}" for severity, count in run.summary.findings_by_severity.items()
    ]
    console.print("Findings: " + ", ".join(severity_parts))
