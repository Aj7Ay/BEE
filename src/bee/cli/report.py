from __future__ import annotations

import re
from pathlib import Path

import typer

from bee.cli.vet import run as vet_run
from bee.core.run import Run
from bee.reports.html import HTMLReport


def report_command(
    path: str = typer.Argument(..., help="Scan run file, run ID from bee history, or target to report on."),
    output: Path = typer.Option("bee-report.html", "--output", "-o", help="Output HTML file."),
    scan: bool = typer.Option(False, "--scan", help="Re-scan the target before generating report."),
) -> None:
    """Generate an HTML report from a BEE scan run."""
    # Check if path matches UUID pattern (36 characters: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, path.lower()):
        typer.echo(f"Loading run from history: {path}...", err=False)
        # TODO: Load from database using run_id = path
        # For now, treat as invalid since db loading not implemented in this PR
        typer.echo(f"Run ID lookup not yet implemented. Use scan run JSON file or local path.", err=True)
        raise typer.Exit(code=1)

    target = Path(path)

    if scan:
        typer.echo(f"Scanning {target}...")
        run_obj = vet_run(target)
        if run_obj is None:
            typer.echo("Scan failed.", err=True)
            raise typer.Exit(code=1)
    elif target.is_file():
        # Load from JSON file
        import json
        data = json.loads(target.read_text())
        from bee.core.run import Run
        run_obj = Run(**data)
    else:
        # Path doesn't exist
        typer.echo(f"Error: path does not exist: {target}", err=True)
        typer.echo("Use --scan to scan a target, or provide an existing run JSON file.", err=True)
        raise typer.Exit(code=1)

    reporter = HTMLReport(run_obj)
    reporter.generate(output)
    typer.echo(f"Report saved to: {output}")
