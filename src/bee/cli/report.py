from __future__ import annotations

import re
from pathlib import Path

import typer

from bee.cli.vet import run as vet_run
from bee.core.run import Run
from bee.reports.html import HTMLReport
from bee.storage.db import load_run


def report_command(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Scan run file, run ID from bee history, or target to report on."),
    output: Path = typer.Option("bee-report.html", "--output", "-o", help="Output HTML file."),
    scan: bool = typer.Option(False, "--scan", help="Re-scan the target before generating report."),
) -> None:
    """Generate an HTML report from a BEE scan run."""
    state = ctx.obj

    # Check if path matches UUID pattern (36 characters: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, path.lower()):
        typer.echo(f"Loading run from history: {path}...", err=False)
        run_obj = load_run(state.db_path, path)
        if not run_obj:
            typer.echo(f"Error: run not found: {path}", err=True)
            raise typer.Exit(code=1)
    else:
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
        elif target.is_dir():
            # Scan the directory
            typer.echo(f"Scanning {target}...")
            run_obj = vet_run(target)
            if run_obj is None:
                typer.echo("Scan failed.", err=True)
                raise typer.Exit(code=1)
        elif not target.exists():
            # Path doesn't exist
            typer.echo(f"Error: path does not exist: {target}", err=True)
            typer.echo("Use --scan to scan a target, or provide an existing run JSON file.", err=True)
            raise typer.Exit(code=1)

    reporter = HTMLReport(run_obj)
    reporter.generate(output)
    typer.echo(f"Report saved to: {output}")
