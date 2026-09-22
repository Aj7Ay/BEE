from __future__ import annotations

from pathlib import Path

import typer

from bee.cli.vet import run as vet_run
from bee.core.run import Run
from bee.reports.html import HTMLReport


def report_command(
    path: str = typer.Argument(..., help="BEE scan run file or target to report on."),
    output: Path = typer.Option("bee-report.html", "--output", "-o", help="Output HTML file."),
    scan: bool = typer.Option(False, "--scan", help="Re-scan the target before generating report."),
) -> None:
    """Generate an HTML report from a BEE scan run."""
    target = Path(path)

    if scan or not target.is_file():
        typer.echo(f"Scanning {target}...")
        run_obj = vet_run(target)
        if run_obj is None:
            typer.echo("Scan failed.", err=True)
            raise typer.Exit(code=1)
    else:
        # Load from JSON file
        import json
        data = json.loads(target.read_text())
        from bee.core.run import Run
        run_obj = Run(**data)

    reporter = HTMLReport(run_obj)
    reporter.generate(output)
    typer.echo(f"Report saved to: {output}")
