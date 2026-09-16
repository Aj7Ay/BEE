from __future__ import annotations

import typer

from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run
from bee.storage.db import load_run


def show_command(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run id, as printed by `bee history`."),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
) -> None:
    """Re-display a previously stored run by id."""
    state = ctx.obj
    threshold = parse_severity_option(fail_on) if fail_on is not None else None
    run = load_run(state.db_path, run_id)
    if run is None:
        typer.echo(f"No run found with id {run_id!r}", err=True)
        raise typer.Exit(code=1)
    if state.output_format is OutputFormat.JSON:
        typer.echo(render_run_json(run))
    else:
        render_run(run)
    exit_if_threshold_met(run.findings, threshold)
