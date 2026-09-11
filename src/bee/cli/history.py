from __future__ import annotations

import typer

from bee.storage.db import list_runs


def history_command(ctx: typer.Context) -> None:
    """List past scan runs stored in the BEE workspace database."""
    state = ctx.obj
    runs = list_runs(state.db_path)
    if not runs:
        typer.echo("No runs recorded yet.")
        return
    for run in runs:
        counts = ", ".join(
            f"{count} {severity.value}"
            for severity, count in run.summary.findings_by_severity.items()
            if count
        )
        counts = counts or "no findings"
        typer.echo(
            f"{run.id}  {run.created_at.isoformat()}  {run.target_path}  "
            f"({run.summary.artifact_count} artifacts, {counts})"
        )
