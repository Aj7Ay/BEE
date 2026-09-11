from __future__ import annotations

import json as jsonlib
from pathlib import Path

import typer

from bee.cli.state import OutputFormat
from bee.core.artifact import Artifact
from bee.evidence.mismatch import check_mismatch


def inspect_command(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, help="File to inspect."
    ),
) -> None:
    """Show a detailed identity and format report for a single artifact."""
    state = ctx.obj
    try:
        artifact = Artifact.from_file(path)
    except OSError as exc:
        typer.echo(f"Error: could not read {path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finding = check_mismatch(artifact)

    if state.output_format is OutputFormat.JSON:
        payload = {
            "artifact": artifact.model_dump(mode="json"),
            "findings": [finding.model_dump(mode="json")] if finding else [],
        }
        typer.echo(jsonlib.dumps(payload, indent=2))
        return

    typer.echo(f"Path:             {artifact.path}")
    typer.echo(f"Size:             {artifact.size} bytes")
    typer.echo(f"SHA-256:          {artifact.sha256}")
    typer.echo(f"SHA-512:          {artifact.sha512}")
    typer.echo(f"Declared format:  {artifact.declared_format}")
    typer.echo(f"Detected format:  {artifact.detected_format} ({artifact.format_confidence.value})")
    typer.echo(f"Magic bytes:      {artifact.magic_bytes_hex}")
    if finding is not None:
        typer.echo(f"Finding:          {finding.id} [{finding.severity.value}] {finding.title}")
    else:
        typer.echo("Finding:          none")
