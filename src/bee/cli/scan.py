from __future__ import annotations

from pathlib import Path

import typer

from bee.cli.state import OutputFormat
from bee.core.artifact import Artifact
from bee.core.run import Run
from bee.evidence.finding import Finding
from bee.evidence.mismatch import check_mismatch
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run
from bee.storage.db import save_run


def _iter_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        p for p in target.rglob("*") if p.is_file() and ".bee" not in p.parts
    )


def scan_command(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to scan."),
) -> None:
    """Scan a local file or directory and report artifact identity, format, and findings."""
    state = ctx.obj
    files = _iter_files(path)

    artifacts: list[Artifact] = []
    findings: list[Finding] = []
    for file_path in files:
        artifact = Artifact.from_file(file_path)
        artifacts.append(artifact)
        finding = check_mismatch(artifact)
        if finding is not None:
            findings.append(finding)

    run = Run.from_scan(target_path=str(path), artifacts=artifacts, findings=findings)
    save_run(state.db_path, run)

    if state.output_format is OutputFormat.JSON:
        typer.echo(render_run_json(run))
    else:
        render_run(run)
