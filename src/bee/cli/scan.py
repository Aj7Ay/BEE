from __future__ import annotations

import sqlite3
from pathlib import Path

import typer

from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run
from bee.storage.db import save_run
from bee.scanning.config import ScanConfig
from bee.scanning.orchestrator import ScanOrchestrator


def scan_command(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to scan."),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use a stable run id and timestamp so identical input produces byte-identical output.",
    ),
    follow_symlinks: bool = typer.Option(
        False,
        "--follow-symlinks",
        help="Read (and hash) a symlink's target even when it resolves outside the scan root.",
    ),
) -> None:
    """Scan a local file or directory for security issues."""
    state = ctx.obj
    threshold = parse_severity_option(fail_on) if fail_on is not None else None

    config = ScanConfig(
        fail_on=fail_on,
        deterministic=deterministic,
        follow_symlinks=follow_symlinks,
        output_dir=None,
    )

    orchestrator = ScanOrchestrator(config)
    run = orchestrator.scan_local(path, workspace_dir=state.db_path.parent)

    if state.output_format is OutputFormat.JSON:
        import json as jsonlib
        output_dict = run.model_dump(mode="json")
        # Add verdict for consistency with vet
        if run.decision:
            verdict = run.decision
        else:
            from bee.evidence.finding import Severity
            if run.severity_count(Severity.CRITICAL) > 0 or run.severity_count(Severity.HIGH) > 0:
                verdict = "block"
            elif run.severity_count(Severity.MEDIUM) > 0:
                verdict = "review"
            else:
                verdict = "allow"
        output_dict["verdict"] = verdict
        typer.echo(jsonlib.dumps(output_dict, indent=2))
    else:
        render_run(run)

    try:
        save_run(state.db_path, run)
    except (sqlite3.Error, OSError) as exc:
        typer.echo(f"Warning: could not save run to {state.db_path}: {exc}", err=True)

    # Fail closed: exit nonzero if critical/high findings present (unless threshold explicitly set lower)
    if threshold is None:
        from bee.evidence.finding import Severity
        if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in run.findings):
            raise typer.Exit(code=1)
    else:
        exit_if_threshold_met(run.findings, threshold)
