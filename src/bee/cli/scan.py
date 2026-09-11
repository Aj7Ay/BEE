from __future__ import annotations

import sqlite3
from pathlib import Path

import typer

from bee.cli.state import OutputFormat
from bee.core.artifact import Artifact
from bee.core.run import Run
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.evidence.mismatch import check_mismatch
from bee.evidence.symlink import check_symlink_escape
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run
from bee.storage.db import save_run

# Severity, most severe first — matches declaration order in Severity itself,
# spelled out here so "meets or exceeds" (--fail-on) reads as "rank <= threshold".
_SEVERITY_ORDER = list(Severity)


def _iter_files(target: Path, workspace_dir: Path) -> list[Path]:
    # Exclude only the actual BEE workspace directory (by location, not by
    # name) — a real directory that happens to be named ".bee" elsewhere in
    # the scanned tree is not our state and must not be silently skipped.
    workspace_abs = workspace_dir.absolute()
    if target.is_file() or target.is_symlink():
        return [target]
    results = []
    for p in target.rglob("*"):
        if not (p.is_file() or p.is_symlink()):
            continue
        if p.absolute().is_relative_to(workspace_abs):
            continue
        results.append(p)
    return sorted(results)


def _parse_severity(value: str) -> Severity:
    try:
        return Severity(value.lower())
    except ValueError as exc:
        valid = ", ".join(s.value for s in Severity)
        raise typer.BadParameter(f"must be one of: {valid}") from exc


def scan_command(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to scan."),
    fail_on: str | None = typer.Option(
        None,
        "--fail-on",
        help="Exit non-zero if any finding's severity meets or exceeds this "
        "level (critical, high, medium, low, info).",
    ),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use a stable run id and timestamp so identical input produces "
        "byte-identical output, for diffing against a baseline.",
    ),
) -> None:
    """Scan a local file or directory and report artifact identity, format, and findings."""
    state = ctx.obj
    threshold = _parse_severity(fail_on) if fail_on is not None else None
    scan_root = path if path.is_dir() else path.parent
    files = _iter_files(path, workspace_dir=state.db_path.parent)

    artifacts: list[Artifact] = []
    findings: list[Finding] = []
    for file_path in files:
        try:
            artifact = Artifact.from_file(file_path)
        except OSError as exc:
            # A file we can't read (permissions, race with deletion, a
            # dangling special file, ...) must not kill the rest of the
            # scan. Record it as a finding and keep going.
            findings.append(
                Finding(
                    id="BEE-IO-001",
                    severity=Severity.HIGH,
                    title="Artifact could not be read",
                    description=f"Reading this file failed: {exc}",
                    artifact_path=str(file_path),
                    evidence=[
                        Evidence(
                            type="os_error",
                            value=str(exc),
                            source="local_filesystem",
                            confidence=Confidence.VERIFIED,
                        ),
                    ],
                )
            )
            continue
        artifacts.append(artifact)
        finding = check_mismatch(artifact)
        if finding is not None:
            findings.append(finding)
        symlink_finding = check_symlink_escape(artifact, scan_root)
        if symlink_finding is not None:
            findings.append(symlink_finding)

    run = Run.from_scan(
        target_path=str(path), artifacts=artifacts, findings=findings, deterministic=deterministic
    )

    # Report first, persist second: a vetting result the operator can see
    # matters more than a row in the history database. An unwritable DB
    # (read-only filesystem, permissions, a .bee/bee.db owned by another
    # user, ...) must not throw away a scan that already completed.
    if state.output_format is OutputFormat.JSON:
        typer.echo(render_run_json(run))
    else:
        render_run(run)

    try:
        save_run(state.db_path, run)
    except (sqlite3.Error, OSError) as exc:
        typer.echo(f"Warning: could not save run to {state.db_path}: {exc}", err=True)

    if threshold is not None:
        threshold_rank = _SEVERITY_ORDER.index(threshold)
        if any(_SEVERITY_ORDER.index(f.severity) <= threshold_rank for f in run.findings):
            raise typer.Exit(code=1)
