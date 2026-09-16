from __future__ import annotations

import sqlite3
from pathlib import Path

import typer

from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.core.artifact import Artifact
from bee.core.run import Run
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.evidence.mismatch import check_mismatch
from bee.evidence.pickle_calls import check_pickle_calls
from bee.evidence.symlink import build_symlink_escape_finding, is_escaping_symlink
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run
from bee.storage.db import save_run


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


def scan_command(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to scan."),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
    deterministic: bool = typer.Option(
        False,
        "--deterministic",
        help="Use a stable run id and timestamp so identical input produces "
        "byte-identical output, for diffing against a baseline.",
    ),
    follow_symlinks: bool = typer.Option(
        False,
        "--follow-symlinks",
        help="Read (and hash) a symlink's target even when it resolves "
        "outside the scan root. BEE-SYM-001 is still recorded either way. "
        "Off by default: a symlink escaping the scan root is not opened at all.",
    ),
) -> None:
    """Scan a local file or directory and report artifact identity, format, and findings."""
    state = ctx.obj
    threshold = parse_severity_option(fail_on) if fail_on is not None else None
    scan_root = path if path.is_dir() else path.parent
    files = _iter_files(path, workspace_dir=state.db_path.parent)

    artifacts: list[Artifact] = []
    findings: list[Finding] = []
    for file_path in files:
        # Decide before hashing: a symlink whose target resolves outside
        # the scan root is flagged without ever being opened, unless the
        # operator explicitly opts in with --follow-symlinks. Checking
        # this after computing an Artifact (and therefore its hash) would
        # mean the hash-harvesting the finding warns about had already
        # happened by the time the warning fires.
        escaping_target = is_escaping_symlink(file_path, scan_root)
        if escaping_target is not None:
            findings.append(
                build_symlink_escape_finding(
                    str(file_path), escaping_target, content_read=follow_symlinks
                )
            )
            if not follow_symlinks:
                artifacts.append(Artifact.unresolved_symlink(file_path, escaping_target))
                continue

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
        pickle_finding = check_pickle_calls(artifact)
        if pickle_finding is not None:
            findings.append(pickle_finding)

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

    exit_if_threshold_met(run.findings, threshold)
