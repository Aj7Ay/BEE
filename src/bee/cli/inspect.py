from __future__ import annotations

import json as jsonlib
from pathlib import Path

import typer

from bee.cli.sanitize import sanitize_for_terminal
from bee.cli.severity import FAIL_ON_HELP, exit_if_threshold_met, parse_severity_option
from bee.cli.state import OutputFormat
from bee.core.artifact import Artifact
from bee.evidence.finding import Finding
from bee.evidence.gguf_bounds import check_gguf_bounds
from bee.evidence.mismatch import check_mismatch
from bee.evidence.pickle_calls import check_pickle_calls
from bee.evidence.safetensors_bounds import check_safetensors_bounds, check_safetensors_gap
from bee.evidence.symlink import build_symlink_escape_finding, is_escaping_symlink
from bee.formats.gguf_ops import analyze_gguf


def inspect_command(
    ctx: typer.Context,
    path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, help="File to inspect."
    ),
    fail_on: str | None = typer.Option(None, "--fail-on", help=FAIL_ON_HELP),
    follow_symlinks: bool = typer.Option(
        False,
        "--follow-symlinks",
        help="Read (and hash) this path's target even when it's a symlink "
        "resolving outside its containing directory. BEE-SYM-001 is still "
        "recorded either way. Off by default -- matches `bee scan`, so the "
        "same path is never read by one command while the other refuses it.",
    ),
) -> None:
    """Show a detailed identity and format report for a single artifact."""
    state = ctx.obj
    threshold = parse_severity_option(fail_on) if fail_on is not None else None
    findings: list[Finding] = []

    # Same decision as `bee scan`, made the same way: from the raw path,
    # before anything is opened. A symlink named directly on the command
    # line is still capable of pointing at an arbitrary file on the host;
    # the user having typed the path is not consent to read whatever it
    # resolves to.
    escaping_target = is_escaping_symlink(path, scan_root=path.parent)
    if escaping_target is not None:
        findings.append(
            build_symlink_escape_finding(str(path), escaping_target, content_read=follow_symlinks)
        )
        if not follow_symlinks:
            artifact = Artifact.unresolved_symlink(path, escaping_target)
            _print_inspection(state, artifact, findings)
            exit_if_threshold_met(findings, threshold)
            return

    try:
        artifact, content = Artifact.from_file_with_content(path)
    except OSError as exc:
        typer.echo(f"Error: could not read {sanitize_for_terminal(str(path))}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    mismatch = check_mismatch(artifact)
    if mismatch is not None:
        findings.append(mismatch)
    pickle_finding = check_pickle_calls(artifact, content)
    if pickle_finding is not None:
        findings.append(pickle_finding)
    bounds_finding = check_safetensors_bounds(artifact, content)
    if bounds_finding is not None:
        findings.append(bounds_finding)
    gap_finding = check_safetensors_gap(artifact, content)
    if gap_finding is not None:
        findings.append(gap_finding)
    gguf_finding = check_gguf_bounds(artifact, content)
    if gguf_finding is not None:
        findings.append(gguf_finding)

    _print_inspection(state, artifact, findings, content)
    exit_if_threshold_met(findings, threshold)


def _gguf_summary(artifact: Artifact, content: bytes | None = None) -> dict | None:
    if artifact.detected_format != "gguf":
        return None
    analysis = analyze_gguf(content if content is not None else Path(artifact.path))
    if analysis is None:
        return None
    architecture = analysis.metadata.get("general.architecture")
    return {
        "version": analysis.version,
        "tensor_count": analysis.tensor_count,
        "metadata_kv_count": analysis.metadata_kv_count,
        "architecture": architecture,
    }


def _print_inspection(
    state, artifact: Artifact, findings: list[Finding], content: bytes | None = None
) -> None:
    gguf_info = _gguf_summary(artifact, content)

    if state.output_format is OutputFormat.JSON:
        payload = {
            "artifact": artifact.model_dump(mode="json"),
            "findings": [f.model_dump(mode="json") for f in findings],
        }
        if gguf_info is not None:
            payload["gguf"] = gguf_info
        typer.echo(jsonlib.dumps(payload, indent=2))
        return

    typer.echo(f"Path:             {sanitize_for_terminal(artifact.path)}")
    typer.echo(f"Size:             {artifact.size} bytes")
    typer.echo(f"SHA-256:          {artifact.sha256}")
    typer.echo(f"SHA-512:          {artifact.sha512}")
    typer.echo(f"Declared format:  {artifact.declared_format}")
    typer.echo(f"Detected format:  {artifact.detected_format} ({artifact.format_confidence.value})")
    typer.echo(f"Magic bytes:      {artifact.magic_bytes_hex}")
    if artifact.is_symlink:
        target = sanitize_for_terminal(artifact.symlink_target or "")
        typer.echo(f"Symlink:          True -> {target}")
    if gguf_info is not None:
        typer.echo(f"GGUF version:     {gguf_info['version']}")
        typer.echo(f"GGUF tensors:     {gguf_info['tensor_count']}")
        typer.echo(f"GGUF metadata kv: {gguf_info['metadata_kv_count']}")
        if gguf_info["architecture"] is not None:
            typer.echo(f"GGUF architecture: {sanitize_for_terminal(gguf_info['architecture'])}")
    if findings:
        for finding in findings:
            typer.echo(f"Finding:          {finding.id} [{finding.severity.value}] {finding.title}")
    else:
        typer.echo("Finding:          none")
