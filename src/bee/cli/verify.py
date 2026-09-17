from __future__ import annotations

import json as jsonlib
from pathlib import Path

import typer

from bee.cli.state import OutputFormat
from bee.core.artifact import compute_file_hashes
from bee.core.run import compute_evidence_hash
from bee.evidence.symlink import is_escaping_symlink
from bee.storage.db import load_run


def verify_command(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run id, as printed by `bee history`."),
) -> None:
    """Re-check a stored run's evidence integrity, and whether its
    artifacts have changed since they were recorded.

    Two independent things are checked: whether the recorded evidence
    (the findings and artifact records BEE originally produced) still
    hashes to what was stored, and whether each artifact's *current*
    file content still matches the hash recorded when it was vetted --
    catching the artifact having been swapped or modified since.

    The evidence check is an unkeyed hash (see
    bee.core.run.compute_evidence_hash) -- it catches accidental
    corruption and naive edits to the stored record, not a capable
    attacker who can write to the database and recompute a matching
    hash after editing it. A clean result here means the record is
    internally self-consistent, not that it's cryptographically
    guaranteed untouched; that guarantee needs a keyed hash or a
    signature, which isn't built yet.
    """
    state = ctx.obj
    run = load_run(state.db_path, run_id)
    if run is None:
        typer.echo(f"No run found with id {run_id!r}", err=True)
        raise typer.Exit(code=1)

    recomputed_hash = compute_evidence_hash(
        run.target_path, run.artifacts, run.findings, run.scanner_version
    )
    evidence_ok = recomputed_hash == run.evidence_sha256

    # Anchor to the absolute location recorded at scan time, not the
    # given target_path as-is: a relative target_path ("./models") only
    # resolves correctly from the exact working directory the scan ran
    # in, and bee verify has no reason to require running from there.
    resolved_target = Path(run.resolved_target_path) if run.resolved_target_path else Path(run.target_path)
    given_target = Path(run.target_path)
    scan_root = resolved_target if resolved_target.is_dir() else resolved_target.parent

    def _resolve_artifact_path(artifact_path: str) -> Path:
        p = Path(artifact_path)
        try:
            relative = p.relative_to(given_target)
        except ValueError:
            return p  # already absolute, or doesn't share the recorded prefix -- use as-is
        return resolved_target if str(relative) == "." else resolved_target / relative

    artifact_results = []
    for artifact in run.artifacts:
        path = _resolve_artifact_path(artifact.path)
        if not artifact.sha256:
            # Nothing was recorded to compare against (an unresolved
            # symlink, an unreadable file at scan time, ...).
            artifact_results.append({"path": artifact.path, "status": "skipped"})
            continue

        # The same decision scan/inspect make, made the same way: a path
        # that wasn't an escaping symlink when it was vetted could have
        # been swapped for one since. Re-hashing it here without the same
        # check would defeat the protection scan/inspect already apply.
        escaping_target = is_escaping_symlink(path, scan_root)
        if escaping_target is not None:
            artifact_results.append({
                "path": artifact.path,
                "status": "escaped",
                "detail": f"now a symlink resolving outside the original scan root, to {escaping_target}",
            })
            continue

        if not path.exists():
            artifact_results.append({"path": artifact.path, "status": "missing"})
            continue

        try:
            current_sha256, _ = compute_file_hashes(path)
        except OSError as exc:
            artifact_results.append({"path": artifact.path, "status": "error", "detail": str(exc)})
            continue

        if current_sha256 == artifact.sha256:
            artifact_results.append({"path": artifact.path, "status": "ok"})
        else:
            artifact_results.append({
                "path": artifact.path,
                "status": "changed",
                "recorded_sha256": artifact.sha256,
                "current_sha256": current_sha256,
            })

    all_ok = evidence_ok and all(r["status"] in ("ok", "skipped") for r in artifact_results)

    if state.output_format is OutputFormat.JSON:
        payload = {
            "run_id": run.id,
            "evidence_ok": evidence_ok,
            "recorded_evidence_sha256": run.evidence_sha256,
            "recomputed_evidence_sha256": recomputed_hash,
            "artifacts": artifact_results,
            "ok": all_ok,
        }
        typer.echo(jsonlib.dumps(payload, indent=2))
    else:
        typer.echo(f"Run:              {run.id}")
        if evidence_ok:
            typer.echo(f"Evidence record:  OK ({run.evidence_sha256[:16]}...)")
        else:
            typer.echo("Evidence record:  TAMPERED")
            typer.echo(f"  recorded:   {run.evidence_sha256}")
            typer.echo(f"  recomputed: {recomputed_hash}")
        typer.echo()
        typer.echo("Artifacts:")
        for result in artifact_results:
            status = result["status"].upper()
            typer.echo(f"  {status:8} {result['path']}")
            if result["status"] == "changed":
                typer.echo(f"           recorded: {result['recorded_sha256']}")
                typer.echo(f"           current:  {result['current_sha256']}")
            elif "detail" in result:
                typer.echo(f"           {result['detail']}")

    if not all_ok:
        raise typer.Exit(code=1)
