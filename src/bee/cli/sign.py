from __future__ import annotations

from pathlib import Path

import typer

from bee.core.run import compute_evidence_hash
from bee.core.signing import DEFAULT_PRIVATE_KEY_PATH, load_private_key, sign_hash
from bee.storage.db import load_run, save_run


def sign_command(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run id, as printed by `bee history`."),
    key: Path = typer.Option(
        DEFAULT_PRIVATE_KEY_PATH,
        "--key",
        help="Private key to sign with. Defaults to ~/.bee/keys/bee_ed25519 "
        "(create one with `bee keygen`).",
    ),
) -> None:
    """Sign a stored run's evidence hash with an Ed25519 private key.

    Signs the evidence recomputed from the run's current stored content,
    not whatever evidence_sha256 happens to already say -- so signing
    also certifies that the stored hash matches the stored data at the
    moment of signing. The public key is stored alongside the signature,
    so `bee verify` never needs access to this key file, only the record.
    """
    state = ctx.obj
    run = load_run(state.db_path, run_id)
    if run is None:
        typer.echo(f"No run found with id {run_id!r}", err=True)
        raise typer.Exit(code=1)

    if not key.exists():
        typer.echo(
            f"No private key at {key}. Generate one with `bee keygen` first.", err=True
        )
        raise typer.Exit(code=1)

    private_key = load_private_key(key)
    evidence_hash = compute_evidence_hash(
        run.target_path, run.artifacts, run.findings, run.scanner_version
    )
    signature_hex, public_key_hex = sign_hash(private_key, evidence_hash)

    run.evidence_sha256 = evidence_hash
    run.signature = signature_hex
    run.public_key = public_key_hex
    save_run(state.db_path, run)

    typer.echo(f"Signed run {run.id}")
    typer.echo(f"  evidence:   {evidence_hash}")
    typer.echo(f"  public key: {public_key_hex}")
