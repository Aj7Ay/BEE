from __future__ import annotations

from pathlib import Path

import typer

from bee.core.signing import DEFAULT_KEY_DIR, KeyPaths, generate_keypair


def keygen_command(
    key_dir: Path = typer.Option(
        DEFAULT_KEY_DIR,
        "--key-dir",
        help="Directory to write the keypair into. Defaults to ~/.bee/keys, "
        "outside any project's .bee/ workspace -- a project database can be "
        "copied or shared freely without also copying the private key.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing keypair. Signatures already made with the "
        "old key remain verifiable (the public key travels with each "
        "signature) -- but you can no longer produce new ones under that "
        "identity once it's overwritten.",
    ),
) -> None:
    """Generate an Ed25519 signing keypair for `bee sign`."""
    paths = KeyPaths(private_key=key_dir / "bee_ed25519", public_key=key_dir / "bee_ed25519.pub")
    if paths.private_key.exists() and not force:
        typer.echo(f"A key already exists at {paths.private_key}. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)

    fp = generate_keypair(paths)
    typer.echo(f"Generated Ed25519 keypair in {paths.private_key.parent}/")
    typer.echo(f"  private key: {paths.private_key} (mode 0600)")
    typer.echo(f"  public key:  {paths.public_key}")
    typer.echo(f"  fingerprint: {fp}")
