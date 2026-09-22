from __future__ import annotations

from pathlib import Path

import typer

from bee.cli.vet import run as scan_for_card
from bee.evidence.model_card_generator import generate_model_card, save_model_card


def modelcard_command(
    target: str = typer.Argument(..., help="File or directory to scan."),
    output: Path = typer.Option("README.md", "--output", "-o", help="Output model card file."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing model card."),
) -> None:
    """Generate a BEE-compliant model card from a scan."""
    target_path = Path(target)

    # Run scan
    typer.echo(f"Scanning {target_path}...")
    run_obj = scan_for_card(target_path)
    if run_obj is None:
        typer.echo("Scan failed.", err=True)
        raise typer.Exit(code=1)

    # Generate card
    existing = target_path / "README.md" if target_path.is_dir() else None
    if existing and existing.is_file() and not overwrite:
        typer.echo(
            f"Model card already exists at {existing}. Use --overwrite to replace.",
            err=True,
        )
        raise typer.Exit(code=1)

    save_model_card(run_obj, output, existing)
    typer.echo(f"Model card saved to: {output}")
