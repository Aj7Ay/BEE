from __future__ import annotations

import typer


def ollama_command(
    action: str = typer.Argument(..., help="ollama vet"),
    model: str = typer.Argument(None, help="Model name (e.g., qwen3:8b)"),
) -> None:
    """Vet local Ollama models. This feature is not yet implemented."""
    if action != "vet":
        typer.echo(f"Error: unknown action '{action}'. Only 'vet' is supported.", err=True)
        raise typer.Exit(code=1)

    if not model:
        typer.echo("Error: model name required for vet (e.g., bee ollama vet qwen3:8b)", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Ollama model vetting not yet implemented: {model}")
    typer.echo("Coming in a future release.")
    raise typer.Exit(code=0)
