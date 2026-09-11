from __future__ import annotations

from pathlib import Path

import typer

from bee import __version__
from bee.cli.init import init_command
from bee.cli.inspect import inspect_command
from bee.cli.scan import scan_command
from bee.cli.state import AppState, OutputFormat

app = typer.Typer(help="BEE - AI model supply-chain vetting CLI.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bee {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    format: OutputFormat = typer.Option(
        OutputFormat.TEXT, "--format", help="Output format."
    ),
    db: Path = typer.Option(
        Path(".bee/bee.db"), "--db", help="Path to the BEE run database."
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """BEE - AI model supply-chain vetting CLI."""
    ctx.obj = AppState(output_format=format, db_path=db)


app.command("init")(init_command)
app.command("scan")(scan_command)
app.command("inspect")(inspect_command)
