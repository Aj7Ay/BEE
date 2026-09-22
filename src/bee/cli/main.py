from __future__ import annotations

from pathlib import Path

import typer

from bee import __version__
from bee.cli.history import history_command
from bee.cli.huggingface import huggingface_command
from bee.cli.init import init_command
from bee.cli.inspect import inspect_command
from bee.cli.keygen import keygen_command
from bee.cli.modelcard import modelcard_command
from bee.cli.ollama import ollama_command
from bee.cli.policy import policy_validate
from bee.cli.report import report_command
from bee.cli.scan import scan_command
from bee.cli.show import show_command
from bee.cli.sign import sign_command
from bee.cli.state import AppState, OutputFormat
from bee.cli.verify import verify_command
from bee.cli.vet import vet_command

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
app.command("history")(history_command)
app.command("show")(show_command)
app.command("verify")(verify_command)
app.command("keygen")(keygen_command)
app.command("sign")(sign_command)
app.command("vet")(vet_command)
app.command("ollama")(ollama_command)
app.command("hf")(huggingface_command)
app.command("report")(report_command)
app.command("modelcard")(modelcard_command)
app.command("policy")(policy_validate)
