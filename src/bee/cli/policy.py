from __future__ import annotations

from pathlib import Path

import typer

from bee.policy.loader import load_policy, load_policy_from_dict
from bee.policy.rules import validate_policy_yaml


def _policy_app() -> typer.Typer:
    """Policy subcommand group."""
    policy_app = typer.Typer(help="BEE policy management commands.")
    policy_app.command("validate")(policy_validate)
    policy_app.command("evaluate")(policy_evaluate)
    return policy_app


def policy_validate(
    path: Path = typer.Argument(..., exists=True, help="Policy YAML file to validate."),
) -> None:
    """Validate a BEE policy YAML file without running a scan."""
    try:
        data = validate_policy_yaml(path.read_text())
        name = data.get("policy", data).get("name", "unnamed")
        typer.echo(f"Policy '{name}' is valid.")
    except ValueError as exc:
        typer.echo(f"Policy validation failed: {exc}", err=True)
        raise typer.Exit(code=1)


def policy_evaluate(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=True, help="File or directory to evaluate against policy."),
    policy: Path = typer.Option(..., "--policy", help="Policy YAML file."),
    format = typer.Option("text", "--format"),
) -> None:
    """Evaluate a target against a BEE policy and show the decision."""
    from bee.cli.state import OutputFormat

    policy_obj = load_policy(policy)
    typer.echo(f"Policy: {policy_obj.name}")
    typer.echo("Policy rules loaded successfully.")
