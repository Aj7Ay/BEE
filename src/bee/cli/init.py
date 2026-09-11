from __future__ import annotations

from pathlib import Path

import typer

from bee.storage.db import init_db

_CONFIG_TEMPLATE = "schema_version: 1\n"


def init_command(ctx: typer.Context) -> None:
    """Initialize a BEE workspace in the current directory."""
    state = ctx.obj
    bee_dir = Path(".bee")
    config_path = bee_dir / "config.yaml"
    already_existed = bee_dir.exists()
    bee_dir.mkdir(exist_ok=True)
    if not config_path.exists():
        config_path.write_text(_CONFIG_TEMPLATE)
    init_db(state.db_path)
    if already_existed:
        typer.echo(f"BEE workspace already exists at {bee_dir}/")
    else:
        typer.echo(f"Initialized BEE workspace at {bee_dir}/")
