from __future__ import annotations

import typer

from bee.storage.db import init_db

_CONFIG_TEMPLATE = "schema_version: 1\n"


def init_command(ctx: typer.Context) -> None:
    """Initialize a BEE workspace in the current directory."""
    state = ctx.obj
    # Derived from --db, not hardcoded: `bee --db other/bee.db init` must
    # not create a workspace split across .bee/config.yaml and other/bee.db.
    bee_dir = state.db_path.parent
    config_path = bee_dir / "config.yaml"
    already_existed = bee_dir.exists()
    bee_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(_CONFIG_TEMPLATE)
    init_db(state.db_path)
    if already_existed:
        typer.echo(f"BEE workspace already exists at {bee_dir}/")
    else:
        typer.echo(f"Initialized BEE workspace at {bee_dir}/")
