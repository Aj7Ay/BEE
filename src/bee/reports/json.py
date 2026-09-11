from __future__ import annotations

from bee.core.run import Run


def render_run_json(run: Run) -> str:
    return run.model_dump_json(indent=2)
