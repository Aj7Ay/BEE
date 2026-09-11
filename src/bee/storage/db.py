from __future__ import annotations

import sqlite3
from pathlib import Path

from bee.core.run import Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    target TEXT NOT NULL,
    data TEXT NOT NULL
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_SCHEMA)


def save_run(db_path: Path, run: Run) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (id, created_at, target, data) VALUES (?, ?, ?, ?)",
            (run.id, run.created_at.isoformat(), run.target_path, run.model_dump_json()),
        )


def load_run(db_path: Path, run_id: str) -> Run | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    return Run.model_validate_json(row[0])


def list_runs(db_path: Path) -> list[Run]:
    """All stored runs, most recently created first."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT data FROM runs ORDER BY created_at DESC").fetchall()
    return [Run.model_validate_json(row[0]) for row in rows]
