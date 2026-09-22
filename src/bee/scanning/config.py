from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ScanConfig(BaseModel):
    """Configuration for a BEE scan run."""
    fail_on: str | None = None
    deterministic: bool = False
    follow_symlinks: bool = False
    policy_path: Path | None = None
    output_dir: Path | None = None
    write_evidence_files: bool = True
