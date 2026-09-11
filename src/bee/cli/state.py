from __future__ import annotations

from enum import Enum
from pathlib import Path


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"


class AppState:
    def __init__(self, output_format: OutputFormat, db_path: Path) -> None:
        self.output_format = output_format
        self.db_path = db_path
