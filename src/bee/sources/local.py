from __future__ import annotations

from pathlib import Path

from bee.sources.base import Source


class LocalSource(Source):
    """Represents a local filesystem path as a source."""

    def __init__(self, path: Path):
        self._path = path
        self._temp_dirs: list[Path] = []

    def get_artifacts(self) -> list[Path]:
        if self._path.is_file():
            return [self._path]
        return list(self._path.rglob("*"))

    def get_source_info(self) -> dict:
        return {
            "provider": "local",
            "repository": None,
            "revision": None,
        }

    def get_model_card(self) -> dict | None:
        card = self._path / "README.md"
        if card.is_file():
            text = card.read_text(errors="replace")
            return {"raw": text}
        return None

    def get_dependencies(self) -> list[Path]:
        deps = []
        for name in ("requirements.txt", "pyproject.toml", "package.json"):
            p = self._path / name
            if p.is_file():
                deps.append(p)
        return deps

    @property
    def source_type(self) -> str:
        return "local"

    def cleanup(self) -> None:
        for d in self._temp_dirs:
            if d.exists():
                import shutil
                shutil.rmtree(d, ignore_errors=True)
