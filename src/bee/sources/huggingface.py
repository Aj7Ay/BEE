from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from bee.sources.base import Source


_MODEL_EXTENSIONS = frozenset({
    ".safetensors", ".gguf", ".bin", ".pt", ".pth", ".pkl",
    ".h5", ".hdf5",
})

_MODEL_FILENAMES = frozenset({
    "pytorch_model", "consolidated", "checkpoint",
    "model.safetensors", "model.gguf",
})


class HuggingFaceSource(Source):
    """Handles hf://org/model and hf://org/model/revision URLs."""

    def __init__(self, url: str):
        self.url = url
        cleaned = url.replace("hf://", "").replace("huggingface://", "")
        parts = cleaned.split("/")
        self.repo_id = f"{parts[0]}/{parts[1]}"
        self.revision = parts[2] if len(parts) > 2 else None
        self._temp_dir: Path | None = None

    def get_artifacts(self) -> list[Path]:
        from huggingface_hub import hf_hub_download, list_repo_files

        self._temp_dir = Path(tempfile.mkdtemp(prefix="bee_hf_"))
        files = list_repo_files(self.repo_id, revision=self.revision)

        artifacts: list[Path] = []
        for filename in files:
            # Validate filename: reject traversal, absolute paths
            if not self._is_safe_filename(filename):
                continue

            if self._is_model_file(filename):
                try:
                    path = hf_hub_download(
                        self.repo_id,
                        filename,
                        revision=self.revision,
                        local_dir=str(self._temp_dir),
                    )
                    artifacts.append(Path(path))
                except Exception:
                    continue

        return artifacts

    def _is_safe_filename(self, filename: str) -> bool:
        """Reject path traversal, absolute paths, and null bytes."""
        if not filename or filename != filename.strip():
            return False
        if filename.startswith("/") or filename.startswith("\\"):
            return False
        if ".." in filename:
            return False
        if "\x00" in filename:
            return False
        return True

    def _is_model_file(self, filename: str) -> bool:
        """Determine if a file is a model artifact worth scanning."""
        path = Path(filename)
        if path.suffix in _MODEL_EXTENSIONS:
            return True
        if path.stem in _MODEL_FILENAMES:
            return True
        if "model" in filename.lower() and any(
            ext in filename for ext in (".safetensors", ".gguf", ".bin", ".pt", ".pth")
        ):
            return True
        return False

    def get_source_info(self) -> dict:
        return {
            "provider": "huggingface",
            "repository": self.repo_id,
            "revision": self.revision,
        }

    def get_model_card(self) -> dict | None:
        from huggingface_hub import hf_hub_download

        try:
            card_path = hf_hub_download(
                self.repo_id, "README.md",
                revision=self.revision,
            )
            card_text = Path(card_path).read_text(errors="replace")
            return self._parse_model_card(card_text)
        except Exception:
            return None

    def _parse_model_card(self, text: str) -> dict[str, Any]:
        """Parse YAML front-matter from a model card."""
        import yaml

        card: dict[str, Any] = {"raw": text}

        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    card["metadata"] = yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    pass

        meta = card.get("metadata", {}) or {}
        card["model_name"] = meta.get("model_name") or meta.get("library_name")
        card["architecture"] = meta.get("config") or meta.get("model_type")
        card["license"] = meta.get("license")
        card["tags"] = meta.get("tags")

        return card

    def get_dependencies(self) -> list[Path]:
        from huggingface_hub import hf_hub_download

        deps = []
        manifest_files = ("requirements.txt", "pyproject.toml", "package.json")

        for name in manifest_files:
            try:
                path = hf_hub_download(
                    self.repo_id, name, revision=self.revision,
                )
                deps.append(Path(path))
            except Exception:
                continue

        return deps

    @property
    def source_type(self) -> str:
        return "huggingface"

    def cleanup(self) -> None:
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
