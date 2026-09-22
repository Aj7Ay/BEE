from __future__ import annotations

import json as jsonlib
import os
import tempfile
from pathlib import Path

import requests

from bee.sources.base import Source


OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaSource(Source):
    """Interacts with a local Ollama installation."""

    def __init__(self, name: str, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._raw_name = name
        self._temp_dir: Path | None = None

    def _is_safe_blob_ref(self, blob_ref: str) -> bool:
        """Reject path traversal, absolute paths, and null bytes."""
        if not blob_ref or blob_ref != blob_ref.strip():
            return False
        if blob_ref.startswith("/") or blob_ref.startswith("\\"):
            return False
        if ".." in blob_ref:
            return False
        if "\x00" in blob_ref:
            return False
        return True

    @property
    def _model_name(self) -> str:
        """Extract just the model name (without tags)."""
        return self._raw_name.split(":")[0]

    @property
    def _repo(self) -> str:
        """Extract the repository path."""
        return self._model_name.replace(":", "/")

    def list_models(self) -> list[dict]:
        """List all Ollama models with metadata."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except requests.RequestException:
            return []

    def inspect_model(self, name: str) -> dict | None:
        """Inspect a specific model's metadata."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"name": name},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def get_artifacts(self) -> list[Path]:
        """Download model file from Ollama library and return local path."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="bee_ollama_"))

        info = self.inspect_model(self._model_name)
        if not info:
            return []

        artifacts: list[Path] = []

        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self._model_name},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            modelfile = data.get("modelfile", "")
            if modelfile:
                for line in modelfile.splitlines():
                    if line.startswith("ADD") and "blob" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            blob_ref = parts[1]
                            # Validate blob_ref: reject traversal/absolute paths
                            if not self._is_safe_blob_ref(blob_ref):
                                continue
                            ollama_dir = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
                            blob_path = ollama_dir / "blobs" / blob_ref
                            if blob_path.is_file():
                                artifacts.append(blob_path)
        except requests.RequestException:
            pass

        if not artifacts:
            try:
                requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self._model_name},
                    timeout=300,
                )

                ollama_dir = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
                repo_dir = ollama_dir / "manifests" / self._repo

                if repo_dir.exists():
                    for manifest_file in repo_dir.rglob("*"):
                        if manifest_file.is_file():
                            artifacts.append(manifest_file)
            except requests.RequestException:
                pass

        return artifacts

    def get_source_info(self) -> dict:
        return {
            "provider": "ollama",
            "repository": self._model_name,
            "revision": None,
        }

    def get_model_card(self) -> dict | None:
        """Extract model info from Ollama metadata."""
        info = self.inspect_model(self._model_name)
        if not info:
            return None

        return {
            "model_name": self._model_name,
            "parameters": info.get("parameters", ""),
            "format": info.get("format", ""),
            "family": info.get("details", {}).get("family", ""),
            "quantization": info.get("details", {}).get("quantization_level", ""),
            "modelfile": info.get("modelfile", ""),
        }

    def get_dependencies(self) -> list[Path]:
        return []

    @property
    def source_type(self) -> str:
        return "ollama"

    def cleanup(self) -> None:
        if self._temp_dir and self._temp_dir.exists():
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)


class OllamaLocalSource(Source):
    """Source that wraps an existing local path as an Ollama-sourced artifact."""

    def __init__(self, path: Path):
        self._path = path

    def get_artifacts(self) -> list[Path]:
        return [self._path] if self._path.is_file() else []

    def get_source_info(self) -> dict:
        return {"provider": "ollama", "repository": None, "revision": None}

    def get_model_card(self) -> dict | None:
        return None

    def get_dependencies(self) -> list[Path]:
        return []

    @property
    def source_type(self) -> str:
        return "ollama"

    def cleanup(self) -> None:
        pass
