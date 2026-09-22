from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Source(ABC):
    """Abstract base class for all artifact sources."""

    @abstractmethod
    def get_artifacts(self) -> list[Path]:
        """Download/fetch artifacts and return local paths."""
        ...

    @abstractmethod
    def get_source_info(self) -> dict:
        """Return provenance source info (provider, repo, revision)."""
        ...

    @abstractmethod
    def get_model_card(self) -> dict | None:
        """Return model card metadata if available."""
        ...

    @abstractmethod
    def get_dependencies(self) -> list[Path]:
        """Return paths to dependency files if any."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any downloaded/temporary files."""
        ...
