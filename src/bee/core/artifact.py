from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from bee.evidence.finding import Confidence
from bee.formats.detector import declared_format_from_extension, detect_format

_HASH_CHUNK_SIZE = 1_048_576
_MAGIC_BYTES_LENGTH = 16


def compute_file_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            sha256.update(chunk)
            sha512.update(chunk)
    return sha256.hexdigest(), sha512.hexdigest()


def read_magic_bytes_hex(path: Path, length: int = _MAGIC_BYTES_LENGTH) -> str:
    with path.open("rb") as f:
        return f.read(length).hex()


class Artifact(BaseModel):
    path: str
    size: int
    sha256: str
    sha512: str
    declared_format: str
    detected_format: str
    format_confidence: Confidence
    magic_bytes_hex: str
    is_symlink: bool = False
    symlink_target: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "Artifact":
        size = path.stat().st_size
        sha256, sha512 = compute_file_hashes(path)
        declared_format = declared_format_from_extension(path)
        detected_format, format_confidence, _evidence = detect_format(path)
        magic_bytes_hex = read_magic_bytes_hex(path)
        is_symlink = path.is_symlink()
        # resolve(strict=False) so a broken symlink still records where it
        # points, instead of raising.
        symlink_target = str(path.resolve(strict=False)) if is_symlink else None
        return cls(
            path=str(path),
            size=size,
            sha256=sha256,
            sha512=sha512,
            declared_format=declared_format,
            detected_format=detected_format,
            format_confidence=format_confidence,
            magic_bytes_hex=magic_bytes_hex,
            is_symlink=is_symlink,
            symlink_target=symlink_target,
        )

    @classmethod
    def unresolved_symlink(cls, path: Path, symlink_target: str) -> "Artifact":
        """An Artifact for a symlink whose target was deliberately never
        opened — because it resolves outside the directory being scanned,
        and reading (and therefore hashing) it would leak information
        about a file this scan wasn't asked to touch. Identity fields are
        empty, the same shape used for a file that couldn't be read at
        all (BEE-IO-001), rather than a guess."""
        return cls(
            path=str(path),
            size=0,
            sha256="",
            sha512="",
            declared_format=declared_format_from_extension(path),
            detected_format="unknown",
            format_confidence=Confidence.UNKNOWN,
            magic_bytes_hex="",
            is_symlink=True,
            symlink_target=symlink_target,
        )
