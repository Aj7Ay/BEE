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

    @classmethod
    def from_file(cls, path: Path) -> "Artifact":
        size = path.stat().st_size
        sha256, sha512 = compute_file_hashes(path)
        declared_format = declared_format_from_extension(path)
        detected_format, format_confidence, _evidence = detect_format(path)
        magic_bytes_hex = read_magic_bytes_hex(path)
        return cls(
            path=str(path),
            size=size,
            sha256=sha256,
            sha512=sha512,
            declared_format=declared_format,
            detected_format=detected_format,
            format_confidence=format_confidence,
            magic_bytes_hex=magic_bytes_hex,
        )
