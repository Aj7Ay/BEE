from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel

from bee.evidence.finding import Confidence, Evidence
from bee.formats.detector import declared_format_from_extension, detect_format
from bee.formats.io_source import TOCTOU_SAFE_MAX_BYTES

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


def _hash_bytes(data: bytes) -> tuple[str, str]:
    return hashlib.sha256(data).hexdigest(), hashlib.sha512(data).hexdigest()


def read_magic_bytes_hex(path: Path, length: int = _MAGIC_BYTES_LENGTH) -> str:
    with path.open("rb") as f:
        return f.read(length).hex()


def _magic_bytes_from_evidence(evidence: list[Evidence]) -> str:
    """The exact bytes a detector matched on, and only those -- not a
    blind fixed-size read from offset 0. A detector reports a
    `type="magic_bytes"` Evidence entry precisely when its signature is a
    real fixed byte sequence at a known offset (GGUF's `GGUF` at 0, tar's
    `ustar` at 257, ...); every such entry already carries the exact
    matched bytes as hex. A detector whose evidence is descriptive instead
    (safetensors' header validity, pickle's protocol, an archive member
    name) has nothing to contribute here, and this returns "" rather than
    substituting an unrelated raw read -- which is exactly the bug this
    replaces: a blind `read(path, 0, 16)` doesn't know that tar's magic
    lives at offset 257, so it recorded the first archive member's
    filename instead."""
    for item in evidence:
        if item.type == "magic_bytes":
            return item.value
    return ""


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
        artifact, _content = cls.from_file_with_content(path)
        return artifact

    @classmethod
    def from_file_with_content(cls, path: Path) -> tuple["Artifact", bytes | None]:
        """Like `from_file`, but also returns the file's exact bytes as
        they stood at read time, when the file is small enough to buffer
        safely (see TOCTOU_SAFE_MAX_BYTES) -- `None` otherwise.

        A caller that goes on to run deep analysis (check_pickle_calls,
        check_safetensors_bounds, check_gguf_bounds, ...) on this same
        artifact should pass that buffer through to them instead of
        letting each one separately re-open `path`: hashing, format
        detection, and deep analysis would otherwise each independently
        re-read the file, and a file replaced on disk between those reads
        could make the hash BEE records and the content BEE actually
        analyzed for danger diverge -- a signed, verified record
        attesting to bytes that were never the ones actually vetted. For
        a file too large to buffer, this still returns a correct Artifact;
        it just can't close that window, the same way it couldn't before
        this method existed.
        """
        size = path.stat().st_size
        content: bytes | None = None
        if size <= TOCTOU_SAFE_MAX_BYTES:
            content = path.read_bytes()
            # Reflects the bytes actually read, not the earlier stat()
            # call -- the two can differ if the file changed in between,
            # and what was hashed is what this record should describe.
            size = len(content)
            sha256, sha512 = _hash_bytes(content)
            detected_format, format_confidence, format_evidence = detect_format(content)
        else:
            sha256, sha512 = compute_file_hashes(path)
            detected_format, format_confidence, format_evidence = detect_format(path)
        declared_format = declared_format_from_extension(path)
        # Derived from the matching detector's own evidence -- see
        # _magic_bytes_from_evidence -- rather than a separate blind read.
        # This is what actually fixes both the general case (nothing is
        # recorded for "unknown", since there's no evidence to draw from)
        # and the specific one (tar's real magic is at offset 257, not 0;
        # a blind fixed-offset read had been recording the first archive
        # member's filename instead of tar's signature).
        magic_bytes_hex = _magic_bytes_from_evidence(format_evidence)
        is_symlink = path.is_symlink()
        # resolve(strict=False) so a broken symlink still records where it
        # points, instead of raising.
        symlink_target = str(path.resolve(strict=False)) if is_symlink else None
        artifact = cls(
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
        return artifact, content

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
