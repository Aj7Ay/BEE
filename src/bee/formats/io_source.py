from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Union

# Every format parser in this package reads either a real file on disk
# (the normal case) or an in-memory buffer captured once, up front, by
# Artifact.from_file for files small enough to buffer safely (see
# TOCTOU_SAFE_MAX_BYTES below) -- the same bytes then get reused for
# hashing, format detection, AND deep analysis, instead of each of those
# stages independently re-opening the path and each potentially seeing
# different content if the file was replaced on disk in between. A
# Source is deliberately either one or the other, never a path string:
# every parser in this package already takes a Path, so a bare `str`
# input would be ambiguous with "the file's actual content as bytes".
Source = Union[Path, bytes]

# Real structural content BEE ever needs to inspect -- a SafeTensors/GGUF
# header, a pickle opcode stream, a small PyTorch zip archive's central
# directory and data.pkl member -- is small. The tensor DATA that makes
# a real checkpoint large is never read for structural analysis in the
# first place (see e.g. gguf_ops.analyze_gguf's own docstring). A file
# at or under this bound can be read into memory exactly once and reused
# for every stage of analysis, closing the window where separate re-opens
# could see different content because the file changed in between. A
# file above it keeps the previous per-stage re-open behavior -- still
# correct, just not race-proof -- rather than buffering potentially many
# gigabytes of tensor data into memory just to close that window.
TOCTOU_SAFE_MAX_BYTES = 64 * 1024 * 1024


def open_source(source: Source) -> IO[bytes]:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    return source.open("rb")


def size_of(source: Source) -> int:
    if isinstance(source, (bytes, bytearray)):
        return len(source)
    return source.stat().st_size
