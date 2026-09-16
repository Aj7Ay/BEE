from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

# Byte size per element for SafeTensors' documented dtype strings.
# https://github.com/huggingface/safetensors -- this is the complete set
# as of the format's current spec.
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


@dataclass
class TensorRange:
    name: str
    start: object
    end: object
    dtype: str
    shape: object


@dataclass
class SafetensorsAnalysis:
    header_len: int
    data_region_size: int
    tensors: list[TensorRange] = field(default_factory=list)


def analyze_safetensors(path: Path) -> SafetensorsAnalysis | None:
    """Parse a SafeTensors header into its declared tensor byte ranges,
    without trusting any of them yet -- that's check_safetensors_bounds'
    job. Deliberately permissive about field types here (a `start` that
    isn't an int, a `shape` that isn't a list) rather than rejecting the
    file outright; a malformed field is exactly what the bounds check
    exists to report, not something to silently swallow by bailing early.
    """
    try:
        with path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return None
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            file_size = path.stat().st_size
            if header_len <= 0 or header_len > file_size - 8:
                return None
            header_bytes = f.read(header_len)
    except OSError:
        return None
    try:
        header = json.loads(header_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        # ValueError (beyond JSONDecodeError) also covers CPython's
        # int-string-conversion digit limit: a header containing an
        # absurdly long integer literal (a bignum-DoS attempt via a
        # shape dimension, say) raises plain ValueError out of the
        # json module's own int parsing, not JSONDecodeError.
        return None
    if not isinstance(header, dict):
        return None

    tensors: list[TensorRange] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            continue
        tensors.append(TensorRange(
            name=name,
            start=offsets[0],
            end=offsets[1],
            dtype=meta.get("dtype", ""),
            shape=meta.get("shape", []),
        ))

    return SafetensorsAnalysis(
        header_len=header_len,
        data_region_size=file_size - 8 - header_len,
        tensors=tensors,
    )
