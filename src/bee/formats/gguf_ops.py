from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

# Mirrors llama.cpp's own reference reader (gguf-py/gguf/gguf_reader.py) --
# its defense against a maliciously large declared count/length, not a
# threshold BEE invented. A real file is nowhere near either bound; a
# tensor/kv count or string/array length past it means the header is
# corrupted or hostile, either way not worth attempting to parse further.
GGUF_MAX_ARRAY_ELEMENTS = 1024 * 1024 * 1024
GGUF_MAX_STRING_LENGTH = 1024 * 1024 * 1024

# No real GGUF file nests an array inside an array more than a couple of
# levels deep. _skip_array_elements recurses on nested arrays; without a
# depth cap, a crafted file can nest deeply enough to blow Python's call
# stack (RecursionError) well before any length/count check above ever
# gets a chance to reject it.
_MAX_ARRAY_NESTING_DEPTH = 100

# GGUFValueType, from llama.cpp's gguf-py/gguf/constants.py.
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_STRUCT: dict[int, tuple[str, int]] = {
    _UINT8: ("B", 1), _INT8: ("b", 1),
    _UINT16: ("H", 2), _INT16: ("h", 2),
    _UINT32: ("I", 4), _INT32: ("i", 4),
    _FLOAT32: ("f", 4),
    _BOOL: ("?", 1),
    _UINT64: ("Q", 8), _INT64: ("q", 8),
    _FLOAT64: ("d", 8),
}

# (elements_per_block, bytes_per_block) for every GGML tensor type this
# module knows how to size-check -- taken directly, values unchanged,
# from llama.cpp's own GGML_QUANT_SIZES table (gguf-py/gguf/constants.py).
# Not re-derived: getting a quantization block layout subtly wrong would
# produce a confidently wrong finding rather than an honest "not checked".
# A ggml_type not in this table (a newer quantization format BEE hasn't
# been updated for) is skipped for the size check, not guessed at.
_QK_K = 256
GGML_QUANT_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2),
    2: (32, 2 + 16), 3: (32, 2 + 2 + 16),
    6: (32, 2 + 4 + 16), 7: (32, 2 + 2 + 4 + 16),
    8: (32, 2 + 32), 9: (32, 4 + 4 + 32),
    10: (256, 2 + 2 + _QK_K // 16 + _QK_K // 4),
    11: (256, 2 + _QK_K // 4 + _QK_K // 8 + 12),
    12: (256, 2 + 2 + _QK_K // 2 + 12),
    13: (256, 2 + 2 + _QK_K // 2 + _QK_K // 8 + 12),
    14: (256, 2 + _QK_K // 2 + _QK_K // 4 + _QK_K // 16),
    15: (256, 4 + _QK_K + _QK_K // 8),
    16: (256, 2 + _QK_K // 4),
    17: (256, 2 + _QK_K // 4 + _QK_K // 32),
    18: (256, 2 + _QK_K // 4 + _QK_K // 8),
    19: (256, 2 + _QK_K // 8 + _QK_K // 16),
    20: (32, 2 + 16),
    21: (256, 2 + _QK_K // 4 + _QK_K // 8 + _QK_K // 32 + 4),
    22: (256, 2 + _QK_K // 4 + _QK_K // 16),
    23: (256, 2 + 2 + _QK_K // 2 + _QK_K // 64),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, _QK_K // 8 + _QK_K // 16 + _QK_K // 32),
    30: (1, 2),
    34: (256, 2 + 4 * 13),
    35: (256, 2 + 64),
    39: (32, 1 + 16),
    40: (64, 4 + 32),
    41: (128, 2 + 16),
    42: (64, 2 + 16),
}

# No real tensor has anywhere near this many elements -- same magnitude
# judgment as the SafeTensors bounds check, and for the same reason: cap
# growth on an attacker-supplied dimension product instead of chasing
# exactness once it's already implausible.
_MAX_PLAUSIBLE_ELEMENTS = 2**62


class _GgufParseError(Exception):
    pass


@dataclass
class GgufTensorInfo:
    name: str
    dimensions: list[int]
    ggml_type: int
    offset: int


@dataclass
class GgufAnalysis:
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: dict[str, Any]
    tensors: list[GgufTensorInfo] = field(default_factory=list)
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    tensor_data_start: int = 0
    file_size: int = 0


def _read_exact(f: IO[bytes], n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise _GgufParseError("unexpected end of file")
    return data


def _read_u32(f: IO[bytes]) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f: IO[bytes]) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f: IO[bytes], file_size: int) -> str:
    length = _read_u64(f)
    if length > GGUF_MAX_STRING_LENGTH:
        raise _GgufParseError(f"string length {length} exceeds maximum {GGUF_MAX_STRING_LENGTH}")
    # A string can never legitimately be longer than the file itself --
    # reject before _read_exact would otherwise try to allocate up to
    # GGUF_MAX_STRING_LENGTH (1 GiB) of buffer for a length claimed by a
    # header in a file that is nowhere near that size.
    if length > file_size:
        raise _GgufParseError(f"string length {length} exceeds file size {file_size}")
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_scalar(f: IO[bytes], vtype: int, file_size: int) -> Any:
    if vtype == _STRING:
        return _read_string(f, file_size)
    if vtype not in _SCALAR_STRUCT:
        raise _GgufParseError(f"unknown scalar value type {vtype}")
    fmt, size = _SCALAR_STRUCT[vtype]
    return struct.unpack("<" + fmt, _read_exact(f, size))[0]


def _skip_array_elements(f: IO[bytes], elem_type: int, count: int, file_size: int, depth: int = 0) -> None:
    """Advance past `count` elements of `elem_type` without materializing
    them -- a real tokenizer vocabulary array is tens of thousands of
    strings, and BEE only needs a handful of scalar metadata keys, not a
    full parse of every array's contents."""
    if depth > _MAX_ARRAY_NESTING_DEPTH:
        raise _GgufParseError(f"array nesting exceeds maximum depth {_MAX_ARRAY_NESTING_DEPTH}")
    if elem_type == _STRING:
        for _ in range(count):
            length = _read_u64(f)
            if length > GGUF_MAX_STRING_LENGTH or length > file_size:
                raise _GgufParseError(f"string length {length} exceeds maximum")
            f.seek(length, 1)
    elif elem_type == _ARRAY:
        for _ in range(count):
            sub_elem_type = _read_u32(f)
            sub_count = _read_u64(f)
            if sub_count > GGUF_MAX_ARRAY_ELEMENTS:
                raise _GgufParseError(f"array length {sub_count} exceeds maximum {GGUF_MAX_ARRAY_ELEMENTS}")
            _skip_array_elements(f, sub_elem_type, sub_count, file_size, depth + 1)
    elif elem_type in _SCALAR_STRUCT:
        _, size = _SCALAR_STRUCT[elem_type]
        f.seek(size * count, 1)
    else:
        raise _GgufParseError(f"unknown array element type {elem_type}")


def _read_kv_value(f: IO[bytes], vtype: int, file_size: int) -> Any:
    if vtype == _ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        if count > GGUF_MAX_ARRAY_ELEMENTS:
            raise _GgufParseError(f"array length {count} exceeds maximum {GGUF_MAX_ARRAY_ELEMENTS}")
        _skip_array_elements(f, elem_type, count, file_size)
        return {"__array__": True, "element_type": elem_type, "length": count}
    return _read_scalar(f, vtype, file_size)


def analyze_gguf(path: Path) -> GgufAnalysis | None:
    """Parse a GGUF file's header, metadata key/value section, and tensor
    info table -- never the tensor data itself. Every declared count and
    length is bounded against the same limits llama.cpp's own reference
    reader enforces (see GGUF_MAX_ARRAY_ELEMENTS/GGUF_MAX_STRING_LENGTH
    above), so a malicious or corrupted header can't make this allocate
    or loop unboundedly just because it says to.
    """
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            magic = _read_exact(f, 4)
            if magic != GGUF_MAGIC:
                return None
            version = _read_u32(f)
            if version not in (2, 3):
                # Version 1 used 32-bit counts/lengths throughout -- a
                # different, long-obsolete layout this module doesn't parse.
                return None
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)
            if tensor_count > GGUF_MAX_ARRAY_ELEMENTS or kv_count > GGUF_MAX_ARRAY_ELEMENTS:
                return None

            metadata: dict[str, Any] = {}
            for _ in range(kv_count):
                key = _read_string(f, file_size)
                vtype = _read_u32(f)
                metadata[key] = _read_kv_value(f, vtype, file_size)

            tensors: list[GgufTensorInfo] = []
            for _ in range(tensor_count):
                name = _read_string(f, file_size)
                n_dims = _read_u32(f)
                if n_dims > 64:  # no real tensor has anywhere near this many dimensions
                    raise _GgufParseError(f"implausible dimension count {n_dims}")
                dimensions = [_read_u64(f) for _ in range(n_dims)]
                ggml_type = _read_u32(f)
                offset = _read_u64(f)
                tensors.append(
                    GgufTensorInfo(name=name, dimensions=dimensions, ggml_type=ggml_type, offset=offset)
                )

            alignment = GGUF_DEFAULT_ALIGNMENT
            align_value = metadata.get("general.alignment")
            if isinstance(align_value, int) and align_value > 0 and (align_value & (align_value - 1)) == 0:
                alignment = align_value

            data_section_start = f.tell()
            padding = data_section_start % alignment
            if padding != 0:
                data_section_start += alignment - padding

        return GgufAnalysis(
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=kv_count,
            metadata=metadata,
            tensors=tensors,
            alignment=alignment,
            tensor_data_start=data_section_start,
            file_size=file_size,
        )
    except (OSError, _GgufParseError, struct.error, UnicodeDecodeError, MemoryError, RecursionError):
        # RecursionError is a backstop, not the primary defense --
        # _skip_array_elements' own _MAX_ARRAY_NESTING_DEPTH check is
        # meant to turn a deeply-nested array into a clean
        # _GgufParseError long before Python's call stack is at risk.
        return None


def compute_tensor_byte_size(ggml_type: int, dimensions: list[int]) -> int | None:
    """Exact byte size for a tensor of `ggml_type` and `dimensions`, using
    the block-size table above. Returns None for a ggml_type the table
    doesn't cover, rather than guessing -- and stops growing the element
    count once it's already implausibly large (see
    _MAX_PLAUSIBLE_ELEMENTS) rather than continuing to multiply further
    attacker-supplied dimensions."""
    sizing = GGML_QUANT_SIZES.get(ggml_type)
    if sizing is None:
        return None
    block_size, block_bytes = sizing
    element_count = 1
    for dim in dimensions:
        if dim < 0:
            return None
        element_count *= dim
        if element_count > _MAX_PLAUSIBLE_ELEMENTS:
            return element_count
    num_blocks = -(-element_count // block_size)  # ceiling division
    return num_blocks * block_bytes
