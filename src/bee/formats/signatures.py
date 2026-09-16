from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

from bee.evidence.finding import Confidence, Evidence
from bee.formats.pickle_ops import analyze_pickle_file

DetectionResult = tuple[str, Confidence, list[Evidence]]

_PREFIX_READ_SIZE = 32


def _read_prefix(path: Path, size: int = _PREFIX_READ_SIZE) -> bytes:
    with path.open("rb") as f:
        return f.read(size)


def detect_safetensors(path: Path) -> DetectionResult | None:
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
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return (
        "safetensors",
        Confidence.SUPPORTED,
        [Evidence(type="header_field", value="valid_safetensors_json_header",
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_gguf(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 4)
    if prefix != b"GGUF":
        return None
    return (
        "gguf",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_numpy(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 6)
    if prefix != b"\x93NUMPY":
        return None
    return (
        "numpy",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_hdf5(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 8)
    if prefix != b"\x89HDF\r\n\x1a\n":
        return None
    return (
        "hdf5",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_zip_based(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 4)
    if prefix != b"PK\x03\x04":
        return None
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return None
    if any(name == "data.pkl" or name.endswith("/data.pkl") for name in names):
        return (
            "pytorch",
            Confidence.SUPPORTED,
            [Evidence(type="archive_member", value="data.pkl",
                       source="local_filesystem", confidence=Confidence.SUPPORTED)],
        )
    return (
        "archive",
        Confidence.SUPPORTED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_gzip(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 2)
    if prefix != b"\x1f\x8b":
        return None
    return (
        "archive",
        Confidence.SUPPORTED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_tar(path: Path) -> DetectionResult | None:
    try:
        with path.open("rb") as f:
            f.seek(257)
            magic = f.read(5)
    except OSError:
        return None
    if magic != b"ustar":
        return None
    return (
        "archive",
        Confidence.SUPPORTED,
        [Evidence(type="header_field", value="ustar_magic",
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_pickle(path: Path) -> DetectionResult | None:
    # Shared with the pickle call-graph analysis (bee.evidence.pickle_calls)
    # -- one opcode-walking implementation, not a second copy that could
    # drift from it. See bee.formats.pickle_ops for why this streams from
    # the file handle bounded by opcode count rather than a byte prefix:
    # an earlier fixed-prefix version reintroduced the exact evasion this
    # detector exists to close.
    analysis = analyze_pickle_file(path)
    if analysis is None:
        return None

    # pickle.load() stops at the first STOP opcode and ignores everything
    # after it — so do we. Anything appended past this point (a trailing
    # null byte, a newline, arbitrary padding) does not change whether the
    # file executes as a pickle, and must not be able to hide it from us.
    protocol_label = (
        f"pickle_protocol_{analysis.protocol}" if analysis.protocol is not None
        else "pickle_protocol_0_or_1"
    )
    return (
        "pickle",
        Confidence.SUPPORTED,
        [Evidence(type="header_field", value=protocol_label,
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def _read_varint(data: bytes, pos: int) -> tuple[int, int] | None:
    """Read a protobuf varint starting at `pos`. Returns (value, next_pos),
    or None if the bytes at `pos` don't form a valid, terminated varint
    within protobuf's 10-byte limit."""
    result = 0
    shift = 0
    start = pos
    while True:
        if pos >= len(data) or pos - start >= 10:
            return None
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1
        if not (byte & 0x80):
            return result, pos
        shift += 7


_ONNX_SNIFF_BYTES = 4096


def _looks_like_protobuf(data: bytes, min_fields: int = 2) -> bool:
    """Generic structural validation of a protobuf byte stream: walk
    `min_fields` consecutive tag/value pairs, checking that each tag decodes
    to a plausible field number and a known wire type, and that
    length-delimited fields declare a length that actually fits. This is
    deliberately schema-agnostic (not specific to ONNX's field numbers) —
    it rejects the overwhelming majority of non-protobuf byte sequences
    without asserting exact knowledge of ONNX's proto definition.
    """
    pos = 0
    for _ in range(min_fields):
        tag = _read_varint(data, pos)
        if tag is None:
            return False
        tag_value, pos = tag
        field_number = tag_value >> 3
        wire_type = tag_value & 0x7
        if field_number == 0 or wire_type not in (0, 1, 2, 5):
            return False
        if wire_type == 0:  # varint
            value = _read_varint(data, pos)
            if value is None:
                return False
            _, pos = value
        elif wire_type == 1:  # 64-bit
            pos += 8
        elif wire_type == 5:  # 32-bit
            pos += 4
        else:  # wire_type == 2: length-delimited
            length_info = _read_varint(data, pos)
            if length_info is None:
                return False
            length, pos = length_info
            if length < 0 or pos + length > len(data):
                return False
            pos += length
        if pos > len(data):
            return False
    return True


def detect_onnx(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, _ONNX_SNIFF_BYTES)
    if not prefix or prefix[0] != 0x08:
        return None
    if not _looks_like_protobuf(prefix):
        return None
    return (
        "onnx",
        Confidence.INFERRED,
        [Evidence(type="header_field", value="protobuf_structure_2_fields",
                   source="local_filesystem", confidence=Confidence.INFERRED)],
    )


DETECTORS: list = [
    detect_safetensors,
    detect_gguf,
    detect_numpy,
    detect_hdf5,
    detect_zip_based,
    detect_gzip,
    detect_tar,
    detect_pickle,
    detect_onnx,
]
