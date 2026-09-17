from __future__ import annotations

import struct
import zipfile

from bee.evidence.finding import Confidence, Evidence
from bee.formats.io_source import Source, open_source
from bee.formats.pickle_ops import analyze_pickle_file
from bee.formats.safetensors_ops import analyze_safetensors

DetectionResult = tuple[str, Confidence, list[Evidence]]

_PREFIX_READ_SIZE = 32


def _read_prefix(source: Source, size: int = _PREFIX_READ_SIZE) -> bytes:
    with open_source(source) as f:
        return f.read(size)


def detect_safetensors(source: Source) -> DetectionResult | None:
    # Shared with the bounds-checking finding (bee.evidence.safetensors_bounds)
    # -- one header-parsing implementation, not a second copy that could
    # drift from it.
    if analyze_safetensors(source) is None:
        return None
    return (
        "safetensors",
        Confidence.SUPPORTED,
        [Evidence(type="header_field", value="valid_safetensors_json_header",
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_gguf(source: Source) -> DetectionResult | None:
    # Deliberately a shallow magic-only check, not upgraded to require a
    # full bee.formats.gguf_ops.analyze_gguf() parse the way safetensors
    # requires a full header parse: "GGUF" is a 4-byte literal match with
    # essentially no collision risk, unlike safetensors' numeric-header
    # heuristic, which does need the deeper validation to mean anything.
    # Structural validation (and the findings it can produce) lives in
    # bee.evidence.gguf_bounds, independent of classification here.
    prefix = _read_prefix(source, 4)
    if prefix != b"GGUF":
        return None
    return (
        "gguf",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_numpy(source: Source) -> DetectionResult | None:
    prefix = _read_prefix(source, 6)
    if prefix != b"\x93NUMPY":
        return None
    return (
        "numpy",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_hdf5(source: Source) -> DetectionResult | None:
    prefix = _read_prefix(source, 8)
    if prefix != b"\x89HDF\r\n\x1a\n":
        return None
    return (
        "hdf5",
        Confidence.VERIFIED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.VERIFIED)],
    )


def detect_zip_based(source: Source) -> DetectionResult | None:
    prefix = _read_prefix(source, 4)
    if prefix != b"PK\x03\x04":
        return None
    try:
        with zipfile.ZipFile(open_source(source)) as zf:
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


def detect_gzip(source: Source) -> DetectionResult | None:
    prefix = _read_prefix(source, 2)
    if prefix != b"\x1f\x8b":
        return None
    return (
        "archive",
        Confidence.SUPPORTED,
        [Evidence(type="magic_bytes", value=prefix.hex(),
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_tar(source: Source) -> DetectionResult | None:
    try:
        with open_source(source) as f:
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


def detect_pickle(source: Source) -> DetectionResult | None:
    # Shared with the pickle call-graph analysis (bee.evidence.pickle_calls)
    # -- one opcode-walking implementation, not a second copy that could
    # drift from it. See bee.formats.pickle_ops for why this streams from
    # the file handle bounded by opcode count rather than a byte prefix:
    # an earlier fixed-prefix version reintroduced the exact evasion this
    # detector exists to close.
    analysis = analyze_pickle_file(source)
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
    if analysis.opcode_cap_hit:
        # STOP was never reached within the opcode cap -- everything past
        # it (including a REDUCE that would call a dangerous primitive)
        # was never inspected. This is still structurally a pickle, not
        # "unknown": returning None here would let padding past the cap
        # evade detection entirely, the same evasion a fixed byte-prefix
        # read already closed once for this format (see analyze_pickle_file).
        return (
            "pickle",
            Confidence.INFERRED,
            [Evidence(type="header_field", value=f"{protocol_label}_opcode_cap_exceeded",
                       source="local_filesystem", confidence=Confidence.INFERRED)],
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


def detect_onnx(source: Source) -> DetectionResult | None:
    prefix = _read_prefix(source, _ONNX_SNIFF_BYTES)
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
