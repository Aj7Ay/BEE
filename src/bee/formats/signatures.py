from __future__ import annotations

import json
import pickletools
import struct
import zipfile
from pathlib import Path

from bee.evidence.finding import Confidence, Evidence

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


# Bounded prefix read for pickle sniffing: real malicious payloads (a few
# opcodes constructing a small object and invoking __reduce__) are tiny —
# a few KB at most. This bound keeps detection O(1) in file size instead of
# genops potentially walking a multi-gigabyte disguised file. A legitimate
# pickle whose STOP opcode lies beyond this prefix is reported as
# "unknown" rather than misclassified — a known, documented limitation,
# not a silent false negative on the attack this detector exists for.
_PICKLE_SNIFF_BYTES = 1_048_576


def detect_pickle(path: Path) -> DetectionResult | None:
    try:
        with path.open("rb") as f:
            prefix = f.read(_PICKLE_SNIFF_BYTES)
    except OSError:
        return None
    if not prefix:
        return None

    protocol: int | None = None
    try:
        for opcode, arg, _pos in pickletools.genops(prefix):
            if opcode.name == "PROTO":
                protocol = arg
            if opcode.name == "STOP":
                break
        else:
            # Exhausted the prefix without ever seeing STOP: either this
            # isn't a pickle, or it's one bigger than our sniff bound.
            # Either way, we can't conclusively call it pickle.
            return None
    except (ValueError, EOFError, IndexError):
        return None

    # pickle.load() stops at the first STOP opcode and ignores everything
    # after it — so do we. Anything appended past this point (a trailing
    # null byte, a newline, arbitrary padding) does not change whether the
    # file executes as a pickle, and must not be able to hide it from us.
    protocol_label = f"pickle_protocol_{protocol}" if protocol is not None else "pickle_protocol_0_or_1"
    return (
        "pickle",
        Confidence.SUPPORTED,
        [Evidence(type="header_field", value=protocol_label,
                   source="local_filesystem", confidence=Confidence.SUPPORTED)],
    )


def detect_onnx(path: Path) -> DetectionResult | None:
    prefix = _read_prefix(path, 1)
    if prefix != b"\x08":
        return None
    return (
        "onnx",
        Confidence.INFERRED,
        [Evidence(type="header_field", value="protobuf_field_tag_0x08",
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
