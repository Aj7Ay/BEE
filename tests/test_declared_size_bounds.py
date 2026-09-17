"""Cross-parser audit: every place BEE reads an attacker-supplied length
or count must (a) bound it against the actual/declared file size before
acting on it, and (b) never crash the process if it doesn't hold up --
degrading to "not parsed"/"suspicious" instead. This module exists
because that exact bug shape has recurred across every binary parser in
this project (pickle's opcode-count cap in 0.1.1, SafeTensors' O(n^2)
overlap check, GGUF's unbounded array-nesting recursion in 0.7.0, and
pickle's opcode declared-length in 0.9.0/0.9.1) -- one declared-size bomb
here should catch the whole family on arrival for a new parser, instead
of being found one release at a time.

Every test in this module must finish fast (well under the timeout) and
must never raise -- a hang or an uncaught exception is itself the
failure being tested for, independent of any specific assertion.
"""

from __future__ import annotations

import struct
import time
import zipfile

from bee.formats.gguf_ops import (
    GGUF_MAX_ARRAY_ELEMENTS,
    GGUF_MAX_STRING_LENGTH,
    analyze_gguf,
)
from bee.formats.pickle_ops import (
    MAX_PICKLE_STREAM_BYTES,
    analyze_pickle_file,
    analyze_pytorch_zip_pickle,
)
from bee.formats.safetensors_ops import analyze_safetensors

_FAST = 2.0  # seconds -- generous, but a real declared-size bomb blows well past this


def _timed(fn, *args):
    start = time.monotonic()
    result = fn(*args)
    elapsed = time.monotonic() - start
    assert elapsed < _FAST, f"{fn.__name__} took {elapsed:.2f}s -- declared-size bound not enforced"
    return result


# ---------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------


def test_gguf_declared_string_length_far_exceeds_tiny_file(tmp_path):
    path = tmp_path / "bomb.gguf"
    # magic + version(3) + tensor_count(0) + kv_count(1), then a key whose
    # declared length is under the hard cap but far past the file's own size.
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
        + struct.pack("<Q", GGUF_MAX_STRING_LENGTH - 1)
    )
    assert _timed(analyze_gguf, path) is None


def test_gguf_declared_tensor_count_at_hard_cap_boundary(tmp_path):
    path = tmp_path / "bomb.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3)
        + struct.pack("<Q", GGUF_MAX_ARRAY_ELEMENTS + 1)  # over the cap
        + struct.pack("<Q", 0)
    )
    assert _timed(analyze_gguf, path) is None


def test_gguf_deeply_nested_array_bounded_by_depth_not_just_length(tmp_path):
    # See tests/test_gguf.py for the full-detail version of this
    # regression; kept here too so the whole declared-size family is
    # visible in one module.
    from tests.test_gguf import _pack_nested_array_kv, _build_gguf

    path = tmp_path / "bomb.gguf"
    path.write_bytes(_build_gguf(tensors=[], raw_kvs=[_pack_nested_array_kv("evil", depth=5000)]))
    assert _timed(analyze_gguf, path) is None


# ---------------------------------------------------------------------------
# SafeTensors
# ---------------------------------------------------------------------------


def test_safetensors_declared_header_length_exceeds_file_size(tmp_path):
    path = tmp_path / "bomb.safetensors"
    # A header_len declaring far more than the file actually has left.
    path.write_bytes(struct.pack("<Q", 8 * 1024**4) + b"{}")  # 8 TiB claimed, ~10 bytes real
    assert _timed(analyze_safetensors, path) is None


def test_safetensors_declared_header_length_near_u64_max(tmp_path):
    path = tmp_path / "bomb.safetensors"
    path.write_bytes(struct.pack("<Q", 2**63) + b"{}")
    assert _timed(analyze_safetensors, path) is None


# ---------------------------------------------------------------------------
# Pickle (raw stream and PyTorch zip-wrapped)
# ---------------------------------------------------------------------------


def test_pickle_declared_opcode_length_exceeds_tiny_file(tmp_path):
    # BINBYTES declaring ~4 GiB on a 7-byte file. This is inert (not a
    # working exploit -- pickle.load() fails on it identically), so
    # "no findings" is correct here, not a gap; what this test actually
    # guards is that it resolves fast and without raising.
    path = tmp_path / "bomb.pkl"
    path.write_bytes(b"\x80\x04B" + (0xFFFFFFFE).to_bytes(4, "little"))
    assert _timed(analyze_pickle_file, path) is None


def test_pickle_declared_opcode_length_near_u64_max(tmp_path):
    path = tmp_path / "bomb.pkl"
    path.write_bytes(b"\x80\x04\x8d" + (2**63).to_bytes(8, "little"))  # BINUNICODE8
    assert _timed(analyze_pickle_file, path) is None


def test_pickle_stream_over_byte_cap_rejected_without_reading_it(tmp_path):
    path = tmp_path / "huge.pkl"
    path.write_bytes(b"\x00" * (MAX_PICKLE_STREAM_BYTES + 1024))
    analysis = _timed(analyze_pickle_file, path)
    assert analysis is not None
    assert analysis.opcode_cap_hit is True


def test_pytorch_zip_member_declared_size_over_cap_not_decompressed(tmp_path):
    path = tmp_path / "bomb.pt"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data.pkl", b"\x00" * (MAX_PICKLE_STREAM_BYTES + 1024))
        zf.writestr("archive/version", "3")
    analysis = _timed(analyze_pytorch_zip_pickle, path)
    assert analysis is not None
    assert analysis.opcode_cap_hit is True


# ---------------------------------------------------------------------------
# Generic zip/archive detection: never opens member content at all, so a
# declared-size bomb in a member's metadata can't reach it regardless.
# ---------------------------------------------------------------------------


def test_zip_detection_never_reads_member_with_huge_declared_size(tmp_path):
    from bee.formats.detector import detect_format

    path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.bin", b"\x00" * (10 * 1024 * 1024))  # real but modest; compresses tiny
    fmt, _, _ = _timed(detect_format, path)
    assert fmt == "archive"
