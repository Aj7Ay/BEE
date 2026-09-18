"""Adversarial tests for BEE's GGUF parser and BEE-GGUF-001 finding.

Each test in this file targets one specific check the code already
claims to make (see the cited line in bee/formats/gguf_ops.py or
bee/evidence/gguf_bounds.py) but that had no test proving it actually
works. This file does not add new checks -- it proves the existing
ones hold, or documents that they do not.
"""

import struct

import pytest

from bee.core.artifact import Artifact
from bee.evidence.gguf_bounds import check_gguf_bounds
from bee.formats.gguf_ops import analyze_gguf
from tests.test_gguf import _GGML_TYPE_F32, _UINT32, _build_gguf


def _artifact_for(path) -> Artifact:
    return Artifact.from_file(path)


def test_tensor_with_more_than_64_dimensions_is_rejected(tmp_path):
    # gguf_ops.py: "if n_dims > 64: raise _GgufParseError(...)". No real
    # tensor has anywhere near 64 dimensions -- a file that declares 65
    # must be treated as unparseable, not silently truncated or crashed on.
    path = tmp_path / "too_many_dims.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[("weight", [1] * 65, _GGML_TYPE_F32, 0)],
            tensor_data=b"\x00" * 32,
        )
    )
    assert analyze_gguf(path) is None


def test_tensor_with_exactly_64_dimensions_still_parses(tmp_path):
    # The boundary case: 64 is allowed, 65 is not. A cap that silently
    # rejects the boundary value itself would be its own (smaller) bug.
    path = tmp_path / "max_dims.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[("weight", [1] * 64, _GGML_TYPE_F32, 0)],
            tensor_data=b"\x00" * 32,
        )
    )
    analysis = analyze_gguf(path)
    assert analysis is not None
    assert len(analysis.tensors[0].dimensions) == 64


def test_non_power_of_two_alignment_metadata_falls_back_to_default(tmp_path):
    # gguf_ops.py only honors general.alignment when it is a positive
    # power of two (the only values a real GGUF loader would treat as a
    # valid alignment). 7 is neither -- the parser must fall back to the
    # documented default of 32, not divide by 7 or crash.
    #
    # general.alignment is stored as a UINT32 value, not a string, so
    # this builds the raw KV bytes directly instead of using
    # _pack_string_kv (which only packs string-typed values).
    key_bytes = "general.alignment".encode("utf-8")
    raw_kv = (
        struct.pack("<Q", len(key_bytes)) + key_bytes
        + struct.pack("<I", _UINT32)
        + struct.pack("<I", 7)
    )
    path = tmp_path / "bad_alignment.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[("weight", [4], _GGML_TYPE_F32, 0)],
            raw_kvs=[raw_kv],
            tensor_data=b"\x00" * 16,
        )
    )
    analysis = analyze_gguf(path)
    assert analysis is not None
    assert analysis.alignment == 32


def test_unknown_ggml_type_does_not_suppress_the_offset_check(tmp_path):
    # An unrecognized ggml_type means compute_tensor_byte_size can't
    # validate the tensor's *size* -- but the offset/alignment checks in
    # _tensor_problem run BEFORE the size check, so a bad offset must
    # still be caught even when the type is unknown.
    path = tmp_path / "unknown_type_bad_offset.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[("weight", [4], 9999, 1_000_000)],  # 9999: not in GGML_QUANT_SIZES
            tensor_data=b"\x00" * 16,
        )
    )
    artifact = _artifact_for(path)
    finding = check_gguf_bounds(artifact)
    assert finding is not None
    assert "past the end" in finding.description


def test_gguf_tensor_offset_field_cannot_be_negative_on_the_wire():
    # gguf_bounds.py checks `tensor.offset < 0`, but offset is read from
    # the file with _read_u64 (an unsigned 64-bit read) -- there is no
    # sequence of real bytes that produces a negative Python int here.
    # This test documents that the check is dead code, not a live
    # defense: struct.pack itself refuses to encode a negative value
    # into the same field format GGUF uses on the wire.
    with pytest.raises(struct.error):
        struct.pack("<Q", -1)