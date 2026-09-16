import json
import struct
import time

from bee.core.artifact import Artifact
from bee.evidence.safetensors_bounds import check_safetensors_bounds, check_safetensors_gap
from bee.formats.safetensors_ops import analyze_safetensors
from tests.fixtures import builders


def _write_raw_safetensors(path, header: dict, data: bytes) -> None:
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)


def _artifact_for(path) -> Artifact:
    return Artifact.from_file(path)


def test_no_finding_for_well_formed_single_tensor(tmp_path):
    # 2x2 F32 tensor = 4 elements * 4 bytes = 16 bytes, exactly the range claimed.
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
        b"\x00" * 16,
    )
    artifact = _artifact_for(path)
    assert artifact.detected_format == "safetensors"
    assert check_safetensors_bounds(artifact) is None


def test_no_finding_for_multiple_non_overlapping_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {
            "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        },
        b"\x00" * 8,
    )
    artifact = _artifact_for(path)
    assert check_safetensors_bounds(artifact) is None


def test_finding_for_range_exceeding_file(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16000]}},
        b"\x00" * 16,  # file only actually has 16 bytes of tensor data
    )
    artifact = _artifact_for(path)
    finding = check_safetensors_bounds(artifact)

    assert finding is not None
    assert finding.id == "BEE-STS-001"
    assert finding.severity.value == "high"
    assert "exceeds" in finding.description


def test_finding_for_inverted_range(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [10, 5]}},
        b"\x00" * 16,
    )
    artifact = _artifact_for(path)
    finding = check_safetensors_bounds(artifact)

    assert finding is not None
    assert "invalid range" in finding.description


def test_finding_for_overlapping_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        },
        b"\x00" * 24,
    )
    artifact = _artifact_for(path)
    finding = check_safetensors_bounds(artifact)

    assert finding is not None
    assert "overlaps" in finding.description


def test_finding_for_shape_dtype_size_mismatch(tmp_path):
    # Declares a 2x2 F32 tensor (16 bytes) but claims only 8 bytes of range --
    # the byte range backing it doesn't match what the shape/dtype implies.
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 8]}},
        b"\x00" * 8,
    )
    artifact = _artifact_for(path)
    finding = check_safetensors_bounds(artifact)

    assert finding is not None
    assert "implies 16 bytes" in finding.description
    assert "claims 8" in finding.description


def test_none_for_non_safetensors_artifact(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    artifact = _artifact_for(path)
    assert check_safetensors_bounds(artifact) is None


def test_none_for_unresolved_symlink(tmp_path):
    link = tmp_path / "link.safetensors"
    artifact = Artifact.unresolved_symlink(link, "/somewhere/outside.safetensors")
    assert check_safetensors_bounds(artifact) is None


def test_overlap_detection_stays_fast_with_many_tensors(tmp_path):
    # Regression test for the O(n^2) pairwise overlap comparison this
    # replaced: a header can declare an arbitrary number of tensors, and
    # a quadratic algorithm turns tensor count directly into CPU cost an
    # attacker controls. 20,000 valid, non-overlapping tensors would take
    # a perceptible fraction of a second even at pure-Python-loop speed
    # under O(n^2) (order 2*10^8 comparisons); the O(n log n) sweep this
    # replaced it with should be near-instant.
    n = 20_000
    header = {
        f"t{i}": {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]}
        for i in range(n)
    }
    path = tmp_path / "many.safetensors"
    _write_raw_safetensors(path, header, b"\x00" * n)

    artifact = _artifact_for(path)
    start = time.monotonic()
    finding = check_safetensors_bounds(artifact)
    elapsed = time.monotonic() - start

    assert finding is None  # all valid, contiguous, non-overlapping
    assert elapsed < 2.0, f"overlap check took {elapsed:.2f}s for {n} tensors"


def test_size_mismatch_check_bounds_bignum_shape_dimension(tmp_path):
    # A shape dimension near Python's own int-string-conversion digit
    # limit (4300 by default) is still a single valid JSON integer -- it
    # must be handled by capping the running product, not by multiplying
    # an unbounded chain of such values.
    huge_dim = 10**4000
    path = tmp_path / "huge_shape.safetensors"
    _write_raw_safetensors(
        path,
        {"weight": {"dtype": "F32", "shape": [huge_dim, huge_dim], "data_offsets": [0, 16]}},
        b"\x00" * 16,
    )
    artifact = _artifact_for(path)

    start = time.monotonic()
    finding = check_safetensors_bounds(artifact)
    elapsed = time.monotonic() - start

    assert finding is not None
    assert "implausible" in finding.description
    assert elapsed < 2.0


def test_analyze_safetensors_does_not_crash_on_oversized_integer_literal(tmp_path):
    # A shape dimension with more digits than Python's int-string
    # conversion limit allows (default 4300) raises a plain ValueError
    # out of json.loads' own integer parsing -- not JSONDecodeError, and
    # must not be an unhandled crash. json.dumps() can't even construct
    # this (stringifying such an int hits the same limit), so the raw
    # JSON text is written directly.
    digits = "9" * 5000
    header_text = (
        '{"weight": {"dtype": "F32", "shape": [' + digits + '], "data_offsets": [0, 4]}}'
    ).encode("utf-8")
    path = tmp_path / "oversized_int.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_text)) + header_text + b"\x00" * 4)

    assert analyze_safetensors(path) is None


def test_gap_finding_for_unreferenced_bytes(tmp_path):
    # Real SafeTensors files (checked against a 47MB timm/resnet18
    # checkpoint and a real GPT-2 export) pack tensors contiguously with
    # zero gap -- any gap at all is worth reporting.
    path = tmp_path / "gap.safetensors"
    _write_raw_safetensors(
        path,
        {
            "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
            "b": {"dtype": "U8", "shape": [8], "data_offsets": [1000, 1008]},
        },
        b"\x00" * 1008,
    )
    artifact = _artifact_for(path)

    assert check_safetensors_bounds(artifact) is None  # both ranges are individually valid
    finding = check_safetensors_gap(artifact)

    assert finding is not None
    assert finding.id == "BEE-STS-002"
    assert finding.severity.value == "low"
    assert "992" in finding.description


def test_no_gap_finding_for_contiguous_tensors(tmp_path):
    path = tmp_path / "contiguous.safetensors"
    _write_raw_safetensors(
        path,
        {"a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}},
        b"\x00" * 8,
    )
    artifact = _artifact_for(path)
    assert check_safetensors_gap(artifact) is None


def test_no_gap_finding_when_bounds_already_flagged(tmp_path):
    # Overlap/out-of-range coverage arithmetic is unreliable -- the bounds
    # check above already reports this file, so the gap check stays quiet
    # rather than adding confusing or contradictory arithmetic on top.
    path = tmp_path / "overlapping.safetensors"
    _write_raw_safetensors(
        path,
        {
            "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
        },
        b"\x00" * 24,
    )
    artifact = _artifact_for(path)
    assert check_safetensors_gap(artifact) is None
