import json
import struct

from bee.core.artifact import Artifact
from bee.evidence.safetensors_bounds import check_safetensors_bounds
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
