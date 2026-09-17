import struct

from bee.core.artifact import Artifact
from bee.evidence.gguf_bounds import check_gguf_bounds
from bee.formats.gguf_ops import analyze_gguf, compute_tensor_byte_size
from tests.fixtures import builders

_STRING = 8
_UINT32 = 4

_GGML_TYPE_F32 = 0  # 1 element/block, 4 bytes/block -- plain float32


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_string_kv(key: str, value: str) -> bytes:
    return _pack_string(key) + struct.pack("<I", _STRING) + _pack_string(value)


def _pack_tensor_info(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    return (
        _pack_string(name)
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<I", ggml_type)
        + struct.pack("<Q", offset)
    )


def _build_gguf(
    tensors: list[tuple[str, list[int], int, int]],
    kvs: list[tuple[str, str]] | None = None,
    version: int = 3,
    alignment: int = 32,
    tensor_data: bytes | None = None,
) -> bytes:
    """Assemble a real, byte-accurate GGUF file: header, metadata KV
    section, tensor info table, alignment padding, then tensor data --
    matching bee.formats.gguf_ops.analyze_gguf's own layout expectations,
    not a simplified stand-in for them."""
    kvs = kvs or []
    kv_section = b"".join(_pack_string_kv(k, v) for k, v in kvs)
    tensor_section = b"".join(_pack_tensor_info(*t) for t in tensors)

    header = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kvs))
        + kv_section
        + tensor_section
    )
    padding_needed = (-len(header)) % alignment
    body = header + b"\x00" * padding_needed
    if tensor_data is not None:
        body += tensor_data
    return body


def _artifact_for(path) -> Artifact:
    return Artifact.from_file(path)


def test_analyze_gguf_parses_valid_file(tmp_path):
    data = _build_gguf(
        tensors=[("weight", [4], _GGML_TYPE_F32, 0)],
        kvs=[("general.architecture", "llama")],
        tensor_data=b"\x00" * 16,
    )
    path = tmp_path / "model.gguf"
    path.write_bytes(data)

    analysis = analyze_gguf(path)
    assert analysis is not None
    assert analysis.version == 3
    assert analysis.tensor_count == 1
    assert analysis.metadata_kv_count == 1
    assert analysis.metadata["general.architecture"] == "llama"
    assert analysis.tensors[0].name == "weight"


def test_analyze_gguf_returns_none_for_bad_magic(tmp_path):
    path = tmp_path / "not_gguf.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 20)
    assert analyze_gguf(path) is None


def test_analyze_gguf_returns_none_for_truncated_header(tmp_path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x00" * 3)  # cut mid tensor_count
    assert analyze_gguf(path) is None


def test_compute_tensor_byte_size_f32():
    # 2x2 F32 tensor = 4 elements * 4 bytes/element = 16 bytes.
    assert compute_tensor_byte_size(_GGML_TYPE_F32, [2, 2]) == 16


def test_compute_tensor_byte_size_unknown_type_returns_none():
    assert compute_tensor_byte_size(9999, [4]) is None


def test_no_finding_for_well_formed_gguf(tmp_path):
    data = _build_gguf(
        tensors=[("weight", [4], _GGML_TYPE_F32, 0)],
        tensor_data=b"\x00" * 16,
    )
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    artifact = _artifact_for(path)
    assert artifact.detected_format == "gguf"
    assert check_gguf_bounds(artifact) is None


def test_finding_for_tensor_data_exceeding_file(tmp_path):
    # Declares 16 bytes of tensor data but the file is truncated to 4 --
    # the real-world signature of an incomplete download.
    data = _build_gguf(
        tensors=[("weight", [4], _GGML_TYPE_F32, 0)],
        tensor_data=b"\x00" * 4,
    )
    path = tmp_path / "truncated.gguf"
    path.write_bytes(data)
    artifact = _artifact_for(path)
    finding = check_gguf_bounds(artifact)

    assert finding is not None
    assert finding.id == "BEE-GGUF-001"
    assert finding.severity.value == "high"
    assert "exceeds" in finding.description


def test_finding_for_misaligned_offset(tmp_path):
    data = _build_gguf(
        tensors=[("weight", [4], _GGML_TYPE_F32, 3)],  # not a multiple of alignment
        tensor_data=b"\x00" * 32,
    )
    path = tmp_path / "misaligned.gguf"
    path.write_bytes(data)
    artifact = _artifact_for(path)
    finding = check_gguf_bounds(artifact)

    assert finding is not None
    assert "not a multiple of the declared alignment" in finding.description


def test_finding_for_offset_past_end_of_file(tmp_path):
    data = _build_gguf(
        tensors=[("weight", [4], _GGML_TYPE_F32, 1_000_000)],
        tensor_data=b"\x00" * 16,
    )
    path = tmp_path / "past_end.gguf"
    path.write_bytes(data)
    artifact = _artifact_for(path)
    finding = check_gguf_bounds(artifact)

    assert finding is not None
    assert "past the end" in finding.description


def test_none_for_non_gguf_artifact(tmp_path):
    path = tmp_path / "model.safetensors"
    builders.write_safetensors(path)
    artifact = _artifact_for(path)
    assert check_gguf_bounds(artifact) is None


def test_none_for_unresolved_symlink(tmp_path):
    link = tmp_path / "link.gguf"
    artifact = Artifact.unresolved_symlink(link, "/somewhere/outside.gguf")
    assert check_gguf_bounds(artifact) is None


def test_none_for_minimal_synthetic_fixture(tmp_path):
    # tests/fixtures/builders.py's write_gguf() writes only a bare 16-byte
    # magic+version+zeroed-counts header -- enough for shallow detection
    # (detect_gguf) but not a real tensor_count/kv_count header analyze_gguf
    # can parse. check_gguf_bounds must degrade to "nothing to report", not
    # crash or fabricate a finding, when the deep parse fails.
    path = tmp_path / "minimal.gguf"
    builders.write_gguf(path)
    artifact = _artifact_for(path)
    assert artifact.detected_format == "gguf"
    assert check_gguf_bounds(artifact) is None
