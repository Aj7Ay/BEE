import hashlib

from bee.core.artifact import Artifact, compute_file_hashes, read_magic_bytes_hex
from bee.evidence.finding import Confidence
from tests.fixtures import builders


def test_compute_file_hashes_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello world")
    sha256, sha512 = compute_file_hashes(path)
    assert sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert sha512 == hashlib.sha512(b"hello world").hexdigest()


def test_read_magic_bytes_hex(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\x01\x02\x03\x04")
    assert read_magic_bytes_hex(path, length=4) == "01020304"


def test_read_magic_bytes_hex_short_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"\xff")
    assert read_magic_bytes_hex(path, length=16) == "ff"


def test_artifact_from_file_safetensors(tmp_path):
    path = tmp_path / "model.pt"  # declared "pytorch" by extension
    builders.write_safetensors(path)  # but structurally safetensors
    artifact = Artifact.from_file(path)
    assert artifact.path == str(path)
    assert artifact.size == path.stat().st_size
    assert artifact.declared_format == "pytorch"
    assert artifact.detected_format == "safetensors"
    assert artifact.format_confidence == Confidence.SUPPORTED
    assert len(artifact.magic_bytes_hex) > 0


def test_artifact_from_file_matching_formats(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    artifact = Artifact.from_file(path)
    assert artifact.declared_format == "gguf"
    assert artifact.detected_format == "gguf"


def test_artifact_from_file_regular_file_is_not_a_symlink(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    artifact = Artifact.from_file(path)
    assert artifact.is_symlink is False
    assert artifact.symlink_target is None


def test_artifact_from_file_records_symlink_target(tmp_path):
    real = tmp_path / "real.gguf"
    builders.write_gguf(real)
    link = tmp_path / "link.gguf"
    link.symlink_to(real)

    artifact = Artifact.from_file(link)

    assert artifact.is_symlink is True
    assert artifact.symlink_target == str(real.resolve())
    # Identity is still computed from the target's actual content.
    assert artifact.detected_format == "gguf"
