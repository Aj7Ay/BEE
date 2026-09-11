from pathlib import Path

from bee.evidence.finding import Confidence
from bee.formats.detector import declared_format_from_extension, detect_format
from tests.fixtures import builders


def test_detect_safetensors(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_safetensors(path)
    fmt, confidence, evidence = detect_format(path)
    assert fmt == "safetensors"
    assert confidence == Confidence.SUPPORTED
    assert evidence


def test_detect_gguf(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_gguf(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "gguf"
    assert confidence == Confidence.VERIFIED


def test_detect_numpy(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_numpy(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "numpy"
    assert confidence == Confidence.VERIFIED


def test_detect_hdf5(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_hdf5(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "hdf5"
    assert confidence == Confidence.VERIFIED


def test_detect_pytorch_zip(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_pytorch_zip(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "pytorch"
    assert confidence == Confidence.SUPPORTED


def test_detect_plain_zip_is_generic_archive(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_plain_zip(path)
    fmt, _, _ = detect_format(path)
    assert fmt == "archive"


def test_detect_gzip(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_gzip(path)
    fmt, _, _ = detect_format(path)
    assert fmt == "archive"


def test_detect_tar(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_tar(path)
    fmt, _, _ = detect_format(path)
    assert fmt == "archive"


def test_detect_pickle(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_pickle(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "pickle"
    assert confidence == Confidence.SUPPORTED


def test_detect_onnx_heuristic(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_onnx_like(path)
    fmt, confidence, _ = detect_format(path)
    assert fmt == "onnx"
    assert confidence == Confidence.INFERRED


def test_detect_unknown(tmp_path):
    path = tmp_path / "model.bin"
    builders.write_unknown(path)
    fmt, confidence, evidence = detect_format(path)
    assert fmt == "unknown"
    assert confidence == Confidence.UNKNOWN
    assert evidence == []


def test_declared_format_from_extension():
    assert declared_format_from_extension(Path("model.safetensors")) == "safetensors"
    assert declared_format_from_extension(Path("model.gguf")) == "gguf"
    assert declared_format_from_extension(Path("model.pt")) == "pytorch"
    assert declared_format_from_extension(Path("model.xyz")) == "unknown"
