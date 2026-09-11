import pickle
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
def test_detect_pickle_all_protocols(tmp_path, protocol):
    path = tmp_path / "model.bin"
    path.write_bytes(pickle.dumps({"x": 1}, protocol=protocol))
    fmt, confidence, _ = detect_format(path)
    assert fmt == "pickle", f"protocol {protocol} not detected"
    assert confidence == Confidence.SUPPORTED


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda payload: payload + b"\x00", id="trailing_null"),
        pytest.param(lambda payload: payload + b"\n", id="trailing_newline"),
        pytest.param(lambda payload: payload + b"\x00" * 1024, id="trailing_1kb_padding"),
        pytest.param(lambda payload: payload + b"garbage-not-a-pickle-opcode", id="trailing_garbage"),
    ],
)
def test_detect_pickle_survives_trailing_bytes(tmp_path, transform):
    # The evasion this guards against: `cat payload >> file; printf '\0' >> file`.
    # pickle.load() stops at STOP and ignores everything after it, so trailing
    # bytes must not be able to hide a pickle from the detector.
    payload = pickle.dumps({"x": 1}, protocol=4)
    path = tmp_path / "evil.safetensors"
    path.write_bytes(transform(payload))

    # Confirm the premise: this file still loads and executes as a pickle.
    assert pickle.loads(payload) == {"x": 1}

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
