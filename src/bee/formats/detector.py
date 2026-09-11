from __future__ import annotations

from pathlib import Path

from bee.evidence.finding import Confidence
from bee.formats.signatures import DETECTORS, DetectionResult

_EXTENSION_MAP = {
    ".safetensors": "safetensors",
    ".gguf": "gguf",
    ".npy": "numpy",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".keras": "hdf5",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".bin": "pytorch",
    ".onnx": "onnx",
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
    ".tgz": "archive",
}


def declared_format_from_extension(path: Path) -> str:
    return _EXTENSION_MAP.get(path.suffix.lower(), "unknown")


def detect_format(path: Path) -> DetectionResult:
    for detector in DETECTORS:
        result = detector(path)
        if result is not None:
            return result
    return ("unknown", Confidence.UNKNOWN, [])
