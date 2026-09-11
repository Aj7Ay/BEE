from __future__ import annotations

import gzip
import json
import pickle
import struct
import tarfile
import zipfile
from pathlib import Path


def write_safetensors(path: Path) -> None:
    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def write_gguf(path: Path) -> None:
    path.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 8)


def write_numpy(path: Path) -> None:
    path.write_bytes(b"\x93NUMPY" + b"\x01\x00" + b"\x00" * 16)


def write_hdf5(path: Path) -> None:
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 16)


def write_pickle(path: Path) -> None:
    path.write_bytes(pickle.dumps({"a": 1}, protocol=4))


def write_pytorch_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"a": 1}, protocol=4))
        zf.writestr("archive/version", "3")


def write_plain_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "hello")


def write_gzip(path: Path) -> None:
    with gzip.open(path, "wb") as f:
        f.write(b"hello")


def write_tar(path: Path) -> None:
    member_path = path.parent / "tar_member.txt"
    member_path.write_text("hello")
    with tarfile.open(path, "w") as tf:
        tf.add(member_path, arcname="hello.txt")
    member_path.unlink()


def write_onnx_like(path: Path) -> None:
    path.write_bytes(b"\x08\x07\x12\x04test")


def write_unknown(path: Path) -> None:
    path.write_bytes(b"just some plain text content, nothing special")
