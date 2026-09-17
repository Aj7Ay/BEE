import hashlib
import pickle

from bee.core.artifact import Artifact, compute_file_hashes, read_magic_bytes_hex
from bee.evidence.finding import Confidence
from bee.evidence.pickle_calls import check_pickle_calls
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
    # safetensors' evidence is header validity, not a fixed magic-byte
    # sequence -- nothing to derive magic_bytes_hex from.
    assert artifact.magic_bytes_hex == ""


def test_artifact_from_file_matching_formats(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    artifact = Artifact.from_file(path)
    assert artifact.declared_format == "gguf"
    assert artifact.detected_format == "gguf"
    # Exactly the 4-byte GGUF magic -- not a blanket 16-byte read that
    # happens to include it plus 12 unrelated bytes.
    assert artifact.magic_bytes_hex == "47475546"


def test_artifact_from_file_tar_does_not_leak_member_filename(tmp_path):
    # Regression test: tar's actual magic ("ustar") lives at offset 257,
    # not 0. A blind fixed-offset-0 read recorded the first archive
    # member's filename instead -- a real secret if that member happened
    # to be named something like "prod-signing-key.pem".
    path = tmp_path / "model.tar"
    builders.write_tar(path)

    artifact = Artifact.from_file(path)

    assert artifact.detected_format == "archive"
    assert artifact.magic_bytes_hex == ""


def test_artifact_from_file_pickle_has_no_magic_bytes(tmp_path):
    # pickle's evidence is its protocol number, not a fixed byte sequence
    # (protocols 0/1 have no header at all) -- nothing to show here.
    path = tmp_path / "model.pkl"
    builders.write_pickle(path)

    artifact = Artifact.from_file(path)

    assert artifact.detected_format == "pickle"
    assert artifact.magic_bytes_hex == ""


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


def test_artifact_from_file_unknown_format_has_no_magic_bytes(tmp_path):
    # Regression test: magic_bytes_hex used to record the first 16 raw
    # bytes of every scanned file unconditionally -- for anything BEE
    # couldn't classify (a .env, a token file, a README) those bytes are
    # just content, not evidence, and shouldn't be recorded at all.
    path = tmp_path / "secrets.env"
    path.write_bytes(b"AWS_SECRET_ACCESS_KEY=fake-not-a-real-key-0123456789")

    artifact = Artifact.from_file(path)

    assert artifact.detected_format == "unknown"
    assert artifact.magic_bytes_hex == ""


def test_from_file_with_content_buffers_small_files(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)

    artifact, content = Artifact.from_file_with_content(path)

    assert content is not None
    assert content == path.read_bytes()
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()


def test_from_file_with_content_returns_none_content_above_size_cap(tmp_path, monkeypatch):
    # Buffering an entire multi-gigabyte real checkpoint just to close a
    # race window would recreate the exact resource-exhaustion problem
    # this project keeps fixing elsewhere -- files above the cap keep the
    # previous per-stage re-open behavior instead.
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    monkeypatch.setattr("bee.core.artifact.TOCTOU_SAFE_MAX_BYTES", 0)

    artifact, content = Artifact.from_file_with_content(path)

    assert content is None
    assert artifact.sha256 == compute_file_hashes(path)[0]


def test_from_file_matches_from_file_with_content(tmp_path):
    # from_file is now a thin wrapper -- must produce an identical
    # Artifact record either way.
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)

    via_wrapper = Artifact.from_file(path)
    via_full, _ = Artifact.from_file_with_content(path)

    assert via_wrapper == via_full


def test_from_file_with_content_closes_toctou_between_hash_and_deep_analysis(tmp_path):
    # The regression this whole mechanism exists to close: a file
    # replaced on disk between the hash pass and pickle_calls' own deep
    # analysis used to mean the recorded hash and the content actually
    # analyzed for danger could be two different files. Buffering once
    # and reusing that buffer for both makes that impossible for a file
    # under the size cap, regardless of what happens to the path
    # afterwards.
    class _OsSystemExploit:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    path = tmp_path / "evil.pt"
    path.write_bytes(pickle.dumps(_OsSystemExploit(), protocol=4))

    artifact, content = Artifact.from_file_with_content(path)
    assert content is not None
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()

    # Swap the file out from under the recorded path -- if check_pickle_calls
    # re-opened `path` instead of using `content`, it would now be
    # analyzing this replacement, not the file that was actually hashed.
    path.write_bytes(b"not a pickle at all, just plain bytes")

    finding = check_pickle_calls(artifact, content)
    assert finding is not None
    assert finding.id == "BEE-PKL-001"  # still sees the original RCE, not the replacement

    # Without the buffer, the stale artifact (still claiming
    # detected_format="pytorch"/"pickle" from before the swap) would now
    # be pointed at content that no longer matches -- confirming the
    # swap actually took effect on disk.
    assert path.read_bytes() != content


def test_unresolved_symlink_has_no_hash_or_content(tmp_path):
    link = tmp_path / "link.gguf"
    link.symlink_to(tmp_path / "somewhere_never_opened.gguf")

    artifact = Artifact.unresolved_symlink(link, "/outside/somewhere_never_opened.gguf")

    assert artifact.is_symlink is True
    assert artifact.symlink_target == "/outside/somewhere_never_opened.gguf"
    assert artifact.sha256 == ""
    assert artifact.sha512 == ""
    assert artifact.detected_format == "unknown"
    assert artifact.format_confidence == Confidence.UNKNOWN
