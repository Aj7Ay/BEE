from bee.evidence.symlink import build_symlink_escape_finding, is_escaping_symlink
from tests.fixtures import builders


def test_none_for_non_symlink(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    assert is_escaping_symlink(path, scan_root=tmp_path) is None


def test_none_when_symlink_target_is_inside_scan_root(tmp_path):
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    real = scan_root / "real.gguf"
    builders.write_gguf(real)
    link = scan_root / "link.gguf"
    link.symlink_to(real)

    assert is_escaping_symlink(link, scan_root=scan_root) is None


def test_target_returned_when_symlink_escapes_scan_root(tmp_path):
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    link = scan_root / "link.gguf"
    link.symlink_to(outside)

    result = is_escaping_symlink(link, scan_root=scan_root)

    assert result == str(outside.resolve())


def test_is_escaping_symlink_never_opens_the_target(tmp_path):
    # The whole point of the fix: this check must be answerable from the
    # symlink alone, without ever reading what it points to. A target that
    # doesn't exist (or isn't readable) must not raise.
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    link = scan_root / "dangling.gguf"
    link.symlink_to(tmp_path / "does_not_exist.gguf")

    result = is_escaping_symlink(link, scan_root=scan_root)

    assert result == str((tmp_path / "does_not_exist.gguf").resolve())


def test_build_finding_notes_content_was_not_read_by_default():
    finding = build_symlink_escape_finding("link.gguf", "/etc/shadow", content_read=False)
    assert finding.id == "BEE-SYM-001"
    assert finding.severity.value == "high"
    assert "was not read" in finding.description


def test_build_finding_notes_content_was_read_when_followed():
    finding = build_symlink_escape_finding("link.gguf", "/etc/shadow", content_read=True)
    assert "read anyway" in finding.description
