from bee.core.artifact import Artifact
from bee.evidence.symlink import check_symlink_escape
from tests.fixtures import builders


def test_no_finding_for_non_symlink(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)
    artifact = Artifact.from_file(path)
    assert check_symlink_escape(artifact, scan_root=tmp_path) is None


def test_no_finding_when_symlink_target_is_inside_scan_root(tmp_path):
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    real = scan_root / "real.gguf"
    builders.write_gguf(real)
    link = scan_root / "link.gguf"
    link.symlink_to(real)

    artifact = Artifact.from_file(link)

    assert check_symlink_escape(artifact, scan_root=scan_root) is None


def test_finding_when_symlink_target_escapes_scan_root(tmp_path):
    scan_root = tmp_path / "models"
    scan_root.mkdir()
    outside = tmp_path / "outside.gguf"
    builders.write_gguf(outside)
    link = scan_root / "link.gguf"
    link.symlink_to(outside)

    artifact = Artifact.from_file(link)
    finding = check_symlink_escape(artifact, scan_root=scan_root)

    assert finding is not None
    assert finding.id == "BEE-SYM-001"
    assert finding.severity.value == "high"
    assert finding.artifact_path == artifact.path
    assert str(outside.resolve()) in finding.description
