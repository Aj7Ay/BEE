"""Integration tests for the scanning orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from bee.scanning.orchestrator import ScanOrchestrator
from bee.scanning.config import ScanConfig


def test_orchestrator_dependencies_appear_in_output():
    """Test that parsed dependencies are included in the Run output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)

        # Create a requirements.txt with known packages
        req_file = target / "requirements.txt"
        req_file.write_text("requests==2.25.0\n")

        config = ScanConfig(
            fail_on=None,
            deterministic=False,
            follow_symlinks=False,
            output_dir=None,
        )
        orchestrator = ScanOrchestrator(config)
        run = orchestrator.scan_local(target, workspace_dir=target.parent)

        # Verify dependencies are in the output
        assert len(run.dependencies) > 0, "Dependencies should be parsed and included in output"
        assert any(d.get("package") == "requests" for d in run.dependencies), \
            "requests package should be in dependencies"


def test_orchestrator_code_findings_appear_in_output():
    """Test that code scanning findings are included in the Run output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)

        # Create a Python file with dangerous code
        code_file = target / "inference.py"
        code_file.write_text("import os\nos.system('echo vulnerable')\n")

        config = ScanConfig(
            fail_on=None,
            deterministic=False,
            follow_symlinks=False,
            output_dir=None,
        )
        orchestrator = ScanOrchestrator(config)
        run = orchestrator.scan_local(target, workspace_dir=target.parent)

        # Verify code findings are in the output
        code_findings = [f for f in run.findings if f.id.startswith("BEE-CODE")]
        assert len(code_findings) > 0, "Code findings should be detected"


def test_orchestrator_multiple_finding_types_with_mocked_osv():
    """Test orchestrator with both code and dependency issues (OSV mocked)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)

        # Create dangerous code
        code_file = target / "loader.py"
        code_file.write_text("import subprocess\nsubprocess.call(['id'])\n")

        # Create requirements with packages
        req_file = target / "requirements.txt"
        req_file.write_text("requests==2.25.0\n")

        config = ScanConfig(
            fail_on=None,
            deterministic=False,
            follow_symlinks=False,
            output_dir=None,
        )
        orchestrator = ScanOrchestrator(config)

        # Mock OSV vulnerability lookup to avoid network calls
        with patch("bee.evidence.vulnerability.lookup_vulnerabilities_for_deps") as mock_osv:
            mock_osv.return_value = []  # No vulnerabilities for offline test

            run = orchestrator.scan_local(target, workspace_dir=target.parent)

        # Verify code findings appear
        code_findings = [f for f in run.findings if f.id.startswith("BEE-CODE")]

        assert len(code_findings) > 0, "Code findings should be detected"
        assert len(run.dependencies) > 0, "Dependencies should be recorded"
