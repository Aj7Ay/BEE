"""End-to-end tests for scan_source (remote vetting paths)."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from bee.scanning.orchestrator import ScanOrchestrator
from bee.scanning.config import ScanConfig


def test_scan_source_with_ollama_source():
    """Test scan_source end-to-end with mocked Ollama source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.gguf"
        model_file.write_bytes(b"fake gguf model")

        with patch("bee.sources.ollama.OllamaSource"):
            mock_source = MagicMock()
            mock_source.get_artifacts.return_value = [model_file]
            mock_source.get_source_info.return_value = {
                "provider": "ollama",
                "repository": "test:latest",
                "revision": None,
            }
            mock_source.get_model_card.return_value = None
            mock_source.source_type = "ollama"
            mock_source.url = "ollama:test:latest"

            config = ScanConfig(
                fail_on=None,
                deterministic=False,
                follow_symlinks=False,
                output_dir=None,
            )
            orchestrator = ScanOrchestrator(config)
            run = orchestrator.scan_source(mock_source)

            assert run is not None
            assert len(run.artifacts) > 0
            assert run.provenance is not None
            assert run.provenance.source.provider == "ollama"


def test_scan_source_with_huggingface_source():
    """Test scan_source with HuggingFace source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.safetensors"
        model_file.write_bytes(b"fake safetensors")

        with patch("bee.sources.huggingface.HuggingFaceSource"):
            mock_source = MagicMock()
            mock_source.get_artifacts.return_value = [model_file]
            mock_source.get_source_info.return_value = {
                "provider": "huggingface",
                "repository": "facebook/opt-350m",
                "revision": "main",
            }
            mock_source.get_model_card.return_value = {"model_name": "opt-350m"}
            mock_source.source_type = "huggingface"
            mock_source.url = "hf://facebook/opt-350m"

            config = ScanConfig(
                fail_on=None,
                deterministic=False,
                follow_symlinks=False,
                output_dir=None,
            )
            orchestrator = ScanOrchestrator(config)
            run = orchestrator.scan_source(mock_source)

            assert run is not None
            assert len(run.artifacts) > 0
            assert run.provenance is not None
            assert run.provenance.source.provider == "huggingface"
            assert run.provenance.source.repository == "facebook/opt-350m"


def test_scan_source_provenance_is_populated():
    """Test scan_source correctly builds provenance for remote sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.bin"
        model_file.write_bytes(b"fake pytorch")

        with patch("bee.sources.huggingface.HuggingFaceSource"):
            mock_source = MagicMock()
            mock_source.get_artifacts.return_value = [model_file]
            mock_source.get_source_info.return_value = {
                "provider": "huggingface",
                "repository": "org/model",
                "revision": "v1",
            }
            mock_source.get_model_card.return_value = None
            mock_source.source_type = "huggingface"
            mock_source.url = "hf://org/model"

            config = ScanConfig(
                fail_on=None,
                deterministic=False,
                follow_symlinks=False,
                output_dir=None,
            )
            orchestrator = ScanOrchestrator(config)
            run = orchestrator.scan_source(mock_source)

            assert run.provenance is not None
            assert run.provenance.source.provider == "huggingface"
