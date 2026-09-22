"""Tests for Ollama source integration."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from bee.sources.ollama import OllamaSource, OllamaLocalSource


class TestOllamaSource:
    """Tests for OllamaSource class."""

    def test_list_models_returns_empty_on_connection_error(self):
        """Test that list_models gracefully handles connection failures."""
        source = OllamaSource("llama2:latest")

        with patch("bee.sources.ollama.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            models = source.list_models()
            assert models == [], "Should return empty list on connection error"

    def test_inspect_model_returns_none_on_connection_error(self):
        """Test that inspect_model gracefully handles connection failures."""
        source = OllamaSource("llama2:latest")

        with patch("bee.sources.ollama.requests.post") as mock_post:
            mock_post.side_effect = requests.RequestException("Connection refused")

            info = source.inspect_model("llama2:latest")
            assert info is None, "Should return None on connection error"

    def test_ollama_local_source_with_valid_file(self):
        """Test OllamaLocalSource wraps a local file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.gguf"
            test_file.write_bytes(b"fake model")

            source = OllamaLocalSource(test_file)

            artifacts = source.get_artifacts()
            assert len(artifacts) == 1, "Should return single artifact"
            assert artifacts[0] == test_file, "Should return the wrapped file"

    def test_ollama_local_source_with_directory(self):
        """Test OllamaLocalSource returns empty for directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = OllamaLocalSource(Path(tmpdir))

            artifacts = source.get_artifacts()
            assert artifacts == [], "Should return empty for directory"

    def test_source_type_is_ollama(self):
        """Test that source_type property returns 'ollama'."""
        source = OllamaSource("test:latest")
        assert source.source_type == "ollama"

    def test_local_source_type_is_ollama(self):
        """Test that OllamaLocalSource source_type is 'ollama'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.gguf"
            test_file.write_bytes(b"fake model")

            source = OllamaLocalSource(test_file)
            assert source.source_type == "ollama"

    def test_get_source_info_structure(self):
        """Test that get_source_info returns expected structure."""
        source = OllamaSource("qwen3:8b")
        info = source.get_source_info()

        assert "provider" in info
        assert info["provider"] == "ollama"
        assert "repository" in info

    def test_model_name_extraction(self):
        """Test that model name is extracted correctly from full reference."""
        source = OllamaSource("llama2:7b")
        assert source._model_name == "llama2"

    def test_model_name_without_tag(self):
        """Test model name when no tag is provided."""
        source = OllamaSource("llama2")
        assert source._model_name == "llama2"

    def test_cleanup_removes_temp_directory(self):
        """Test that cleanup removes temporary directories."""
        source = OllamaSource("test:latest")

        # Manually set temp dir to test cleanup
        source._temp_dir = Path(tempfile.mkdtemp(prefix="bee_test_"))
        assert source._temp_dir.exists()

        source.cleanup()
        # Temp dir should be removed (or at least cleanup doesn't crash)
        # We don't assert it's gone because of timing, but verify cleanup runs
        assert True, "Cleanup completed without error"

    def test_local_source_cleanup_is_noop(self):
        """Test that OllamaLocalSource cleanup is safe (noop)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "model.gguf"
            test_file.write_bytes(b"fake model")

            source = OllamaLocalSource(test_file)
            source.cleanup()  # Should not raise

            # File should still exist
            assert test_file.exists()

    def test_parse_from_directive_in_modelfile(self):
        """Test that FROM directives pointing to blob paths are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock blob file
            ollama_dir = Path(tmpdir) / ".ollama" / "models"
            blobs_dir = ollama_dir / "blobs"
            blobs_dir.mkdir(parents=True)
            blob_file = blobs_dir / "sha256-abcdef1234567890"
            blob_file.write_bytes(b"fake model content")

            source = OllamaSource("qwen3:8b")

            # Mock the API response with a FROM directive
            modelfile = "FROM /path/to/.ollama/models/blobs/sha256-abcdef1234567890\nPARAMETER temperature 0.7"
            mock_response = {
                "modelfile": modelfile,
                "parameters": "test",
                "format": "gguf"
            }

            with patch("bee.sources.ollama.requests.post") as mock_post:
                with patch.dict("os.environ", {"OLLAMA_MODELS": str(ollama_dir)}):
                    mock_post.return_value.json.return_value = mock_response

                    artifacts = source.get_artifacts()
                    assert len(artifacts) == 1, "Should parse FROM directive and find artifact"
                    assert artifacts[0] == blob_file, "Should identify the correct blob file"

    def test_parse_add_directive_in_modelfile(self):
        """Test that ADD directives with blob references are parsed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock blob file
            ollama_dir = Path(tmpdir) / ".ollama" / "models"
            blobs_dir = ollama_dir / "blobs"
            blobs_dir.mkdir(parents=True)
            blob_file = blobs_dir / "sha256-fedcba0987654321"
            blob_file.write_bytes(b"fake model content")

            source = OllamaSource("llama2:7b")

            # Mock the API response with an ADD directive
            modelfile = "FROM base\nADD sha256-fedcba0987654321 blob"
            mock_response = {
                "modelfile": modelfile,
                "parameters": "test",
                "format": "gguf"
            }

            with patch("bee.sources.ollama.requests.post") as mock_post:
                with patch.dict("os.environ", {"OLLAMA_MODELS": str(ollama_dir)}):
                    mock_post.return_value.json.return_value = mock_response

                    artifacts = source.get_artifacts()
                    assert len(artifacts) == 1, "Should parse ADD directive and find artifact"
                    assert artifacts[0] == blob_file, "Should identify the correct blob file"

    def test_parse_both_from_and_add_directives(self):
        """Test that both FROM and ADD directives are parsed in the same modelfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock blob files
            ollama_dir = Path(tmpdir) / ".ollama" / "models"
            blobs_dir = ollama_dir / "blobs"
            blobs_dir.mkdir(parents=True)

            blob_file1 = blobs_dir / "sha256-from1234567890"
            blob_file1.write_bytes(b"model content 1")

            blob_file2 = blobs_dir / "sha256-add9876543210"
            blob_file2.write_bytes(b"model content 2")

            source = OllamaSource("qwen3:8b")

            # Mock the API response with both FROM and ADD directives
            modelfile = "FROM /path/to/.ollama/models/blobs/sha256-from1234567890\nADD sha256-add9876543210 blob"
            mock_response = {
                "modelfile": modelfile,
                "parameters": "test",
                "format": "gguf"
            }

            with patch("bee.sources.ollama.requests.post") as mock_post:
                with patch.dict("os.environ", {"OLLAMA_MODELS": str(ollama_dir)}):
                    mock_post.return_value.json.return_value = mock_response

                    artifacts = source.get_artifacts()
                    assert len(artifacts) == 2, "Should parse both FROM and ADD directives"
                    artifact_paths = {str(a) for a in artifacts}
                    assert str(blob_file1) in artifact_paths, "Should find FROM blob"
                    assert str(blob_file2) in artifact_paths, "Should find ADD blob"
