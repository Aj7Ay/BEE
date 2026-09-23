"""Tests for HuggingFace source integration."""

import tempfile
from pathlib import Path

import pytest

from bee.sources.huggingface import HuggingFaceSource


class TestHuggingFaceSource:
    """Tests for HuggingFaceSource class."""

    def test_repo_id_parsing_with_slash(self):
        """Test that org/model format is parsed correctly."""
        source = HuggingFaceSource("hf://meta-llama/Llama-2-7b")
        assert source.repo_id == "meta-llama/Llama-2-7b"

    def test_repo_id_parsing_without_prefix(self):
        """Test that org/model without hf:// is normalized."""
        source = HuggingFaceSource("meta-llama/Llama-2-7b")
        assert source.repo_id == "meta-llama/Llama-2-7b"

    def test_revision_parsing(self):
        """Test that revision is extracted when present."""
        source = HuggingFaceSource("hf://meta-llama/Llama-2-7b/main")
        assert source.revision == "main"

    def test_revision_is_none_when_absent(self):
        """Test that revision is None when not specified."""
        source = HuggingFaceSource("meta-llama/Llama-2-7b")
        assert source.revision is None

    def test_source_type_is_huggingface(self):
        """Test that source_type property returns 'huggingface'."""
        source = HuggingFaceSource("test/model")
        assert source.source_type == "huggingface"

    def test_get_source_info_structure(self):
        """Test that get_source_info returns expected structure."""
        source = HuggingFaceSource("meta-llama/Llama-2-7b")
        info = source.get_source_info()

        assert "provider" in info
        assert info["provider"] == "huggingface"
        assert "repository" in info
        assert info["repository"] == "meta-llama/Llama-2-7b"

    def test_model_file_detection(self):
        """Test that model files are correctly identified."""
        source = HuggingFaceSource("test/model")

        # Should detect model files
        assert source._is_model_file("model.safetensors")
        assert source._is_model_file("pytorch_model.bin")
        assert source._is_model_file("model.gguf")
        assert source._is_model_file("consolidated.00.pth")

        # Should reject non-model files
        assert not source._is_model_file("README.md")
        assert not source._is_model_file("config.json")
        assert not source._is_model_file("tokenizer.json")

    def test_onnx_files_excluded(self):
        """Test that ONNX files are not detected as model files."""
        source = HuggingFaceSource("test/model")

        # ONNX files should be rejected (no longer in _MODEL_EXTENSIONS)
        assert not source._is_model_file("model.onnx")
        assert not source._is_model_file("model-quantized.onnx")
        assert not source._is_model_file("encoder.onnx")

    def test_cleanup_is_safe(self):
        """Test that cleanup handles missing temp directory gracefully."""
        source = HuggingFaceSource("test/model")
        source.cleanup()  # Should not raise
        assert True, "Cleanup completed without error"

    def test_url_normalization_huggingface_prefix(self):
        """Test that huggingface:// prefix is normalized."""
        source = HuggingFaceSource("huggingface://org/model")
        assert source.repo_id == "org/model"
