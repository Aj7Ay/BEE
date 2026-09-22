from __future__ import annotations

from pathlib import Path

from typing import Any


def parse_model_card(readme_path: Path) -> dict[str, Any] | None:
    """Parse a HuggingFace-style model card (README.md with YAML front-matter)."""
    try:
        text = readme_path.read_text(errors="replace")
    except OSError:
        return None

    return _parse_model_card_text(text)


def _parse_model_card_text(text: str) -> dict[str, Any]:
    """Parse model card text with optional YAML front-matter."""
    import yaml

    card: dict[str, Any] = {}

    # Extract YAML front-matter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                card["metadata"] = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                card["metadata"] = {}

    # Parse metadata fields
    meta = card.get("metadata", {}) or {}

    # Core fields
    card["model_name"] = meta.get("model_name") or meta.get("model-index", [{}])[0].get("name")
    card["architecture"] = meta.get("config") or meta.get("model_type")
    card["license"] = meta.get("license")
    card["tags"] = meta.get("tags")
    card["pipeline_tag"] = meta.get("pipeline_tag")
    card["language"] = meta.get("language")

    # Parameter info
    card["parameter_count"] = meta.get("parameter_count") or meta.get("size")

    # Training info
    card["training_data"] = meta.get("training_data") or meta.get("train_data")
    card["base_model"] = meta.get("base_model")
    card["intended_use"] = meta.get("intended_use")
    card["limitations"] = meta.get("limitations")
    card["risks"] = meta.get("risks")
    card["bias"] = meta.get("bias")

    # Datasets
    datasets = meta.get("datasets")
    if isinstance(datasets, list):
        card["datasets"] = datasets
    elif isinstance(datasets, str):
        card["datasets"] = [datasets]

    # Quantization
    card["quantization"] = meta.get("quantization") or meta.get("quantization_config", {}).get("load_in_4bit") or meta.get("quantization_config", {}).get("load_in_8bit")

    # Check the raw text for additional info
    card["raw_text"] = text[:2000]  # First 2000 chars for context

    return card


def extract_model_info_from_gguf_metadata(gguf_metadata: dict) -> dict[str, Any]:
    """Extract model info from GGUF metadata (gleaned from file header)."""
    info: dict[str, Any] = {}

    info["architecture"] = gguf_metadata.get("general.architecture")
    info["quantization"] = gguf_metadata.get("general.file_type")

    # Parameter count
    param_str = gguf_metadata.get("tokenizer.ggml.tokens", "")

    if info["architecture"]:
        arch = info["architecture"]
        # Common GGUF metadata keys
        info["layer_norm_epsilon"] = gguf_metadata.get(f"{arch}.attention.layer_norm_epsilon")
        info["vocab_size"] = gguf_metadata.get(f"{arch}.vocab_size")
        info["context_length"] = gguf_metadata.get(f"{arch}.context_length")
        info["embedding_length"] = gguf_metadata.get(f"{arch}.embedding_length")
        info["feed_forward_length"] = gguf_metadata.get(f"{arch}.feed_forward_length")
        info["attention_heads"] = gguf_metadata.get(f"{arch}.attention.head_count")
        info["key_heads"] = gguf_metadata.get(f"{arch}.attention.head_count_kv")
        info["block_count"] = gguf_metadata.get(f"{arch}.block_count")

    return info
