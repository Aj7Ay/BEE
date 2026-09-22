from __future__ import annotations

import re
from pathlib import Path


# SPDX identifiers mapped to their common filename patterns
SPDX_LICENSES = {
    "MIT": ["MIT", "MIT-License", "mit"],
    "Apache-2.0": ["Apache", "Apache-2.0", "apache", "LICENSE.APACHE"],
    "GPL-3.0": ["GPL", "GPL-3.0", "gpl", "GNU GPL"],
    "GPL-2.0": ["GPL-2.0", "gpl-2.0"],
    "BSD-3-Clause": ["BSD", "BSD-3-Clause", "bsd"],
    "BSD-2-Clause": ["BSD-2-Clause", "bsd-2"],
    "ISC": ["ISC-License", "isc"],
    "MPL-2.0": ["MPL-2.0", "mpl"],
    "LGPL-3.0": ["LGPL", "lgpl"],
    "Unlicense": ["Unlicense", "unlicense"],
    "CC-BY-4.0": ["CC-BY", "cc-by"],
    "CC-BY-SA-4.0": ["CC-BY-SA", "cc-by-sa"],
    "CC-BY-NC-4.0": ["CC-BY-NC", "cc-by-nc"],
    "ODC-BY": ["ODC-BY"],
    "OpenRail-M": ["OpenRail-M"],
    "Open Weight": ["open-weights", "open weights"],
}

# Reverse mapping: normalized text -> SPDX ID
_TEXT_TO_SPDX: dict[str, str] = {}
for spdx, names in SPDX_LICENSES.items():
    for name in names:
        _TEXT_TO_SPDX[name.lower()] = spdx


def detect_license_from_text(text: str) -> str | None:
    """Detect SPDX license identifier from text content."""
    text_lower = text.lower()

    # Check for SPDX identifier in text
    for spdx, patterns in SPDX_LICENSES.items():
        for pattern in patterns:
            if pattern.lower() in text_lower:
                return spdx

    # Check for common license phrases
    if "permission is hereby granted, free of charge" in text_lower:
        return "MIT"
    if "apache license" in text_lower and "version 2.0" in text_lower:
        return "Apache-2.0"
    if "gnu general public license" in text_lower:
        if "version 3" in text_lower:
            return "GPL-3.0"
        return "GPL-2.0"
    if "redistribution and use" in text_lower and "source" in text_lower:
        if "3-clause" in text_lower or "three clause" in text_lower:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "creative commons" in text_lower:
        if "attribution" in text_lower and "noncommercial" in text_lower:
            return "CC-BY-NC-4.0"
        if "sharealike" in text_lower:
            return "CC-BY-SA-4.0"
        if "attribution" in text_lower:
            return "CC-BY-4.0"

    return None


def detect_license_from_file(path: Path) -> dict | None:
    """Detect license from a LICENSE file."""
    if not path.is_file():
        return None

    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None

    spdx = detect_license_from_text(text)
    if spdx:
        return {"spdx": spdx, "source": str(path), "type": "file"}
    return None


def detect_license_from_pyproject(path: Path) -> dict | None:
    """Detect license from pyproject.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    try:
        data = tomllib.loads(path.read_text())
        license_field = data.get("project", {}).get("license", {})
        if isinstance(license_field, str):
            # "MIT", "MIT OR Apache-2.0", etc.
            return {"spdx": license_field, "source": str(path), "type": "metadata"}
        elif isinstance(license_field, dict):
            text = license_field.get("text", "")
            if text:
                spdx = detect_license_from_text(text)
                if spdx:
                    return {"spdx": spdx, "source": str(path), "type": "metadata"}
    except Exception:
        pass
    return None


def detect_license_from_package_json(path: Path) -> dict | None:
    """Detect license from package.json."""
    import json as jsonlib

    try:
        data = jsonlib.loads(path.read_text())
        license_val = data.get("license")
        if isinstance(license_val, str):
            return {"spdx": license_val, "source": str(path), "type": "metadata"}
        elif isinstance(license_val, dict):
            spdx = license_val.get("type", "")
            if spdx:
                return {"spdx": spdx, "source": str(path), "type": "metadata"}
    except Exception:
        pass
    return None


def detect_license_from_hf_metadata(metadata: dict) -> dict | None:
    """Detect license from HuggingFace model card metadata."""
    license_val = metadata.get("license")
    if license_val:
        return {"spdx": license_val, "source": "huggingface_metadata", "type": "metadata"}

    tags = metadata.get("tags", [])
    if tags:
        for tag in tags:
            if tag.startswith("license:"):
                spdx = tag.split(":", 1)[1]
                return {"spdx": spdx, "source": "huggingface_tags", "type": "tag"}

    return None


LICENSE_MANIFESTS = {
    "LICENSE": lambda p: detect_license_from_file(p),
    "LICENSE.md": lambda p: detect_license_from_file(p),
    "LICENSE.txt": lambda p: detect_license_from_file(p),
    "LICENSE-MIT": lambda p: detect_license_from_file(p),
    "pyproject.toml": lambda p: detect_license_from_pyproject(p),
    "package.json": lambda p: detect_license_from_package_json(p),
}


def detect_license_in_directory(scan_root: Path) -> list[dict]:
    """Scan a directory for license information from multiple sources."""
    licenses: list[dict] = []
    seen_spdx: set[str] = set()

    # Check LICENSE files
    for manifest_name, parser in LICENSE_MANIFESTS.items():
        manifest_path = scan_root / manifest_name
        result = parser(manifest_path)
        if result and result.get("spdx") not in seen_spdx:
            licenses.append(result)
            seen_spdx.add(result["spdx"])

    return licenses


def assess_license_compatibility(
    detected_licenses: list[dict],
    allowed_licenses: list[str] | None = None,
) -> dict:
    """Assess compatibility of detected licenses against an allowed list.

    Returns: {
        "compatible": bool,
        "licenses": [...],
        "incompatible": ["..."],
    }
    """
    if not allowed_licenses:
        return {"compatible": True, "licenses": detected_licenses, "incompatible": []}

    incompatible = []
    for lic in detected_licenses:
        spdx = lic.get("spdx", "")
        if spdx not in allowed_licenses:
            incompatible.append(spdx)

    return {
        "compatible": len(incompatible) == 0,
        "licenses": detected_licenses,
        "incompatible": incompatible,
    }
