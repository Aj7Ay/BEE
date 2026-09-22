from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from bee.core.run import Provenance, SourceInfo, PublisherInfo, ArtifactInfo, AcquisitionInfo, SignatureInfo


def build_provenance(
    provider: str,
    repository: str | None = None,
    revision: str | None = None,
    artifact_path: str | None = None,
    sha256: str | None = None,
    artifact_size: int | None = None,
    artifact_filename: str | None = None,
    acquisition_method: str = "local",
    publisher_name: str | None = None,
    publisher_verified: bool = False,
) -> Provenance:
    """Build a provenance chain for the scanned artifact."""
    return Provenance(
        source=SourceInfo(
            provider=provider,
            repository=repository,
            revision=revision,
        ),
        publisher=PublisherInfo(
            name=publisher_name,
            verified=publisher_verified,
        ),
        artifact=ArtifactInfo(
            filename=artifact_filename or (Path(artifact_path).name if artifact_path else "unknown"),
            sha256=sha256 or "",
            size=artifact_size or 0,
        ),
        acquisition=AcquisitionInfo(method=acquisition_method),
    )


def build_provenance_graph(provenance: Provenance, has_signature: bool = False) -> list[dict]:
    """Build the visual provenance chain for the HTML report.

    Returns a list of chain nodes in order:
    Publisher → Repository → Revision → Build → Artifact → SHA-256 → BEE Evidence → [Signed Evidence]
    """
    nodes = []

    if provenance.publisher.name:
        nodes.append({
            "step": "publisher",
            "label": provenance.publisher.name,
            "verified": provenance.publisher.verified,
            "icon": "user" if provenance.publisher.verified else "user-dot",
        })

    if provenance.source.repository:
        nodes.append({
            "step": "repository",
            "label": provenance.source.repository,
            "verified": True,
            "icon": "git-branch",
        })

    if provenance.source.revision:
        nodes.append({
            "step": "revision",
            "label": provenance.source.revision[:12],
            "verified": True,
            "icon": "code-commit",
        })

    nodes.append({
        "step": "build",
        "label": provenance.acquisition.method,
        "verified": True,
        "icon": "download",
    })

    nodes.append({
        "step": "artifact",
        "label": provenance.artifact.filename,
        "verified": True,
        "icon": "file",
    })

    if provenance.artifact.sha256:
        nodes.append({
            "step": "sha256",
            "label": provenance.artifact.sha256[:16] + "...",
            "verified": True,
            "icon": "hash",
        })

    nodes.append({
        "step": "evidence",
        "label": "BEE Analysis",
        "verified": True,
        "icon": "shield-check",
    })

    if has_signature:
        nodes.append({
            "step": "signature",
            "label": "Signed",
            "verified": True,
            "icon": "shield-check",
        })
    else:
        nodes.append({
            "step": "signature",
            "label": "Unsigned",
            "verified": False,
            "icon": "shield",
        })

    return nodes
