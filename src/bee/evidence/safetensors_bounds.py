from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.formats.safetensors_ops import DTYPE_SIZES, TensorRange, analyze_safetensors

_MAX_REPORTED_PROBLEMS = 10


def _range_problem(tensor: TensorRange, data_region_size: int) -> str | None:
    start, end = tensor.start, tensor.end
    if not (isinstance(start, int) and isinstance(end, int)):
        return f"{tensor.name}: data_offsets are not integers ({start!r}, {end!r})"
    if start < 0 or end < start:
        return f"{tensor.name}: invalid range [{start}, {end})"
    if end > data_region_size:
        return (
            f"{tensor.name}: range [{start}, {end}) exceeds the file's data "
            f"region ({data_region_size} bytes available)"
        )
    return None


def _size_mismatch_problem(tensor: TensorRange) -> str | None:
    dtype_size = DTYPE_SIZES.get(tensor.dtype)
    if dtype_size is None or not isinstance(tensor.shape, list):
        return None
    try:
        element_count = 1
        for dim in tensor.shape:
            element_count *= int(dim)
    except (TypeError, ValueError):
        return None
    expected_bytes = element_count * dtype_size
    actual_bytes = tensor.end - tensor.start
    if expected_bytes != actual_bytes:
        return (
            f"{tensor.name}: shape {tensor.shape} of {tensor.dtype} implies "
            f"{expected_bytes} bytes, but data_offsets claims {actual_bytes}"
        )
    return None


def check_safetensors_bounds(artifact: Artifact) -> Finding | None:
    """A file can be structurally recognizable as SafeTensors (a valid
    8-byte length prefix followed by valid JSON) while its header still
    lies about where each tensor's bytes actually are -- a range that
    runs past the end of the file, two tensors claiming overlapping
    bytes, or a declared shape/dtype that doesn't match the byte range
    backing it. None of that is checked by format detection, which only
    confirms the container is well-formed enough to identify.
    """
    if artifact.detected_format != "safetensors":
        return None
    analysis = analyze_safetensors(Path(artifact.path))
    if analysis is None:
        return None

    problems: list[str] = []
    accepted_ranges: list[tuple[int, int, str]] = []

    for tensor in analysis.tensors:
        problem = _range_problem(tensor, analysis.data_region_size)
        if problem is not None:
            problems.append(problem)
            continue

        size_problem = _size_mismatch_problem(tensor)
        if size_problem is not None:
            problems.append(size_problem)

        for other_start, other_end, other_name in accepted_ranges:
            if tensor.start < other_end and other_start < tensor.end:
                problems.append(f"{tensor.name} overlaps {other_name}")
        accepted_ranges.append((tensor.start, tensor.end, tensor.name))

    if not problems:
        return None

    shown = problems[:_MAX_REPORTED_PROBLEMS]
    remainder_note = (
        f" (+{len(problems) - _MAX_REPORTED_PROBLEMS} more)"
        if len(problems) > _MAX_REPORTED_PROBLEMS
        else ""
    )
    return Finding(
        id="BEE-STS-001",
        severity=Severity.HIGH,
        title="SafeTensors header declares invalid or overlapping tensor ranges",
        description=(
            "This file's SafeTensors header contains tensor byte ranges that "
            "don't hold up: " + "; ".join(shown) + remainder_note
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(type="safetensors_bounds", value=p, source="local_filesystem",
                      confidence=Confidence.VERIFIED)
            for p in shown
        ],
    )
