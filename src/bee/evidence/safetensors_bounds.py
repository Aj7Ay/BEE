from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.formats.safetensors_ops import DTYPE_SIZES, TensorRange, analyze_safetensors

_MAX_REPORTED_PROBLEMS = 10

# A header declaring more tensors than this is itself the finding, rather
# than something exhaustively bounds-checked -- an attacker controls the
# header's tensor count directly, and no real checkpoint approaches it.
_MAX_TENSORS_CHECKED = 200_000

# No real tensor has anywhere near this many elements. Used to cap the
# shape x dtype multiplication below: an attacker-supplied shape is
# exactly the kind of value that could otherwise turn that into a chain
# of expensive bignum multiplications, each one bounded in isolation by
# json's own integer-digit limit but not by how many of them a shape
# array can declare.
_MAX_PLAUSIBLE_ELEMENTS = 2**62


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


def _element_count(shape: object) -> int | None:
    """Product of shape's dimensions, stopping as soon as it's already
    conclusively larger than any real tensor could be -- multiplying
    further attacker-supplied bignums past that point doesn't change the
    outcome (still "implausible"), only the CPU cost of finding out."""
    if not isinstance(shape, list):
        return None
    try:
        count = 1
        for dim in shape:
            count *= int(dim)
            if count > _MAX_PLAUSIBLE_ELEMENTS:
                return count
    except (TypeError, ValueError):
        return None
    return count


def _size_mismatch_problem(tensor: TensorRange) -> str | None:
    dtype_size = DTYPE_SIZES.get(tensor.dtype)
    if dtype_size is None:
        return None
    element_count = _element_count(tensor.shape)
    if element_count is None:
        return None
    if element_count > _MAX_PLAUSIBLE_ELEMENTS:
        return (
            f"{tensor.name}: shape {tensor.shape} declares an implausible "
            f"element count (> {_MAX_PLAUSIBLE_ELEMENTS})"
        )
    expected_bytes = element_count * dtype_size
    actual_bytes = tensor.end - tensor.start
    if expected_bytes != actual_bytes:
        return (
            f"{tensor.name}: shape {tensor.shape} of {tensor.dtype} implies "
            f"{expected_bytes} bytes, but data_offsets claims {actual_bytes}"
        )
    return None


def _overlap_problems(tensors: list[TensorRange]) -> list[str]:
    """O(n log n): sort by start, then a single sweep tracking the
    furthest end seen so far is enough to detect every overlapping pair
    without comparing all n^2 combinations. A header can declare an
    arbitrary number of tensors, and pairwise comparison turns that
    directly into a CPU cost quadratic in something the attacker
    controls -- the same class of issue as detect_pickle's old
    byte-prefix bound, just one layer up."""
    problems = []
    max_end_so_far = -1
    holder_name = ""
    for tensor in sorted(tensors, key=lambda t: t.start):
        if tensor.start < max_end_so_far:
            problems.append(f"{tensor.name} overlaps {holder_name}")
        if tensor.end > max_end_so_far:
            max_end_so_far = tensor.end
            holder_name = tensor.name
    return problems


def _format_problems(problems: list[str]) -> str:
    shown = problems[:_MAX_REPORTED_PROBLEMS]
    remainder = (
        f" (+{len(problems) - _MAX_REPORTED_PROBLEMS} more)"
        if len(problems) > _MAX_REPORTED_PROBLEMS
        else ""
    )
    return "; ".join(shown) + remainder


def check_safetensors_bounds(artifact: Artifact, content: bytes | None = None) -> Finding | None:
    """A file can be structurally recognizable as SafeTensors (a valid
    8-byte length prefix followed by valid JSON) while its header still
    lies about where each tensor's bytes actually are -- a range that
    runs past the end of the file, two tensors claiming overlapping
    bytes, or a declared shape/dtype that doesn't match the byte range
    backing it. None of that is checked by format detection, which only
    confirms the container is well-formed enough to identify.

    `content`, when provided, is this same artifact's already-buffered
    bytes (see Artifact.from_file / bee.formats.io_source), analyzed
    directly instead of re-opening artifact.path.
    """
    if artifact.detected_format != "safetensors":
        return None
    analysis = analyze_safetensors(content if content is not None else Path(artifact.path))
    if analysis is None:
        return None

    if len(analysis.tensors) > _MAX_TENSORS_CHECKED:
        return Finding(
            id="BEE-STS-001",
            severity=Severity.HIGH,
            title="SafeTensors header declares an implausible number of tensors",
            description=(
                f"This file's SafeTensors header declares {len(analysis.tensors)} "
                f"tensors, more than BEE bounds-checks exhaustively "
                f"({_MAX_TENSORS_CHECKED}). That alone is atypical of a real "
                "checkpoint."
            ),
            artifact_path=artifact.path,
            evidence=[
                Evidence(type="safetensors_bounds", value=f"tensor_count={len(analysis.tensors)}",
                          source="local_filesystem", confidence=Confidence.VERIFIED),
            ],
        )

    problems: list[str] = []
    valid_ranged: list[TensorRange] = []

    for tensor in analysis.tensors:
        problem = _range_problem(tensor, analysis.data_region_size)
        if problem is not None:
            problems.append(problem)
            continue
        size_problem = _size_mismatch_problem(tensor)
        if size_problem is not None:
            problems.append(size_problem)
        valid_ranged.append(tensor)

    problems.extend(_overlap_problems(valid_ranged))

    if not problems:
        return None

    return Finding(
        id="BEE-STS-001",
        severity=Severity.HIGH,
        title="SafeTensors header declares invalid or overlapping tensor ranges",
        description=(
            "This file's SafeTensors header contains tensor byte ranges that "
            "don't hold up: " + _format_problems(problems)
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(type="safetensors_bounds", value=p, source="local_filesystem",
                      confidence=Confidence.VERIFIED)
            for p in problems[:_MAX_REPORTED_PROBLEMS]
        ],
    )


def check_safetensors_gap(artifact: Artifact, content: bytes | None = None) -> Finding | None:
    """Bytes in a SafeTensors file's data region that no tensor's
    data_offsets range covers. Not caught by the bounds check above: each
    individual range can be perfectly valid and non-overlapping while
    still leaving room unaccounted for. Every real SafeTensors file
    checked during development (a 47MB timm/resnet18 checkpoint, a real
    GPT-2 export) packed its tensors contiguously with exactly zero such
    gap, so any gap at all is reported -- no tolerance threshold that
    could hide a small one alongside real alignment padding that, in
    practice, doesn't appear to exist.

    Only evaluated when the file has no bounds problems already: overlap
    and out-of-range issues make "bytes covered" arithmetic unreliable
    (double-counted or negative), and check_safetensors_bounds already
    reports those files.
    """
    if artifact.detected_format != "safetensors":
        return None
    analysis = analyze_safetensors(content if content is not None else Path(artifact.path))
    if analysis is None or not analysis.tensors:
        return None
    if len(analysis.tensors) > _MAX_TENSORS_CHECKED:
        return None

    for tensor in analysis.tensors:
        if _range_problem(tensor, analysis.data_region_size) is not None:
            return None
    if _overlap_problems(analysis.tensors):
        return None

    covered = sum(tensor.end - tensor.start for tensor in analysis.tensors)
    gap = analysis.data_region_size - covered
    if gap <= 0:
        return None

    return Finding(
        id="BEE-STS-002",
        severity=Severity.LOW,
        title="SafeTensors data region contains bytes no tensor references",
        description=(
            f"{gap} of {analysis.data_region_size} bytes in this file's data "
            "region are not covered by any tensor's data_offsets range -- "
            "bytes that ride along in the file without being loaded as part "
            "of any declared tensor."
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(type="safetensors_gap", value=str(gap), source="local_filesystem",
                      confidence=Confidence.VERIFIED),
        ],
    )
