from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.formats.gguf_ops import GgufTensorInfo, analyze_gguf, compute_tensor_byte_size

_MAX_REPORTED_PROBLEMS = 10

# A header declaring more tensors than this is itself the finding,
# rather than something exhaustively bounds-and-overlap-checked -- an
# attacker controls the header's declared tensor count directly, and no
# real model approaches this. Matches the same-purpose, same-value cap
# in bee.evidence.safetensors_bounds (_MAX_TENSORS_CHECKED).
_MAX_TENSORS_CHECKED = 200_000


def _tensor_problem(tensor: GgufTensorInfo, tensor_data_start: int, alignment: int, file_size: int) -> str | None:
    if tensor.offset < 0:
        return f"{tensor.name}: negative offset {tensor.offset}"
    if tensor.offset % alignment != 0:
        return (
            f"{tensor.name}: offset {tensor.offset} is not a multiple of the "
            f"declared alignment ({alignment})"
        )
    absolute_start = tensor_data_start + tensor.offset
    if absolute_start > file_size:
        return (
            f"{tensor.name}: data starts at byte {absolute_start}, past the end "
            f"of the file ({file_size} bytes)"
        )
    byte_size = compute_tensor_byte_size(tensor.ggml_type, tensor.dimensions)
    if byte_size is None:
        # An unrecognized ggml_type (a newer quantization format this
        # module's table doesn't cover yet) -- not checked, not guessed.
        return None
    absolute_end = absolute_start + byte_size
    if absolute_end > file_size:
        return (
            f"{tensor.name}: data range [{absolute_start}, {absolute_end}) "
            f"exceeds the file ({file_size} bytes) -- consistent with a "
            "truncated or incomplete file"
        )
    return None


def _overlap_problems(
    tensors: list[GgufTensorInfo], tensor_data_start: int
) -> list[str]:
    """O(n log n): sort by absolute start, then a single sweep tracking
    the furthest end seen so far is enough to detect every overlapping
    pair without comparing all n^2 combinations -- the same sweep
    bee.evidence.safetensors_bounds._overlap_problems already uses for
    SafeTensors, adapted here for GGUF's offset-plus-computed-size
    ranges instead of an explicit data_offsets pair.

    Only given tensors whose absolute range is already known to be
    valid (see check_gguf_bounds below): a tensor with an unrecognized
    ggml_type has no computable end, and one already out of bounds is
    already reported by _tensor_problem -- reporting it again here as
    an "overlap" would just be confusing, not more informative.
    """
    ranges: list[tuple[int, int, str]] = []
    for tensor in tensors:
        byte_size = compute_tensor_byte_size(tensor.ggml_type, tensor.dimensions)
        if byte_size is None:
            continue
        absolute_start = tensor_data_start + tensor.offset
        ranges.append((absolute_start, absolute_start + byte_size, tensor.name))

    problems = []
    max_end_so_far = -1
    holder_name = ""
    for start, end, name in sorted(ranges, key=lambda r: r[0]):
        if start < max_end_so_far:
            problems.append(f"{name} overlaps {holder_name}")
        if end > max_end_so_far:
            max_end_so_far = end
            holder_name = name
    return problems


def check_gguf_bounds(artifact: Artifact, content: bytes | None = None) -> Finding | None:
    """A file can be structurally recognizable as GGUF (correct magic,
    a header that parses) while its tensor table still lies about where
    each tensor's bytes actually are -- an offset past the end of the
    file, one that isn't aligned the way the file's own declared
    alignment requires, or a declared shape/type that doesn't fit in
    what's left. The single most common real-world trigger for this is
    an incomplete download, not an attack -- but either way, a file a
    naive loader would read past its own end is worth flagging.

    Also checks that no two tensors claim the same bytes -- individually
    valid ranges that still overlap each other, which a per-tensor
    end-of-file check alone can never catch.

    `content`, when provided, is this same artifact's already-buffered
    bytes (see Artifact.from_file / bee.formats.io_source) -- analyzed
    directly instead of re-opening artifact.path, so this can't see
    different content than what was actually hashed if the file changed
    on disk in between.
    """
    if artifact.detected_format != "gguf":
        return None
    analysis = analyze_gguf(content if content is not None else Path(artifact.path))
    if analysis is None:
        return None

    if len(analysis.tensors) > _MAX_TENSORS_CHECKED:
        return Finding(
            id="BEE-GGUF-001",
            severity=Severity.HIGH,
            title="GGUF header declares an implausible number of tensors",
            description=(
                f"This file's GGUF header declares {len(analysis.tensors)} "
                f"tensors, more than BEE bounds-checks exhaustively "
                f"({_MAX_TENSORS_CHECKED}). That alone is atypical of a real model."
            ),
            artifact_path=artifact.path,
            evidence=[
                Evidence(type="gguf_bounds", value=f"tensor_count={len(analysis.tensors)}",
                          source="local_filesystem", confidence=Confidence.VERIFIED),
            ],
        )

    problems = []
    valid_tensors: list[GgufTensorInfo] = []
    for tensor in analysis.tensors:
        problem = _tensor_problem(tensor, analysis.tensor_data_start, analysis.alignment, analysis.file_size)
        if problem is not None:
            problems.append(problem)
        else:
            valid_tensors.append(tensor)

    problems.extend(_overlap_problems(valid_tensors, analysis.tensor_data_start))

    if not problems:
        return None

    shown = problems[:_MAX_REPORTED_PROBLEMS]
    remainder = (
        f" (+{len(problems) - _MAX_REPORTED_PROBLEMS} more)"
        if len(problems) > _MAX_REPORTED_PROBLEMS
        else ""
    )
    return Finding(
        id="BEE-GGUF-001",
        severity=Severity.HIGH,
        title="GGUF tensor table declares invalid or out-of-bounds data",
        description=(
            "This file's GGUF tensor table contains entries that don't hold up: "
            + "; ".join(shown) + remainder
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(type="gguf_bounds", value=p, source="local_filesystem", confidence=Confidence.VERIFIED)
            for p in shown
        ],
    )
