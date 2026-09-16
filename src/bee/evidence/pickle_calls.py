from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.formats.pickle_ops import (
    UNRESOLVED_STACK_GLOBAL,
    PickleAnalysis,
    analyze_pickle_file,
    analyze_pytorch_zip_pickle,
)

# Callables whose presence in a pickle's GLOBAL/STACK_GLOBAL references
# means the file can execute arbitrary code or shell commands the moment
# it is loaded -- not "might be able to," a direct, well-known primitive.
# Kept as an exact-match list deliberately, not a module-prefix match
# (e.g. "os."): a prefix would also catch harmless references like
# os.path.join under the same CRITICAL severity, and a CRITICAL finding
# that's wrong even occasionally costs more trust than it's worth.
# Anything not on this list, and not on the allowlist below, still gets
# flagged -- at MEDIUM, as "unrecognized" rather than "definitely
# dangerous." That's the safety net for an os.* function this list
# doesn't happen to name.
DANGEROUS_GLOBALS = frozenset({
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.fork", "os.forkpty", "os.remove", "os.unlink", "os.rmdir",
    "posix.system", "nt.system",
    "subprocess.run", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.Popen",
    "builtins.eval", "builtins.exec", "builtins.compile", "builtins.__import__",
    "__builtin__.eval", "__builtin__.exec", "__builtin__.__import__",
    "eval", "exec", "compile", "__import__",
    "pickle.loads", "pickle.load",
    "importlib.import_module",
    "pty.spawn",
    "ctypes.CDLL", "ctypes.PyDLL", "ctypes.cdll.LoadLibrary",
    "socket.socket",
    "shutil.rmtree",
    "runpy._run_module_as_main", "runpy.run_module", "runpy.run_path",
    "webbrowser.open",
})

# Callables real ML checkpoints reference constantly, that are not
# themselves code-execution primitives -- reconstructing a tensor, a
# dict, a numpy array. Referencing one of these alone is not flagged, so
# a normal checkpoint doesn't get buried in "unrecognized global"
# findings on every single scan.
ALLOWED_GLOBALS = frozenset({
    "collections.OrderedDict",
    "builtins.dict", "builtins.set", "builtins.list", "builtins.tuple",
    "builtins.frozenset", "builtins.bytearray", "builtins.complex", "builtins.slice",
    "__builtin__.dict", "__builtin__.set", "__builtin__.list", "__builtin__.tuple",
    "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_device_tensor_from_numpy",
    "torch.Tensor", "torch.Size", "torch.dtype", "torch.device", "torch.Storage",
    "torch.serialization._get_layout", "torch._C._nn._parse_to",
    "numpy.core.multiarray._reconstruct", "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct", "numpy._core.multiarray.scalar",
    "numpy.ndarray", "numpy.dtype",
})


def _analysis_for(artifact: Artifact) -> PickleAnalysis | None:
    path = Path(artifact.path)
    if artifact.detected_format == "pickle":
        return analyze_pickle_file(path)
    if artifact.detected_format == "pytorch":
        return analyze_pytorch_zip_pickle(path)
    return None


def check_pickle_calls(artifact: Artifact) -> Finding | None:
    """For a file already determined to be (or embed) a pickle, identify
    what it actually references via GLOBAL/STACK_GLOBAL, and whether
    REDUCE -- the opcode that calls one of them -- is present. This is
    what turns "this is a pickle" into "this pickle calls os.system": a
    finding a format-mismatch check alone can never produce for a file
    that's honestly named .pt and genuinely is one.
    """
    analysis = _analysis_for(artifact)
    if analysis is None:
        return None

    dangerous = sorted({g for g in analysis.globals_referenced if g in DANGEROUS_GLOBALS})
    if dangerous:
        reduce_note = (
            f" REDUCE (the opcode that calls it) appears {analysis.reduce_count} time(s)."
            if analysis.reduce_count
            else ""
        )
        return Finding(
            id="BEE-PKL-001",
            severity=Severity.CRITICAL,
            title="Pickle references a dangerous primitive",
            description=(
                f"This pickle's opcode stream references {', '.join(dangerous)}."
                f"{reduce_note}"
            ),
            artifact_path=artifact.path,
            evidence=[
                Evidence(type="pickle_global", value=name, source="local_filesystem",
                          confidence=Confidence.VERIFIED)
                for name in dangerous
            ],
        )

    unresolved_count = analysis.globals_referenced.count(UNRESOLVED_STACK_GLOBAL)
    unrecognized = sorted({
        g for g in analysis.globals_referenced
        if g not in ALLOWED_GLOBALS and g != UNRESOLVED_STACK_GLOBAL
    })
    if not unrecognized and not unresolved_count:
        return None

    parts = list(unrecognized)
    if unresolved_count:
        parts.append(f"{unresolved_count} unresolved STACK_GLOBAL reference(s)")
    return Finding(
        id="BEE-PKL-002",
        severity=Severity.MEDIUM,
        title="Pickle references an unrecognized global",
        description=(
            "This pickle's opcode stream references callables/classes not "
            "recognized as either a known-safe checkpoint helper or a "
            f"known-dangerous primitive: {', '.join(parts)}. Manual review recommended."
        ),
        artifact_path=artifact.path,
        evidence=[
            Evidence(type="pickle_global", value=name, source="local_filesystem",
                      confidence=Confidence.INFERRED)
            for name in unrecognized
        ],
    )
