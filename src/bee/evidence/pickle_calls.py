from __future__ import annotations

from pathlib import Path

from bee.core.artifact import Artifact
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.formats.pickle_ops import (
    MAX_PICKLE_OPCODES,
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
#
# The torch.*Storage entries are calibrated against a real corpus, not
# guessed: 10 actual checkpoints downloaded from Hugging Face (tiny-gpt2,
# tiny-bert, tiny-t5, tiny-gpt-neo, tiny-ViT, tiny-distilbert,
# tiny-roberta, CLIP, across both hf-internal-testing and sshleifer) were
# scanned, and torch.FloatStorage/LongStorage/ByteStorage were the
# recurring unrecognized globals across 9 of them -- legacy per-dtype
# storage classes referenced by PyTorch's own pickle-based save format
# via _rebuild_tensor_v2, present in essentially every non-safetensors
# checkpoint. The remaining dtype variants below are the same
# well-documented family (one class per torch dtype, same role), not a
# separate guess.
ALLOWED_GLOBALS = frozenset({
    "collections.OrderedDict",
    "builtins.dict", "builtins.set", "builtins.list", "builtins.tuple",
    "builtins.frozenset", "builtins.bytearray", "builtins.complex", "builtins.slice",
    "__builtin__.dict", "__builtin__.set", "__builtin__.list", "__builtin__.tuple",
    "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_device_tensor_from_numpy",
    "torch.Tensor", "torch.Size", "torch.dtype", "torch.device", "torch.Storage",
    "torch.serialization._get_layout", "torch._C._nn._parse_to",
    "torch.storage.TypedStorage", "torch.storage._TypedStorage",
    "torch.FloatStorage", "torch.DoubleStorage", "torch.HalfStorage",
    "torch.LongStorage", "torch.IntStorage", "torch.ShortStorage",
    "torch.CharStorage", "torch.ByteStorage", "torch.BoolStorage",
    "torch.BFloat16Storage", "torch.ComplexFloatStorage", "torch.ComplexDoubleStorage",
    "torch.QUInt8Storage", "torch.QInt8Storage", "torch.QInt32Storage",
    "torch.QUInt4x2Storage", "torch.QUInt2x4Storage",
    "numpy.core.multiarray._reconstruct", "numpy.core.multiarray.scalar",
    "numpy._core.multiarray._reconstruct", "numpy._core.multiarray.scalar",
    "numpy.ndarray", "numpy.dtype",
})


def _module_of(global_name: str) -> str:
    return global_name.rsplit(".", 1)[0] if "." in global_name else global_name


# Modules that DANGEROUS_GLOBALS names live in, derived rather than
# hand-maintained so it can't drift out of sync as that list changes.
# An *unrecognized* reference into one of these modules (os.setuid, say --
# dangerous, just not one of the specific names above) is treated with
# the same suspicion as a listed one. An unrecognized reference to
# anything else -- almost always a user's own training-script class
# (__main__.MyModel) or a library type the allowlist doesn't happen to
# name -- is not: real checkpoints from custom code reference such
# classes constantly, and a finding that fires on nearly every real
# checkpoint gets muted, taking the genuine "os.setuid" case down with
# it. See DANGEROUS_GLOBALS/ALLOWED_GLOBALS above for why this can't just
# be "allowlist everything else."
_RISKY_MODULES = frozenset(_module_of(g) for g in DANGEROUS_GLOBALS if "." in g)


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
        # A dangerous global visible before the cap was ever hit is
        # reported as-is (CRITICAL) regardless of opcode_cap_hit -- this
        # is strictly worse information than "we couldn't finish looking".
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

    if analysis.opcode_cap_hit:
        # STOP was never reached: everything past the cap, including a
        # REDUCE that would call a dangerous primitive, was never
        # inspected. A benign checkpoint's pickle stream never needs
        # anywhere near this many opcodes -- a file that does is itself
        # the anomaly, so this fails closed (a finding) rather than open
        # (silently falling through to "no dangerous/unrecognized globals
        # found", which is only true of what could be inspected).
        return Finding(
            id="BEE-PKL-003",
            severity=Severity.HIGH,
            title="Pickle too large to fully analyze",
            description=(
                f"This pickle's opcode stream exceeds the {MAX_PICKLE_OPCODES:,}-opcode "
                "analysis limit without ever reaching STOP. No real model checkpoint "
                "needs anywhere near this many opcodes -- a file that does is worth "
                "treating as suspicious on its own. Everything past this point, "
                "including any call to a dangerous primitive, was never inspected."
            ),
            artifact_path=artifact.path,
            evidence=[
                Evidence(type="pickle_opcode_cap_exceeded", value=str(MAX_PICKLE_OPCODES),
                          source="local_filesystem", confidence=Confidence.VERIFIED)
            ],
        )

    unresolved_count = analysis.globals_referenced.count(UNRESOLVED_STACK_GLOBAL)
    unrecognized = sorted({
        g for g in analysis.globals_referenced
        if g not in ALLOWED_GLOBALS and g != UNRESOLVED_STACK_GLOBAL
    })
    if not unrecognized and not unresolved_count:
        return None

    risky = [g for g in unrecognized if _module_of(g) in _RISKY_MODULES]
    # An unresolved STACK_GLOBAL target that is then actually invoked
    # (REDUCE fires) is worse than one merely referenced: a memo-tracking
    # gap or some other opcode sequence this module doesn't model made
    # its target unrecoverable, and it's being called anyway. That's
    # treated with more suspicion than a resolved-but-unlisted risky-
    # module reference, not less.
    if unresolved_count and analysis.reduce_count:
        severity = Severity.HIGH
        title = "Pickle calls an unresolved global reference"
    elif risky or unresolved_count:
        severity = Severity.MEDIUM
        title = "Pickle references an unrecognized global in a sensitive module"
    else:
        severity = Severity.LOW
        title = "Pickle references an unrecognized global"

    parts = list(unrecognized)
    if unresolved_count:
        parts.append(f"{unresolved_count} unresolved STACK_GLOBAL reference(s)")
    return Finding(
        id="BEE-PKL-002",
        severity=severity,
        title=title,
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
