from __future__ import annotations

import pickletools
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

# Bound analysis by OPCODE COUNT, not by a byte-offset/prefix-size cutoff.
# A fixed byte prefix read means any payload padded past that prefix (e.g.
# embedded in a multi-megabyte disguised checkpoint) falls outside the
# sniffed window and is misreported as "unknown" -- recreating the exact
# evasion this detector exists to close. Streaming genops() from an open
# handle instead means STOP is found at whatever byte offset it actually
# occurs at, however large the file -- a real payload (malicious or not)
# is a handful of opcodes regardless of how much padding surrounds it.
# The opcode cap exists only to bound CPU against a pathological "opcode
# bomb" (millions of tiny opcodes before ever reaching STOP), not to limit
# how far into the file we're willing to look.
MAX_PICKLE_OPCODES = 100_000

# Opcodes that push exactly one string value onto the interpreter stack.
# Relevant only for resolving STACK_GLOBAL (protocol 4+), which pops the
# two most recently pushed strings as (module, qualname) rather than
# carrying its target as its own opcode argument the way GLOBAL does.
_STRING_PUSH_OPS = frozenset({
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
})

# MEMOIZE (protocol 4+) records a reference to the current stack top for
# later BINGET/LONG_BINGET use -- it doesn't push or pop anything itself,
# so it must not break the (string, string, STACK_GLOBAL) sequence this
# module is watching for.
_STACK_NEUTRAL_OPS = frozenset({"MEMOIZE"})

UNRESOLVED_STACK_GLOBAL = "<unresolved_stack_global>"


@dataclass
class PickleAnalysis:
    protocol: int | None
    globals_referenced: list[str] = field(default_factory=list)
    reduce_count: int = 0


def _analyze_stream(stream: IO[bytes]) -> PickleAnalysis | None:
    protocol: int | None = None
    found_stop = False
    globals_seen: list[str] = []
    reduce_count = 0
    # Best-effort tracking of the last two pushed string literals, to
    # resolve STACK_GLOBAL's (module, qualname) pair. This is not a full
    # stack simulation -- it doesn't need to be: CPython's own pickler
    # always writes a GLOBAL/STACK_GLOBAL reference as two consecutive
    # string-push opcodes (optionally MEMOIZE'd) immediately followed by
    # STACK_GLOBAL, because that's the only sequence the pickle VM accepts
    # for it. Any other opcode appearing in between means whatever was
    # being tracked is stale, so it's dropped rather than risk attributing
    # the wrong pair to a later STACK_GLOBAL.
    pending_strings: list[str] = []

    try:
        for i, (opcode, arg, _pos) in enumerate(pickletools.genops(stream)):
            if i >= MAX_PICKLE_OPCODES:
                break
            name = opcode.name
            if name == "PROTO":
                protocol = arg
            elif name == "GLOBAL":
                if isinstance(arg, str):
                    # read_stringnl_noescape_pair joins "module" and
                    # "qualname" with a single space.
                    globals_seen.append(arg.replace(" ", ".", 1))
                pending_strings.clear()
            elif name == "STACK_GLOBAL":
                if len(pending_strings) >= 2:
                    module, qualname = pending_strings[-2], pending_strings[-1]
                    globals_seen.append(f"{module}.{qualname}")
                else:
                    globals_seen.append(UNRESOLVED_STACK_GLOBAL)
                pending_strings.clear()
            elif name == "REDUCE":
                reduce_count += 1
            elif name in _STRING_PUSH_OPS and isinstance(arg, str):
                pending_strings.append(arg)
                del pending_strings[:-2]
            elif name not in _STACK_NEUTRAL_OPS:
                pending_strings.clear()

            if name == "STOP":
                found_stop = True
                break
    except (ValueError, EOFError, IndexError):
        return None

    if not found_stop:
        return None
    return PickleAnalysis(protocol=protocol, globals_referenced=globals_seen, reduce_count=reduce_count)


def analyze_pickle_file(path: Path) -> PickleAnalysis | None:
    """Walk every opcode of `path` as a raw pickle stream."""
    try:
        with path.open("rb") as f:
            return _analyze_stream(f)
    except OSError:
        return None


def analyze_pytorch_zip_pickle(path: Path) -> PickleAnalysis | None:
    """Walk the opcodes of the `data.pkl` member embedded in a PyTorch
    zip-format checkpoint -- the actual pickle stream a real `.pt`/`.pth`
    file almost always is, as opposed to a raw pickle file."""
    try:
        with zipfile.ZipFile(path) as zf:
            member_name = next(
                (n for n in zf.namelist() if n == "data.pkl" or n.endswith("/data.pkl")), None
            )
            if member_name is None:
                return None
            with zf.open(member_name) as f:
                return _analyze_stream(f)
    except (OSError, zipfile.BadZipFile):
        return None
