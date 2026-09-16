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

# Opcodes that record the current stack-top value into the memo table for
# later recall, and the ones that recall it. MEMOIZE (protocol 4+) uses an
# implicit index (the memo table's current size, exactly as CPython's own
# unpickler assigns it); PUT/BINPUT/LONG_BINPUT (older, but not actually
# barred from appearing in any stream -- pickle's VM doesn't enforce
# protocol consistency, only a real pickler does) carry an explicit index.
# GET/BINGET/LONG_BINGET recall by that same index.
#
# This module tracks memo indirection specifically because a hand-crafted
# pickle (not one written by CPython's own pickler) can route a
# STACK_GLOBAL's module/qualname strings through the memo instead of
# pushing them immediately beforehand -- MEMOIZE the module string,
# MEMOIZE the qualname string, do something else, then BINGET each one
# right before STACK_GLOBAL. That's a real, verified evasion: a pickle
# built this way loads and executes exactly like one that doesn't, and an
# implementation that only watches for immediately-preceding string
# pushes reports it as an unresolved reference instead of the dangerous
# primitive it actually calls.
_MEMO_STORE_OPS_IMPLICIT = frozenset({"MEMOIZE"})
_MEMO_STORE_OPS_EXPLICIT = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_MEMO_RECALL_OPS = frozenset({"GET", "BINGET", "LONG_BINGET"})

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
    # Tracks the last two string-valued items placed on top of the
    # interpreter stack -- whether pushed directly (a string-push opcode)
    # or recalled from the memo (BINGET et al.) -- to resolve
    # STACK_GLOBAL's (module, qualname) pair. Not a full stack simulation:
    # any opcode this module doesn't specifically recognize as producing
    # or recalling a string clears it, so a stale pair is never attributed
    # to a later STACK_GLOBAL.
    pending_strings: list[str] = []
    # The memo table, keyed by index. Only string values are ever stored
    # (a memoized non-string is recorded as None, still occupying its
    # slot so implicit MEMOIZE indices -- which are just the table's
    # current size -- stay correctly aligned with what a real unpickler
    # would assign).
    memo: dict[int, str | None] = {}
    # The string most recently placed on top of the stack by an opcode
    # this module tracks, valid only until the next opcode -- what a
    # MEMOIZE occurring right now would be recording.
    stack_top_string: str | None = None

    try:
        for i, (opcode, arg, _pos) in enumerate(pickletools.genops(stream)):
            if i >= MAX_PICKLE_OPCODES:
                break
            name = opcode.name
            next_stack_top_string: str | None = None  # what this opcode leaves on top, if a tracked string

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
                next_stack_top_string = arg
            elif name in _MEMO_STORE_OPS_IMPLICIT:
                # Stack-neutral: MEMOIZE records the current top without
                # popping it, so both the memo write and the tracking
                # below use stack_top_string as it already stood.
                memo[len(memo)] = stack_top_string
                next_stack_top_string = stack_top_string
            elif name in _MEMO_STORE_OPS_EXPLICIT and isinstance(arg, int):
                memo[arg] = stack_top_string
                next_stack_top_string = stack_top_string
            elif name in _MEMO_RECALL_OPS and isinstance(arg, int):
                recalled = memo.get(arg)
                if recalled is not None:
                    pending_strings.append(recalled)
                    del pending_strings[:-2]
                    next_stack_top_string = recalled
                else:
                    pending_strings.clear()
            else:
                pending_strings.clear()

            stack_top_string = next_stack_top_string

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
