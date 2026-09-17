from __future__ import annotations

import pickletools
import zipfile
from dataclasses import dataclass, field
from typing import IO

from bee.formats.io_source import Source, open_source, size_of

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

# The opcode-count cap above bounds *how many* opcodes run, but not what
# any single one of them is allowed to claim -- a lone BINBYTES/
# BINUNICODE8 opcode backed by real file bytes can declare a
# multi-gigabyte length and still count as exactly one opcode toward
# that cap, forcing an attempt to read/allocate all of it before analysis
# can continue to whatever comes after. Bounding the whole stream's size
# before analysis ever starts closes that regardless of which single
# opcode would have tried to claim it. No real `data.pkl` (PyTorch's
# actual tensor bytes live in separate zip members, not the pickle
# stream itself) or standalone metadata pickle is anywhere near this
# size; one that is is itself the anomaly.
MAX_PICKLE_STREAM_BYTES = 64 * 1024 * 1024

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
    # True when analysis was cut short before ever reaching STOP --
    # either because the opcode cap (MAX_PICKLE_OPCODES) was hit, or
    # because an opcode declared a length/count that raised an exception
    # (a stdlib bounds-check ValueError, or MemoryError) after at least
    # one real opcode had already been parsed. Either way, everything
    # past this point, including a REDUCE that would call a dangerous
    # primitive, was never inspected. This must not be conflated with
    # "not a pickle": the file can still load and execute exactly like
    # one; analysis was just cut short.
    opcode_cap_hit: bool = False


def _analyze_stream(stream: IO[bytes]) -> PickleAnalysis | None:
    protocol: int | None = None
    found_stop = False
    hit_opcode_cap = False
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
                hit_opcode_cap = True
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
    except (ValueError, EOFError, IndexError, MemoryError):
        # A declared length/count that doesn't fit what's left in the
        # stream (pickletools' own bounds check -- ValueError) makes a
        # file that pickle.load() itself could never execute either: the
        # declared bytes aren't there, so there's no working payload
        # hiding behind it. Returning None ("not a pickle") for that case
        # is correct, not a gap -- unlike the opcode-count cap, where the
        # padding *was* real, loadable bytes with a real payload after it.
        # MemoryError is caught as a pure backstop: the opcode-count cap
        # above already bounds how many opcodes can run, and
        # MAX_PICKLE_STREAM_BYTES (enforced by both callers below) bounds
        # the size of the stream this ever gets to run against at all, so
        # a single opcode's declared length can never legitimately exceed
        # that ceiling by the time it would try to honor it.
        return None

    if not found_stop and not hit_opcode_cap:
        # Genuinely ran out of stream without ever finding STOP and
        # without hitting the cap -- not a valid pickle, not a "we
        # couldn't finish looking" case.
        return None
    return PickleAnalysis(
        protocol=protocol,
        globals_referenced=globals_seen,
        reduce_count=reduce_count,
        opcode_cap_hit=hit_opcode_cap,
    )


def _stream_too_large_analysis() -> PickleAnalysis:
    # Reported the same way as hitting the opcode cap: analysis never
    # ran at all, so nothing (dangerous or otherwise) can be named -- but
    # a stream this large is itself the anomaly, and check_pickle_calls'
    # BEE-PKL-003 handling of opcode_cap_hit already means "cut short,
    # treat as suspicious" regardless of which bound caused that.
    return PickleAnalysis(protocol=None, globals_referenced=[], reduce_count=0, opcode_cap_hit=True)


def analyze_pickle_file(source: Source) -> PickleAnalysis | None:
    """Walk every opcode of `source` as a raw pickle stream. `source` is
    either a Path (opened and read here) or the file's already-read bytes
    (see bee.formats.io_source)."""
    try:
        if size_of(source) > MAX_PICKLE_STREAM_BYTES:
            return _stream_too_large_analysis()
        with open_source(source) as f:
            return _analyze_stream(f)
    except (OSError, MemoryError):
        # MemoryError is a pure backstop here: MAX_PICKLE_STREAM_BYTES
        # above already keeps this from ever being attempted against a
        # stream large enough to plausibly exhaust memory in the first
        # place, the same belt-and-suspenders role RecursionError plays
        # for the GGUF parser's own depth cap.
        return None


def analyze_pytorch_zip_pickle(source: Source) -> PickleAnalysis | None:
    """Walk the opcodes of the `data.pkl` member embedded in a PyTorch
    zip-format checkpoint -- the actual pickle stream a real `.pt`/`.pth`
    file almost always is, as opposed to a raw pickle file. `source` is
    either a Path or the file's already-read bytes (see
    bee.formats.io_source)."""
    try:
        with zipfile.ZipFile(open_source(source)) as zf:
            member_name = next(
                (n for n in zf.namelist() if n == "data.pkl" or n.endswith("/data.pkl")), None
            )
            if member_name is None:
                return None
            # Zip metadata's declared uncompressed size is attacker-
            # controlled and not verified against the real decompressed
            # length until it's actually read -- checked here, before
            # ever opening the member, for exactly the same reason a
            # tiny multi-gigabyte-uncompressed zip ("zip bomb") is a well
            # known attack: a small compressed size is not evidence of a
            # small amount of work to decompress it.
            if zf.getinfo(member_name).file_size > MAX_PICKLE_STREAM_BYTES:
                return _stream_too_large_analysis()
            with zf.open(member_name) as f:
                return _analyze_stream(f)
    except (OSError, zipfile.BadZipFile, MemoryError):
        return None
