"""Seeded random fuzzing of bee.formats.gguf_ops.analyze_gguf.

analyze_gguf's own contract is: return a GgufAnalysis, or return None --
never raise, no matter what bytes it is given. This test tries to break
that contract with random and structured-random inputs. A fixed seed
means a failure here is reproducible: rerun with the same seed printed
in the failure message to get the exact same input again.
"""

import random
import struct
import time

from bee.formats.gguf_ops import GGUF_MAGIC, analyze_gguf

_SEED = 20260918
_ITERATIONS = 2000
_MAX_TOTAL_SECONDS = 15.0


def _random_bytes(rng: random.Random, min_len: int, max_len: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(rng.randint(min_len, max_len)))


def _mutate_valid_prefix(rng: random.Random) -> bytes:
    # Start from a real, well-formed GGUF header shape, then flip or
    # replace random byte ranges -- more likely to reach deep code paths
    # (past the magic-bytes check) than pure random bytes are.
    header = (
        GGUF_MAGIC
        + struct.pack("<I", rng.choice([1, 2, 3, 4, 999999]))
        + struct.pack("<Q", rng.choice([0, 1, 2, 2**32, 2**63, 2**64 - 1]))
        + struct.pack("<Q", rng.choice([0, 1, 2, 2**32, 2**63, 2**64 - 1]))
    )
    tail = _random_bytes(rng, 0, 4096)
    return header + tail


def test_analyze_gguf_never_raises_on_random_or_mutated_input(tmp_path):
    rng = random.Random(_SEED)
    start = time.monotonic()

    for i in range(_ITERATIONS):
        if i % 2 == 0:
            data = _random_bytes(rng, 0, 512)
        else:
            data = _mutate_valid_prefix(rng)

        path = tmp_path / "fuzz_input.gguf"
        path.write_bytes(data)

        try:
            analyze_gguf(path)
        except Exception as exc:  # noqa: BLE001 -- the fuzzer's whole job is catching this
            raise AssertionError(
                f"analyze_gguf raised {type(exc).__name__}: {exc} "
                f"on iteration {i} (seed={_SEED}). "
                f"Input ({len(data)} bytes): {data!r}"
            ) from exc

    elapsed = time.monotonic() - start
    assert elapsed < _MAX_TOTAL_SECONDS, (
        f"{_ITERATIONS} fuzz iterations took {elapsed:.2f}s -- "
        f"a real caller would see this as a hang, not a crash"
    )