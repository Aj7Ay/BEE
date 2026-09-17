# BEE
![BEE](banner-bee.png)

**B**inary, **E**vidence & **E**valuation.

BEE is a CLI for vetting AI/ML model artifacts before they enter your
environment: it examines the **binary** artifact itself, gathers
**evidence** about what it actually is, and produces an **evaluation** —
a concrete, explained finding rather than a bare pass/fail label. Today
that means establishing an artifact's identity, detecting its real
structural format (never trusting the file extension), and flagging
mismatches between the two.

This is early. Beyond what's below (format-mismatch detection, pickle
call-graph analysis, SafeTensors bounds checking, GGUF metadata
inspection, evidence integrity and signing), provenance, supply-chain
checks, licensing, and policy enforcement are planned in later
releases.

## Install

```bash
pip install bee-guard
```

or, for local development:

```bash
git clone https://github.com/Aj7Ay/BEE.git
cd BEE
uv sync
```

## Usage

```bash
# Initialize a workspace in the current directory
bee init

# Scan a file or directory
bee scan ./models

# Inspect a single artifact in detail
bee inspect ./models/model.safetensors

# JSON output, for scripting or CI
# --format is a root option, so it comes before the subcommand
bee --format json scan ./models

# Fail the build if anything at or above a severity is found
# --fail-on works identically on scan, inspect, and show
bee scan ./models --fail-on high
bee inspect ./model.safetensors --fail-on critical

# Reproducible output: identical input -> byte-identical JSON
bee --format json scan ./models --deterministic

# Past runs, and re-displaying one by id -- e.g. re-checking a stored
# run in CI without re-scanning
bee history
bee show <run-id>
bee show <run-id> --fail-on critical

# Has anything changed since this was vetted?
bee verify <run-id>

# Cryptographically sign a vetting record, and verify it later
bee keygen
bee sign <run-id>
bee verify <run-id>   # now checks the signature, not just a plain hash
```

### Example

```
$ bee scan ./models
BEE SCAN
Target: models
Artifacts scanned: 2
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ PATH              ┃ FORMAT ┃ SIZE ┃ FINDINGS ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│ models/model.gguf │ gguf   │ 16   │ -        │
│ models/weights.pt │ numpy  │ 24   │ 1        │
└───────────────────┴────────┴──────┴──────────┘
Findings: 0 critical, 0 high, 0 medium, 1 low, 0 info
```

`weights.pt` is flagged (`BEE-FMT-001`) because its extension claims
PyTorch but the file is structurally a NumPy array — exactly the kind of
mismatch a renamed or mislabeled artifact would produce.

## Pickle call-graph analysis

A pickle-based file can be exactly what it claims to be — no format
mismatch, correctly named `.pt` — and still execute arbitrary code the
moment it's loaded. BEE reads the actual opcode stream (for both raw
pickle files and PyTorch's zip-wrapped checkpoints) and reports what it
references — including resolving a target reached through memo
indirection (`MEMOIZE`/`PUT` + `GET`/`BINGET`) rather than only one
pushed immediately before the reference, which a hand-crafted (not
`pickle.dumps()`-produced) payload can use to reach the same call while
evading a naive "last two strings" tracker:

- **`BEE-PKL-001` (critical)** — references a known code-execution or
  destructive primitive (`os.system`, `subprocess.Popen`, `eval`,
  `shutil.rmtree`, ...) and names exactly which one
- **`BEE-PKL-002` (medium)** — references something unrecognized in a
  module that *also* contains known-dangerous primitives (an `os.*` or
  `subprocess.*` function not on the exact list above) — as suspicious as
  an exact match, just not one BEE can name with full confidence
- **`BEE-PKL-002` (low)** — references something else unrecognized (a
  user's own training-script class, an uncommon library type) that's
  neither dangerous nor a known-safe checkpoint helper
  (`torch._utils._rebuild_tensor_v2`, `collections.OrderedDict`, ...).
  This is the common case for a real checkpoint from custom code, which
  is exactly why it's LOW and not MEDIUM — a finding that fires on nearly
  every real model gets muted, taking the genuine `os.*` case down with it
- **`BEE-PKL-003` (high)** — analysis was cut short before ever reaching
  `STOP`, for either of two reasons. (1) The opcode stream exceeds BEE's
  100,000-opcode limit: analysis is bounded by opcode *count*, not a
  byte-offset prefix, specifically so padding can't push a payload out of
  a fixed sniff window — but the cap itself was, for one release, a
  second way to do the same thing (pad past 100,000 trivial opcodes and
  a dangerous primitive placed after the cutoff was never reached,
  reported as `unknown` with zero findings). (2) The stream itself
  exceeds 64MB: a single opcode can legitimately claim a huge amount of
  *real* data (a multi-gigabyte `BINBYTES` blob backed by actual bytes,
  not just a declared length) and still count as one opcode toward the
  count cap — bounding the whole stream's size closes that regardless of
  which opcode would have tried to claim it. No real `data.pkl` (a real
  checkpoint's tensor bytes live in separate zip members, never the
  pickle stream itself) needs anywhere near either limit, so hitting
  either now fails closed — reported as suspicious — instead of falling
  through to a clean, unanalyzed "unknown"

```
$ bee scan ./models
BEE SCAN
Target: models
Artifacts scanned: 2
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ PATH                    ┃ FORMAT  ┃ SIZE ┃ FINDINGS ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│ models/clean.pt         │ pytorch │ 287  │ -        │
│ models/legit_looking.pt │ pytorch │ 297  │ 1        │
└─────────────────────────┴─────────┴──────┴──────────┘
Findings: 1 critical, 0 high, 0 medium, 0 low, 0 info

Critical/High findings:
  BEE-PKL-001  models/legit_looking.pt: Pickle references a dangerous primitive
```

Both files here are honestly named, correctly formatted PyTorch
checkpoints — no `BEE-FMT-001` involved. `legit_looking.pt` is flagged
because its embedded pickle references `posix.system` (how `os.system`
resolves internally) and calls it via `REDUCE` on load.

The allowlist behind the LOW/MEDIUM split is calibrated against a real
corpus, not guessed: 10 checkpoints downloaded from Hugging Face across
GPT-2, BERT, T5, GPT-Neo, ViT, DistilBERT, RoBERTa, and CLIP. Before
calibration, 9 of the 10 tripped `BEE-PKL-002` on legacy
`torch.*Storage` classes referenced by PyTorch's own save format — real
noise, not a real finding. After, all 10 scan clean.

## SafeTensors bounds checking

SafeTensors' container format can be well-formed — a valid length
prefix, valid JSON — while its header still lies about where a tensor's
bytes actually live. `BEE-STS-001` (high) checks every declared
`data_offsets` range against the file, the other tensors, and the
shape/dtype that's supposed to back it:

- a range that runs past the end of the file
- two tensors claiming overlapping bytes
- a declared shape × dtype that doesn't match the byte range claimed for it
- an implausible tensor count or element count, bounded rather than
  computed exactly — an attacker-controlled header shouldn't be able to
  turn "check the bounds" into its own CPU/memory exhaustion attack

`BEE-STS-002` (low) separately flags bytes in the data region that no
tensor's range covers at all — every real file checked scans with zero
such gap, so any gap is worth a look, not something a naive loader would
ever see since it only reads what a tensor points at.

```
$ bee inspect attacked.safetensors --fail-on high
Detected format:  safetensors (supported)
Finding:          BEE-STS-001 [high] SafeTensors header declares invalid or overlapping tensor ranges
```

(`attacked.safetensors` here is a real, otherwise-valid GPT-2 checkpoint
with one tensor's `data_offsets` end pushed 10MB past the actual file —
the header still parses as valid JSON, so format detection alone would
call this a clean safetensors file.)

## GGUF metadata inspection

`bee inspect` parses a GGUF file's header, metadata key/value section,
and tensor info table — never the tensor data itself — and surfaces the
architecture, tensor count, and metadata count alongside the usual
identity fields:

```
$ bee inspect tinyllama.gguf
Detected format:  gguf (verified)
GGUF version:     3
GGUF tensors:     201
GGUF metadata kv: 23
GGUF architecture: llama
Finding:          none
```

`BEE-GGUF-001` (high) cross-checks every tensor's declared offset and
shape/type against the file: an offset that isn't a multiple of the
file's own declared alignment, one that starts past the end of the
file, or a declared shape whose byte size runs off the end of it.
Tensor byte sizes are computed exactly for every GGML quantization
format (F32 down through the K-quants and beyond), not just plain
float/int types. The single most common real-world trigger is an
incomplete download, not an attack — but a naive loader would still
read past the file's own end either way:

```
$ bee inspect small.gguf --fail-on high
Detected format:  gguf (verified)
GGUF version:     3
GGUF tensors:     201
GGUF metadata kv: 21
GGUF architecture: llama
Finding:          BEE-GGUF-001 [high] GGUF tensor table declares invalid or out-of-bounds data
```

(`small.gguf` here is a real TinyLlama Q2_K checkpoint that was cut off
mid-download at 310,558,210 of its actual 483,116,416 bytes — the magic
bytes, header, and metadata are all intact, so format detection alone
would call this a valid GGUF file.)

## Evidence integrity and re-verification

Every stored run carries an `evidence_sha256` — a hash of exactly what
was scanned and what was found (the target, every artifact record, every
finding, the scanner version), independent of the run's id or timestamp.
Two scans of identical, unchanged input get the same evidence hash even
without `--deterministic`; the run id identifies *which* recorded run
found it, the evidence hash identifies *what* it found.

`bee verify <run-id>` checks two different things against that record:
whether it's internally self-consistent with what's stored, and whether
each artifact's *current* file content still matches the hash recorded
when it was vetted:

```
$ bee verify b74ff8e6-d316-4d7a-9dc5-ff536d4f5deb
Evidence record:  OK (3863d8f944c2a230...)

Artifacts:
  CHANGED  models/weights.gguf
           recorded: 1ef5107f394ec3b832bbcf48f4723c94100a3fe3da695bce60392a882777b7ff
           current:  dae83aba02090c4963c2573ce22cbb2f466afb554547dc60524c6b90273809d3
```

A path that was a normal file (or a safe in-root symlink) when it was
vetted, but has since been replaced with a symlink escaping the original
scan root, is flagged as `ESCAPED` rather than silently re-hashed — the
same protection `scan`/`inspect` apply during vetting also holds during
re-verification.

**What an unsigned record does and doesn't prove:** `evidence_sha256` is
a plain, unkeyed sha256 stored right next to the data it covers. It
catches accidental corruption and naive edits to the stored record —
not a capable attacker who can write to the database, since the
algorithm is public and such an attacker can simply recompute a
matching hash after editing it. A clean `bee verify` on an *unsigned*
run means the record is internally self-consistent; it isn't a
cryptographic guarantee that no one who understands this format has
touched it. `bee sign` is what closes that gap — see below.

**What the recorded hash is guaranteed to match:** hashing, format
detection, and deep analysis (pickle opcodes, SafeTensors/GGUF headers)
used to each independently re-open the scanned path — meaning a file
replaced on disk mid-scan could make the hash BEE recorded and the
content it actually analyzed for danger two different files. For any
file at or under 64MB, BEE now reads it into memory exactly once and
reuses that same buffer for every stage, closing that window entirely
regardless of what happens to the path afterwards. A file larger than
that still uses the previous per-stage re-open behavior — buffering a
multi-gigabyte checkpoint just to close a race window would trade one
resource-exhaustion problem for another, so the bound is deliberate,
not an oversight. Closing it above that size would require staying on
one open file descriptor for the artifact's entire processing window,
which is a larger structural change than this release makes.

## Signing

`bee keygen` generates an Ed25519 keypair in `~/.bee/keys/` by default —
outside any project's `.bee/` workspace, so a project's database can be
freely copied or shared without the private key going with it. `bee sign
<run-id>` signs the run's evidence hash; `bee verify` then checks the
signature instead of falling back to the plain hash comparison.

The difference matters exactly where the plain hash fails: an attacker
who edits a signed run's findings and recomputes `evidence_sha256` to
match — the same forgery a plain hash comparison can't detect — still
can't produce a signature that verifies against the new content,
because the private key never touches the database:

```
$ bee verify a8d7761e-f342-4865-b38a-1f1a3d96e531
Evidence record:  SIGNED, INVALID (signer 305ddefe24f7baf3)
  the signature does not verify against the current recorded content
```

The public key travels with the signature inside the record itself, so
verifying never requires access to the signer's key files — only the
record and the signature it already carries.

**Pin an expected signer — this is the part that makes signing actually
load-bearing.** Without a pin, `bee verify` only proves the record is
signed by *some* key embedded in it: an attacker who forges content can
just discard your signature and re-sign under a key of their own, and
`bee verify` correctly reports `VALID` — it never claimed to check
*whose* key. Pinning with `--signer <fingerprint>` or `--pubkey
<path>` closes that:

```
$ bee verify <run-id> --signer a361824b3549aa8c
Evidence record:  SIGNED BY UNEXPECTED KEY (got 5da0a589531a65a9, expected a361824b3549aa8c)
```

That's a real forge-and-re-sign attack caught: the same evidence-tampering
forgery from above, but this time the attacker also generated their own
keypair and signed the forged record with it. The signature is
perfectly valid — under the wrong key. `--signer`/`--pubkey` is what a
CI gate actually needs: not "this record is self-consistent," but
"this record was vouched for by someone I trust."

## What BEE detects today

Structural signatures for: SafeTensors, GGUF, NumPy, HDF5/Keras, Pickle
(all protocols, resistant to trailing-byte padding), PyTorch (zip-based),
ONNX (structural heuristic), and generic zip/tar/gzip archives. Anything
else is reported as `unknown` rather than guessed.

`bee scan` and `bee inspect` both flag:
- symlinks whose target resolves outside the scanned/inspected
  directory (`BEE-SYM-001`) — the target is never opened (so never
  hashed) unless you pass `--follow-symlinks`; the same rule applies
  whether you point `inspect` at the symlink directly or `scan` finds
  it while walking a directory
- files it couldn't read, without aborting the rest of the scan
  (`BEE-IO-001`)

The magic-bytes field shown by `inspect` (and stored per-artifact by
`scan`) records exactly the bytes a detector matched on — nothing more.
For a format whose signature is a real fixed byte sequence (GGUF, NumPy,
HDF5, a zip/gzip magic), that's the signature itself, at whatever offset
it actually lives at. For anything else — `unknown`, but also
safetensors, pickle, PyTorch, tar, ONNX, whose evidence is descriptive
rather than a raw byte match — it's left empty, never a blind fixed-size
read from the start of the file that could just as easily land on
someone's `.env` contents or an archive member's filename.

## Development

```bash
uv sync
uv run pytest -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
