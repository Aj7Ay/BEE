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
call-graph analysis, SafeTensors bounds checking), deeper static
security analysis (GGUF metadata inspection and more), provenance,
supply-chain checks, licensing, and policy enforcement are planned in
later releases.

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

## Evidence integrity and re-verification

Every stored run carries an `evidence_sha256` — a hash of exactly what
was scanned and what was found (the target, every artifact record, every
finding, the scanner version), independent of the run's id or timestamp.
Two scans of identical, unchanged input get the same evidence hash even
without `--deterministic`; the run id identifies *which* recorded run
found it, the evidence hash identifies *what* it found.

`bee verify <run-id>` checks two different things against that record:
whether it's been altered since it was written (a hand-edited database
row, say), and whether each artifact's *current* file content still
matches the hash recorded when it was vetted:

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
