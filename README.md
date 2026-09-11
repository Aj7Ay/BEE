# BEE

BEE is a CLI for vetting AI/ML model artifacts before they enter your
environment. It establishes an artifact's identity, detects its real
structural format (never trusting the file extension), and flags
mismatches between the two.

This is early — **0.1, the foundation release**. It covers local-file
identity and format detection only. Real static security analysis
(pickle call-graph analysis, SafeTensors bounds checks, GGUF metadata
inspection, and more), provenance, supply-chain checks, licensing, and
policy enforcement are planned in later releases — see
[`docs/superpowers/specs`](docs/superpowers/specs) for the design behind
where this is headed.

## Install

```bash
pip install bee
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
bee scan ./models --format json
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
Findings: 0 critical, 0 high, 1 medium, 0 low, 0 info
```

`weights.pt` is flagged (`BEE-FMT-001`) because its extension claims
PyTorch but the file is structurally a NumPy array — exactly the kind of
mismatch a renamed or mislabeled artifact would produce.

## What BEE detects today

Structural signatures for: SafeTensors, GGUF, NumPy, HDF5/Keras, Pickle,
PyTorch (zip-based), ONNX (heuristic), and generic zip/tar/gzip archives.
Anything else is reported as `unknown` rather than guessed.

## Development

```bash
uv sync
uv run pytest -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
