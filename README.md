
# 🐝 BEE - AI Model Security & Supply-Chain Vetting

<p align="center">
<img src="banner-bee.png" alt="BEE - AI Model Security" width="900"/>
</p>

<p align="center">
<strong>Scan • Verify • Vet • Evidence • Secure AI Models</strong>
</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Aj7Ay%2FBEE-black)](https://github.com/Aj7Ay/BEE)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Aj7Ay/BEE/releases/tag/1.0.0)
[![Tests](https://img.shields.io/badge/tests-286%20passing-brightgreen.svg)](#testing)

</p>

---

## What is BEE?

**BEE** is an open-source security tool for vetting AI/ML model artifacts before they enter development, CI/CD pipelines, model registries, or production environments.

BEE analyzes model artifacts for:

- **Malicious serialization** (pickle RCE, dangerous opcodes)
- **Format violations** (GGUF/SafeTensors bounds, overlaps, tensor misalignment)
- **Custom code security** (os.system, subprocess, eval, network access)
- **Vulnerable dependencies** (OSV.dev integration, CVE lookup)
- **Artifact integrity** (SHA-256 hashing, Ed25519 signatures)
- **Provenance tracking** (source, repository, revision, acquisition method)
- **Policy enforcement** (YAML-based ALLOW/REVIEW/BLOCK gates)

BEE produces **machine-readable security evidence** (JSON) and human-readable reports (HTML, terminal) that can be used in CI/CD and AI model supply-chain workflows.

> **BEE doesn't create trust. BEE helps you verify trust.**

---

## Why BEE?

AI models are software supply-chain artifacts.

A model repository can contain much more than model weights:

```
AI Model Repository
│
├── model.safetensors
├── model.gguf
├── pytorch_model.bin
├── config.json
├── tokenizer.json
├── modeling_custom.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

A model can therefore introduce risk through serialization, format, custom code, dependencies, provenance, and integrity.

BEE helps security and engineering teams inspect these components before deployment.

---

## Core Security Features

### 🔍 Format Detection & Bounds Checking

- GGUF: magic/version, tensor table, offsets, alignment, byte-size, quantization
- SafeTensors: tensor ranges, out-of-bounds data, overlaps, shape/size mismatches
- Pickle: opcode analysis, dangerous callable detection
- PyTorch, ONNX, NumPy, HDF5/Keras, archives
- Format mismatch detection (doesn't trust file extensions)

### 💀 Pickle Security

Detects dangerous behavior without executing:

```
os.system
subprocess
eval / exec
dangerous callable reconstruction
suspicious opcode patterns
all pickle protocol versions (0-5)
evasion techniques (trailing bytes, opcode-cap padding, memo indirection)
```

### 🧱 Tensor Validation

- Tensor overlap detection
- Out-of-bounds access
- Implausible dimensions
- Size/resource exhaustion
- Declared-size bombs

### 💻 Custom Code Analysis

Scans for security-sensitive patterns:

```
os.system, subprocess, eval, exec
network access (requests, urllib, socket)
environment/credential access
dynamic imports
file operations
download-and-execute patterns
```

### 📦 Dependency Analysis

- Parses: requirements.txt, pyproject.toml, package.json, poetry.lock, Pipfile.lock, environment.yml
- Identifies: package, version, source, direct/transitive relationships
- Enriches with OSV.dev vulnerability lookup (when network available)

### 🔐 Artifact Integrity

- SHA-256 hashing
- Ed25519 digital signatures
- Tamper detection
- Signature verification with key pinning

### 📋 Provenance Tracking

Records and validates:

```json
{
  "source": {
    "provider": "huggingface",
    "repository": "organization/model",
    "revision": "abc123"
  },
  "artifact": {
    "filename": "model.safetensors",
    "sha256": "...",
    "size": 5242880
  }
}
```

### 🛡️ Policy Enforcement

YAML-based security policies with fail-closed defaults:

```yaml
findings:
  critical: block
  high: block
  medium: review
  low: allow

formats:
  blocked:
    - pickle

vulnerabilities:
  critical: block
  high: block
```

Decisions: **ALLOW** | **REVIEW** | **BLOCK**

### 🐳 Remote Model Vetting

- **Ollama**: Vet local Ollama models (`bee ollama vet qwen3:8b`)
- **HuggingFace**: Scan models from the Hub (`bee hf meta-llama/Llama-2-7b`)
- Filename validation (rejects path traversal, absolute paths)

### 📊 Security Reports

- **JSON**: Machine-readable evidence for automation, CI/CD, AIBOM
- **HTML**: Dark-themed security report with findings, severity, evidence
- **Model Cards**: BEE-compliant README.md generation
- **Terminal**: Color-coded output with findings summary

---

## Installation

### PyPI

```bash
pip install bee-guard
```

### uv

```bash
uv pip install bee-guard
```

### From source

```bash
git clone https://github.com/Aj7Ay/BEE.git
cd BEE
uv sync
```

Verify:

```bash
bee --version
```

---

## Quick Start

### Scan a model

```bash
bee scan model.gguf
bee scan ./models/
```

### Full vetting with policy

```bash
bee vet --policy security.yaml model.gguf
```

### Vet remote models

```bash
bee hf meta-llama/Llama-2-7b
bee ollama vet qwen3:8b
```

### Generate reports

```bash
# JSON for automation
bee --format json vet model.gguf

# HTML report
bee report --scan model.gguf -o security-report.html

# Model card
bee modelcard --scan model.gguf -o README.md
```

### Sign and verify

```bash
bee keygen              # Generate signing key
bee sign model.gguf     # Sign artifact
bee verify model.gguf   # Verify signature
```

---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `bee scan <target>` | Scan model artifacts |
| `bee inspect <target>` | Inspect model metadata |
| `bee vet <target>` | Full security vetting |
| `bee report <target>` | Generate HTML security report |
| `bee modelcard <target>` | Generate BEE model card |
| `bee policy <file>` | Validate security policy |
| `bee verify <file>` | Verify signed artifacts |
| `bee sign <file>` | Sign artifacts/evidence |
| `bee keygen` | Generate signing keys |
| `bee history <target>` | Show scan history |
| `bee show <run-id>` | Display scan results |
| `bee ollama vet <model>` | Vet a local Ollama model |
| `bee hf <org/model>` | Vet a HuggingFace model |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Scan complete, no blocking issues |
| `1` | Security findings at threshold level |
| `2` | Usage error (missing file, bad policy) |

---

## Threat Model

BEE addresses AI model supply-chain threats:

| Threat | Detection | Action |
|--------|-----------|--------|
| Malicious Pickle | Opcode analysis | BLOCK |
| GGUF tensor overlap | Structural analysis | BLOCK |
| Tensor size bombs | Bounds checking | BLOCK |
| SafeTensors overlap | Bounds checking | BLOCK |
| Dangerous custom code | Static code analysis | REVIEW/BLOCK |
| Vulnerable dependencies | OSV lookup | REVIEW/BLOCK |
| Artifact tampering | SHA-256/signatures | BLOCK |
| Unknown provenance | Provenance analysis | REVIEW |
| Format mismatch | Format validation | REVIEW/BLOCK |

---

## Machine-Readable Evidence

```bash
bee --format json vet model.gguf
```

JSON output includes:

```json
{
  "id": "run-abc123",
  "target": "model.gguf",
  "verdict": "allow",
  "decision": "allow",
  "findings": [],
  "severity_count": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "info": 0
  },
  "provenance": {},
  "timestamp": "2026-09-22T15:30:00Z"
}
```

Consumed by: CI/CD, AIBOM, security platforms, compliance workflows.

---

## What BEE Does NOT Claim

A result of `0 findings` means:

> No issue was detected by the configured BEE checks.

It does **not** mean the model is universally safe.

Complete model security requires:

- Publisher verification
- Provenance verification
- Human review
- Model behavior testing
- Runtime isolation
- Access control
- Monitoring
- Organizational policy

**BEE is one layer of a defense-in-depth AI security architecture.**

---

## Testing

```bash
uv sync
uv run pytest
uv run ruff check src/
```

BEE's test suite includes 286+ tests covering:

- Adversarial model fixtures
- Malformed structures
- Evasion techniques
- Integration paths
- Policy logic
- Full regression

---

## Contributing

Contributions welcome in:

- New model-format analyzers
- Security test fixtures
- Parser hardening
- Fuzzing
- Provenance integrations
- Policy rules
- Vulnerability integrations
- CI/CD integrations
- Documentation

---

## Security

If you discover a security vulnerability in BEE, please do not create a public issue.

Use GitHub's private vulnerability reporting mechanism or contact the maintainers directly.

For questions, open a GitHub discussion or issue.

---

## License

Apache License 2.0

See [LICENSE](LICENSE).

---

## Philosophy

AI models are becoming software supply-chain artifacts.

They should be:

```
Discovered
    ↓
Identified
    ↓
Analyzed
    ↓
Verified
    ↓
Vetted
    ↓
Documented
    ↓
Policy Checked
    ↓
Audited
```

Not blindly downloaded and deployed.

> **BEE — Scan. Verify. Vet. Secure AI Models.**

