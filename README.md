# BEE - AI/ML Model Supply-Chain Security Vetting

![Version](https://img.shields.io/badge/version-0.14.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)
![Status](https://img.shields.io/badge/status-production--grade-brightgreen.svg)

**BEE** is a production-ready CLI tool for comprehensive security vetting of AI/ML model artifacts. Detect malicious code, track provenance, enforce policies, and generate security reports.

## Features

### 🔍 **Security Analysis**
- **Format Detection & Validation** - Detects SafeTensors, GGUF, Pickle, PyTorch, and more
- **Pickle RCE Detection** - Identifies dangerous opcodes (os.system, subprocess, eval)
- **Custom Code Scanning** - Detects os.system, subprocess, eval, network access, credential leaks, dynamic imports
- **Dependency Analysis** - Parses requirements.txt, pyproject.toml, package.json, poetry.lock, and more
- **Vulnerability Lookup** - Queries OSV.dev for known CVEs in dependencies
- **Bounds Checking** - Validates tensor metadata, detects overlaps and size bombs in GGUF/SafeTensors

### 📋 **Provenance & Policy**
- **Provenance Tracking** - Records source, hash, size, acquisition method
- **Policy Engine** - YAML-based security policies with ALLOW/REVIEW/BLOCK verdicts
- **Fail-Closed by Default** - Critical/high findings block without explicit policy override

### 📊 **Reporting**
- **JSON Output** - Machine-readable verdicts for automation
- **HTML Reports** - Dark-themed security reports with detailed findings
- **Model Cards** - BEE-compliant README.md generation from scans
- **Terminal Output** - Rich, color-coded scan summaries

## Installation

```bash
pip install bee-guard
# or with uv
uv pip install bee-guard
```

## Quick Start

### 1. Scan a Model File

```bash
# Scan a single file
bee scan model.gguf

# Scan a directory recursively
bee scan ./models/

# Get JSON output for downstream tools
bee --format json scan model.safetensors
```

### 2. Vet with Security Gate

```bash
# Vet and apply policy
bee vet --policy security-policy.yaml model.gguf

# Fail on high/critical findings
bee vet model.gguf --fail-on high
```

### 3. Generate Reports

```bash
# Generate HTML report
bee report --scan model.gguf -o security-report.html

# Generate model card from scan
bee modelcard --scan model.gguf -o README.md
```

### 4. Validate Policies

```bash
# Check policy YAML syntax
bee policy security-policy.yaml
```

## Commands

| Command | Purpose |
|---------|---------|
| `bee scan <target>` | Scan model files for security issues |
| `bee inspect <target>` | Inspect model artifact metadata |
| `bee vet <target>` | Full security vetting with policy support |
| `bee report <target>` | Generate HTML security report |
| `bee modelcard <target>` | Generate BEE-compliant model card |
| `bee policy <file>` | Validate security policy YAML |
| `bee verify <file>` | Verify signed model files |
| `bee sign <file>` | Digitally sign model artifacts |
| `bee keygen` | Generate signing keypair |
| `bee history <target>` | Show scan history from database |
| `bee show <run-id>` | Display scan results by ID |
| `bee ollama vet <model>` | Vet local Ollama models (experimental) |

## Policy Example

```yaml
integrity:
  require_sha256: true

provenance:
  require_publisher: false
  require_repository: false
  require_revision: false

formats:
  blocked: ["pickle"]

findings:
  critical: block
  high: block
  medium: review
  low: allow
  info: allow

custom_code:
  allowed: true

licenses:
  allowed: ["MIT", "Apache-2.0"]

vulnerabilities:
  critical: block
  high: block
  medium: allow
  low: allow
```

Save as `policy.yaml` and use:
```bash
bee vet --policy policy.yaml model.gguf
```

## Threat Model

BEE detects and prevents:

| Threat | Detection | Prevention |
|--------|-----------|-----------|
| **Malicious Pickle** | Opcode analysis (os.system, subprocess, eval) | BLOCK by format |
| **Size Bombs** | Bounds checking, overlap detection | BLOCK on mismatch |
| **Vulnerable Dependencies** | OSV.dev lookup | BLOCK on critical |
| **Dangerous Code** | Pattern scanning (20+ patterns) | BLOCK by policy |
| **Tampered Artifacts** | SHA-256 validation, signature verification | BLOCK on mismatch |
| **Unknown Provenance** | Source tracking, publisher verification | REVIEW without policy |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Scan complete, no blocking issues |
| `1` | Security findings at threshold level |
| `2` | Usage error (missing file, bad policy, etc.) |

## Output Formats

### JSON Mode
```bash
bee --format json vet model.gguf
```

Returns:
```json
{
  "id": "run-abc123",
  "target": "model.gguf",
  "verdict": "allow",
  "decision": "allow",
  "findings": [...],
  "severity_count": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "info": 0
  },
  "provenance": {...},
  "timestamp": "2026-09-22T15:30:00Z"
}
```

### Text Mode (default)
```
BEE SCAN
Target: model.gguf
Artifacts scanned: 1
┏━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━┓
┃ PATH          ┃ FORMAT ┃ SIZE ┃ FINDINGS ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━┩
│ model.gguf    │ gguf   │ 5.2G │ -        │
└───────────────┴────────┴──────┴──────────┘
Findings: 0 critical, 0 high, 0 medium, 0 low, 0 info
```

## Development

### Install from source
```bash
git clone https://github.com/Aj7Ay/BEE.git
cd BEE
uv install
```

### Run tests
```bash
uv run pytest
```

### Run linting
```bash
uv run ruff check src/
```

## Version History

- **0.14.0** - HTML report integration, code & dependency scanners
- **0.13.0** - Full code and dependency scanning with vulnerability lookup
- **0.12.0** - Policy engine wiring, fail-closed verdict logic
- **0.11.0** - Experimental: provenance, policy, licensing (scaffolding)
- **0.10.0** - GGUF tensor-overlap detection
- **0.9.2** - Python 3.10+ support
- **0.9.1** - Pickle declared-size bounds
- **0.9.0** - Opcode-cap detection bypass fix, terminal sanitization
- Earlier: Foundational format detection, signing, verification

## Status by Phase

| Feature | Version | Status |
|---------|---------|--------|
| Core scanning (scan, inspect) | 0.1-0.10 | ✅ Production |
| Signing & verification | 0.7-0.9 | ✅ Production |
| Policy engine | 0.12 | ✅ Production |
| Code & dependency scanners | 0.13 | ✅ Production |
| HTML reports | 0.14 | ✅ Production |
| Ollama support | 0.15 | 🔄 Experimental |
| HuggingFace support | 0.16 | 🔄 Experimental |

## License

Apache License 2.0 — see LICENSE file

## Contributing

We welcome contributions. Please open an issue or submit a pull request on [GitHub](https://github.com/Aj7Ay/BEE).

## Security

Report security vulnerabilities to [security@example.com](mailto:security@example.com). Do not open public issues for security bugs.

---

**BEE**: Because model safety is not optional.
