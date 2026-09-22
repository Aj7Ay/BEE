# BEE 1.0 — Implementation Plan (v2, revised after code audit)

**Version:** 0.10.0 → 1.0.0
**Status:** A first attempt at this plan was built already. It does not work. This document replaces the first plan.

---

## 1. Why this plan is different from the first one

The first plan tried to build 9 phases at once, with no tests. I checked the result. Here is what I found.

| Command | Result |
|---|---|
| `bee vet` | Crashes on every file. Wrong variable name in the code. |
| `bee ollama` (list, inspect, or vet) | Crashes on the first line. Wrong number of arguments. |
| `bee report` | Crashes. It calls `vet` inside it. |
| `bee modelcard` | Crashes. It calls `vet` inside it. |
| `bee policy` | Does not exist. The code is there, but nothing connects it to the menu. |

Other problems found in the same batch of code:

- Zero new tests. The first plan asked for 500+. None were written.
- A second, new GGUF file reader (`evidence/gguf.py`) with none of the safety limits we spent four releases adding to the real one. It has no cap on string size or array size.
- A dependency scanner that will crash on Python 3.10 — the same Python version we just added support for — because it needs a package (`tomli`) that is not installed.
- An HTML report that crashes when it has real vulnerability data, from a case-sensitive text bug.
- Three new hard dependencies (`requests`, `pyyaml`, `huggingface_hub`) added for every user, even users who never touch these new features.
- A fallback import that reaches into `pip`'s private internal code. This can break at any time.
- A finished-looking command (`bee ollama vet`) that actually just prints `[TBD: full vet integration with orchestrator]` and does nothing.

**Root cause:** the whole batch was built in one pass, with no test after each piece, and no run of the real `bee` command to check it. This plan fixes that by never doing that again.

---

## 2. The one rule for this plan

**One phase. One release. Real tests. Real files. Then the next phase starts.**

No phase starts until the phase before it is:
1. Fully tested (new tests pass, all old tests still pass).
2. Run against a real file on this Mac, not just a made-up test file.
3. Committed, pushed, green in CI on all 4 Python versions, tagged, released, and live on PyPI.

This is the same rule this project has used for every release from 0.1.0 to 0.10.0. The first version of this plan broke that rule. It failed because of that.

---

## 3. Decision: cut `bee ollama list` and `bee ollama inspect`

Ollama already has its own `ollama list` command. Copying it into BEE adds no security check — it only shows a name and a size. It does BEE's job for exactly zero seconds.

**Kept:** `bee ollama vet <model>`. This is the one Ollama command worth building, because it is the only one that finds the model's real file on disk and runs BEE's real checks on it.

**Cut:** `bee ollama list`, `bee ollama inspect`. Removed from the plan and, when we reach that phase, removed from the code too.

---

## 4. What else gets cut or paused (scope control)

The first plan had 9 phases and asked for dashboards, compliance mappings (SOC2, NIST, ISO), an audit trail system, and a CI/CD GitHub Action package. That is enterprise-scale scope. Building it before the core is solid is how we got a broken batch last time.

**Cut from this plan, for now:**
- Dashboard (`dashboard/render.py`)
- Compliance mappings (`compliance/soc2.py`, `nist_ai_rmf.py`, `iso27001.py`, `iso42001.py`)
- Audit trail module (`audit/trail.py`)
- Packaged GitHub Action (`bee-guard/action@v1`)
- `jinja2` and `semver` dependencies (not needed — the HTML report already works with plain Python string building, no template engine required)

These can come back as their own phases after 1.0.0, once the core commands are proven solid. Adding them now would repeat the same mistake.

**Deleted, not fixed:**
- `src/bee/evidence/gguf.py` — a second, unsafe GGUF reader. BEE already has a real one (`formats/gguf_ops.py`) with years of hardening in it. This file duplicates it badly and is not used by any working code path.
- `src/bee/evidence/ml_artifacts.py` — depends on the file above, and guesses a file's type mostly from its extension. That is the exact habit BEE was built to avoid. Also not used by any working code path.

---

## 5. Phase order

| Phase | Version | What it delivers |
|---|---|---|
| 0 | 0.10.1 | Stop the bleeding: fix crashes, delete dead/unsafe files, no new features |
| 1 | 0.11.0 | Provenance for local scans (where did this file come from) |
| 2 | 0.12.0 | Policy engine (`bee policy validate`, `bee vet --policy`) |
| 3 | 0.13.0 | License detection + model card reading |
| 4 | 0.14.0 | HTML report (`bee report`) |
| 5 | 0.15.0 | `bee ollama vet` — the one real Ollama command |
| 6 | 0.16.0 | `bee vet hf://org/model` — Hugging Face support |
| 7 | 0.17.0 | Custom code scanner + dependency scanner, rebuilt with size limits |
| 8 | 0.18.0 | Vulnerability lookup (OSV only, cached, never blocks a scan on network failure) |
| — | 1.0.0 | Freeze: lock the CLI commands, lock the JSON shape, final full regression check |

Each phase is small enough to test in one sitting. Each phase ships on its own.

---

## Phase 0 — Stop the bleeding (0.10.1)

**Goal:** the code that already exists must stop crashing. No new features in this phase.

**Files to fix:**
- `src/bee/scanning/orchestrator.py` — fix `_build_local_provenance`: it uses a variable named `artifact_objs`, but the method's real parameter is named `artifacts`. This is the crash in every `bee vet` call.
- `src/bee/sources/ollama.py` and `src/bee/cli/ollama.py` — fix `OllamaSource()` being built with no arguments when its constructor needs one. Match the constructor to how the CLI actually calls it.
- `src/bee/evidence/dependency.py` — remove the `tomli` fallback import, or add `tomli` as a real dependency for Python < 3.11. Pick one and make it actually work on Python 3.10.
- `src/bee/reports/html.py` — fix `_build_vulnerabilities`: it builds `Severity("CRITICAL")` (uppercase) but `Severity`'s values are lowercase (`"critical"`). Lowercase the string before building the enum.
- `src/bee/policy/rules.py` — remove the `import pip._vendor.yaml` fallback. `pyyaml` is already a real dependency; just `import yaml`.
- `src/bee/cli/main.py` — either wire up `bee policy` for real (`app.add_typer` or `app.command`), or remove the dead `_policy_app` import until Phase 2 is ready.
- `src/bee/cli/ollama.py` — remove the `list` and `inspect` actions (see Section 3). Keep only `vet`, and make it say plainly "not built yet" instead of `[TBD: ...]` until Phase 5 actually builds it — a placeholder must say it is a placeholder, not look like a working message.

**Files to delete:**
- `src/bee/evidence/gguf.py`
- `src/bee/evidence/ml_artifacts.py`

**Tests to add:**
- `tests/test_vet_command.py` — running `bee vet` on a real small GGUF file must exit 0 with no findings, and on a file with a real finding (reuse an existing fixture, e.g. a pickle RCE) must show it and exit 1 under `--fail-on`.
- `tests/test_ollama_cli.py` — `bee ollama vet` with no model given must print a clear error and exit 1, not crash.

**Real-file check on this Mac:**
- `bee vet` on a real GGUF file from the existing test corpus. Must show the same output shape as `bee scan`/`bee inspect` already do.
- `bee ollama list`/`inspect` no longer exist — confirm `bee ollama --help` only shows `vet`.

**Done when:** `uv run pytest -q` passes, and `bee vet`, `bee ollama vet <anything>`, `bee report`, `bee modelcard` all run without a Python traceback (they may say "not implemented yet" — they must never crash).

---

## Phase 1 — Provenance for local scans (0.11.0)

**Goal:** every scan can say, at minimum, "this came from a local path, here is its hash and size" — the honest, no-network-needed version of provenance. Hugging Face and Ollama provenance come later, in their own phases, once those sources exist for real.

**Files:**
- `src/bee/core/run.py` — the `Provenance`/`SourceInfo`/`ArtifactInfo` models already added; review them for correctness, keep what's right.
- `src/bee/evidence/provenance.py` — `build_provenance`, matched to what `orchestrator.py` actually calls (fix the mismatched keyword arguments found in the audit).
- `src/bee/cli/scan.py` and `src/bee/cli/inspect.py` — add provenance to the existing, working commands. Do **not** introduce a new `vet` command in this phase; extend what already works.

**Tests:**
- `tests/test_provenance.py` — building provenance for a local file, a directory, and a symlink.
- Regression: `bee scan`/`bee inspect` output on existing fixtures must be unchanged except for the new provenance field.

**Real-file check:** `bee scan` on a real GGUF file from the corpus, confirm the provenance block shows the correct sha256, size, and "local" provider.

---

## Phase 2 — Policy engine (0.12.0)

**Goal:** `bee scan --policy policy.yaml` produces an ALLOW/REVIEW/BLOCK decision.

**Files:**
- `src/bee/policy/rules.py`, `loader.py`, `evaluator.py` — review the existing draft logic (it reads reasonably), fix the `pip._vendor` import, add real tests.
- `src/bee/cli/policy.py` — `bee policy validate <file>`. Wire it into `main.py` for real.
- `src/bee/cli/scan.py` — add `--policy` option.

**Tests:**
- `tests/test_policy_loader.py` — valid YAML, invalid YAML, unknown keys.
- `tests/test_policy_evaluator.py` — one test per rule (formats, findings, code, provenance, licenses).
- `tests/test_policy_adversarial.py` — a policy file that is not a mapping, a policy file with a huge string, a policy file that is actually a symlink.

**Real-file check:** a real GGUF file, once with a permissive policy (ALLOW) and once with a policy that blocks its format (BLOCK), confirm the exit code changes.

---

## Phase 3 — License + model card (0.13.0)

**Goal:** detect a `LICENSE` file and read a Hugging-Face-style model card's YAML header, for local directories only (no network yet).

**Files:** `src/bee/evidence/license.py`, `src/bee/evidence/model_card.py`.

**Tests:** real SPDX license text samples (MIT, Apache-2.0, GPL-3.0), a model card with valid YAML front matter, one with broken YAML front matter (must not crash).

**Real-file check:** run against this repository's own `LICENSE` file and confirm the correct SPDX id comes back.

---

## Phase 4 — HTML report (0.14.0)

**Goal:** `bee report <run-id> --output report.html` produces a working HTML file, with the bug from the audit fixed.

**Files:** `src/bee/reports/html.py`, `src/bee/cli/report.py`.

**Tests:** generate a report for a run with zero findings, one with every severity level, and one with vulnerability data (this is the case that crashed in the audit — it must have its own test so it can never silently break again).

**Real-file check:** open the generated HTML file in a browser on this Mac and visually confirm it renders.

---

## Phase 5 — `bee ollama vet` (0.15.0)

**Goal:** the one real Ollama command. Given a model name, find its real GGUF file on disk (Ollama stores pulled models under its own blob store) and run BEE's existing, tested checks on it.

**Files:** `src/bee/sources/ollama.py`, `src/bee/cli/ollama.py`.

**Tests:** mock the Ollama HTTP API (no real Ollama install required for CI) for the "model found" and "model not found" cases. A real end-to-end check against a locally installed Ollama, run by hand on this Mac when Ollama is available, is a bonus check — not something CI can depend on.

**Real-file check:** if Ollama is installed on this Mac at the time, `bee ollama vet <a real local model>` end to end. If not installed, confirm the command fails with a clear, honest message instead of a crash.

---

## Phase 6 — Hugging Face support (0.16.0)

**Goal:** `bee vet hf://org/model` downloads and scans a real Hugging Face repository's model files.

**Files:** `src/bee/sources/huggingface.py`, `src/bee/sources/base.py`, a real `bee vet` command that finally replaces the stub from Phase 0.

**Must fix from the audit:**
- No cap today on how much gets downloaded. Add a max total size and a max file count, with a clear message when a repo is skipped for being too large.
- No cleanup on the failure path — wrap the download in a way that always cleans up its temp directory, even when it fails partway through.
- Malformed `hf://` URL (missing the model name) must give a clear error, not an `IndexError`.

**Tests:** URL parsing (valid, missing model name, with and without a revision), a mocked download that hits the new size cap, a mocked download that fails partway through (confirm the temp directory is gone afterward).

**Real-file check:** one real, small, public Hugging Face model repo, downloaded and scanned for real on this Mac.

---

## Phase 7 — Custom code + dependency scanners, rebuilt (0.17.0)

**Goal:** re-do `evidence/custom_code.py` and `evidence/dependency.py` with the size limits this project already uses everywhere else.

**Must fix from the audit:**
- `custom_code.py` reads whole files into memory with no size cap. Add the same kind of stream-or-skip bound used for pickle files.
- Findings must be grouped per file (one Finding, many pieces of evidence, capped and counted), the same shape `BEE-PKL-001`/`BEE-STS-001`/`BEE-GGUF-001` already use — not one raw Finding per matched line, which floods the output on a large file.
- `dependency.py`'s `parse_pyproject` silently swallows every real dependency because it builds a `Dependency` object missing its required `version` field, and the wrapping `except Exception: pass` hides the failure. Fix the model construction, and stop the blanket exception swallowing from hiding a broken parser as if it found nothing.

**Tests:** one real test per file format (`requirements.txt`, `pyproject.toml`, `package.json`, `poetry.lock`, `Pipfile.lock`, `environment.yml`), a malformed version of each, and a declared-size-bomb test for the code scanner (matching `tests/test_declared_size_bounds.py`'s existing style).

**Real-file check:** run against this repository's own `pyproject.toml` and confirm the real dependency list comes back correctly.

---

## Phase 8 — Vulnerability lookup (0.18.0)

**Goal:** enrich a dependency list with OSV.dev results only (drop CVE/GHSA for now — they need API keys and add real complexity for real value later, not now).

**Must fix from the audit:** a vulnerability lookup must never turn a scan failure into a security failure. If OSV.dev is unreachable, the scan must still complete, with a clear note that vulnerability data is missing — never a crash, never a silent "no vulnerabilities found" that actually means "the network call failed."

**Tests:** mocked OSV response (found, not found, malformed response, timeout).

**Real-file check:** a real `requirements.txt` with a package known to have a real, published OSV advisory.

---

## 1.0.0 — Freeze

Once Phases 0 through 8 are each shipped and stable:

1. Lock the CLI: `vet`, `scan`, `inspect`, `ollama vet`, `report`, `policy validate`, `sign`, `verify`, `keygen`, `history`, `show`. No renames after this point without a real deprecation window.
2. Lock the JSON shape of `Run`/`Finding`/`Provenance`. Document it as a schema other tools can depend on.
3. Full regression run: every test, every real-file check from every phase above, run together, on this Mac, on all 4 supported Python versions.
4. Update the README's threat-model section to describe exactly what 1.0 does and does not check — the same honesty this project has kept at every release so far.

---

## What we learned from the first attempt

Write the plan phase by phase. Build one phase. Test it for real. Ship it. Only then start the next phase. The first version of this plan skipped every one of those steps at once, and every new command it produced was broken. This version of the plan exists so that does not happen again.
