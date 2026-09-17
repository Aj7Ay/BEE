import json

from bee.core.artifact import Artifact
from bee.core.run import Run
from bee.evidence.finding import Confidence, Evidence, Finding, Severity
from bee.reports.json import render_run_json
from bee.reports.terminal import render_run


def _run() -> Run:
    artifact = Artifact(
        path="model.bin", size=10, sha256="a" * 64, sha512="b" * 128,
        declared_format="pytorch", detected_format="safetensors",
        format_confidence=Confidence.SUPPORTED, magic_bytes_hex="00",
    )
    finding = Finding(
        id="BEE-FMT-001", severity=Severity.MEDIUM, title="t", description="d",
        artifact_path="model.bin",
        evidence=[Evidence(type="x", value="y", source="local_filesystem",
                             confidence=Confidence.SUPPORTED)],
    )
    return Run.from_scan(target_path="./models", artifacts=[artifact], findings=[finding])


def test_render_run_json_is_valid_json_with_expected_fields():
    payload = json.loads(render_run_json(_run()))
    assert payload["target_path"] == "./models"
    assert len(payload["artifacts"]) == 1
    assert payload["artifacts"][0]["path"] == "model.bin"
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["id"] == "BEE-FMT-001"


def test_render_run_terminal_does_not_raise(capsys):
    render_run(_run())
    captured = capsys.readouterr()
    assert "model.bin" in captured.out
    assert "BEE SCAN" in captured.out


def _run_with_path(path: str) -> Run:
    artifact = Artifact(
        path=path, size=10, sha256="a" * 64, sha512="b" * 128,
        declared_format="gguf", detected_format="gguf",
        format_confidence=Confidence.VERIFIED, magic_bytes_hex="00",
    )
    finding = Finding(
        id="BEE-FMT-001", severity=Severity.CRITICAL, title="t", description="d",
        artifact_path=path,
        evidence=[Evidence(type="x", value="y", source="local_filesystem",
                             confidence=Confidence.SUPPORTED)],
    )
    return Run.from_scan(target_path=path, artifacts=[artifact], findings=[finding])


def test_render_run_terminal_escapes_raw_ansi_in_path(capsys):
    # An attacker who controls a filename in a scanned directory must not
    # be able to inject a raw ANSI escape byte into the operator's
    # terminal -- it could otherwise recolor text, move the cursor, or
    # overwrite a real finding line with a fabricated "clean" one.
    evil_path = "esc\x1b[31mred.gguf"
    render_run(_run_with_path(evil_path))
    captured = capsys.readouterr()

    # capsys-captured output is non-tty, so Rich itself never emits color
    # codes here -- any ESC byte present could only have come from the
    # artifact path, and must not survive unescaped.
    assert "\x1b" not in captured.out
    assert "\\x1b" in captured.out


def test_render_run_terminal_does_not_interpret_markup_in_path(capsys):
    # A filename that looks like Rich markup ("[bold red]...") must
    # render as literal text, not be interpreted as a style directive --
    # the same forged-output risk as the raw-ANSI case, via Rich's own
    # markup syntax instead of terminal control bytes.
    evil_path = "[bold red]FAKE-CLEAN.gguf"
    render_run(_run_with_path(evil_path))
    captured = capsys.readouterr()

    assert "[bold red]FAKE-CLEAN" in captured.out
