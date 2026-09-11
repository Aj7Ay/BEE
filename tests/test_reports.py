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
