import pickle

from bee.core.artifact import Artifact
from bee.evidence.pickle_calls import check_pickle_calls
from bee.formats.pickle_ops import analyze_pickle_file, analyze_pytorch_zip_pickle
from tests.fixtures import builders


class _OsSystemExploit:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


class _BenignOrderedDict:
    def __reduce__(self):
        import collections

        return (collections.OrderedDict, ())


class _UnrecognizedGlobal:
    def __reduce__(self):
        import math

        return (math.sqrt, (4,))


class _UnrecognizedRiskyModuleGlobal:
    def __reduce__(self):
        import os

        # A real, dangerous os.* call that just isn't on the exact-match
        # denylist -- the case the denylist's own safety net exists for.
        return (os.setuid, (0,))


class _CustomTrainingClass:
    """Stands in for a real checkpoint's custom user-defined class
    (commonly __main__.MyModel or similar) -- something the allowlist
    was never going to name, that isn't itself dangerous."""

    def __init__(self, value):
        self.value = value


def _write(path, obj, protocol):
    path.write_bytes(pickle.dumps(obj, protocol=protocol))


def _artifact_for(path) -> Artifact:
    return Artifact.from_file(path)


# ---------------------------------------------------------------------------
# formats.pickle_ops: opcode-level analysis, both GLOBAL (proto <4) and
# STACK_GLOBAL (proto >=4) code paths.
# ---------------------------------------------------------------------------


def test_analyze_pickle_file_resolves_global_opcode_protocol_2(tmp_path):
    path = tmp_path / "evil.pkl"
    _write(path, _OsSystemExploit(), protocol=2)

    analysis = analyze_pickle_file(path)

    assert analysis is not None
    assert "posix.system" in analysis.globals_referenced
    assert analysis.reduce_count == 1


def test_analyze_pickle_file_resolves_stack_global_opcode_protocol_4(tmp_path):
    path = tmp_path / "evil.pkl"
    _write(path, _OsSystemExploit(), protocol=4)

    analysis = analyze_pickle_file(path)

    assert analysis is not None
    assert "posix.system" in analysis.globals_referenced
    assert analysis.reduce_count == 1


def test_analyze_pickle_file_no_globals_for_plain_data(tmp_path):
    path = tmp_path / "plain.pkl"
    _write(path, {"x": 1}, protocol=4)

    analysis = analyze_pickle_file(path)

    assert analysis is not None
    assert analysis.globals_referenced == []
    assert analysis.reduce_count == 0


def test_analyze_pytorch_zip_pickle_resolves_embedded_global(tmp_path):
    # A real .pt/.pth checkpoint is a zip with the pickle in data.pkl --
    # this must be analyzed too, since that's the common case, not raw
    # pickle files.
    import zipfile

    path = tmp_path / "evil.pt"
    payload = pickle.dumps(_OsSystemExploit(), protocol=4)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/version", "3")

    analysis = analyze_pytorch_zip_pickle(path)

    assert analysis is not None
    assert "posix.system" in analysis.globals_referenced


def test_analyze_pytorch_zip_pickle_returns_none_without_data_pkl(tmp_path):
    path = tmp_path / "plain.zip"
    builders.write_plain_zip(path)
    assert analyze_pytorch_zip_pickle(path) is None


# ---------------------------------------------------------------------------
# evidence.pickle_calls: the Finding produced from that analysis.
# ---------------------------------------------------------------------------


def test_check_pickle_calls_flags_dangerous_global_as_critical(tmp_path):
    path = tmp_path / "evil.pt"  # honestly-named, no format mismatch at all
    _write(path, _OsSystemExploit(), protocol=4)

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-001"
    assert finding.severity.value == "critical"
    assert "posix.system" in finding.description


def test_check_pickle_calls_allows_known_checkpoint_helpers(tmp_path):
    path = tmp_path / "benign.pkl"
    _write(path, _BenignOrderedDict(), protocol=4)

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is None


def test_check_pickle_calls_flags_unrecognized_global_in_ordinary_module_as_low(tmp_path):
    # A reference to something in an unremarkable module (math) that's
    # neither dangerous nor allowlisted -- worth a look, but not the same
    # concern as an unrecognized function in os/subprocess/etc.
    path = tmp_path / "unknown.pkl"
    _write(path, _UnrecognizedGlobal(), protocol=4)

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-002"
    assert finding.severity.value == "low"
    assert "math.sqrt" in finding.description


def test_check_pickle_calls_flags_unrecognized_risky_module_global_as_medium(tmp_path):
    # os.setuid isn't on the exact-match denylist, but it's in a module
    # (os) that other denylisted names live in -- this is the case the
    # denylist's safety net exists for, and it must not be diluted to the
    # same LOW severity as an unrelated custom class.
    path = tmp_path / "risky.pkl"
    _write(path, _UnrecognizedRiskyModuleGlobal(), protocol=4)

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-002"
    assert finding.severity.value == "medium"
    # os.setuid resolves as posix.setuid on this platform, same as
    # os.system does -- "posix" is risky because posix.system is
    # denylisted, so an unrecognized posix.* function inherits that.
    assert "posix.setuid" in finding.description


def test_check_pickle_calls_flags_custom_training_class_as_low_not_medium(tmp_path):
    # Regression test: a checkpoint referencing the user's own training
    # code (or any library class the allowlist doesn't name) is the
    # overwhelmingly common case for real checkpoints, not a rare one --
    # flagging it at the same MEDIUM severity as a genuinely suspicious
    # os.* reference would make the finding fire on nearly every real
    # model and get muted, taking the case that matters down with it.
    path = tmp_path / "custom.pkl"
    _write(path, _CustomTrainingClass(1), protocol=4)

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-002"
    assert finding.severity.value == "low"


def test_check_pickle_calls_none_for_plain_data(tmp_path):
    path = tmp_path / "plain.pkl"
    _write(path, {"x": 1}, protocol=4)

    artifact = _artifact_for(path)
    assert check_pickle_calls(artifact) is None


def test_check_pickle_calls_none_for_non_pickle_artifact(tmp_path):
    path = tmp_path / "model.gguf"
    builders.write_gguf(path)

    artifact = _artifact_for(path)
    assert check_pickle_calls(artifact) is None


def test_check_pickle_calls_none_for_unresolved_symlink(tmp_path):
    # A refused escaping symlink has detected_format forced to "unknown" --
    # this must never try to reopen content BEE already decided not to read.
    link = tmp_path / "link.pt"
    artifact = Artifact.unresolved_symlink(link, "/somewhere/outside.pt")
    assert check_pickle_calls(artifact) is None


def test_check_pickle_calls_flags_dangerous_global_in_pytorch_zip(tmp_path):
    import zipfile

    path = tmp_path / "evil.pt"
    payload = pickle.dumps(_OsSystemExploit(), protocol=4)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/version", "3")

    artifact = _artifact_for(path)
    assert artifact.detected_format == "pytorch"
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-001"
    assert finding.severity.value == "critical"


def test_allowed_globals_covers_the_calibrated_real_checkpoint_corpus():
    # Calibrated against 10 real checkpoints downloaded from Hugging Face
    # (hf-internal-testing/tiny-random-{gpt2,bert,BertModel,t5,gpt_neo,
    # ViTModel,distilbert,roberta}, sshleifer/tiny-gpt2, and the CLIP
    # zero-shot-image-classification tiny model) -- these are the exact
    # unrecognized globals that appeared across 9 of the 10 before this
    # calibration, all legacy per-dtype storage classes PyTorch's own
    # pickle save format references via _rebuild_tensor_v2.
    from bee.evidence.pickle_calls import ALLOWED_GLOBALS

    observed_in_real_corpus = {
        "torch.FloatStorage", "torch.LongStorage", "torch.ByteStorage",
        "torch._utils._rebuild_tensor_v2", "collections.OrderedDict",
    }
    assert observed_in_real_corpus <= ALLOWED_GLOBALS
