import pickle

from bee.core.artifact import Artifact
from bee.evidence.pickle_calls import check_pickle_calls
from bee.formats.pickle_ops import UNRESOLVED_STACK_GLOBAL, analyze_pickle_file, analyze_pytorch_zip_pickle
from tests.fixtures import builders


def _short_binunicode(s: str) -> bytes:
    b = s.encode("utf-8")
    return b"\x8c" + bytes([len(b)]) + b


def _build_memo_indirected_rce(command: str) -> bytes:
    """A hand-crafted pickle stream (not produced by pickle.dumps) that
    resolves a STACK_GLOBAL target through memo indirection instead of
    pushing the module/qualname strings immediately beforehand: MEMOIZE
    both strings, then BINGET each one right before STACK_GLOBAL. This is
    a real, verified evasion of naive "last two string pushes" tracking --
    the resulting bytes load and execute via plain pickle.loads() exactly
    like a normal pickle.dumps()-produced one would.
    """
    return (
        b"\x80\x04"  # PROTO 4
        + _short_binunicode("os") + b"\x94"      # SHORT_BINUNICODE 'os', MEMOIZE (memo 0)
        + _short_binunicode("system") + b"\x94"  # SHORT_BINUNICODE 'system', MEMOIZE (memo 1)
        + b"h\x00"    # BINGET 0 -> 'os'
        + b"h\x01"    # BINGET 1 -> 'system'
        + b"\x93"     # STACK_GLOBAL
        + _short_binunicode(command) + b"\x94"
        + b"\x85"     # TUPLE1
        + b"R"        # REDUCE
        + b"."        # STOP
    )


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


def test_memo_indirected_stack_global_payload_actually_executes(tmp_path):
    # Ground the regression test in reality first: confirm this hand-built
    # byte sequence is not just plausible-looking but a real, loadable,
    # executing pickle -- via a filesystem side effect, not a mocked call.
    marker = tmp_path / "executed"
    payload = _build_memo_indirected_rce(f"touch {marker}")
    assert not marker.exists()
    pickle.loads(payload)
    assert marker.exists()


def test_analyze_pickle_file_resolves_memo_indirected_stack_global(tmp_path):
    # The actual regression: this must resolve to os.system, not fall back
    # to <unresolved_stack_global>, even though the module/qualname
    # strings reach STACK_GLOBAL via BINGET rather than immediately
    # preceding it.
    path = tmp_path / "evil.pkl"
    path.write_bytes(_build_memo_indirected_rce("echo pwned"))

    analysis = analyze_pickle_file(path)

    assert analysis is not None
    assert UNRESOLVED_STACK_GLOBAL not in analysis.globals_referenced
    assert "os.system" in analysis.globals_referenced or "posix.system" in analysis.globals_referenced
    assert analysis.reduce_count == 1


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


def test_check_pickle_calls_flags_memo_indirected_rce_as_critical_not_medium(tmp_path):
    # The regression this guards against: this hand-crafted pickle is a
    # real, verified os.system RCE (see
    # test_memo_indirected_stack_global_payload_actually_executes), and
    # before the memo-tracking fix it resolved to
    # <unresolved_stack_global> -- BEE-PKL-002 medium, not critical, and
    # invisible to `--fail-on high`.
    path = tmp_path / "evil.pkl"
    path.write_bytes(_build_memo_indirected_rce("echo pwned"))

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-001"
    assert finding.severity.value == "critical"


def test_check_pickle_calls_flags_memo_indirected_rce_in_pytorch_zip(tmp_path):
    import zipfile

    path = tmp_path / "evil.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", _build_memo_indirected_rce("echo pwned"))
        zf.writestr("archive/version", "3")

    artifact = _artifact_for(path)
    assert artifact.detected_format == "pytorch"
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-001"
    assert finding.severity.value == "critical"


def test_check_pickle_calls_unresolved_and_reduced_is_high_not_medium(tmp_path):
    # Defense in depth for whatever memo-tracking doesn't cover: a
    # STACK_GLOBAL that genuinely can't be resolved (both BINGETs recall
    # memo slots that were never stored) but is still invoked via REDUCE
    # is treated with more suspicion than a merely-referenced one.
    path = tmp_path / "unresolved.pkl"
    path.write_bytes(
        b"\x80\x04"
        + b"h\x00"  # BINGET 0 -- nothing was ever memoized at index 0
        + b"h\x01"  # BINGET 1 -- likewise
        + b"\x93"   # STACK_GLOBAL
        + _short_binunicode("x") + b"\x94"
        + b"\x85"   # TUPLE1
        + b"R"      # REDUCE
        + b"."
    )

    artifact = _artifact_for(path)
    finding = check_pickle_calls(artifact)

    assert finding is not None
    assert finding.id == "BEE-PKL-002"
    assert finding.severity.value == "high"


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
