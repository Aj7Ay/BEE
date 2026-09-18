"""Documents a real gap, not a passing guarantee: BEE-GGUF-001 checks
each tensor's range against the end of the file, but never checks two
tensors against each other. SafeTensors has this check
(bee/evidence/safetensors_bounds.py::_overlap_problems); GGUF does not.

If this test starts FAILING after a future change, that is good news:
it means someone added the missing check, and this test (and this
comment) should be deleted, not "fixed" to pass again.
"""

from bee.core.artifact import Artifact
from bee.evidence.gguf_bounds import check_gguf_bounds
from tests.test_gguf import _GGML_TYPE_F32, _build_gguf


def test_gguf_does_not_currently_detect_two_tensors_sharing_the_same_bytes(tmp_path):
    # Two 4-byte F32 tensors, both individually well inside the file,
    # both correctly aligned -- but "a" and "b" claim the exact same
    # 4 bytes of tensor data. A real loader reading both would silently
    # alias one tensor's memory onto the other's.
    path = tmp_path / "overlap.gguf"
    path.write_bytes(
        _build_gguf(
            tensors=[
                ("a", [1], _GGML_TYPE_F32, 0),
                ("b", [1], _GGML_TYPE_F32, 0),  # same offset as "a"
            ],
            tensor_data=b"\x00" * 4,
        )
    )
    artifact = Artifact.from_file(path)
    # Documents current behavior: no finding, even though the two
    # tensors provably share every byte of their declared data.
    assert check_gguf_bounds(artifact) is None