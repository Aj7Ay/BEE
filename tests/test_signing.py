from bee.core.signing import KeyPaths, fingerprint, generate_keypair, load_private_key, sign_hash, verify_signature


def _keypair(tmp_path):
    paths = KeyPaths(private_key=tmp_path / "key", public_key=tmp_path / "key.pub")
    generate_keypair(paths)
    return paths


def test_generate_keypair_writes_both_files_with_restrictive_private_permissions(tmp_path):
    paths = _keypair(tmp_path)
    assert paths.private_key.exists()
    assert paths.public_key.exists()
    mode = paths.private_key.stat().st_mode & 0o777
    assert mode == 0o600


def test_sign_and_verify_round_trip(tmp_path):
    paths = _keypair(tmp_path)
    private_key = load_private_key(paths.private_key)
    evidence_hash = "a" * 64

    signature_hex, public_key_hex = sign_hash(private_key, evidence_hash)

    assert verify_signature(evidence_hash, signature_hex, public_key_hex) is True


def test_verify_rejects_tampered_hash(tmp_path):
    paths = _keypair(tmp_path)
    private_key = load_private_key(paths.private_key)
    signature_hex, public_key_hex = sign_hash(private_key, "a" * 64)

    # The signature was made over "a"*64; verifying it against different
    # content (what a tampered-and-rehashed record would produce) must fail.
    assert verify_signature("b" * 64, signature_hex, public_key_hex) is False


def test_verify_rejects_signature_from_a_different_key(tmp_path):
    paths_a = KeyPaths(private_key=tmp_path / "a", public_key=tmp_path / "a.pub")
    paths_b = KeyPaths(private_key=tmp_path / "b", public_key=tmp_path / "b.pub")
    generate_keypair(paths_a)
    generate_keypair(paths_b)

    key_a = load_private_key(paths_a.private_key)
    signature_hex, _ = sign_hash(key_a, "a" * 64)

    # A forged record: attacker's own public key, claiming to match a
    # signature it didn't actually produce.
    key_b_public_hex = paths_b.public_key.read_bytes().hex()
    assert verify_signature("a" * 64, signature_hex, key_b_public_hex) is False


def test_verify_rejects_malformed_signature_or_key_without_raising():
    assert verify_signature("a" * 64, "not-hex", "also-not-hex") is False
    assert verify_signature("a" * 64, "00" * 64, "00" * 32) is False  # well-formed but wrong


def test_fingerprint_is_stable_and_short():
    fp_a = fingerprint(b"\x01" * 32)
    fp_b = fingerprint(b"\x01" * 32)
    fp_c = fingerprint(b"\x02" * 32)
    assert fp_a == fp_b
    assert fp_a != fp_c
    assert len(fp_a) == 16
