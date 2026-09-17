from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

DEFAULT_KEY_DIR = Path.home() / ".bee" / "keys"
DEFAULT_PRIVATE_KEY_PATH = DEFAULT_KEY_DIR / "bee_ed25519"
DEFAULT_PUBLIC_KEY_PATH = DEFAULT_KEY_DIR / "bee_ed25519.pub"


@dataclass
class KeyPaths:
    private_key: Path
    public_key: Path


def default_key_paths() -> KeyPaths:
    return KeyPaths(private_key=DEFAULT_PRIVATE_KEY_PATH, public_key=DEFAULT_PUBLIC_KEY_PATH)


def generate_keypair(paths: KeyPaths) -> str:
    """Generate a new Ed25519 keypair at `paths`, outside any project's
    .bee/ workspace by default -- a project's database can be freely
    copied or shared without also copying the private key, unless the
    operator deliberately points --key elsewhere. Returns the public
    key's fingerprint (sha256 of the raw public key bytes, hex)."""
    paths.private_key.parent.mkdir(parents=True, exist_ok=True)
    paths.private_key.parent.chmod(0o700)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes_raw()
    public_bytes = public_key.public_bytes_raw()

    paths.private_key.write_bytes(private_bytes)
    paths.private_key.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600: owner read/write only
    paths.public_key.write_bytes(public_bytes)
    paths.public_key.chmod(0o644)

    return fingerprint(public_bytes)


def fingerprint(public_key_bytes: bytes) -> str:
    return hashlib.sha256(public_key_bytes).hexdigest()[:16]


def load_private_key(path: Path) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())


def sign_hash(private_key: Ed25519PrivateKey, evidence_hash_hex: str) -> tuple[str, str]:
    """Sign the (hex-encoded) evidence hash. Returns (signature_hex,
    public_key_hex) -- the public key travels with the signature so a
    verifier never needs access to the signer's key files, only the
    record itself."""
    message = bytes.fromhex(evidence_hash_hex)
    signature = private_key.sign(message)
    public_bytes = private_key.public_key().public_bytes_raw()
    return signature.hex(), public_bytes.hex()


def verify_signature(evidence_hash_hex: str, signature_hex: str, public_key_hex: str) -> bool:
    """Verify `signature_hex` over the (hex-encoded) evidence hash using
    the embedded public key. Unlike a plain hash comparison, this cannot
    be defeated by an attacker who edits the record and recomputes the
    hash: producing a signature that verifies against `public_key_hex`
    requires the corresponding private key, which the record never
    contains."""
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(bytes.fromhex(signature_hex), bytes.fromhex(evidence_hash_hex))
        return True
    except (InvalidSignature, ValueError):
        return False
