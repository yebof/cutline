"""cutline.crypto - hashes, MACs, HKDF and the two signature schemes.

Hashes and MACs
- ``H(b) -> bytes32``: SHA-256. ``keccak256(b) -> bytes32``: Ethereum Keccak-256 (eth-hash; NOT hashlib.sha3_256).
- ``hmac256(k, m) -> bytes32``: HMAC-SHA256. ``hkdf_sha256(ikm, salt, info, L) -> bytes``: RFC 5869, an empty salt
  means 32 zero bytes.

P-256 (device key, sigma_dev)
- ``N_P256``: the group order.
- ``p256_key_from_label(label: str | bytes) -> int``: ``sk = H(label) mod (n - 1) + 1`` (label str is UTF-8).
- ``p256_pub(sk: int) -> bytes``: 64 bytes ``pk_x || pk_y``.
- ``p256_sign_lows(sk: int, digest32) -> (r, s)``: RFC 6979 deterministic ECDSA over the digest as given (no second
  hash), normalised to low-s (s <= n/2).
- ``p256_sign_raw(sk, digest32) -> (r, s)``: the same without the low-s normalisation (for negative test cases).
- ``p256_verify(digest, r, s, pk64, require_low_s=True) -> bool``.

secp256k1 (owner, delegate, holder, relays)
- ``N_K1``: the group order. ``k1_key_from_label(label) -> int``: ``sk = H(label) mod (n - 1) + 1``.
- ``k1_sign(sk: int, digest32) -> bytes``: 65 bytes ``r || s || v``, RFC 6979, low-s, ``v = 27 + recid``.
- ``k1_recover(digest32, sig65) -> bytes20 | None``: ``ecrecover`` semantics (v in {27, 28}, 1 <= r, s < n, no low-s
  check, as the EVM precompile); None when invalid or when the recovered address is zero.
- ``address_of(sk: int) -> bytes20``; ``address_of_pub(pub64) -> bytes20``.

Private keys are Python ints throughout (convert a 32-byte key with ``int.from_bytes(k, "big")``).
"""
from __future__ import annotations

import hashlib
import hmac as _hmac

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.exceptions import InvalidSignature
from eth_hash.auto import keccak as _keccak
from eth_keys import keys as _ethkeys

N_P256 = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
N_K1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def H(b: bytes) -> bytes:
    return hashlib.sha256(bytes(b)).digest()


def keccak256(b: bytes) -> bytes:
    return _keccak(bytes(b))


def hmac256(k: bytes, m: bytes) -> bytes:
    return _hmac.new(bytes(k), bytes(m), hashlib.sha256).digest()


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, L: int) -> bytes:
    """RFC 5869 HKDF-SHA256; an empty salt is 32 zero bytes."""
    if not 0 < L <= 255 * 32:
        raise ValueError("HKDF output length out of range")
    prk = hmac256(salt if salt else b"\x00" * 32, ikm)
    okm, t, i = b"", b"", 1
    while len(okm) < L:
        t = hmac256(prk, t + bytes(info) + bytes([i]))
        okm += t
        i += 1
    return okm[:L]


def _label_bytes(label) -> bytes:
    return label.encode("utf-8") if isinstance(label, str) else bytes(label)


def _check32(d: bytes):
    if len(d) != 32:
        raise ValueError("digest must be 32 bytes")


# ------------------------------------------------------------------------------------------------ P-256
def p256_key_from_label(label) -> int:
    return int.from_bytes(H(_label_bytes(label)), "big") % (N_P256 - 1) + 1


def _p256_sk(sk: int):
    return ec.derive_private_key(int(sk), ec.SECP256R1())


def p256_pub(sk: int) -> bytes:
    pn = _p256_sk(sk).public_key().public_numbers()
    return pn.x.to_bytes(32, "big") + pn.y.to_bytes(32, "big")


def p256_sign_raw(sk: int, digest32: bytes) -> tuple[int, int]:
    """RFC 6979 deterministic ECDSA on the 32-byte digest as given, without low-s normalisation."""
    _check32(digest32)
    der = _p256_sk(sk).sign(bytes(digest32), ec.ECDSA(utils.Prehashed(hashes.SHA256()), deterministic_signing=True))
    return utils.decode_dss_signature(der)


def p256_sign_lows(sk: int, digest32: bytes) -> tuple[int, int]:
    r, s = p256_sign_raw(sk, digest32)
    return r, (N_P256 - s if s > N_P256 // 2 else s)


def p256_verify(digest: bytes, r: int, s: int, pk64: bytes, require_low_s: bool = True) -> bool:
    if len(digest) != 32 or len(pk64) != 64:
        return False
    if not (0 < r < N_P256 and 0 < s < N_P256):
        return False
    if require_low_s and s > N_P256 // 2:
        return False
    try:
        pub = ec.EllipticCurvePublicNumbers(int.from_bytes(pk64[:32], "big"), int.from_bytes(pk64[32:], "big"),
                                            ec.SECP256R1()).public_key()
        pub.verify(utils.encode_dss_signature(r, s), bytes(digest), ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        return True
    except (InvalidSignature, ValueError):
        return False


# ------------------------------------------------------------------------------------------------ secp256k1
def k1_key_from_label(label) -> int:
    return int.from_bytes(H(_label_bytes(label)), "big") % (N_K1 - 1) + 1


def _k1_sk(sk: int):
    if not 0 < int(sk) < N_K1:
        raise ValueError("secp256k1 private key out of range")
    return _ethkeys.PrivateKey(int(sk).to_bytes(32, "big"))


def address_of_pub(pub64: bytes) -> bytes:
    return keccak256(bytes(pub64))[12:]


def address_of(sk: int) -> bytes:
    return address_of_pub(_k1_sk(sk).public_key.to_bytes())


def k1_sign(sk: int, digest32: bytes) -> bytes:
    """65-byte ``r || s || v``: RFC 6979 (eth-keys), normalised to low-s, v = 27 + recid of the normalised form."""
    _check32(digest32)
    sig = _k1_sk(sk).sign_msg_hash(bytes(digest32))
    r, s, v = sig.r, sig.s, sig.v
    if s > N_K1 // 2:
        s, v = N_K1 - s, v ^ 1
    return r.to_bytes(32, "big") + s.to_bytes(32, "big") + bytes([27 + v])


def k1_recover(digest32: bytes, sig65: bytes) -> bytes | None:
    """ecrecover: the signer address, or None (invalid signature, v not in {27, 28}, or the zero address)."""
    if len(digest32) != 32 or len(sig65) != 65:
        return None
    r, s, v = int.from_bytes(sig65[:32], "big"), int.from_bytes(sig65[32:64], "big"), sig65[64]
    if v not in (27, 28) or not (0 < r < N_K1 and 0 < s < N_K1):
        return None
    try:
        pub = _ethkeys.Signature(vrs=(v - 27, r, s)).recover_public_key_from_msg_hash(bytes(digest32))
    except Exception:            # eth-keys raises BadSignature / ValidationError on points off the curve
        return None
    a = address_of_pub(pub.to_bytes())
    return None if a == b"\x00" * 20 else a
