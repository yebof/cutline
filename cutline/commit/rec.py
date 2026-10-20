"""cutline.commit.rec - capture record rec v3.

rec v3 ABI encoding (416 B, 13 static words, domain "CUTLINE-REC-v3"), keyed R_M and the stroke hash H_m, the capture
policy digest pi_cap, the nullifier, and the device signature sigma_dev (P-256, low-s).

- ``encode_rec(r: Rec) -> bytes`` (416 B; ``Rec.encode``).
- ``decode_rec_strict(b) -> Rec``: the validity checks, in this order: length (``E.REC.LEN``), domain word
  (``E.REC.DOMAIN``), the zero high bytes of words 4, 9, 10, 11, 12 in word order (``E.REC.NONCANONICAL``), owner != 0
  (``E.REC.OWNER_ZERO``). Raises :class:`cutline.errors.RecError`.
- ``rec_hash(b) = H(rec)``; ``nf(rec_hash) = H("nf" || recHash)``.
- ``pi_cap_record(...) -> dict`` and ``pi_cap(policy: dict) -> bytes32`` = H(JCS(policy)); the dict must have exactly
  the capture-policy keys (``PICAP_KEYS``), else ``ValueError``.
- ``r_m(k_cap, M) = HMAC(k_cap, "master" || M)`` (M uint8[H0, W0, 3] or its raw bytes); ``R_M_NONE`` = 32 zero bytes.
- ``h_m(k_cap, strokes) = HMAC(k_cap, "strokes" || packbits(m))`` (m a bool H0 x W0 mask; ``h_m_none(k_cap, H0, W0)``
  for the all-zero mask).
- ``sign_rec(sk, rec_hash) -> (r, s_low)`` (RFC 6979, low-s; sk a Python int); ``verify_rec_sig(rec_hash, r, s,
  pk64) -> bool`` (low-s enforced); ``sig_bytes(r, s) -> 64 B`` wire form r || s.
"""
from __future__ import annotations

import numpy as np

from ..consts import HM_MASTER, HM_STROKES, PFX_NF, PICAP_V, REC_DOMAIN
from ..crypto import H, hmac256, p256_sign_lows, p256_verify
from ..enc import jcs, packbits
from ..errors import RecError
from ..types import Rec

REC_LEN = 416
R_M_NONE = b"\x00" * 32
PICAP_KEYS = ("a_max_px_per_MP", "app", "dilation", "input", "psi", "seg", "thr_ppm", "v")
DILATION_V1 = "ceil(0.01*diag),ellipse"
PSI_NONE = "00" * 32

# (word index, number of leading zero bytes required) for the non-bytes32 words
_CANON = ((4, 24), (9, 12), (10, 28), (11, 30), (12, 30))


def encode_rec(r: Rec) -> bytes:
    return r.encode()


def decode_rec_strict(b: bytes) -> Rec:
    b = bytes(b)
    if len(b) != REC_LEN:
        raise RecError("E.REC.LEN", f"rec length {len(b)} != 416")
    w = [b[32 * i:32 * i + 32] for i in range(13)]
    if w[0] != REC_DOMAIN:
        raise RecError("E.REC.DOMAIN")
    for i, z in _CANON:
        if any(w[i][:z]):
            raise RecError("E.REC.NONCANONICAL", f"word {i}")
    owner = w[9][12:]
    if owner == b"\x00" * 20:
        raise RecError("E.REC.OWNER_ZERO")
    iv = lambda i: int.from_bytes(w[i], "big")  # noqa: E731
    return Rec(R_cut=w[1], R_occ=w[2], R_M=w[3], h_beta=iv(4), beta=w[5], pi_cap=w[6], H_m=w[7], key_id=w[8],
               owner=owner, epoch=iv(10), slot=iv(11), liveness=iv(12), domain=w[0])


def rec_hash(b: bytes) -> bytes:
    return H(bytes(b))


def nf(rec_hash_: bytes) -> bytes:
    if len(rec_hash_) != 32:
        raise ValueError("recHash must be 32 bytes")
    return H(PFX_NF + bytes(rec_hash_))


def pi_cap_record(app: str, seg_sha256_hex: str, input_px: int, thr_ppm: int = 500000,
                  a_max_px_per_MP: int = 2048, psi: str | None = None, dilation: str = DILATION_V1) -> dict:
    return {"a_max_px_per_MP": int(a_max_px_per_MP), "app": app, "dilation": dilation, "input": int(input_px),
            "psi": psi or PSI_NONE, "seg": seg_sha256_hex.lower(), "thr_ppm": int(thr_ppm), "v": PICAP_V}


def pi_cap(policy: dict) -> bytes:
    if tuple(sorted(policy)) != PICAP_KEYS:
        raise ValueError(f"pi_cap record must have exactly the keys {PICAP_KEYS}")
    return H(jcs(policy))


def _master_bytes(M) -> bytes:
    if isinstance(M, (bytes, bytearray, memoryview)):
        return bytes(M)
    a = np.asarray(M)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("master must be uint8[H0, W0, 3]")
    return np.ascontiguousarray(a).tobytes()


def r_m(k_cap: bytes, M) -> bytes:
    """R_M = HMAC(k_cap, "master" || M), M row-major pixel-interleaved RGB."""
    return hmac256(k_cap, HM_MASTER + _master_bytes(M))


def h_m(k_cap: bytes, strokes) -> bytes:
    """H_m = HMAC(k_cap, "strokes" || packbits(m)); m is the H0 x W0 binary stroke mask."""
    return hmac256(k_cap, HM_STROKES + packbits(np.asarray(strokes)))


def h_m_none(k_cap: bytes, H0: int, W0: int) -> bytes:
    return hmac256(k_cap, HM_STROKES + b"\x00" * ((H0 * W0 + 7) // 8))


def sign_rec(sk: int, rec_hash_: bytes) -> tuple[int, int]:
    """sigma_dev = P256Sign(sk_D, recHash) normalised to low-s."""
    return p256_sign_lows(sk, rec_hash_)


def verify_rec_sig(rec_hash_: bytes, r: int, s: int, pk64: bytes) -> bool:
    return p256_verify(rec_hash_, r, s, pk64, require_low_s=True)


def sig_bytes(r: int, s: int) -> bytes:
    return int(r).to_bytes(32, "big") + int(s).to_bytes(32, "big")
