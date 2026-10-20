"""cutline.region.uncovered - C_adj, the U rule, the platform U record and the carry.

U is a non-adaptive heuristic: given S_pub (heuristic) and Sil (exact), everything here is an exact integer
function. C_adj is evaluated once on Y0 (Chebyshev distance 8), and the U verdict is decided once, on Y0's grid, for
the claim: renditions inherit it and do not re-test it (the carry below is informational).

- ``c_adj(S_pub, Sil) -> bool[H, W]``: NS = S_pub & ~Sil; the union of the 8-connected components of NS that
  meet ``Dilate(Sil, 17 x 17 square)`` (``cv2.connectedComponents(..., connectivity=8)`` and ``cv2.dilate`` with
  ``np.ones((17, 17))`` and the default border). Empty Sil -> empty.
- ``u_exceeds(cadj_px, W, H, tau=TAU_U) -> bool``: ``10^6 * cadj_px > 64 * tau * W * H`` (tau 32: ``2048 * W * H``).
- ``u_value(cadj_px, W, H) -> Fraction`` = cadj_px / 64 / (W*H / 10^6) exactly; ``u_print(cadj_px, W, H) -> str``
  with two decimals (round half to even of the exact value; the decision always uses ``u_exceeds``).
- ``u_float(px, W, H) -> float``: the float ``px / 64 / (H*W / 1e6)`` (reporting only).
- Alternative U measures (reported, never level criteria), each a pixel count: ``u_px(S_pub, Sil)`` = |NS|;
  ``u_touch(S_pub, Sil)``: components of NS meeting ``Dilate(Sil, 3 x 3)``; ``u_win(S_pub, Sil, w)``: pixels of NS
  within Chebyshev distance w of Sil (``(2w+1)`` square); ``u_adj8(S_pub, Sil)`` = |c_adj|.
- ``UREC_HEADER = 87``; ``urec_len(W, H) = 87 + 2 * ceil(W*H/8)``.
- ``encode_urec(rY, W, H, cadj_px, digest, S_pub, C_adj) -> bytes``: ``"CUTLINE-urec-v1" || rY || u16 W ||
  u16 H || u32 cadjPx || pubsegDigest || packbits(S_pub) || packbits(C_adj)``; ``urec_id(urec) = H(urec)``.
- ``decode_urec(b) -> URec(r_y, W, H, cadj_px, digest, s_pub, c_adj)``: ``E.U.MASK_SIZE`` when the length is not
  ``urec_len(W, H)``, and also (the only structural error defined for the record) when the prefix is wrong, W or H
  is 0, or a packbits padding bit is 1.
- ``urec_header(b)``: the header and length checks only (no unpacking; also ``E.U.MASK_SIZE`` beyond the canvas
  bound W, H <= 8192, W*H <= 2^25).
- ``use_record(urec, core_rY, accepted_digests, Sil) -> C_adj`` (``use_record_full`` also returns the decoded
  ``URec``; one decode), in this order: header and length (``E.U.MASK_SIZE``);
  ``rY == core.rY`` (``E.U.RECORD_RY``); digest accepted (``E.U.RECORD_DIGEST``); the record's (W, H) equal to Sil's
  grid (``E.U.MASK_SIZE``); the masks are unpacked only now (padding bits: ``E.U.MASK_SIZE``); C_adj recomputed
  from the record's S_pub and the caller's own Sil must equal the record's C_adj and cadjPx
  (``E.U.RECORD_MISMATCH``). Returns the recomputed C_adj. On any error the party runs PubSeg itself.
- ``make_urec(S_pub, Sil, rY, digest) -> (urec, C_adj, cadj_px)``: the ingestion side (P_E / P_I).
- ``carry(C_adj, op) -> bool[op.out]`` = ``cutline.verify.hop.push_through(C_adj, op, 128)`` (imported, not
  reimplemented); ``carry_count(C_adj, op) -> (C_ch, carried_px, [ow, oh])``: the carried area on the rendition's
  grid, **informational only** (reports and drift measurements; never a level input).
- ``evidence_fields(uncovered_px, grid_wh, digest, S_pub, C_adj) -> dict``: the five evidence-record U fields
  (``uncovered_px``, ``grid_wh``, ``pubseg_digest``, ``s_pub_packbits``, ``c_adj_packbits``); the masks are Y0's
  (the U record's), and ``uncovered_px`` / ``grid_wh`` are |C_adj(Y0)| and Y0's (W, H) for every evaluated file,
  renditions included.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..consts import ADJ_CHEBYSHEV, PFX_UREC, TAU_U
from ..enc import packbits, rd_u16, rd_u32, u16, u32, unpackbits_strict
from ..errors import CutlineError, UError

UREC_HEADER = len(PFX_UREC) + 32 + 2 + 2 + 4 + 32          # 87
assert UREC_HEADER == 87
UREC_MAX_SIDE = 8192                                        # the Y0 grid (canvas) bound
UREC_MAX_PIXELS = 1 << 25


# ------------------------------------------------------------------------------------------------ C_adj
def _masks(S_pub, Sil):
    S_pub = np.asarray(S_pub, bool)
    Sil = np.asarray(Sil, bool)
    if S_pub.shape != Sil.shape or S_pub.ndim != 2:
        raise ValueError(f"S_pub {S_pub.shape} and Sil {Sil.shape} must be the same 2-D grid")
    return S_pub, Sil


def _components_meeting(S_pub, Sil, k: int) -> np.ndarray:
    """Union of the 8-connected components of NS = S_pub & ~Sil that meet Dilate(Sil, k x k square)."""
    import cv2
    S_pub, Sil = _masks(S_pub, Sil)
    un = S_pub & ~Sil
    n, lab = cv2.connectedComponents(un.astype(np.uint8), connectivity=8)
    ring = cv2.dilate(Sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    hit = np.zeros(n, bool)
    if n > 1:
        hit[np.unique(lab[ring & un])] = True
        hit[0] = False
    return hit[lab]


def c_adj(S_pub, Sil) -> np.ndarray:
    return _components_meeting(S_pub, Sil, 2 * ADJ_CHEBYSHEV + 1)


def u_adj8(S_pub, Sil) -> int:
    return int(c_adj(S_pub, Sil).sum())


def u_px(S_pub, Sil) -> int:
    S_pub, Sil = _masks(S_pub, Sil)
    return int((S_pub & ~Sil).sum())


def u_touch(S_pub, Sil) -> int:
    return int(_components_meeting(S_pub, Sil, 3).sum())


def u_win(S_pub, Sil, w: int) -> int:
    import cv2
    S_pub, Sil = _masks(S_pub, Sil)
    win = cv2.dilate(Sil.astype(np.uint8), np.ones((2 * w + 1, 2 * w + 1), np.uint8)) > 0
    return int((S_pub & ~Sil & win).sum())


# ------------------------------------------------------------------------------------------------ U rule
def u_exceeds(cadj_px: int, W: int, H: int, tau: int = TAU_U) -> bool:
    return 10 ** 6 * int(cadj_px) > 64 * int(tau) * int(W) * int(H)


def u_value(cadj_px: int, W: int, H: int) -> Fraction:
    return Fraction(10 ** 6 * int(cadj_px), 64 * int(W) * int(H))


def u_print(cadj_px: int, W: int, H: int) -> str:
    c = round(u_value(cadj_px, W, H) * 100)                 # Fraction.__round__: half to even, exact
    return f"{c // 100}.{c % 100:02d}"


def u_float(px: int, W: int, H: int) -> float:
    return float(int(px) / 64 / (int(H) * int(W) / 1e6))


# ------------------------------------------------------------------------------------------------ U record
def urec_len(W: int, H: int) -> int:
    return UREC_HEADER + 2 * (-(-int(W) * int(H) // 8))


def urec_id(urec: bytes) -> bytes:
    return hashlib.sha256(bytes(urec)).digest()


@dataclass(frozen=True)
class URec:
    r_y: bytes
    W: int
    H: int
    cadj_px: int
    digest: bytes
    s_pub: np.ndarray
    c_adj: np.ndarray


def encode_urec(rY: bytes, W: int, H: int, cadj_px: int, digest: bytes, S_pub, C_adj) -> bytes:
    if len(rY) != 32 or len(digest) != 32:
        raise ValueError("rY and the PubSeg digest are 32 bytes")
    S_pub = np.asarray(S_pub, bool)
    C_adj = np.asarray(C_adj, bool)
    if S_pub.shape != (H, W) or C_adj.shape != (H, W):
        raise ValueError(f"masks {S_pub.shape}/{C_adj.shape} != (H, W) = ({H}, {W})")
    if int(C_adj.sum()) != int(cadj_px):
        raise ValueError("cadj_px != |C_adj|")
    b = PFX_UREC + bytes(rY) + u16(W) + u16(H) + u32(cadj_px) + bytes(digest) + packbits(S_pub) + packbits(C_adj)
    assert len(b) == urec_len(W, H)
    return b


def urec_header(b: bytes) -> tuple:
    """The 87-byte header and the length checks, **without unpacking the masks**: ``(r_y, W, H, cadj_px, digest)``.
    ``E.U.MASK_SIZE`` when the prefix is wrong, W or H is 0, the grid exceeds the canvas bound (W, H <= 8192,
    W*H <= 2^25; a valid Y0 grid never does), or the length is not ``urec_len``."""
    if len(b) < UREC_HEADER:
        raise UError("E.U.MASK_SIZE", f"U record of {len(b)} bytes is shorter than its 87-byte header")
    if b[:len(PFX_UREC)] != PFX_UREC:
        raise UError("E.U.MASK_SIZE", "U record prefix is not CUTLINE-urec-v1")
    o = len(PFX_UREC)
    r_y = bytes(b[o:o + 32])
    W, H = rd_u16(b, o + 32), rd_u16(b, o + 34)
    cadj_px = rd_u32(b, o + 36)
    digest = bytes(b[o + 40:o + 72])
    if W == 0 or H == 0:
        raise UError("E.U.MASK_SIZE", "U record grid is empty")
    if W > UREC_MAX_SIDE or H > UREC_MAX_SIDE or W * H > UREC_MAX_PIXELS:
        raise UError("E.U.MASK_SIZE", f"U record grid {W} x {H} exceeds the canvas bound")
    if len(b) != urec_len(W, H):
        raise UError("E.U.MASK_SIZE", f"U record of {len(b)} bytes != 87 + 2*ceil({W}*{H}/8) = {urec_len(W, H)}")
    return r_y, W, H, cadj_px, digest


def _unpack_masks(b: bytes, W: int, H: int) -> tuple:
    n = (len(b) - UREC_HEADER) // 2
    try:
        s_pub = unpackbits_strict(b[UREC_HEADER:UREC_HEADER + n], H, W, "E.U.MASK_SIZE")
        cadj = unpackbits_strict(b[UREC_HEADER + n:], H, W, "E.U.MASK_SIZE")
    except CutlineError as e:               # enc raises the base class; E.U errors are UError
        raise UError(e.name, e.detail) from None
    return s_pub, cadj


def decode_urec(b: bytes) -> URec:
    b = bytes(b)
    r_y, W, H, cadj_px, digest = urec_header(b)
    s_pub, cadj = _unpack_masks(b, W, H)
    return URec(r_y, W, H, cadj_px, digest, s_pub, cadj)


def make_urec(S_pub, Sil, rY: bytes, digest: bytes) -> tuple:
    S_pub, Sil = _masks(S_pub, Sil)
    C = c_adj(S_pub, Sil)
    H, W = S_pub.shape
    n = int(C.sum())
    return encode_urec(rY, W, H, n, digest, S_pub, C), C, n


def use_record_full(urec: bytes, core_rY: bytes, accepted_digests, Sil) -> tuple:
    """Check a U record with a single decode: header and length (``E.U.MASK_SIZE``) -> rY (``E.U.RECORD_RY``) ->
    digest (``E.U.RECORD_DIGEST``) -> (W, H) against Sil's grid (``E.U.MASK_SIZE``) -> only then unpack the masks
    (padding bits: ``E.U.MASK_SIZE``) -> recompute (``E.U.RECORD_MISMATCH``). Returns ``(C_adj, URec)``."""
    b = bytes(urec)
    r_y, W, H, cadj_px, digest = urec_header(b)
    if r_y != bytes(core_rY):
        raise UError("E.U.RECORD_RY", "U record rY != core.rY")
    if digest not in {bytes(d) for d in accepted_digests}:
        raise UError("E.U.RECORD_DIGEST", "PubSeg digest not accepted")
    Sil = np.asarray(Sil, bool)
    if Sil.shape != (H, W):
        raise UError("E.U.MASK_SIZE", f"U record grid ({W}, {H}) != Y0 grid {Sil.shape[::-1]}")
    s_pub, cadj = _unpack_masks(b, W, H)
    rec = URec(r_y, W, H, cadj_px, digest, s_pub, cadj)
    C = c_adj(rec.s_pub, Sil)
    if int(C.sum()) != rec.cadj_px or not np.array_equal(C, rec.c_adj):
        raise UError("E.U.RECORD_MISMATCH", f"recomputed |C_adj| = {int(C.sum())}, record {rec.cadj_px}")
    return C, rec


def use_record(urec: bytes, core_rY: bytes, accepted_digests, Sil) -> np.ndarray:
    return use_record_full(urec, core_rY, accepted_digests, Sil)[0]


# ------------------------------------------------------------------------------------------------ carry
def carry(C_adj, op) -> np.ndarray:
    from ..verify import hop
    return np.asarray(hop.push_through(np.asarray(C_adj, bool), op, 128), bool)


def carry_count(C_adj, op) -> tuple:
    C = carry(C_adj, op)
    oh, ow = C.shape
    return C, int(C.sum()), [ow, oh]


# ------------------------------------------------------------------------------------------------ evidence
def evidence_fields(uncovered_px: int, grid_wh, digest: bytes, S_pub, C_adj) -> dict:
    return {"uncovered_px": int(uncovered_px), "grid_wh": [int(grid_wh[0]), int(grid_wh[1])],
            "pubseg_digest": bytes(digest), "s_pub_packbits": packbits(np.asarray(S_pub, bool)),
            "c_adj_packbits": packbits(np.asarray(C_adj, bool))}
