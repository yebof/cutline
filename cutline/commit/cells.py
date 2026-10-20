"""cutline.commit.cells - 16x16 cells, HMAC(k_cap) salts, cell leaves, level roots, occupancy, R_cut, R_occ.

- ``salt(k_cap, l, cy, cx) = HMAC(k_cap, "salt" || u8 l || u16 cy || u16 cx)[0:16]``.
- cell bytes: ``RGBA[16][16][4]`` row-major, pixel = premultiplied R, G, B then alpha; outside the level ``00 00 00 00``.
- ``cell_leaf = H(0x00 || "CUTLINE-cell-v1" || u8 l || u16 cy || u16 cx || salt16 || RGBA[1024])`` (1,061 B).
- ``R_l = MerkleRoot(leaves in row-major cell order)``; ``O_l[cy, cx] = 1`` iff some alpha of the cell is > 0.
- ``R_cut = H(0x02 || "CUTLINE-roots" || cut_pre)``, ``cut_pre = u16 H0 || u16 W0 || u8 Lmax || R_0 .. R_Lmax``.
- ``R_occ = H(0x03 || H(packbits(O_0)) || ... || H(packbits(O_Lmax)))``.

API (used by ``commit.capture`` and the bundle code):
- ``salt(k_cap, l, cy, cx) -> bytes16``; ``cell_bytes(C, alpha, cy, cx) -> bytes1024``;
  ``cell_leaf(l, cy, cx, salt16, data) -> bytes32``; ``occupancy(alpha) -> bool[ny, nx]``.
- ``commit_level(k_cap, l, C, alpha) -> LevelCommit(leaves, root, occ, occ_bits, occ_hash)``.
- ``commitment(M, A, k_cap, Lmax=None) -> types.Commitment`` (levels[l] = (C, alpha, leaves)).
- ``cut_pre(H0, W0, Lmax, roots) -> bytes``; ``r_cut(cut_pre) -> bytes32``; ``r_occ(occ_hashes) -> bytes32``.
- ``open_cells(commitment, k_cap, l, cells) -> (cellidx_bytes, cells_bytes, multiproof)``: ``cells`` are (cy, cx)
  pairs (sorted into raster order here; duplicates rejected); ``cellidx`` = N records ``u16 cy || u16 cx || salt16``,
  ``cells`` = N x 1,024 RGBA bytes in the same order, ``multiproof`` = the MultiProof of the leaves at
  indices cy nx_l + cx in the level-l tree (``[]`` for N = 0).
- ``parse_cellidx(b) -> list[(cy, cx, salt16)]`` (exact multiple of 20 bytes, else ``E.BUNDLE.CELLS``) and
  ``verify_cells(level_root, l, ny, nx, cellidx, cells, proof) -> True`` (raises MerkleError / BundleError).
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

from ..consts import HM_SALT, TAG_CELL, TAG_OCC, TAG_ROOTS
from ..crypto import H
from ..enc import packbits, u8, u16
from ..errors import BundleError
from ..types import Commitment
from . import merkle
from .pyramid import pyramid

CELL = 16
CELL_BYTES = CELL * CELL * 4
CELLIDX_REC = 20


def salt(k_cap: bytes, l: int, cy: int, cx: int) -> bytes:
    return hmac.new(bytes(k_cap), HM_SALT + u8(l) + u16(cy) + u16(cx), hashlib.sha256).digest()[:16]


def _grid(h: int, w: int) -> tuple:
    return -(-h // CELL), -(-w // CELL)


def cell_bytes(C, alpha, cy: int, cx: int) -> bytes:
    blk = np.zeros((CELL, CELL, 4), np.uint8)
    c = C[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
    a = alpha[cy * CELL:(cy + 1) * CELL, cx * CELL:(cx + 1) * CELL]
    blk[:c.shape[0], :c.shape[1], :3] = c
    blk[:a.shape[0], :a.shape[1], 3] = a
    return blk.tobytes()


def _rgba_cells(C, alpha) -> np.ndarray:
    """All cells of a level as uint8[ny, nx, 16, 16, 4] (zero outside the level)."""
    h, w = alpha.shape
    ny, nx = _grid(h, w)
    buf = np.zeros((ny * CELL, nx * CELL, 4), np.uint8)
    buf[:h, :w, :3] = C
    buf[:h, :w, 3] = alpha
    return np.ascontiguousarray(buf.reshape(ny, CELL, nx, CELL, 4).transpose(0, 2, 1, 3, 4))


def cell_leaf(l: int, cy: int, cx: int, salt16: bytes, data: bytes) -> bytes:
    if len(salt16) != 16 or len(data) != CELL_BYTES:
        raise ValueError("salt16 must be 16 bytes and data 1,024 bytes")
    return H(TAG_CELL + u8(l) + u16(cy) + u16(cx) + bytes(salt16) + bytes(data))


def occupancy(alpha) -> np.ndarray:
    a = np.asarray(alpha)
    h, w = a.shape
    ny, nx = _grid(h, w)
    pad = np.zeros((ny * CELL, nx * CELL), bool)
    pad[:h, :w] = a > 0
    return pad.reshape(ny, CELL, nx, CELL).any(axis=(1, 3))


@dataclass(eq=False)
class LevelCommit:
    leaves: list
    root: bytes
    occ: np.ndarray
    occ_bits: bytes
    occ_hash: bytes


def commit_level(k_cap: bytes, l: int, C, alpha) -> LevelCommit:
    cells = _rgba_cells(C, alpha)
    ny, nx = cells.shape[:2]
    pre = TAG_CELL + u8(l)
    leaves = []
    for cy in range(ny):
        row = cells[cy]
        for cx in range(nx):
            leaves.append(H(pre + u16(cy) + u16(cx) + salt(k_cap, l, cy, cx) + row[cx].tobytes()))
    occ = cells[..., 3].reshape(ny, nx, -1).any(axis=2)
    ob = packbits(occ)
    return LevelCommit(leaves, merkle.merkle_root(leaves), occ, ob, H(ob))


def cut_pre(H0: int, W0: int, Lmax: int, roots) -> bytes:
    return u16(H0) + u16(W0) + u8(Lmax) + b"".join(roots)


def r_cut(cut_pre_bytes: bytes) -> bytes:
    return H(TAG_ROOTS + cut_pre_bytes)


def r_occ(occ_hashes) -> bytes:
    return H(TAG_OCC + b"".join(occ_hashes))


def commitment(M, A, k_cap: bytes, Lmax: int | None = None) -> Commitment:
    """The capture commitment (Lmax defaults to lmax(H0, W0))."""
    if len(k_cap) != 32:
        raise ValueError("k_cap must be 32 bytes")
    A = np.asarray(A)
    H0, W0 = A.shape
    levels, roots, occ_bits, occ_hashes = [], [], [], []
    pyr = pyramid(M, A, Lmax)
    L = len(pyr) - 1
    for l, (C, al) in enumerate(pyr):
        lc = commit_level(k_cap, l, C, al)
        levels.append((C, al, lc.leaves))
        roots.append(lc.root)
        occ_bits.append(lc.occ_bits)
        occ_hashes.append(lc.occ_hash)
    cp = cut_pre(H0, W0, L, roots)
    return Commitment(H0=H0, W0=W0, Lmax=L, roots=roots, occ_bits=occ_bits, occ_hashes=occ_hashes, R_cut=r_cut(cp),
                      R_occ=r_occ(occ_hashes), cut_pre=cp, levels=levels)


def level_occupancy(cm: Commitment, l: int) -> np.ndarray:
    return occupancy(cm.levels[l][1])


def open_cells(cm: Commitment, k_cap: bytes, l: int, cells) -> tuple:
    C, al, leaves = cm.levels[l]
    ny, nx = _grid(*al.shape)
    cl = sorted({(int(cy), int(cx)) for cy, cx in cells})
    if len(cl) != len(list(cells)):
        raise ValueError("duplicate cells")
    for cy, cx in cl:
        if not (0 <= cy < ny and 0 <= cx < nx):
            raise ValueError(f"cell ({cy}, {cx}) outside the {ny} x {nx} grid of level {l}")
    idx = b"".join(u16(cy) + u16(cx) + salt(k_cap, l, cy, cx) for cy, cx in cl)
    data = b"".join(cell_bytes(C, al, cy, cx) for cy, cx in cl)
    proof = merkle.multiproof(leaves, [cy * nx + cx for cy, cx in cl]) if cl else []
    return idx, data, proof


def parse_cellidx(b: bytes) -> list:
    if len(b) % CELLIDX_REC:
        raise BundleError("E.BUNDLE.CELLS", "cellidx.bin length not a multiple of 20")
    out = []
    for o in range(0, len(b), CELLIDX_REC):
        out.append((int.from_bytes(b[o:o + 2], "big"), int.from_bytes(b[o + 2:o + 4], "big"), bytes(b[o + 4:o + 20])))
    return out


def verify_cells(root: bytes, l: int, ny: int, nx: int, cellidx: bytes, cells: bytes, proof) -> bool:
    """Framing (raster order, lengths; E.BUNDLE.CELLS) and the level-l multiproof (MerkleError)."""
    recs = parse_cellidx(cellidx)
    if len(cells) != CELL_BYTES * len(recs):
        raise BundleError("E.BUNDLE.CELLS", "cells.bin length != 1024 N")
    I = [cy * nx + cx for cy, cx, _ in recs]
    if any(not (cy < ny and cx < nx) for cy, cx, _ in recs) or any(b <= a for a, b in zip(I, I[1:])):
        raise BundleError("E.BUNDLE.CELLS", "cell records not in strictly ascending raster order inside the grid")
    leaves = [cell_leaf(l, cy, cx, s, cells[k * CELL_BYTES:(k + 1) * CELL_BYTES]) for k, (cy, cx, s) in enumerate(recs)]
    return merkle.multi_verify(ny * nx, I, leaves, list(proof), root)
