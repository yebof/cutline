"""cutline.compiler.fill - the public fill layer: fill leaves, rFill and fill openings.

    fillLeaf(m) = H(0x00 || "CUTLINE-fill-v1" || u32 m || RGB[p][p][3])   # Bg pixels of canvas MCU m, row-major, R,G,B
    rFill       = MerkleRoot(fillLeaf(0), ..., fillLeaf(nMCU - 1))           # every canvas MCU, raster order

p = 8 (4:4:4, sampling 0) or 16 (4:2:0, sampling 1); the canvas MCU grid is (Hc div p) x (Wc div p).
The bundle opens the fill of every MCU of M_cert whose min alpha < 255 (tier E), or the whole fill (the *constant*
background class); the set may be empty (an empty MultiVerify opening).

- ``mcu_size(sampling) -> p``; ``mcu_grid(Hc, Wc, sampling) -> (ny, nx)``.
- ``fill_block(bg, m, sampling) -> bytes`` (3 p^2 bytes); ``fill_leaf(m, rgb) -> bytes32``.
- ``fill_leaves(bg, sampling) -> list[bytes32]``; ``r_fill(bg, sampling) -> bytes32``.
- ``fill_open_set(m_cert, alpha, sampling, whole=False) -> sorted list[int]``.
- ``open_fill(bg, sampling, mcus) -> (mcus, rgb bytes, proof list[bytes32])`` (the FillFileV1 fields).
- ``verify_fill(n_mcu, mcus, rgb, proof, r_fill, sampling) -> True`` (E.BUNDLE.FILL on framing, MerkleError on the
  proof).
"""
from __future__ import annotations

import numpy as np

from ..consts import TAG_FILL
from ..crypto import H
from ..enc import u32
from ..errors import BundleError
from ..commit import merkle


def mcu_size(sampling: int) -> int:
    if sampling not in (0, 1):
        raise ValueError("sampling must be 0 (4:4:4) or 1 (4:2:0)")
    return 8 if sampling == 0 else 16


def mcu_grid(Hc: int, Wc: int, sampling: int) -> tuple:
    p = mcu_size(sampling)
    return Hc // p, Wc // p


def _blocks(bg, sampling) -> np.ndarray:
    bg = np.asarray(bg)
    if bg.ndim != 3 or bg.shape[2] != 3 or bg.dtype != np.uint8:
        raise ValueError("bg must be uint8 (Hc, Wc, 3)")
    p = mcu_size(sampling)
    ny, nx = mcu_grid(bg.shape[0], bg.shape[1], sampling)
    return np.ascontiguousarray(bg[:ny * p, :nx * p].reshape(ny, p, nx, p, 3).transpose(0, 2, 1, 3, 4)
                                .reshape(ny * nx, p * p * 3))


def fill_block(bg, m: int, sampling: int) -> bytes:
    p = mcu_size(sampling)
    ny, nx = mcu_grid(bg.shape[0], bg.shape[1], sampling)
    if not 0 <= m < ny * nx:
        raise ValueError("MCU index outside the canvas")
    my, mx = divmod(m, nx)
    return np.ascontiguousarray(bg[my * p:(my + 1) * p, mx * p:(mx + 1) * p]).astype(np.uint8).tobytes()


def fill_leaf(m: int, rgb: bytes) -> bytes:
    if len(rgb) not in (192, 768):
        raise ValueError("a fill block is 192 (4:4:4) or 768 (4:2:0) bytes")
    return H(TAG_FILL + u32(m) + bytes(rgb))


def fill_leaves(bg, sampling: int) -> list:
    blk = _blocks(bg, sampling)
    return [H(TAG_FILL + u32(m) + blk[m].tobytes()) for m in range(blk.shape[0])]


def r_fill(bg, sampling: int) -> bytes:
    return merkle.merkle_root(fill_leaves(bg, sampling))


def fill_open_set(m_cert, alpha, sampling: int, whole: bool = False) -> list:
    """The fill MCUs a tier-E bundle opens: M_cert MCUs with min alpha < 255 (or every canvas MCU if ``whole``)."""
    a = np.asarray(alpha)
    ny, nx = mcu_grid(a.shape[0], a.shape[1], sampling)
    if whole:
        return list(range(ny * nx))
    p = mcu_size(sampling)
    amin = a[:ny * p, :nx * p].reshape(ny, p, nx, p).min(axis=(1, 3)).reshape(-1)
    return sorted(int(m) for m in set(m_cert) if amin[m] < 255)


def open_fill(bg, sampling: int, mcus) -> tuple:
    mcus = sorted(int(m) for m in mcus)
    if len(set(mcus)) != len(mcus):
        raise ValueError("duplicate MCU")
    if not mcus:
        return [], b"", []
    blk = _blocks(bg, sampling)
    leaves = [H(TAG_FILL + u32(m) + blk[m].tobytes()) for m in range(blk.shape[0])]
    rgb = b"".join(blk[m].tobytes() for m in mcus)
    return mcus, rgb, merkle.multiproof(leaves, mcus)


def verify_fill(n_mcu: int, mcus, rgb: bytes, proof, root: bytes, sampling: int) -> bool:
    p = mcu_size(sampling)
    sz = 3 * p * p
    mcus = list(mcus)
    if len(rgb) != sz * len(mcus) or any(b <= a for a, b in zip(mcus, mcus[1:])) or any(m >= n_mcu for m in mcus):
        raise BundleError("E.BUNDLE.FILL")
    leaves = [fill_leaf(m, rgb[k * sz:(k + 1) * sz]) for k, m in enumerate(mcus)]
    return merkle.multi_verify(n_mcu, mcus, leaves, list(proof), root)
