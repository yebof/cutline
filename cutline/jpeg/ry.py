"""cutline.jpeg.ry - the listing content address R_Y v2.

- ``ry_header(g) -> bytes`` (197 B): ``u16 W ‖ u16 H ‖ u8 sampling ‖ q[0] ‖ q[1] ‖ q[2]`` (natural order; a table shared
  by Cb and Cr is written twice).
- ``mcu_grid(g) -> (my, mx)``.
- ``mcu_real_blocks(g, coefs, mcu) -> list[(comp, blk, int16[8,8])]``: the MCU's real blocks in hashing order (luma
  TL, TR, BL, BR restricted to real blocks, then Cb, Cr); ``blk`` is the block position (0-3 for 4:2:0 luma, else 0).
  An MCU index outside the grid raises ``E.COURT.INDEX``; ``r_y_from`` with a header that is not 197 bytes raises
  ``E.COURT.HEADER`` (as the court does; never IndexError / ValueError on caller input).
- ``mcu_leaf(mcu, blocks) -> bytes32`` = H(0x00 ‖ "CUTLINE-mcu-v2" ‖ u32 mcu ‖ i16be(block)…).
- ``mcu_leaves(g, coefs) -> list[bytes32]`` (raster order); ``r_mcu(leaves)``; ``r_y_from(header, R_mcu)``.
- ``r_y_v2(g, coefs) -> bytes32``; ``r_y_parts(g, coefs) -> (header, R_mcu, R_Y)``.
- ``r_y_v2_bytes(data, upload=False) -> bytes32``: ContainerOK v2.1, then ``read_coefs``, then the hash.
Dummy blocks are never hashed (they are not in the decoder's arrays); the values are the stored quantised JCOEFs.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from cutline.commit.merkle import merkle_root
from cutline.consts import TAG_LISTING_V2, TAG_MCU_V2
from cutline.errors import CourtError
from cutline.types import Grammar


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def ry_header(g: Grammar) -> bytes:
    h = struct.pack(">HHB", g.W, g.H, g.sampling) + b"".join(bytes(t) for t in g.q3())
    assert len(h) == 197
    return h


def mcu_grid(g: Grammar) -> tuple:
    m = 2 if g.sampling else 1
    return _ceil(g.H, 8 * m), _ceil(g.W, 8 * m)


def _positions(g: Grammar):
    return ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)) if g.sampling else ((0, 0, 0),)


def mcu_real_blocks(g: Grammar, coefs, mcu: int) -> list:
    Y, Cb, Cr = coefs
    m = 2 if g.sampling else 1
    my, mx = mcu_grid(g)
    if not 0 <= mcu < my * mx:                                        # court step 10
        raise CourtError("E.COURT.INDEX", f"mcu {mcu} not in [0, {my * mx})")
    i, j = divmod(mcu, mx)
    yhb, ywb = Y.shape[:2]
    out = []
    for di, dj, blk in _positions(g):
        r, c = m * i + di, m * j + dj
        if r < yhb and c < ywb:
            out.append((0, blk, Y[r, c]))
    out.append((1, 0, Cb[i, j]))
    out.append((2, 0, Cr[i, j]))
    return out


def enc_block(block) -> bytes:
    return np.asarray(block, dtype=np.int64).astype(">i2").reshape(64).tobytes()


def mcu_leaf(mcu: int, blocks) -> bytes:
    return hashlib.sha256(TAG_MCU_V2 + struct.pack(">I", mcu) + b"".join(enc_block(b) for b in blocks)).digest()


def mcu_leaves(g: Grammar, coefs) -> list:
    Y, Cb, Cr = (np.ascontiguousarray(np.asarray(a).astype(">i2")) for a in coefs)
    m = 2 if g.sampling else 1
    my, mx = mcu_grid(g)
    yhb, ywb = Y.shape[:2]
    pos = _positions(g)
    leaves = []
    sha = hashlib.sha256
    for i in range(my):
        for j in range(mx):
            parts = [TAG_MCU_V2, struct.pack(">I", i * mx + j)]
            for di, dj, _ in pos:
                r, c = m * i + di, m * j + dj
                if r < yhb and c < ywb:
                    parts.append(Y[r, c].tobytes())
            parts.append(Cb[i, j].tobytes())
            parts.append(Cr[i, j].tobytes())
            leaves.append(sha(b"".join(parts)).digest())
    return leaves


def r_mcu(leaves) -> bytes:
    return merkle_root(leaves)


def r_y_from(header: bytes, R_mcu: bytes) -> bytes:
    if len(header) != 197:                                           # court step 9
        raise CourtError("E.COURT.HEADER", f"header has {len(header)} bytes, not 197")
    return hashlib.sha256(TAG_LISTING_V2 + header + R_mcu).digest()


def r_y_parts(g: Grammar, coefs) -> tuple:
    header = ry_header(g)
    R = r_mcu(mcu_leaves(g, coefs))
    return header, R, r_y_from(header, R)


def r_y_v2(g: Grammar, coefs) -> bytes:
    return r_y_parts(g, coefs)[2]


def r_y_v2_bytes(data: bytes, upload: bool = False) -> bytes:
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    g = container_ok_v2(data, upload)
    return r_y_v2(g, read_coefs(data, g))
