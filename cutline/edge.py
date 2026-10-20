"""cutline.edge - the libjpeg edge-expansion model of partial MCUs (EdgeExpand).

- ``pad(A, h, w)``: replicate A's last column to width w, then its last row to height h (no-op where already large).
- ``edge_expand(X1, sampling, front="phi"|"go") -> (Y, Cb, Cr)``: the front-end planes of the canvas X1 (Hch x Wch RGB)
  padded so that every **real** block of the file is an aligned 8x8 block of the planes:
  4:4:4 -> each plane padded to (8·ceil(H/8), 8·ceil(W/8)); 4:2:0 -> Y padded to (16·ceil(H/16), 8·ceil(W/8)) and cut
  to 8·ceil(H/8) rows; Cb/Cr (Phi, libjpeg order) -> F = pad(plane, H + (H mod 2), 16·wbc), D = h2v2(F),
  pad(D, 8·hbc, 8·wbc); Cb/Cr (Phi_go, class 5, Go image/jpeg order) -> F = pad(plane, 16·ceil(H/16), 16·wbc),
  D = h2v2_go(F) ((sum + 2) >> 2), keep the first 8·hbc rows; wbc = ceil(ceil(W/2)/8), hbc = ceil(ceil(H/2)/8).
  The two orders coincide whenever H is odd or H mod 16 = 0.
- ``reference_blocks(X1, sampling, cls=1) -> (Yb, Cbb, Crb)``: the level-shifted int64 reference blocks
  ``[hb, wb, 8, 8]`` of the real blocks only (dummy blocks are never tested or hashed); Phi_go for class 5.
- ``hop_reference(parent_rgb, crop, out_hw, kernel, sampling, cls=1) -> (X1, (Yb, Cbb, Crb))``: crop (y, x, h, w) of
  the decoded parent, ``kernels.resize_exact`` to out_hw with the exact kernel, then ``reference_blocks`` (the
  reference of a verified hop).
"""
from __future__ import annotations

import numpy as np

from cutline import phi as PHI
from cutline.dct import level_shift_blocks


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def pad(A, h: int, w: int):
    A = np.asarray(A)
    ph, pw = max(0, h - A.shape[0]), max(0, w - A.shape[1])
    if ph == 0 and pw == 0:
        return A
    return np.pad(A, ((0, ph), (0, pw)), mode="edge")


def edge_expand(X1, sampling: int, front: str = "phi") -> tuple:
    if front not in ("phi", "go"):
        raise ValueError("front must be 'phi' or 'go'")
    conv, down = (PHI.phi, PHI.h2v2) if front == "phi" else (PHI.phi_go, PHI.h2v2_go)
    H, W = np.asarray(X1).shape[:2]
    y, cb, cr = conv(X1)
    if sampling == 0:
        hb, wb = _ceil(H, 8), _ceil(W, 8)
        return tuple(pad(p, 8 * hb, 8 * wb) for p in (y, cb, cr))
    Y = pad(y, 16 * _ceil(H, 16), 8 * _ceil(W, 8))[:8 * _ceil(H, 8)]
    wbc, hbc = _ceil(_ceil(W, 2), 8), _ceil(_ceil(H, 2), 8)
    out = [Y]
    for p in (cb, cr):
        if front == "go":  # Go clamps to the full 16-row MCU before the 2x2 average (writer.go toYCbCr/scale)
            out.append(down(pad(p, 16 * _ceil(H, 16), 16 * wbc))[:8 * hbc])
        else:  # libjpeg: duplicate an odd last row, downsample, replicate the last downsampled row (jcprepct.c)
            out.append(pad(down(pad(p, H + (H % 2), 16 * wbc)), 8 * hbc, 8 * wbc))
    return tuple(out)


def reference_blocks(X1, sampling: int, cls: int = 1) -> tuple:
    planes = edge_expand(X1, sampling, "go" if cls == 5 else "phi")
    return tuple(level_shift_blocks(p) for p in planes)


def hop_reference(parent_rgb, crop, out_hw, kernel, sampling: int, cls: int = 1) -> tuple:
    from cutline.compiler.kernels import resize_exact
    y, x, h, w = crop
    P = np.asarray(parent_rgb)[y:y + h, x:x + w]
    X1 = resize_exact(P, out_hw[0], out_hw[1], kernel).astype(np.uint8)
    return X1, reference_blocks(X1, sampling, cls)
