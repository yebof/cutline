"""cutline.phi - the integer colour front ends.

- ``phi(X) -> (Y, Cb, Cr)``: libjpeg ``jccolor.c`` ``rgb_ycc_convert`` (SCALEBITS 16, one constant set), int64 planes
  at full resolution.
- ``h2v2(p)``: libjpeg ``h2v2_downsample``, ``(sum of 4 + b_j) >> 2`` with bias b_j = 1 (even output column j), 2 (odd).
- ``phi_go(X)``: Go ``image/color.RGBToYCbCr`` (Y as Phi; chroma with offset 257 << 15 and the 0xff000000 clamp).
- ``h2v2_go(p)``: Go 4:2:0 chroma ``(sum of 4 + 2) >> 2``.
- ``front_end(X, sampling, cls) -> (Y, Cb, Cr)``: Phi for classes 1, 3, 7 (and every tier-T use), Phi_go for class 5;
  at sampling 1 the chroma planes are downsampled (the matching h2v2). The input must have even H and W at sampling 1;
  non-aligned renditions go through ``cutline.edge.edge_expand``.
numpy int64 only; no floating point.
"""
from __future__ import annotations

import numpy as np

from cutline.consts import PHI_CB, PHI_CR, PHI_GO_CB, PHI_GO_CR, PHI_Y


def _rgb(X):
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("expected an RGB array [H, W, 3]")
    return (X[..., 0].astype(np.int64), X[..., 1].astype(np.int64), X[..., 2].astype(np.int64))


def phi(X) -> tuple:
    R, G, B = _rgb(X)
    Y = (PHI_Y[0] * R + PHI_Y[1] * G + PHI_Y[2] * B + PHI_Y[3]) >> 16
    Cb = (PHI_CB[0] * R + PHI_CB[1] * G + PHI_CB[2] * B + PHI_CB[3]) >> 16
    Cr = (PHI_CR[0] * R + PHI_CR[1] * G + PHI_CR[2] * B + PHI_CR[3]) >> 16
    return Y, Cb, Cr


def _go_clamp(x):
    return np.where((x & 0xFF000000) == 0, x >> 16, np.where(x < 0, 0, 255))


def phi_go(X) -> tuple:
    R, G, B = _rgb(X)
    Y = (PHI_Y[0] * R + PHI_Y[1] * G + PHI_Y[2] * B + PHI_Y[3]) >> 16
    Cb = _go_clamp(PHI_GO_CB[0] * R + PHI_GO_CB[1] * G + PHI_GO_CB[2] * B + PHI_GO_CB[3])
    Cr = _go_clamp(PHI_GO_CR[0] * R + PHI_GO_CR[1] * G + PHI_GO_CR[2] * B + PHI_GO_CR[3])
    return Y, Cb, Cr


def _sum4(p):
    p = np.asarray(p).astype(np.int64)
    if p.shape[0] % 2 or p.shape[1] % 2:
        raise ValueError("h2v2 needs even plane dimensions (pad first, cutline.edge)")
    return p[0::2, 0::2] + p[0::2, 1::2] + p[1::2, 0::2] + p[1::2, 1::2]


def h2v2(p):
    s = _sum4(p)
    bias = np.where(np.arange(s.shape[1]) % 2 == 0, 1, 2)[None, :]
    return (s + bias) >> 2


def h2v2_go(p):
    return (_sum4(p) + 2) >> 2


def front(cls: int):
    """(colour conversion, chroma downsampler) of an encoder class (Phi_go only for class 5)."""
    return (phi_go, h2v2_go) if cls == 5 else (phi, h2v2)


def front_end(X, sampling: int, cls: int = 1) -> tuple:
    conv, down = front(cls)
    Y, Cb, Cr = conv(X)
    if sampling == 0:
        return Y, Cb, Cr
    return Y, down(Cb), down(Cr)
