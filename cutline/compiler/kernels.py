"""cutline.compiler.kernels - exact resize kernels.

Exact rational area and triangle (antialiased bilinear) kernels, applied separably in int64 with one final
round-half-up division (no intermediate rounding).

- ``weights(n_in, n_out, kind) -> np.ndarray int64[n_out, n_in]``: the dense weight matrix
  (kind 0 = area: ``max(0, min((i+1) n_out, (j+1) n_in) - max(i n_out, j n_in))``;
  kind 1 = triangle: ``max(0, 2 max(n_in, n_out) - |(2i+1) n_out - (2j+1) n_in|)``).
- ``resize_exact(img, oh, ow, kind) -> np.ndarray uint8``: ResizeExact for any integer image
  ``(h, w)`` or ``(h, w, C)``; the output keeps the input's number of dimensions (``(oh, ow)`` or ``(oh, ow, C)``).
  ``out = clip(floor((2 u + den) / (2 den)), 0, 255)`` with ``den = toty[j'] * totx[j]``. With n_out = n_in both
  kernels are the identity. Internally the nonzero taps of each output are applied (sparse rows, memory-bounded
  chunks) in output-row tiles that widen only the input rows they read; the result equals the dense formula exactly.
- ``resize_rows(img, oh, ow, kind, j0, j1, ncols=None)``: rows j0..j1-1 and columns 0..ncols-1 of the same output
  (Compile composes the canvas tile by tile with it); ``tile_rows(h, oh, ncols, C, kind)`` the tile height.
- ``kind`` accepts 0/1 or ``"area"``/``"triangle"``.
- Court twin: ``taps(n_in, n_out, kind, j)``, ``axis_taps``, ``footprint``, ``court_block`` recompute one listing
  block from its opened cells exactly as the on-chain court does.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_KINDS = {0: 0, 1: 1, "area": 0, "triangle": 1}
_CHUNK_ELEMS = 1 << 21          # int64 elements per gathered chunk (16 MiB)
_TILE_ELEMS = 1 << 21           # int64 elements of one output-row tile's horizontal pass (16 MiB)


def _kind(kind) -> int:
    try:
        return _KINDS[kind]
    except (KeyError, TypeError):
        raise ValueError(f"kernel kind must be 0 (area) or 1 (triangle), got {kind!r}") from None


def weights(n_in: int, n_out: int, kind) -> np.ndarray:
    """Dense integer weights w[j][i] (int64[n_out, n_in])."""
    k = _kind(kind)
    if n_in < 1 or n_out < 1:
        raise ValueError("sizes must be >= 1")
    i = np.arange(n_in, dtype=np.int64)[None, :]
    j = np.arange(n_out, dtype=np.int64)[:, None]
    if k == 0:
        w = np.minimum((i + 1) * n_out, (j + 1) * n_in) - np.maximum(i * n_out, j * n_in)
    else:
        w = 2 * max(n_in, n_out) - np.abs((2 * i + 1) * n_out - (2 * j + 1) * n_in)
    return np.maximum(w, 0)


@lru_cache(maxsize=256)
def _rows(n_in: int, n_out: int, k: int):
    """Sparse form of weights(): (start int64[n_out], taps int64[n_out, K], tot int64[n_out]); zero taps trimmed."""
    rows = []
    for j in range(n_out):
        if k == 0:
            lo, hi = j * n_in, (j + 1) * n_in
            i0, i1 = lo // n_out, -(-hi // n_out)
            w = [max(0, min((i + 1) * n_out, hi) - max(i * n_out, lo)) for i in range(i0, i1)]
        else:
            s2, c2 = 2 * max(n_in, n_out), (2 * j + 1) * n_in
            # nonzero iff |(2i+1) n_out - c2| < s2
            i0 = max(0, (c2 - s2) // (2 * n_out) - 1)
            i1 = min(n_in, (c2 + s2) // (2 * n_out) + 2)
            w = [max(0, s2 - abs((2 * i + 1) * n_out - c2)) for i in range(i0, i1)]
        while w and w[0] == 0:
            w.pop(0)
            i0 += 1
        while w and w[-1] == 0:
            w.pop()
        if not w:
            raise AssertionError("an output with no nonzero tap")      # cannot happen for n_in, n_out >= 1
        rows.append((i0, w))
    K = max(len(w) for _, w in rows)
    start = np.empty(n_out, np.int64)
    taps = np.zeros((n_out, K), np.int64)
    tot = np.empty(n_out, np.int64)
    for j, (i0, w) in enumerate(rows):
        s = min(i0, max(0, n_in - K))          # keep the K-window inside [0, n_in) when possible
        start[j] = s
        taps[j, i0 - s:i0 - s + len(w)] = w
        tot[j] = sum(w)
    return start, taps, tot


def _apply(x: np.ndarray, start: np.ndarray, taps: np.ndarray, n_in: int, axis: int) -> np.ndarray:
    """Weighted sums along axis 0 or 1 of an int64 (H, W, C) array, gathering at most _CHUNK_ELEMS at a time."""
    n_out, K = taps.shape
    idx = start[:, None] + np.arange(K)[None, :]
    valid = idx < n_in                        # only when K > n_in (then the extra taps are zero)
    idx = np.where(valid, idx, 0)
    if axis == 1:
        H, _, C = x.shape
        out = np.empty((H, n_out, C), np.int64)
        per_row = n_out * K * C
        step = max(1, _CHUNK_ELEMS // max(1, per_row))
        for r0 in range(0, H, step):
            g = x[r0:r0 + step][:, idx]       # (h, n_out, K, C)
            out[r0:r0 + step] = np.einsum("hjkc,jk->hjc", g, taps)
        return out
    _, W, C = x.shape
    out = np.empty((n_out, W, C), np.int64)
    per_col = n_out * K * C
    step = max(1, _CHUNK_ELEMS // max(1, per_col))
    for c0 in range(0, W, step):
        g = x[:, c0:c0 + step][idx]           # (n_out, K, w, C)
        out[:, c0:c0 + step] = np.einsum("jkwc,jk->jwc", g, taps)
    return out


def _prep(img, oh: int, ow: int, kind):
    k = _kind(kind)
    a = np.asarray(img)
    if a.ndim not in (2, 3):
        raise ValueError("image must be (h, w) or (h, w, C)")
    if not np.issubdtype(a.dtype, np.integer) and a.dtype != np.bool_:
        raise ValueError("ResizeExact is defined on integer images")
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    h, w = a.shape[:2]
    if oh < 1 or ow < 1 or h < 1 or w < 1:
        raise ValueError("sizes must be >= 1")
    return a, k, squeeze, h, w


def _tile(a, k, h, w, oh, ow, j0: int, j1: int, ncols: int) -> np.ndarray:
    """uint8 (j1 - j0, ncols, C): output rows j0..j1-1 and columns 0..ncols-1 of ResizeExact, from only the input
    rows those outputs read (their vertical taps). The int64 arithmetic per output is the whole-image formula's."""
    sx, tx, nx = _rows(w, ow, k)
    sy, ty, ny = _rows(h, oh, k)
    Ky = ty.shape[1]
    sy_t = sy[j0:j1]
    r_lo = int(sy_t.min())
    r_hi = min(h, int(sy_t.max()) + Ky)
    x = a[r_lo:r_hi].astype(np.int64)
    t = _apply(x, sx[:ncols], tx[:ncols], w, 1)                     # (r_hi - r_lo, ncols, C)
    u = _apply(t, sy_t - r_lo, ty[j0:j1], h - r_lo, 0)              # (j1 - j0, ncols, C)
    den = (ny[j0:j1, None] * nx[None, :ncols])[..., None]
    return ((2 * u + den) // (2 * den)).clip(0, 255).astype(np.uint8)


def tile_rows(h: int, oh: int, ncols: int, C: int, kind) -> int:
    """Output rows per tile so that one tile's int64 intermediates stay near ``_TILE_ELEMS`` elements."""
    k = _kind(kind)
    ky = 3 if k == 0 else 4
    per_out = max(1, -(-h // oh)) * max(1, ncols) * max(1, C)
    return max(1, (_TILE_ELEMS // per_out) - ky)


def resize_rows(img, oh: int, ow: int, kind, j0: int, j1: int, ncols: int | None = None) -> np.ndarray:
    """Output rows j0..j1-1 (columns 0..ncols-1, default ow) of ``resize_exact(img, oh, ow, kind)``, bit for bit,
    computed in row tiles (memory bounded by the tile, not by the image)."""
    a, k, squeeze, h, w = _prep(img, oh, ow, kind)
    nc = ow if ncols is None else int(ncols)
    if not (0 <= j0 <= j1 <= oh) or not 0 <= nc <= ow:
        raise ValueError("row/column range outside the output")
    C = a.shape[2]
    out = np.empty((j1 - j0, nc, C), np.uint8)
    if j1 > j0 and nc > 0:
        step = tile_rows(h, oh, nc, C, k)
        for r in range(j0, j1, step):
            e = min(j1, r + step)
            out[r - j0:e - j0] = _tile(a, k, h, w, oh, ow, r, e, nc)
    return out[..., 0] if squeeze else out


def resize_exact(img, oh: int, ow: int, kind) -> np.ndarray:
    """ResizeExact (exact integers, one rounding half up); keeps the input's ndim. Computed in output-row tiles
    (``resize_rows``): only the input rows a tile reads are widened to int64."""
    return resize_rows(img, oh, ow, kind, 0, oh, None)


# ================================================================================================ court twin
# Python twin of the on-chain court's per-block recomputation (`taps`, `axis_taps`, `footprint`, `court_block`).
# A listing block is the 8 x 8 pixel block (bx, by) of the canvas (block index = by * (Wc / 8) + bx, as the
# court's p.mcu).
COURT_KMAX = 12        # the court's per-output tap buffer; rule 3 bounds it by 3 (area) / 4 (triangle)


def taps(n_in: int, n_out: int, kind, j: int) -> tuple:
    """(i0, w): the nonzero taps w[j][i0 .. i0 + len(w) - 1] of output j (zero taps trimmed at both ends)."""
    k = _kind(kind)
    if not 0 <= j < n_out:
        raise ValueError("output index out of range")
    if k == 0:
        lo, hi = j * n_in, (j + 1) * n_in
        i0, i1 = lo // n_out, -(-hi // n_out)
        w = [max(0, min((i + 1) * n_out, hi) - max(i * n_out, lo)) for i in range(i0, i1)]
    else:
        s2, c2 = 2 * max(n_in, n_out), (2 * j + 1) * n_in
        i0 = max(0, (c2 - s2) // (2 * n_out) - 1)
        i1 = min(n_in, (c2 + s2) // (2 * n_out) + 2)
        w = [max(0, s2 - abs((2 * i + 1) * n_out - c2)) for i in range(i0, i1)]
    while w and w[0] == 0:
        w.pop(0)
        i0 += 1
    while w and w[-1] == 0:
        w.pop()
    if not w:
        raise AssertionError("an output with no nonzero tap")
    return i0, w


def axis_taps(program, axis: int, b: int) -> list:
    """Taps of the 8 outputs of block coordinate b along axis 0 (y) or 1 (x); None where outside the placement."""
    y0, x0, ch, cw = program.crop
    n_in = ch if axis == 0 else cw
    n_out = program.out[axis]
    o = program.place[axis]
    res = []
    for d in range(8):
        c = b * 8 + d
        res.append(taps(n_in, n_out, program.kernel, c - o) if o <= c < o + n_out else None)
    return res


def footprint(program, bx: int, by: int):
    """Crop-relative source ranges and the level-l* cell rectangle of listing block (bx, by); None if the block is
    outside the placed rectangle on either axis (the court reverts E.COURT.NOT_CERTIFIED)."""
    ty, tx = axis_taps(program, 0, by), axis_taps(program, 1, bx)
    if all(t is None for t in ty) or all(t is None for t in tx):
        return None
    rmin = min(t[0] for t in ty if t)
    rmax = max(t[0] + len(t[1]) - 1 for t in ty if t)
    cmin = min(t[0] for t in tx if t)
    cmax = max(t[0] + len(t[1]) - 1 for t in tx if t)
    y0, x0 = program.crop[0], program.crop[1]
    cells = [(cy, cx) for cy in range((y0 + rmin) // 16, (y0 + rmax) // 16 + 1)
             for cx in range((x0 + cmin) // 16, (x0 + cmax) // 16 + 1)]
    return {"ty": ty, "tx": tx, "rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax, "cells": cells}


def court_block(program, bx: int, by: int, opened: dict, fill_blk) -> tuple:
    """(X uint8[8,8,3], certified): listing block (bx, by) recomputed only from the opened cells {(cy, cx): 1024 B}
    of the footprint and the block's fill pixels uint8[8,8,3], as the court does."""
    from ..errors import CourtError
    fp = footprint(program, bx, by)
    if fp is None:
        raise CourtError("E.COURT.NOT_CERTIFIED", "block outside the placed rectangle")
    lut = np.frombuffer(bytes(program.lut), np.uint8).reshape(3, 256)
    y0, x0 = program.crop[0], program.crop[1]
    fh, fw = fp["rmax"] - fp["rmin"] + 1, fp["cmax"] - fp["cmin"] + 1
    src = np.zeros((fh, fw, 4), np.int64)
    for r in range(fh):
        for c in range(fw):
            Y, X_ = y0 + fp["rmin"] + r, x0 + fp["cmin"] + c
            data = opened.get((Y // 16, X_ // 16))
            if data is not None:
                o = ((Y % 16) * 16 + X_ % 16) * 4
                src[r, c] = list(data[o:o + 4])
    hor = {}
    for dx, t in enumerate(fp["tx"]):
        if t:
            i0, wq = t
            hor[dx] = (sum(wv * src[:, i0 - fp["cmin"] + k] for k, wv in enumerate(wq)), sum(wq))
    fill_blk = np.asarray(fill_blk)
    Xb = np.zeros((8, 8, 3), np.uint8)
    certified = False
    for dy in range(8):
        for dx in range(8):
            Bp = fill_blk[dy, dx].astype(np.int64)
            ty = fp["ty"][dy]
            if ty is None or dx not in hor:
                comp = Bp
            else:
                i0, wq = ty
                u = sum(wv * hor[dx][0][i0 - fp["rmin"] + k] for k, wv in enumerate(wq))
                D2 = sum(wq) * hor[dx][1]
                out = np.minimum((2 * u + D2) // (2 * D2), 255)
                a = int(out[3])
                certified |= a > 0
                comp = np.minimum(out[:3] + ((255 - a) * Bp + 127) // 255, 255)
            Xb[dy, dx] = [lut[c][comp[c]] for c in range(3)]
    return Xb, certified
