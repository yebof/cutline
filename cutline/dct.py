"""cutline.dct - exact integer transforms.

- ``blocks(plane) -> [..., hb, wb, 8, 8]`` and ``unblocks``; ``level_shift_blocks(plane)`` = blocks(plane) − 128.
- ``fdct_islow(P) -> t``: libjpeg ``jpeg_fdct_islow`` (``jfdctint.c`` of libjpeg-turbo 2.1.0, CONST_BITS 13, PASS1_BITS
  2, DESCALE = arithmetic shift) on level-shifted int blocks ``P[..., 8(y), 8(x)]``; output ``t[..., 8(v), 8(u)]``
  scaled by 8. Vectorised over the leading axes (int64).
- ``quant_islow(t, q) -> c``: ``sign(t)·⌊(|t| + 4Q)/(8Q)⌋`` (q: 64 natural-order entries or [8, 8]).
- ``d80(P) -> list[int]``: the C40 fixed-point DCT ``D80_k = Σ_y C40[v][y] Σ_x C40[u][x] P[y][x]`` (Python ints; can
  exceed 2^90, never int64); ``cell_fail(c, q, D80, eta20) -> bool``: ``|c·q·2^80 − D80| > (q·2^19 + eta20 + q)·2^60``.
- ``idct_islow(coef, q) -> uint8[..., 8, 8]``: ``jpeg_idct_islow`` (``jidctint.c``) incl. its ``range_limit`` table with
  ``RANGE_MASK`` wrap-around.
- ``dec_star(data) -> uint8[H, W, 3]``: the normative Dec* call (jpeglib turbo210 ``read_spatial``, ``JCS_RGB``,
  ``JDCT_ISLOW``, fancy upsampling, the library defaults).
- ``dec_star_numpy(g, coefs) -> uint8[H, W, 3]``: a numpy port (IDCT, ``h2v2_fancy_upsample`` of ``jdsample.c`` with
  libjpeg's context-row replication at the top and bottom edges, or plain 2x2 replication when the downsampled
  width is at most 2, as ``jinit_upsampler`` selects; ``ycc_rgb_convert`` of ``jdcolor.c``). It counts as
  Dec* only where it has been differentially tested against ``dec_star``.
"""
from __future__ import annotations

import numpy as np

from cutline import consts

CONST_BITS = 13
PASS1_BITS = 2


# ------------------------------------------------------------------------------------------------ blocks
def blocks(plane, n: int = 8):
    p = np.asarray(plane)
    h, w = p.shape[-2:]
    if h % n or w % n:
        raise ValueError("plane dimensions must be multiples of the block size")
    return p.reshape(p.shape[:-2] + (h // n, n, w // n, n)).swapaxes(-3, -2)


def unblocks(b):
    b = np.asarray(b)
    hb, wb, n, _ = b.shape[-4:]
    return b.swapaxes(-3, -2).reshape(b.shape[:-4] + (hb * n, wb * n))


def level_shift_blocks(plane):
    return blocks(np.asarray(plane).astype(np.int64)) - 128


# ------------------------------------------------------------------------------------------------ forward islow
def _desc(x, n):
    return (x + (1 << (n - 1))) >> n


def _fdct1d(v, pass_no):
    d = [v[..., i] for i in range(8)]
    tmp0, tmp7 = d[0] + d[7], d[0] - d[7]
    tmp1, tmp6 = d[1] + d[6], d[1] - d[6]
    tmp2, tmp5 = d[2] + d[5], d[2] - d[5]
    tmp3, tmp4 = d[3] + d[4], d[3] - d[4]
    tmp10, tmp13 = tmp0 + tmp3, tmp0 - tmp3
    tmp11, tmp12 = tmp1 + tmp2, tmp1 - tmp2
    CB, PB = CONST_BITS, PASS1_BITS
    if pass_no == 1:
        o0, o4, sh = (tmp10 + tmp11) << PB, (tmp10 - tmp11) << PB, CB - PB
    else:
        o0, o4, sh = _desc(tmp10 + tmp11, PB), _desc(tmp10 - tmp11, PB), CB + PB
    z1 = (tmp12 + tmp13) * 4433
    o2 = _desc(z1 + tmp13 * 6270, sh)
    o6 = _desc(z1 - tmp12 * 15137, sh)
    z1, z2, z3, z4 = tmp4 + tmp7, tmp5 + tmp6, tmp4 + tmp6, tmp5 + tmp7
    z5 = (z3 + z4) * 9633
    tmp4, tmp5, tmp6, tmp7 = tmp4 * 2446, tmp5 * 16819, tmp6 * 25172, tmp7 * 12299
    z1, z2, z3, z4 = z1 * -7373, z2 * -20995, z3 * -16069 + z5, z4 * -3196 + z5
    o7 = _desc(tmp4 + z1 + z3, sh)
    o5 = _desc(tmp5 + z2 + z4, sh)
    o3 = _desc(tmp6 + z2 + z3, sh)
    o1 = _desc(tmp7 + z1 + z4, sh)
    return np.stack([o0, o1, o2, o3, o4, o5, o6, o7], axis=-1)


def fdct_islow(P):
    """jpeg_fdct_islow on level-shifted blocks P[..., y, x] -> t[..., v, u] (scaled by 8), int64."""
    b = np.asarray(P).astype(np.int64)
    p1 = _fdct1d(b, 1)                                     # rows: [y][u]
    return np.swapaxes(_fdct1d(np.swapaxes(p1, -1, -2), 2), -1, -2)


def _q88(q):
    q = np.asarray(q, dtype=np.int64)
    return q.reshape(8, 8) if q.size == 64 else q


def quant_islow(t, q):
    """libjpeg islow quantiser: divisor 8Q, round half away from zero."""
    t = np.asarray(t, dtype=np.int64)
    qv = _q88(q) << 3
    a = np.abs(t)
    return np.sign(t) * ((a + (qv >> 1)) // qv)


# ------------------------------------------------------------------------------------------------ CELL (C40)
def d80(P) -> list:
    """D80_k (k = 8v + u) of one level-shifted 8x8 block, exact Python ints."""
    C40 = consts.C40
    P = [[int(x) for x in row] for row in np.asarray(P).reshape(8, 8).tolist()]
    row = [[sum(C40[8 * u + x] * P[y][x] for x in range(8)) for u in range(8)] for y in range(8)]   # [y][u]
    return [sum(C40[8 * v + y] * row[y][u] for y in range(8)) for v in range(8) for u in range(8)]


def cell_fail(c_k: int, q_k: int, D80_k: int, eta20: int) -> bool:
    return abs(int(c_k) * int(q_k) * (1 << 80) - int(D80_k)) > ((int(q_k) << 19) + int(eta20) + int(q_k)) << 60


# ------------------------------------------------------------------------------------------------ inverse islow
def _idct1(d, first):
    """One pass of jpeg_idct_islow (jidctint.c) over the last axis (d dequantised in pass 1, workspace in pass 2)."""
    z2, z3 = d[..., 2], d[..., 6]
    z1 = (z2 + z3) * 4433
    tmp2, tmp3 = z1 + z3 * -15137, z1 + z2 * 6270
    tmp0, tmp1 = (d[..., 0] + d[..., 4]) << CONST_BITS, (d[..., 0] - d[..., 4]) << CONST_BITS
    t10, t13, t11, t12 = tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2
    tmp0, tmp1, tmp2, tmp3 = d[..., 7], d[..., 5], d[..., 3], d[..., 1]
    z1, z2, z3, z4 = tmp0 + tmp3, tmp1 + tmp2, tmp0 + tmp2, tmp1 + tmp3
    z5 = (z3 + z4) * 9633
    tmp0, tmp1, tmp2, tmp3 = tmp0 * 2446, tmp1 * 16819, tmp2 * 25172, tmp3 * 12299
    z1, z2, z3, z4 = z1 * -7373, z2 * -20995, z3 * -16069 + z5, z4 * -3196 + z5
    tmp0, tmp1, tmp2, tmp3 = tmp0 + z1 + z3, tmp1 + z2 + z4, tmp2 + z2 + z3, tmp3 + z1 + z4
    sh = CONST_BITS - PASS1_BITS if first else CONST_BITS + PASS1_BITS + 3
    return np.stack([_desc(t10 + tmp3, sh), _desc(t11 + tmp2, sh), _desc(t12 + tmp1, sh), _desc(t13 + tmp0, sh),
                     _desc(t13 - tmp0, sh), _desc(t12 - tmp1, sh), _desc(t11 - tmp2, sh), _desc(t10 - tmp3, sh)], -1)


def range_limit(x):
    """libjpeg IDCT_range_limit(x & RANGE_MASK): x + 128 clamped for |x| < 384, libjpeg's wrap-around beyond."""
    i = np.asarray(x) & 1023
    return np.where(i < 128, i + 128, np.where(i < 512, 255, np.where(i < 896, 0, i - 896)))


def idct_islow(coef, q):
    """coef[..., v, u] (quantised), q (64 natural or [8, 8]) -> samples[..., y, x] uint8, exactly jpeg_idct_islow."""
    d = np.asarray(coef).astype(np.int64) * _q88(q)
    ws = _idct1(np.swapaxes(d, -1, -2), True)              # pass 1 over columns: [u][y]
    return range_limit(_idct1(np.swapaxes(ws, -1, -2), False)).astype(np.uint8)   # pass 2 over rows: [y][x]


# ------------------------------------------------------------------------------------------------ Dec*
def dec_star(data: bytes):
    """The pinned Dec* call: libjpeg-turbo 2.1.0 read_spatial, JCS_RGB, JDCT_ISLOW, fancy upsampling."""
    import jpeglib

    from cutline.jpeg._io import temp_jpeg
    with temp_jpeg(bytes(data)) as p:
        with jpeglib.version("turbo210"):
            im = jpeglib.read_spatial(p, out_color_space=jpeglib.JCS_RGB, dct_method=jpeglib.JDCT_ISLOW)
            im.load()
            return np.ascontiguousarray(im.spatial, dtype=np.uint8)


def h2v2_fancy_upsample(p, out_h: int, out_w: int):
    """jdsample.c 4:2:0 chroma upsampling on a chroma plane p (rows >= downsampled_height, cols >= downsampled_width):
    ``h2v2_fancy_upsample`` (the triangle filter; libjpeg's context rows replicate the first row above the image and
    the last real row below it, jdmainct.c) when downsampled_width > 2, else ``h2v2_upsample`` (2x2 replication;
    jinit_upsampler uses the fancy filter only for downsampled_width > 2). Returns [out_h, out_w] int64."""
    dh, dw = -(-out_h // 2), -(-out_w // 2)
    cur = np.asarray(p).astype(np.int64)[:dh, :dw]
    if dw <= 2:
        return np.repeat(np.repeat(cur, 2, 0), 2, 1)[:out_h, :out_w]
    above = np.concatenate([cur[:1], cur[:-1]], 0)
    below = np.concatenate([cur[1:], cur[-1:]], 0)
    out = np.empty((2 * dh, 2 * dw), np.int64)
    for r, nb in ((0, above), (1, below)):
        this = cur * 3 + nb                                 # column sums [dh, dw]
        last = np.concatenate([this[:, :1], this[:, :-1]], 1)
        nxt = np.concatenate([this[:, 1:], this[:, -1:]], 1)
        even = (this * 3 + last + 8) >> 4
        odd = (this * 3 + nxt + 7) >> 4
        even[:, 0] = (this[:, 0] * 4 + 8) >> 4
        odd[:, -1] = (this[:, -1] * 4 + 7) >> 4
        out[r::2, 0::2] = even
        out[r::2, 1::2] = odd
    return out[:out_h, :out_w]


def ycc_rgb(Y, Cb, Cr):
    """jdcolor.c ycc_rgb_convert (SCALEBITS 16) -> uint8 [H, W, 3]."""
    ONE_HALF = 1 << 15
    fix = lambda v: int(v * 65536 + 0.5)  # noqa: E731
    y = np.asarray(Y, np.int64)
    cb = np.asarray(Cb, np.int64) - 128
    cr = np.asarray(Cr, np.int64) - 128
    r = y + ((fix(1.40200) * cr + ONE_HALF) >> 16)
    g = y + ((-fix(0.34414) * cb + ONE_HALF + -fix(0.71414) * cr) >> 16)
    b = y + ((fix(1.77200) * cb + ONE_HALF) >> 16)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def dec_star_numpy(g, coefs):
    """Port of Dec* on the stored coefficients of a ContainerOK file (Grammar g, coefs from read_coefs)."""
    q = [np.frombuffer(t, np.uint8).astype(np.int64) for t in g.q3()]
    planes = [unblocks(idct_islow(np.asarray(coefs[c]), q[c])).astype(np.int64) for c in range(3)]
    H, W = g.H, g.W
    Y = planes[0][:H, :W]
    if g.sampling == 0:
        Cb, Cr = planes[1][:H, :W], planes[2][:H, :W]
    else:
        dh = -(-H // 2)
        Cb, Cr = (h2v2_fancy_upsample(p[:dh], H, W) for p in planes[1:])
    return ycc_rgb(Y, Cb, Cr)
