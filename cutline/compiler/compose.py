"""cutline.compiler.compose - the opened set, `partial`, and Compile.

Compile(pi, opened cells at l*, Bg) (v1, no overlay), exact integers only:

    Xfg   <- ResizeExact(C_l*[y0:y0+ch, x0:x0+cw], (oh, ow), kernel)          # per channel, premultiplied
    a     <- ResizeExact(alpha_l*[crop], (oh, ow), kernel)
    alpha <- Hc x Wc zeros; alpha[oy:oy+oh, ox:ox+ow] <- a
    comp  <- Bg; comp[placed] <- clip(Xfg + floor(((255 - a) Bg[placed] + 127) / 255), 0, 255)
    X[c]  <- LUT[c][comp[c]]                                                    # LUT applied after compositing
    M_cert <- {canvas MCU m : max_{p in m} alpha(p) > 0} minus D
    Sil    <- {p : alpha(p) >= 128} minus (pixels of the MCUs in D)

The compiler does NOT enforce the static rules (the verifier does), so e.g. a program with l* = 1 upscaling still
compiles. Out-of-range geometry is handled without error: crop pixels outside the level are zero RGBA (as unopened
cells), and the placed rectangle is clipped to the canvas. The canvas MCU grid is (Hc div p) x (Wc div p) (p = 8 for
sampling 0, 16 for 1), as in static rule 5(8); pixels beyond it (only when rule 5(3) is violated) belong to no MCU.

- ``opened_set(occ_lstar, program, hstar, wstar) -> sorted list[(cy, cx)]``.
- ``occupied_bbox(occ_lstar, hstar, wstar) -> (top, left, bottom, right) | None``.
- ``partial(occ_lstar, program, hstar, wstar) -> bool``: True iff the crop does not contain the occupied bbox.
- ``assemble_crop(opened, crop, hstar=None, wstar=None) -> (C uint8[ch,cw,3], alpha uint8[ch,cw])`` from opened
  cells ``{(cy, cx): 1024 bytes}``.
- ``compile_listing(program, opened, hstar, wstar, bg) -> types.Compiled``.
- ``compile_arrays(C, alpha, crop, out, place, canvas, kernel, lut, D=(), bg=None, sampling=0) -> types.Compiled``
  (C, alpha are the whole level-l* planes).
- ``m_cert_of(alpha, D, sampling) -> sorted list[int]``; ``sil_of(alpha, D, sampling) -> bool[Hc, Wc]``.
"""
from __future__ import annotations

import numpy as np

from ..types import Compiled, Program
from .fill import mcu_size
from .kernels import resize_rows
from .program import lut_bytes

CELL = 16


# ------------------------------------------------------------------------------------------------ opened set
def _crop_cells(y0, x0, ch, cw, ny, nx):
    if ch < 1 or cw < 1:
        return range(0), range(0)
    cy0, cy1 = y0 // CELL, min(ny - 1, (y0 + ch - 1) // CELL)
    cx0, cx1 = x0 // CELL, min(nx - 1, (x0 + cw - 1) // CELL)
    return range(cy0, cy1 + 1), range(cx0, cx1 + 1)


def opened_set(occ_lstar, program: Program, hstar: int, wstar: int) -> list:
    """{(cy, cx) : O_l*[cy, cx] = 1 and the cell's pixel rectangle intersects the crop}, raster order."""
    occ = np.asarray(occ_lstar, bool)
    ny, nx = occ.shape
    y0, x0, ch, cw = program.crop
    rys, rxs = _crop_cells(y0, x0, ch, cw, ny, nx)
    return [(cy, cx) for cy in rys for cx in rxs if occ[cy, cx]]


def occupied_bbox(occ_lstar, hstar: int, wstar: int):
    occ = np.asarray(occ_lstar, bool)
    if not occ.any():
        return None
    ys, xs = np.nonzero(occ)
    top, left = int(ys.min()) * CELL, int(xs.min()) * CELL
    bottom = min(hstar, (int(ys.max()) + 1) * CELL)
    right = min(wstar, (int(xs.max()) + 1) * CELL)
    return top, left, bottom, right


def partial(occ_lstar, program: Program, hstar: int, wstar: int) -> bool:
    bb = occupied_bbox(occ_lstar, hstar, wstar)
    if bb is None:
        return False
    top, left, bottom, right = bb
    y0, x0, ch, cw = program.crop
    return not (y0 <= top and x0 <= left and bottom <= y0 + ch and right <= x0 + cw)


# ------------------------------------------------------------------------------------------------ Compile
def assemble_crop(opened: dict, crop, hstar: int | None = None, wstar: int | None = None) -> tuple:
    """Level-l* crop planes from the opened cells; unopened cells (and pixels outside the level) are zero RGBA."""
    y0, x0, ch, cw = crop
    buf = np.zeros((ch, cw, 4), np.uint8)
    if ch < 1 or cw < 1:
        return buf[..., :3], buf[..., 3]
    for cy in range(y0 // CELL, (y0 + ch - 1) // CELL + 1):
        for cx in range(x0 // CELL, (x0 + cw - 1) // CELL + 1):
            data = opened.get((cy, cx))
            if data is None:
                continue
            cell = np.frombuffer(bytes(data), np.uint8).reshape(CELL, CELL, 4)
            ya, yb = max(y0, cy * CELL), min(y0 + ch, (cy + 1) * CELL)
            xa, xb = max(x0, cx * CELL), min(x0 + cw, (cx + 1) * CELL)
            buf[ya - y0:yb - y0, xa - x0:xb - x0] = cell[ya - cy * CELL:yb - cy * CELL, xa - cx * CELL:xb - cx * CELL]
    if hstar is not None and wstar is not None:      # pixels outside the level are zero (as in the committed cells)
        buf[max(0, hstar - y0):] = 0
        buf[:, max(0, wstar - x0):] = 0
    return buf[..., :3], buf[..., 3]              # views of one RGBA buffer (no copies)


def _mcu_max_min(alpha, sampling):
    p = mcu_size(sampling)
    Hc, Wc = alpha.shape
    ny, nx = Hc // p, Wc // p
    a = alpha[:ny * p, :nx * p].reshape(ny, p, nx, p)
    return a.max(axis=(1, 3)), a.min(axis=(1, 3)), ny, nx, p


def m_cert_of(alpha, D=(), sampling: int = 0) -> list:
    amax, _, ny, nx, _ = _mcu_max_min(np.asarray(alpha), sampling)
    Ds = set(int(d) for d in D)
    return [int(m) for m in np.flatnonzero(amax.reshape(-1) > 0) if int(m) not in Ds]


def _d_mask(shape, D, sampling):
    p = mcu_size(sampling)
    Hc, Wc = shape
    nx = Wc // p
    ny = Hc // p
    m = np.zeros(shape, bool)
    for d in D:
        d = int(d)
        if 0 <= d < ny * nx:
            my, mx = divmod(d, nx)
            m[my * p:(my + 1) * p, mx * p:(mx + 1) * p] = True
    return m


def sil_of(alpha, D=(), sampling: int = 0) -> np.ndarray:
    a = np.asarray(alpha)
    return (a >= 128) & ~_d_mask(a.shape, D, sampling)


def _cert_mask(shape, m_cert, sampling):
    p = mcu_size(sampling)
    nx = shape[1] // p
    m = np.zeros(shape, bool)
    for d in m_cert:
        my, mx = divmod(d, nx)
        m[my * p:(my + 1) * p, mx * p:(mx + 1) * p] = True
    return m


def _lut_array(lut) -> np.ndarray:
    return np.frombuffer(lut_bytes(lut), np.uint8).reshape(3, 256)


COMPOSE_TILE_PX = 1 << 18       # canvas pixels per composition tile (int64 temporaries ~ 8 MiB per channel)


def _compose(Ccrop, acrop, out, place, canvas, kernel, lut, D, bg, sampling) -> Compiled:
    """Composition, tiled by canvas rows: each tile resizes only the output rows it places
    (``kernels.resize_rows``, bit-identical to ``resize_exact``) and widens only its own rows to int64; the outputs
    are the uint8 X / alpha and the bool masks. Equal, bit for bit, to the whole-canvas form."""
    oh, ow = out
    oy, ox = place
    Hc, Wc = canvas
    L = _lut_array(lut)
    if bg is not None:
        bg = np.asarray(bg)
        if bg.shape != (Hc, Wc, 3):
            raise ValueError("bg must be (Hc, Wc, 3)")
    X = np.empty((Hc, Wc, 3), np.uint8)
    alpha = np.zeros((Hc, Wc), np.uint8)
    ya, yb, xa, xb = oy, min(Hc, oy + oh), ox, min(Wc, ox + ow)
    placed = oh >= 1 and ow >= 1 and Ccrop.shape[0] >= 1 and Ccrop.shape[1] >= 1 and ya < yb and xa < xb
    step = max(1, COMPOSE_TILE_PX // max(1, Wc))
    for r0 in range(0, Hc, step):
        r1 = min(Hc, r0 + step)
        comp = np.zeros((r1 - r0, Wc, 3), np.int64) if bg is None else bg[r0:r1].astype(np.int64)
        pa, pb = max(r0, ya), min(r1, yb)
        if placed and pa < pb:
            j0, j1 = pa - oy, pb - oy
            Xfg = resize_rows(Ccrop, oh, ow, kernel, j0, j1, xb - xa).astype(np.int64)
            a = resize_rows(acrop, oh, ow, kernel, j0, j1, xb - xa).astype(np.int64)
            Bs = comp[pa - r0:pb - r0, xa:xb]
            comp[pa - r0:pb - r0, xa:xb] = (Xfg + ((255 - a)[..., None] * Bs + 127) // 255).clip(0, 255)
            alpha[pa:pb, xa:xb] = a.astype(np.uint8)
        for c in range(3):
            X[r0:r1, :, c] = L[c][comp[..., c]]
    mc = m_cert_of(alpha, D, sampling)
    return Compiled(X=X, alpha=alpha, m_cert=mc, sil=sil_of(alpha, D, sampling),
                    cert_mask=_cert_mask(alpha.shape, mc, sampling))


def compile_arrays(C, alpha, crop, out, place, canvas, kernel, lut, D=(), bg=None, sampling: int = 0) -> Compiled:
    """Compile from the whole level-l* planes (C uint8[h,w,3], alpha uint8[h,w]); array form of Compile."""
    y0, x0, ch, cw = crop
    C = np.asarray(C)
    alpha = np.asarray(alpha)
    h, w = alpha.shape
    Cc = np.zeros((max(ch, 0), max(cw, 0), 3), np.uint8)
    ac = np.zeros((max(ch, 0), max(cw, 0)), np.uint8)
    yb, xb = min(h, y0 + ch), min(w, x0 + cw)
    if y0 < yb and x0 < xb:
        Cc[:yb - y0, :xb - x0] = C[y0:yb, x0:xb]
        ac[:yb - y0, :xb - x0] = alpha[y0:yb, x0:xb]
    return _compose(Cc, ac, out, place, canvas, kernel, lut, D, bg, sampling)


def compile_listing(program: Program, opened: dict, hstar: int, wstar: int, bg) -> Compiled:
    """Compile(pi, opened cells at l*, Bg) from the opened cells {(cy, cx): 1024 bytes}."""
    Cc, ac = assemble_crop(opened, program.crop, hstar, wstar)
    return _compose(Cc, ac, program.out, program.place, program.canvas, program.kernel, program.lut, program.D, bg,
                    program.sampling)
