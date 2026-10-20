"""cutline.jpeg.coefs - quantised coefficient read.

``read_coefs(data, g) -> (Y, Cb, Cr)``: int16 arrays ``[hb, wb, 8, 8]`` indexed ``[by][bx][v][u]`` (natural order
k = 8v + u), exactly the ``JCOEF`` values the pinned libjpeg-turbo 2.1.0 stores (jpeglib ``version("turbo210")`` +
``read_dct``; the file goes through a temporary file in ``$TMPDIR``). Real blocks only (jpeglib never returns MCU
dummy blocks). Asserts ``E.RY.BLOCK_GRID`` (shapes vs the grid derived from ``g``) and ``E.RY.TABLE_MISMATCH``
(the decoder's latched tables vs the grammar's). There is no pure-Python entropy decoder.

``block_grid(g) -> ((yhb, ywb), (chb, cwb))`` gives the real-block grid of the luma and chroma planes.

A Grammar beyond the SOF size bound raises ``E.CONTAINER.SOF_TOO_LARGE`` before jpeglib allocates anything
(defence in depth; ``container_ok_v2`` already rejects such a file).
"""
from __future__ import annotations

import numpy as np

from cutline.errors import ContainerError, RYError
from cutline.jpeg._io import temp_jpeg
from cutline.types import Grammar


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def block_grid(g: Grammar) -> tuple:
    m = 2 if g.sampling else 1
    return (_ceil(g.H, 8), _ceil(g.W, 8)), (_ceil(g.H, 8 * m), _ceil(g.W, 8 * m))


def read_coefs(data: bytes, g: Grammar) -> tuple:
    """(Y, Cb, Cr) int16 [hb, wb, 8, 8] of a ContainerOK file, read with the pinned turbo 2.1.0."""
    import jpeglib
    from cutline.jpeg.grammar import SOF_MAX_PIXELS, SOF_MAX_SIDE
    if g.W > SOF_MAX_SIDE or g.H > SOF_MAX_SIDE or g.W * g.H > SOF_MAX_PIXELS:   # SOF size guard (defence in depth)
        raise ContainerError("E.CONTAINER.SOF_TOO_LARGE", f"{g.W} x {g.H}")
    with temp_jpeg(bytes(data)) as p:
        with jpeglib.version("turbo210"):
            im = jpeglib.read_dct(p)
            im.load()
            Y, Cb, Cr = (np.ascontiguousarray(a, dtype=np.int16) for a in (im.Y, im.Cb, im.Cr))
            qt, tn = np.asarray(im.qt), list(im.quant_tbl_no)
    (yhb, ywb), (chb, cwb) = block_grid(g)
    if Y.shape != (yhb, ywb, 8, 8) or Cb.shape != (chb, cwb, 8, 8) or Cr.shape != (chb, cwb, 8, 8):
        raise RYError("E.RY.BLOCK_GRID", f"{Y.shape} {Cb.shape} {Cr.shape} vs {(yhb, ywb)} {(chb, cwb)}")
    for c in range(3):
        dec = bytes(int(v) for v in qt[tn[c]].reshape(-1))
        if dec != bytes(g.tables[g.tq[c]]):
            raise RYError("E.RY.TABLE_MISMATCH", f"component {c}")
    return Y, Cb, Cr
