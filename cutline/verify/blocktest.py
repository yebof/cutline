"""cutline.verify.blocktest - per-block codec tests and TE rows.

A TE row is ``(mcu, comp, blk, k, c, w)``: the MCU index, the component, the block inside the MCU, the coefficient
index (natural order), the file's quantised coefficient and the integer witness. Rows are ascending by
(mcu, comp, blk, k).

- ``exact_block(P, c, q) -> (ok, fails)``: class 1 (EXACT). P = level-shifted 8x8 reference block [y][x]; c = the
  file's 64 quantised coefficients (natural order or [8, 8]); q = the table. ``fails`` = [(k, c_k, t_k)] where
  ``t = jpeg_fdct_islow(P)`` is the integer witness.
- ``cell_block(P, c, q, eta20) -> (ok, fails)``: CELL (classes 3, 5; eta20 = 52,429 or 262,144). ``fails`` =
  [(k, c_k, D80_k)] (Python ints).
- ``e0_block_fails(X, coefs, g) -> (Yf, Cbf, Crf)``: bool masks [hb, wb, 8, 8] of failing coefficients of every real
  block of an E0 (class 1) file against its canvas X (EdgeExpand at non-aligned sizes), vectorised int64.
- ``e0_file_fails(X, coefs, g, cert_mcus=None) -> list[TE]``: the EXACT test's TE rows, restricted to the MCUs in
  ``cert_mcus`` when given.
- ``block_mcu_index(g, comp, r, c) -> (mcu, blk)`` and ``block_mcu_map(g, comp, hb, wb)`` (vectorised): the MCU and
  the ``blk`` index of real block (r, c) of a component.
- ``court_q(header, comp, k) -> q``: the court's table read; ``E.COURT.ZERO_Q`` when the header entry is 0 (a zero
  entry is a static-rule violation, proved through the static path).
- ``cell_file_fails(X, coefs, g, eta20, front="phi"|"go", cert_mcus=None) -> list[TE]``: the CELL test of every real
  block of the file (EdgeExpand at non-aligned sizes), rows ``(mcu, comp, blk, k, c, D80_k)``. A float fast path
  never decides alone: a vectorised float64 prefilter F = (C40/2^40)·P·(C40/2^40)^T decides every coefficient whose
  margin to the threshold q·(1/2 + 2^-20) + eta20·2^-20 exceeds the guard (2q + 1)·2^-20 (the float error is below
  10^-9 here); every block with a failing or a near-threshold coefficient is re-tested exactly (Python-int D80), so
  the rows, witnesses included, are exactly the integer path's. ``cell_prefilter(P, cf, q, eta20) -> (fail, near)``
  and ``cell_blocks_fails(P, cf, q, eta20, mask=None)`` are the block-array layers (rows ``(r, c, k, c_k, D80_k)``).
- ``einf_file(X, coefs, g, eta20, front="phi", cert_mcus=None) -> list[(mcu, comp, blk, einf_2p30)]``: E_inf·2^30
  for every tested real block, ascending, zeros included. The tie set A and σ come from the same prefilter; a block
  where some coefficient's tie distance lies within the guard of (eta20 + q)·2^-20, or a tie member lies within the
  guard of its boundary (σ undecided), is recomputed with the exact ``excess.e_inf_block``; the others use the exact
  integer formula on (A, σ, q, B30S) in int64 (|value| < 2^43).
- ``class_test_file(enc_class, X, coefs, g, cert_mcus=None, ep=None, cls=None) -> list[TE]``: the class test on a
  file: 1 = EXACT (``e0_file_fails``); 3 = CELL at eta20 52,429 with Phi, precondition every table entry <= Q_pol
  (else ``E.HOP.CLASS``); 5 = CELL at 262,144 with Phi_go; 7 = EP: re-encode X with the EP entry
  (``cutline.verify.ep``; ``ep`` = an entry id or an ``EPSpec``) and compare every coefficient, rows
  ``(mcu, comp, blk, k, c, c_ep)`` (EP has no integer witness; the re-encoded coefficient is recorded). ``cls`` = the
  class registry (default ``consts.CLASS_REGISTRY``); an undefined or reserved class raises ``E.HOP.CLASS``.
  ``cert_mcus`` None = every MCU. ``class3_tables_ok(g)``.
- ``iter_class_rows`` / ``iter_class_row_chunks`` / ``iter_einf``: the same rows generated band by band (whole MCU
  rows), so memory is bounded by one band; ``first_row`` / ``head_rows`` take the verdict or a capped report.
- ``e0_hit(X, coefs, g, cert_mcus=None) -> bool``: the EXACT test passes on every tested block.
"""
from __future__ import annotations

import numpy as np

from cutline import consts
from cutline.dct import cell_fail, d80, fdct_islow, quant_islow
from cutline.errors import CourtError, HopError


def _flat(c):
    return [int(v) for v in np.asarray(c).reshape(64)]


def exact_block(P, c, q) -> tuple:
    t = fdct_islow(np.asarray(P, dtype=np.int64).reshape(8, 8))
    ref = quant_islow(t, q).reshape(64)
    cf, tf = _flat(c), [int(v) for v in t.reshape(64)]
    fails = [(k, cf[k], tf[k]) for k in range(64) if cf[k] != int(ref[k])]
    return (not fails), fails


def cell_block(P, c, q, eta20: int) -> tuple:
    D = d80(P)
    cf, qf = _flat(c), _flat(q)
    fails = [(k, cf[k], D[k]) for k in range(64) if cell_fail(cf[k], qf[k], D[k], eta20)]
    return (not fails), fails


def block_mcu_index(g, comp: int, r: int, c: int) -> tuple:
    m = 2 if g.sampling else 1
    mx = -(-g.W // (8 * m))
    if comp == 0 and m == 2:
        return (r // 2) * mx + (c // 2), 2 * (r % 2) + (c % 2)
    return r * mx + c, 0


def _qs(g):
    return [np.frombuffer(bytes(t), np.uint8).astype(np.int64).reshape(8, 8) for t in g.q3()]


def e0_block_fails(X, coefs, g) -> tuple:
    from cutline.edge import reference_blocks
    X = np.asarray(X)
    if X.shape[:2] != (g.H, g.W):
        raise ValueError(f"canvas {X.shape[:2]} != file {(g.H, g.W)}")
    refs = reference_blocks(X, g.sampling, 1)
    out = []
    for c, (P, q) in enumerate(zip(refs, _qs(g))):
        cf = np.asarray(coefs[c]).astype(np.int64)
        if P.shape != cf.shape:
            raise ValueError(f"component {c}: reference {P.shape} vs coefficients {cf.shape}")
        out.append(quant_islow(fdct_islow(P), q) != cf)
    return tuple(out)


def _e0_band(X, coefs, g, cert_mcus=None) -> list:
    from cutline.edge import reference_blocks
    masks = e0_block_fails(X, coefs, g)
    cert = None if cert_mcus is None else set(int(v) for v in cert_mcus)
    rows = []
    refs = None
    for comp, mk in enumerate(masks):
        bad = np.argwhere(mk.any(axis=(2, 3)))
        if len(bad) and refs is None:
            refs = reference_blocks(np.asarray(X), g.sampling, 1)
        for r, c in bad:
            mcu, blk = block_mcu_index(g, comp, int(r), int(c))
            if cert is not None and mcu not in cert:
                continue
            t = fdct_islow(refs[comp][r, c]).reshape(64)
            cf = np.asarray(coefs[comp][r, c]).reshape(64)
            for k in np.flatnonzero(mk[r, c].reshape(64)):
                rows.append((mcu, comp, blk, int(k), int(cf[k]), int(t[k])))
    rows.sort(key=lambda x: x[:4])
    return rows


def _e0_band_arrays(X, coefs, g, cert_mcus=None) -> np.ndarray:
    """``_e0_band``'s rows as one int64 array [n, 6] (mcu, comp, blk, k, c, t), vectorised (an all-failing listing
    costs numpy work, not a Python tuple per coefficient); same rows, same order."""
    from cutline.edge import reference_blocks
    X = _check_canvas(X, g)
    refs = reference_blocks(X, g.sampling, 1)
    cert = None if cert_mcus is None else np.unique(np.asarray([int(v) for v in cert_mcus], np.int64))
    parts = []
    for comp, (P, q) in enumerate(zip(refs, _qs(g))):
        cf = np.asarray(coefs[comp]).astype(np.int64)
        if P.shape != cf.shape:
            raise ValueError(f"component {comp}: reference {P.shape} vs coefficients {cf.shape}")
        t = fdct_islow(P)
        r, c, v, u = np.nonzero(quant_islow(t, q) != cf)
        if not len(r):
            continue
        hb, wb = cf.shape[:2]
        mcu_m, blk_m = block_mcu_map(g, comp, hb, wb)
        mcu = np.asarray(mcu_m[r, c], np.int64)
        if cert is not None:
            keep = np.isin(mcu, cert)
            r, c, v, u, mcu = r[keep], c[keep], v[keep], u[keep], mcu[keep]
        parts.append(np.stack([mcu, np.full(len(r), comp, np.int64), np.asarray(blk_m[r, c], np.int64),
                               (8 * v + u).astype(np.int64), cf[r, c, v, u], t[r, c, v, u]], axis=1))
    A = np.concatenate(parts) if parts else np.zeros((0, 6), np.int64)
    return A[np.lexsort((A[:, 3], A[:, 2], A[:, 1], A[:, 0]))]


def court_q(header: bytes, comp: int, k: int) -> int:
    q = header[5 + 64 * comp + k]
    if q < 1:
        raise CourtError("E.COURT.ZERO_Q", f"q[{comp}][{k}] = 0: static violation, use path a'")
    return q


# ================================================================================================ float prefilter
TWO_M20 = 2.0 ** -20


def block_mcu_map(g, comp: int, hb: int, wb: int) -> tuple:
    m = 2 if g.sampling else 1
    mx = -(-g.W // (8 * m))
    r = np.arange(hb, dtype=np.int64)[:, None]
    c = np.arange(wb, dtype=np.int64)[None, :]
    if comp == 0 and m == 2:
        return (r // 2) * mx + (c // 2), 2 * (r % 2) + (c % 2)
    return np.broadcast_to(r * mx + c, (hb, wb)), np.zeros((hb, wb), np.int64)


def _float_dct(P):
    C = consts.c40_np().astype(np.float64) / float(1 << 40)            # exact: every |C40| < 2^40
    return np.einsum("vy,...yx,ux->...vu", C, np.asarray(P, np.float64), C, optimize=True)


def _cert_mask(g, comp, hb, wb, cert_mcus):
    if cert_mcus is None:
        return np.ones((hb, wb), bool)
    mcu, _ = block_mcu_map(g, comp, hb, wb)
    return np.isin(mcu, np.fromiter((int(v) for v in cert_mcus), np.int64))


def _refs(X, g, front):
    from cutline.edge import reference_blocks
    if front not in ("phi", "go"):
        raise ValueError("front must be 'phi' or 'go'")
    X = np.asarray(X)
    if X.shape[:2] != (g.H, g.W):
        raise ValueError(f"canvas {X.shape[:2]} != file {(g.H, g.W)}")
    return reference_blocks(X, g.sampling, 5 if front == "go" else 1)


def cell_prefilter(P, cf, q, eta20: int) -> tuple:
    """(fail, near) bool[..., 8, 8] of the float64 prefilter over level-shifted blocks P and coefficients cf (natural
    order [..., v, u]); ``near`` = within the guard (2q + 1)·2^-20 of the threshold, where only the exact test
    decides."""
    qf = np.asarray(q).reshape(8, 8).astype(np.float64)
    margin = np.abs(np.asarray(cf, np.int64) * qf - _float_dct(P)) - (qf * (0.5 + TWO_M20) + eta20 * TWO_M20)
    guard = (2 * qf + 1) * TWO_M20
    return margin > guard, np.abs(margin) <= guard


def cell_blocks_fails(P, cf, q, eta20: int, mask=None) -> list:
    """Every failing (r, c, k, c_k, D80_k) of the blocks P[hb, wb] (``mask`` bool[hb, wb] restricts the blocks):
    the prefilter selects the blocks with a failing or near-threshold coefficient, ``cell_block`` decides them."""
    fail, near = cell_prefilter(P, cf, q, eta20)
    cand = (fail | near).any(axis=(-2, -1))
    if mask is not None:
        cand &= mask
    qq = np.asarray(q).reshape(8, 8)
    idx = np.argwhere(cand)
    out = []
    for s0 in range(0, len(idx), 4096):
        part = idx[s0:s0 + 4096]
        b, k, ck, w = _cell_exact_batch(P[part[:, 0], part[:, 1]], cf[part[:, 0], part[:, 1]], qq, eta20)
        rr, cc = part[b, 0].tolist(), part[b, 1].tolist()
        out.extend(zip(rr, cc, k.tolist(), ck.tolist(), w.tolist()))
    return out


_C40_OBJ = None


def _cell_exact_batch(Pb, cfb, q, eta20: int) -> tuple:
    """``cell_block`` on a batch of blocks, exactly (numpy instead of a Python loop per block):
    row = P·C40^T in int64 (|row| < 2^51), D80 = C40·row in Python ints (object dtype), then the CELL predicate
    |c·q·2^80 − D80| > (q·2^19 + eta20 + q)·2^60 per coefficient. Returns (block index, k, c_k, D80_k) of every failing
    coefficient, ascending by (block, k), with D80 an object array of Python ints; equal to ``cell_block`` block by
    block."""
    global _C40_OBJ
    if _C40_OBJ is None:
        _C40_OBJ = np.array([int(v) for v in consts.C40], dtype=object).reshape(8, 8)
    Pb = np.asarray(Pb, np.int64).reshape(-1, 8, 8)
    n = len(Pb)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, object)
    C64 = np.asarray(consts.C40, np.int64).reshape(8, 8)
    row = np.einsum("nyx,ux->nyu", Pb, C64)                             # [n, y, u], exact in int64
    D = np.matmul(_C40_OBJ, row.astype(object)).reshape(n, 64)         # [n, v*8+u], Python ints
    qf = [int(v) for v in np.asarray(q).reshape(64)]
    cfl = np.asarray(cfb, np.int64).reshape(n, 64)
    lhs = (cfl * np.asarray(qf, np.int64)).astype(object) * (1 << 80) - D
    thr = np.array([((qk << 19) + int(eta20) + qk) << 60 for qk in qf], dtype=object)
    fail = np.abs(lhs) > thr[None, :]
    fail = np.asarray(fail, bool)
    b, k = np.nonzero(fail)
    return b, k, cfl[b, k], D[b, k]


class WideRows:
    """A chunk of CELL rows: ``A`` int64 [n, 5] (mcu, comp, blk, k, c) and ``W`` object [n] (the D80 witnesses,
    Python ints up to about 2^94), ascending by (mcu, comp, blk, k)."""
    __slots__ = ("A", "W")

    def __init__(self, A, W):
        self.A, self.W = A, W

    def __len__(self):
        return len(self.A)

    def rows(self):
        return [tuple(a) + (w,) for a, w in zip(self.A.tolist(), self.W.tolist())]


def _cell_band_chunk(X, coefs, g, eta20: int, front: str = "phi", cert_mcus=None):
    """``_cell_band``'s rows as one ``WideRows`` chunk (vectorised mapping and sort)."""
    As, Ws = [], []
    for comp, (P, q) in enumerate(zip(_refs(X, g, front), _qs(g))):
        cf = np.asarray(coefs[comp]).astype(np.int64)
        if P.shape != cf.shape:
            raise ValueError(f"component {comp}: reference {P.shape} vs coefficients {cf.shape}")
        hb, wb = cf.shape[:2]
        mcu_m, blk_m = block_mcu_map(g, comp, hb, wb)
        fail, near = cell_prefilter(P, cf, q, eta20)
        cand = (fail | near).any(axis=(-2, -1)) & _cert_mask(g, comp, hb, wb, cert_mcus)
        idx = np.argwhere(cand)
        qq = np.asarray(q).reshape(8, 8)
        for s0 in range(0, len(idx), 4096):
            part = idx[s0:s0 + 4096]
            b, k, ck, w = _cell_exact_batch(P[part[:, 0], part[:, 1]], cf[part[:, 0], part[:, 1]], qq, eta20)
            if not len(b):
                continue
            r, c = part[b, 0], part[b, 1]
            As.append(np.stack([np.asarray(mcu_m[r, c], np.int64), np.full(len(b), comp, np.int64),
                                np.asarray(blk_m[r, c], np.int64), k.astype(np.int64), ck.astype(np.int64)], axis=1))
            Ws.append(w)
    if not As:
        return WideRows(np.zeros((0, 5), np.int64), np.zeros(0, object))
    A, W = np.concatenate(As), np.concatenate(Ws)
    o = np.lexsort((A[:, 3], A[:, 2], A[:, 1], A[:, 0]))
    return WideRows(A[o], W[o])


def _cell_band(X, coefs, g, eta20: int, front: str = "phi", cert_mcus=None) -> list:
    return _cell_band_chunk(X, coefs, g, eta20, front, cert_mcus).rows()


def _einf_band(X, coefs, g, eta20: int, front: str, cert_mcus, chunk: int) -> list:
    from cutline.verify.excess import e_inf_block
    B = consts.b30s_np()                                                # [k, p]
    out = []
    for comp, (P, q) in enumerate(zip(_refs(X, g, front), _qs(g))):
        hb, wb = P.shape[:2]
        qflat = q.reshape(64)
        qf = qflat.astype(np.float64)
        F = _float_dct(P).reshape(hb, wb, 64)
        b = (2 * np.floor(F / qf) + 1) * qf / 2
        dist = np.abs(F - b)
        tie = (eta20 + qf) * TWO_M20
        guard = (2 * qf + 1) * TWO_M20
        inA = dist <= tie
        amb = ((np.abs(dist - tie) <= guard) | (inA & (dist <= guard))).any(axis=2)
        W = np.where(inA, np.where(b > F, 1, -1) * qflat.astype(np.int64), 0)          # σ_k·q_k on A
        mcu_m, blk_m = block_mcu_map(g, comp, hb, wb)
        idx = np.argwhere(_cert_mask(g, comp, hb, wb, cert_mcus))
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            M = W[part[:, 0], part[:, 1]][:, :, None] * B[None, :, :]                  # [n, k, p]
            e = np.maximum(np.where(M > 0, M, 0).sum(axis=1), np.where(M < 0, -M, 0).sum(axis=1)).max(axis=1)
            for j, (r, c) in enumerate(part):
                v = e_inf_block(d80(P[r, c]), qflat, eta20)[0] if amb[r, c] else int(e[j])
                out.append((int(mcu_m[r, c]), comp, int(blk_m[r, c]), int(v)))
    out.sort(key=lambda x: x[:3])
    return out


def iter_einf(X, coefs, g, eta20: int, front: str = "phi", cert_mcus=None, chunk: int = 256,
              band_px: int | None = None):
    """``einf_file``'s tuples, generated band by band (MCU rows) in the same ascending order."""
    X = _check_canvas(X, g)
    if front not in ("phi", "go"):
        raise ValueError("front must be 'phi' or 'go'")
    mx = -(-int(g.W) // (16 if g.sampling else 8))
    cert = None if cert_mcus is None else np.unique(np.fromiter((int(v) for v in cert_mcus), np.int64))
    for off, R0, R1, Xb, cob, gb in bands(X, coefs, g, band_px):
        cb = _local_cert(cert, off, R1 * mx)
        if cb is not None and len(cb) == 0:
            continue
        for m, comp, blk, v in _einf_band(Xb, cob, gb, eta20, front, cb, chunk):
            yield m + off, comp, blk, v


def einf_file(X, coefs, g, eta20: int, front: str = "phi", cert_mcus=None, chunk: int = 256) -> list:
    return list(iter_einf(X, coefs, g, eta20, front, cert_mcus, chunk))


def _class_of(enc_class: int, cls=None):
    reg = consts.CLASS_REGISTRY if cls is None else cls
    e = reg.get(int(enc_class))
    if e is None or int(enc_class) in consts.RESERVED_CLASSES:
        raise HopError("E.HOP.CLASS", f"encClass {enc_class} is undefined or reserved")
    return e


def class3_tables_ok(g) -> bool:
    """Class 3 admits a file only if 1 <= q[c][k] <= Q_pol[c][k] for every (c, k)."""
    return all(1 <= v <= p for v, p in zip(b"".join(g.q3()), consts.Q_POL))


# ================================================================================================ bands
BAND_PX = 1 << 18            # canvas pixels per band of MCU rows (every block test is local to its MCU row)
ROW_CAP = 1024               # rows a report keeps (verify_e, VerifyHop, the vote's returned rows); verdicts never cap


class BandGrammar:
    """The parts of a file's grammar the block tests read (H, W, sampling, q3), for a band of whole MCU rows."""

    def __init__(self, g, H: int):
        self.H, self.W, self.sampling = int(H), int(g.W), int(g.sampling)
        self._q = g.q3()

    def q3(self):
        return self._q


def bands(X, coefs, g, band_px: int | None = None):
    """Yield ``(mcu_offset, R0, R1, X_band, coefs_band, g_band)`` over bands of whole MCU rows [R0, R1), in order.
    Every block test is local to an MCU row (Phi and h2v2 are per pixel / per 2x2 inside the MCU, and the edge
    padding only replicates the last row and column), so testing the bands and adding ``mcu_offset`` =
    R0·mx to each row's mcu gives exactly the whole-file rows, in the same ascending order."""
    p = 16 if g.sampling else 8
    nmy, mx = -(-int(g.H) // p), -(-int(g.W) // p)
    per = max(1, (BAND_PX if band_px is None else int(band_px)) // (p * max(1, int(g.W))))
    for R0 in range(0, nmy, per):
        R1 = min(nmy, R0 + per)
        y0, y1 = R0 * p, min(int(g.H), R1 * p)
        if g.sampling:
            cob = (coefs[0][2 * R0:2 * R1], coefs[1][R0:R1], coefs[2][R0:R1])
        else:
            cob = tuple(coefs[c][R0:R1] for c in range(3))
        yield R0 * mx, R0, R1, X[y0:y1], cob, BandGrammar(g, y1 - y0)


def _local_cert(cert_arr, lo: int, hi: int):
    if cert_arr is None:
        return None
    a, b = np.searchsorted(cert_arr, [lo, hi])
    return cert_arr[a:b] - lo


def _check_canvas(X, g):
    X = np.asarray(X)
    if X.shape[:2] != (g.H, g.W):
        raise ValueError(f"canvas {X.shape[:2]} != file {(g.H, g.W)}")
    return X


def iter_class_rows(enc_class: int, X, coefs, g, cert_mcus=None, ep=None, cls=None, band_px: int | None = None):
    """The rows of ``class_test_file``, generated band by band in the same ascending order: memory is
    bounded by one band, and a caller that needs only the verdict stops at the first row. The class checks
    (``E.HOP.CLASS``) run before the first row is produced."""
    _, eta20 = _class_of(enc_class, cls)
    c = int(enc_class)
    if c == 3 and not class3_tables_ok(g):
        raise HopError("E.HOP.CLASS", "class 3 needs every table entry within Q_pol")
    if c not in (1, 3, 5, 7):
        raise HopError("E.HOP.CLASS", f"encClass {enc_class} has no v1 block test")
    return _iter_rows(c, eta20, X, coefs, g, cert_mcus, ep, band_px)


def iter_class_row_chunks(enc_class: int, X, coefs, g, cert_mcus=None, ep=None, cls=None,
                          band_px: int | None = None):
    """``iter_class_rows`` grouped per band (arrays for class 1, lists otherwise); the vote hashes these."""
    _, eta20 = _class_of(enc_class, cls)
    c = int(enc_class)
    if c == 3 and not class3_tables_ok(g):
        raise HopError("E.HOP.CLASS", "class 3 needs every table entry within Q_pol")
    if c not in (1, 3, 5, 7):
        raise HopError("E.HOP.CLASS", f"encClass {enc_class} has no v1 block test")
    if c == 7:
        from cutline.verify.ep import ep_iter_rows
        return (lambda it: ([r] for r in it))(ep_iter_rows(ep, X, coefs, g, cert_mcus, band_px))
    return _banded_chunks(c, eta20, "go" if c == 5 else "phi", X, coefs, g, cert_mcus, band_px)


def _iter_rows(c, eta20, X, coefs, g, cert_mcus, ep, band_px):
    if c == 7:
        from cutline.verify.ep import ep_iter_rows
        yield from ep_iter_rows(ep, X, coefs, g, cert_mcus, band_px)
        return
    yield from _banded(c, eta20, "go" if c == 5 else "phi")(X, coefs, g, cert_mcus, band_px)


def first_row(rows_iter):
    """The first row of a row generator, or None (the verdict: fail iff not None)."""
    return next(iter(rows_iter), None)


def head_rows(rows_iter, cap: int = ROW_CAP) -> list:
    """At most ``cap`` rows from the front of a row generator (the report; the generator stops there)."""
    out = []
    for r in rows_iter:
        out.append(r)
        if len(out) >= cap:
            break
    return out


def _banded_chunks(c, eta20, front, X, coefs, g, cert_mcus=None, band_px=None):
    """One chunk per band, in order: an int64 array [n, 6] (class 1) or a ``WideRows`` (CELL; big witnesses)."""
    X = _check_canvas(X, g)
    mx = -(-int(g.W) // (16 if g.sampling else 8))
    cert = None if cert_mcus is None else np.unique(np.fromiter((int(v) for v in cert_mcus), np.int64))
    for off, R0, R1, Xb, cob, gb in bands(X, coefs, g, band_px):
        cb = _local_cert(cert, off, R1 * mx)
        if cb is not None and len(cb) == 0:
            continue
        if c == 1:
            A = _e0_band_arrays(Xb, cob, gb, cb)
            if len(A):
                A[:, 0] += off
                yield A
        else:
            ch = _cell_band_chunk(Xb, cob, gb, eta20, front, cb)
            if len(ch):
                ch.A[:, 0] += off
                yield ch


def rows_of_chunks(chunks):
    """Flatten row chunks (arrays or lists) into Python-int tuples, lazily."""
    for ch in chunks:
        if isinstance(ch, np.ndarray):
            yield from map(tuple, ch.tolist())
        elif isinstance(ch, WideRows):
            yield from ch.rows()
        else:
            yield from ch


def _banded(c, eta20, front):
    def run(X, coefs, g, cert_mcus=None, band_px=None):
        yield from rows_of_chunks(_banded_chunks(c, eta20, front, X, coefs, g, cert_mcus, band_px))
    return run


def e0_file_fails(X, coefs, g, cert_mcus=None) -> list:
    """TE rows of the EXACT test over the file (banded; the same rows and order as the whole-file form)."""
    return list(_banded(1, None, None)(X, coefs, g, cert_mcus))


def cell_file_fails(X, coefs, g, eta20: int, front: str = "phi", cert_mcus=None) -> list:
    if front not in ("phi", "go"):
        raise ValueError("front must be 'phi' or 'go'")
    return list(_banded(3, eta20, front)(X, coefs, g, cert_mcus))


def class_test_file(enc_class: int, X, coefs, g, cert_mcus=None, ep=None, cls=None) -> list:
    """Every failing row (a list). Unbounded by construction: verifiers use ``iter_class_rows`` with ``first_row`` /
    ``head_rows`` and the vote streams its transcript; this form is for small inputs."""
    return list(iter_class_rows(enc_class, X, coefs, g, cert_mcus, ep, cls))


def e0_hit(X, coefs, g, cert_mcus=None) -> bool:
    """The EXACT test passes on every tested block; stops at the first failing coefficient."""
    return first_row(iter_class_rows(1, X, coefs, g, cert_mcus)) is None
