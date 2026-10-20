"""cutline.verify.verify_t - VerifyT, the tier-T verifier.

- ``core_mask(alpha, D=(), sampling=0) -> bool[H, W]``: Erode({alpha = 255}, 3 px) with a 7x7 square (pixels
  outside the canvas do not erode, as ``cv2.erode``'s default border) **minus every pixel of a canvas MCU listed in
  D**: MCU size p = 8 (sampling 0) or 16 (sampling 1), canvas MCU grid (H div p) x (W div p); an entry outside the
  grid removes nothing (the static rules make it a violation). The one-argument call is the plain eroded core.
- ``tiert_statistics(PY, PX, core, table) -> (fails, tested_blocks)``: PY, PX = the Phi planes (Y, Cb, Cr) at full
  resolution of Dec*(listing) and of the compiled canvas; ``table`` = ``TierTTable`` (or a dict with keys ``id``,
  ``T`` {2: (Y, Cb, Cr), 4: …, 8: …}, ``G4``, ``G8``). For b ∈ {2, 4, 8} the aligned b x b blocks
  (0 ≤ by < ⌊H/b⌋, 0 ≤ bx < ⌊W/b⌋, pos = by·⌊W/b⌋ + bx) wholly inside the core are tested:
  statId 1 (block sum S of PY_c − PX_c, every channel, |S| > T[b][c]); statId 2 (b = 4, luma, level-1 Haar
  S1(PY) − S1(PX), |·| > G4); statId 3 (b = 8, luma, level-2 Haar S2(PY) − S2(PX), |·| > G8).
  ``fails`` = TT rows ``(tableId, b, pos, channel, statId, value)`` ascending by (b, pos, channel, statId);
  ``tested_blocks`` = Σ_b 3·(number of tested b-blocks) (every b, every channel).
- ``iter_tiert_rows(Y_rgb, X_rgb, core, table)``: the same rows from the RGB planes, generated band by band in the
  same order; ``tiert_count(core)`` = tested_blocks (the verifier and the vote use these).
- ``block_stats(PY, PX, core, b) -> dict``: the raw per-block arrays.
- ``TierTTable``, ``encode_table_record(t) -> 55 B``, ``decode_table_record(b)`` (``E.REG.TABLE_LEN``),
  ``table_digest(record) = H("CUTLINE-tierT-v1" ‖ record)``: defined in ``cutline.registries`` and re-exported here.
- ``verify_t(raw, ledger, *, h=None, listing=None) -> stages.VerifyResult``: from a ``RawBundle`` that passed
  stage A, in Evaluate's order: stage B (static rules, with the ledger's tier-T tables; a violation →
  ``static_violation``, G1); stage C without the fill coverage check (tier T needs no fill leaf) →
  ``g0_reason = "openings"`` on failure; Compile (Bg = the opened fill blocks and 0 elsewhere; the kernel is
  enc(π)'s, which equals exact(tierT[encClass].kernelFamily) by static rule 8c); ``tested_blocks = 0`` or
  ``sil_px = 0`` → ``"nothing_certified"`` (D can remove every tested block); then the tier-T statistics under
  **the claim's own table** ``ledger.tier_t(core.encClass)`` (whatever successors exist) on ``Dec*(listing)``
  (``dct.dec_star``) and X, both through Phi at full resolution; ``verdict`` "pass" / "fail", ``rows`` = the first
  TT rows, at most ``row_cap`` (default ``blocktest.ROW_CAP``; streamed band by band with ``iter_tiert_rows`` and
  ``tiert_count``). Facts: ``tier = "T"``, ``enc_classes = [encClass]``, ``e0_hit`` and ``excess`` None,
  ``excess_sum = excess_max = 0`` (the upload contributes no excess), Y0's own resolution class and four integers,
  ``tone``, ``background`` original or generated-declared (tier T has no class test), the claim facts of
  ``stages.claim_facts``. ``versions = {"classes": [], "tier_t": {"table", "family", "exact", "noise_milli"}}``.
  A ledger entry without its 55-byte record cannot be tested: ``E.EVAL.TIER_T_TABLE`` (reference-internal).
"""
from __future__ import annotations

import numpy as np

# The tier-T table record lives in cutline.registries; re-exported here unchanged.
from cutline.registries import BS, TierTTable, decode_table_record, encode_table_record, table_digest  # noqa: F401


# ------------------------------------------------------------------------------------------------ core
def core_mask(alpha, D=(), sampling: int = 0) -> np.ndarray:
    a = np.asarray(alpha) == 255
    H, W = a.shape
    # the 7x7 count of {alpha = 255} (outside the canvas counts as 255), as separable uint8 box sums (<= 49, so
    # exact in uint8; 1 byte per pixel instead of an int64 integral image)
    p = np.pad(a, 3, mode="constant", constant_values=True).astype(np.uint8)
    v = p[0:H].copy()
    for d in range(1, 7):
        v += p[d:d + H]
    del p
    s = v[:, 0:W].copy()
    for d in range(1, 7):
        s += v[:, d:d + W]
    del v
    core = s == 49
    if len(D):
        m = 8 if sampling == 0 else 16
        ny, nx = H // m, W // m
        for d in D:
            d = int(d)
            if 0 <= d < ny * nx:
                my, mx = divmod(d, nx)
                core[my * m:(my + 1) * m, mx * m:(mx + 1) * m] = False
    return core


# ------------------------------------------------------------------------------------------------ statistics
def _crop(P, b):
    P = np.asarray(P).astype(np.int64)
    H, W = P.shape
    return P[:H // b * b, :W // b * b]


def _bsum(P, b):
    P = _crop(P, b)
    h, w = P.shape
    return P.reshape(h // b, b, w // b, b).sum(axis=(1, 3))


def _det(P):
    p, q, r, s = P[0::2, 0::2], P[0::2, 1::2], P[1::2, 0::2], P[1::2, 1::2]
    return np.abs(p - q + r - s) + np.abs(p + q - r - s) + np.abs(p - q - r + s), p + q + r + s


def haar_s1(P):
    """S1 per aligned 4x4 block (level-1 Haar, unnormalised)."""
    det, _ = _det(_crop(P, 4))
    return _bsum(det, 2)


def haar_s2(P):
    """S2 per aligned 8x8 block (level-2 Haar on the 2x2 LL sums)."""
    _, ll = _det(_crop(P, 8))
    det2, _ = _det(ll)
    return _bsum(det2, 2)


def tested(core, b):
    c = _crop(core, b).astype(bool)
    h, w = c.shape
    return c.reshape(h // b, b, w // b, b).all(axis=(1, 3))


def block_stats(PY, PX, core, b) -> dict:
    out = {"tested": tested(core, b)}
    for c in range(3):
        out[f"S{c}"] = _bsum(np.asarray(PY[c]).astype(np.int64) - np.asarray(PX[c]).astype(np.int64), b)
    if b == 4:
        out["H1"] = haar_s1(PY[0]) - haar_s1(PX[0])
    if b == 8:
        out["H2"] = haar_s2(PY[0]) - haar_s2(PX[0])
    return out


def _tget(table, key):
    return table[key] if isinstance(table, dict) else getattr(table, key)


def _tested_count(core, b) -> int:
    c = np.asarray(core, bool)
    H, W = c.shape
    c = c[:H // b * b, :W // b * b]
    return int(c.reshape(H // b, b, W // b, b).all(axis=(1, 3)).sum())


def tiert_count(core) -> int:
    """``tiert_statistics``' tested_blocks from the core alone: Σ_b 3·(tested b-blocks)."""
    return sum(3 * _tested_count(core, b) for b in BS)


BAND_PX_T = 1 << 18


def iter_tiert_row_chunks(Y_rgb, X_rgb, core, table, band_px: int | None = None):
    """The rows of ``iter_tiert_rows`` as one int64 array [n, 6] per (b, band), in the same order (vectorised)."""
    from cutline.phi import phi
    tid = int(_tget(table, "id"))
    T = _tget(table, "T")
    G4, G8 = int(_tget(table, "G4")), int(_tget(table, "G8"))
    core = np.asarray(core, bool)
    H, W = core.shape
    step = max(8, ((BAND_PX_T if band_px is None else int(band_px)) // max(1, W)) // 8 * 8)
    for b in BS:
        wb = W // b
        for y0 in range(0, H, step):
            y1 = min(H, y0 + step)
            if (y1 - y0) < b:
                continue
            st = block_stats(phi(Y_rgb[y0:y1]), phi(X_rgb[y0:y1]), core[y0:y1], b)
            tb = st["tested"]
            cand = [(1, c, st[f"S{c}"], int(T[b][c])) for c in range(3)]
            if b == 4:
                cand.append((2, 0, st["H1"], G4))
            if b == 8:
                cand.append((3, 0, st["H2"], G8))
            parts = []
            oy = y0 // b
            for stat, c, V, thr in cand:
                by, bx = np.nonzero(tb & (np.abs(V) > thr))
                if len(by):
                    n = len(by)
                    parts.append(np.stack([np.full(n, tid, np.int64), np.full(n, b, np.int64),
                                           ((by + oy) * wb + bx).astype(np.int64), np.full(n, c, np.int64),
                                           np.full(n, stat, np.int64), np.asarray(V[by, bx], np.int64)], axis=1))
            if parts:
                A = np.concatenate(parts)
                yield A[np.lexsort((A[:, 4], A[:, 3], A[:, 2]))]


def iter_tiert_rows(Y_rgb, X_rgb, core, table, band_px: int | None = None):
    """``tiert_statistics(phi(Y_rgb), phi(X_rgb), core, table)[0]``, generated in the same (b, pos, channel, statId)
    order, band by band (rows of a multiple of 8 px: every aligned b-block, b <= 8, lies inside one band), so that
    memory is one band's Phi planes and a caller that needs only the verdict stops at the first row."""
    for A in iter_tiert_row_chunks(Y_rgb, X_rgb, core, table, band_px):
        yield from map(tuple, A.tolist())


def tiert_statistics(PY, PX, core, table) -> tuple:
    tid = int(_tget(table, "id"))
    T = _tget(table, "T")
    G4, G8 = int(_tget(table, "G4")), int(_tget(table, "G8"))
    rows = []
    n_tested = 0
    for b in BS:
        st = block_stats(PY, PX, core, b)
        tb = st["tested"]
        wb = tb.shape[1]
        n_tested += 3 * int(tb.sum())
        cand = []
        for c in range(3):
            S = st[f"S{c}"]
            cand.append((1, c, S, int(T[b][c])))
        if b == 4:
            cand.append((2, 0, st["H1"], G4))
        if b == 8:
            cand.append((3, 0, st["H2"], G8))
        for stat, c, V, thr in cand:
            for by, bx in np.argwhere(tb & (np.abs(V) > thr)):
                rows.append((tid, b, int(by) * wb + int(bx), c, stat, int(V[by, bx])))
    rows.sort(key=lambda r: (r[1], r[2], r[3], r[4]))
    return rows, n_tested


# ------------------------------------------------------------------------------------------------ VerifyT
def tier_t_versions(entry, kfam=None) -> dict:
    from cutline.registries import exact_of
    return {"table": entry.table_id, "family": entry.kernel_family, "exact": exact_of(entry.kernel_family, kfam),
            "noise_milli": entry.noise_milli}


def verify_t(raw, ledger, *, h=None, listing=None, row_cap: int | None = None):
    from cutline.dct import dec_star
    from cutline.errors import CutlineError, EvalError
    from cutline.phi import phi
    from cutline.verify.stages import (NOTHING_CERTIFIED, VerifyResult, claim_facts, listing_of, res_facts,
                                       stage_b, stage_c, tone_facts)
    core = raw.claim.core
    lst = listing if listing is not None else listing_of(raw)
    res = VerifyResult(tier="T", listing=lst, versions={"classes": [], "tier_t": None})
    sv = stage_b(raw, ledger, h, lst.header)
    if sv is not None:
        res.static_violation = tuple(sv)
        return res
    entry = ledger.tier_t(core.enc_class, h) if ledger is not None else None
    if entry is None or entry.record is None:
        raise EvalError("E.EVAL.TIER_T_TABLE", f"the ledger holds no 55-byte record for tierT[{core.enc_class}]")
    res.versions["tier_t"] = tier_t_versions(entry, ledger.kfam(h))
    try:
        op = stage_c(raw, None, ledger, "T")
    except CutlineError as e:
        res.g0_reason, res.error = "openings", e.name
        return res
    res.openings, res.compiled = op, op.compiled
    p = op.program
    comp = op.compiled
    facts = claim_facts(raw, op, "T")
    cm = core_mask(comp.alpha, p.D, p.sampling)
    tested = tiert_count(cm)
    facts.update({"enc_classes": [int(core.enc_class)], "tested_blocks": tested, "e0_hit": None, "excess": None,
                  "excess_sum": 0, "excess_max": 0})
    res.facts = facts
    if tested == 0 or facts["sil_px"] == 0:
        res.g0_reason = NOTHING_CERTIFIED
        return res
    from cutline.verify.blocktest import ROW_CAP, head_rows
    cap = ROW_CAP if row_cap is None else int(row_cap)
    rows = head_rows(iter_tiert_rows(dec_star(raw.members["listing.jpg"]), comp.X, cm, entry.table), cap)
    res.verdict = "fail" if rows else "pass"
    res.rows = rows
    res.rows_capped = len(rows) >= cap
    facts.update(res_facts(lst.g))
    facts.update(tone_facts(raw, op))
    D0 = len(p.D) == 0 and bool((np.asarray(comp.alpha) == 255).all())
    facts["background"] = "original" if D0 else "generated-declared"
    return res
