"""cutline.verify.verify_e - VerifyE, the tier-E verifier; also the seller's self-check.

``verify_e(raw, ledger=None, *, h=None, all_rows=True, fraud=True, listing=None) -> VerifyResult`` continues from a
``RawBundle`` that passed stage A (``bundle.stage_a``), in Evaluate's order, stopping at the first failing step:

1. stage B (``stages.stage_b``): a violated static rule → ``static_violation`` (G1; nothing else is evaluated);
2. stage C (``stages.stage_c``, tier E, with the fill coverage check): a failure → ``g0_reason = "openings"`` and
   ``error`` = its E.* name (``E.BUNDLE.PROOF``, ``E.VERIFYE.OPENING_SET``, ``E.MERKLE.*``, ``E.BUNDLE.ZLIB``,
   ``E.BUNDLE.ABI``, ``E.BUNDLE.FILL``, ``E.VERIFYE.FILL``);
3. Compile's facts; ``tested_blocks = 0`` (M_cert = ∅) or ``sil_px = 0`` → ``g0_reason = "nothing_certified"``;
4. the class test of ``core.encClass`` (``blocktest.iter_class_rows``) on every real block of every MCU of M_cert
   (D excluded): ``verdict`` "pass" / "fail" and ``rows`` = every failing TE row ``(mcu, comp, blk, k, c, w)``
   ascending, **at most ``row_cap``** (default ``blocktest.ROW_CAP`` = 1024; ``rows_capped`` tells when the cap was
   reached; ``all_rows=False`` keeps only the first). The test is streamed band by band and stops once the report is
   full: the verdict is unchanged, memory is one band, and an all-failing listing costs no more than the first
   ``row_cap`` rows. After a fail the remaining record fields are still computed (the full canonical record), but
   each boolean test stops at its first failing row (``e0_hit``, the constant-background test).

Facts (``VerifyResult.evidence()`` copies them into ``types.Evidence``): ``tier = "E"``, ``enc_classes = [encClass]``,
``mcert_count``, ``tested_blocks``, ``sil_px``, the resolution class of Y0 and its four integers, ``excess`` (EXACT
and EP: map ``[]`` and zeros; CELL: ``blocktest.iter_einf`` over the tested blocks, ``map_2p30`` the values > 0,
p50 / p99 / max by nearest rank over all tested blocks), ``excess_sum`` and ``excess_max`` (the upload's
E_inf(file)), ``e0_hit`` (class ≠ 1 and the EXACT test passes on every tested block), ``tone``, ``background``
(*original* iff D = ∅ and every canvas alpha = 255; *constant* iff not original, the whole fill is opened and is one
colour c, and every canvas MCU passes the class test against the composite over c; else *generated-declared*),
``geometry`` / ``cum_scale``, ``partial``, ``d_count``, ``d_frac_ppm``, ``d_gen``, ``category``, ``liveness``.
``versions`` = ``{"classes": [[encClass, kind, eta20]], "tier_t": None}``.

``fraud``: the first failing unit as a ``types.FraudUnit``; for a 4:4:4 claim its ``proof`` is the ProofIn v2 built
with ``ledger.proofin.build_proofin`` (``verify.fraud.fraud_unit``); 4:2:0 claims are challenged through the
sidecar vote (path b) and carry no ProofIn.

The ledger (``verify.ledger_view.LedgerView``) supplies the tier-T tables and kernel families of stage B and the
class registry (``cls``); ``None`` means the genesis registries.
"""
from __future__ import annotations

from cutline.consts import CLASS_REGISTRY
from cutline.errors import CutlineError
from cutline.verify import blocktest
from cutline.verify.excess import excess_summary
from cutline.verify.stages import (NOTHING_CERTIFIED, VerifyResult, claim_facts, is_original, listing_of, res_facts,
                                   stage_b, stage_c, tone_facts)


def _cls_map(ledger, enc_class: int) -> dict:
    if ledger is None:
        return dict(CLASS_REGISTRY)
    e = ledger.cls(enc_class)
    return {} if e is None else {int(enc_class): tuple(e)}


def excess_facts(enc_class: int, X, coefs, g, cert, cls: dict) -> dict:
    kind, eta20 = cls.get(int(enc_class), (None, None))
    if enc_class in (3, 5):
        vals, emap = [], []
        for m, c, b, v in blocktest.iter_einf(X, coefs, g, eta20, "go" if enc_class == 5 else "phi", cert):
            vals.append(v)
            if v > 0:
                emap.append([m, c, b, v])
        s = excess_summary(vals)
        ex = {"map_2p30": emap, "p50": s["p50"], "p99": s["p99"], "max": s["max"]}
        e = s["max"] or 0
    else:
        ex = {"map_2p30": [], "p50": 0, "p99": 0, "max": 0}
        e = 0
    return {"excess": ex, "excess_sum": e, "excess_max": e}


def background_class(op, enc_class: int, coefs, g, cls: dict, ep=None) -> str:
    """The background class for tier E (see the module docstring); the constant test stops at its first failing
    row."""
    if is_original(op):
        return "original"
    if not op.whole_fill:
        return "generated-declared"
    blocks = list(op.fill_blocks.values())
    c = blocks[0][:3]
    if any(b != c * (len(b) // 3) for b in blocks):
        return "generated-declared"
    # the whole fill is one colour c, so rFill recomputed from c is the verified root; test every canvas MCU
    bad = blocktest.first_row(blocktest.iter_class_rows(enc_class, op.compiled.X, coefs, g, None, ep=ep, cls=cls))
    return "generated-declared" if bad is not None else "constant"


def verify_e(raw, ledger=None, *, h=None, all_rows: bool = True, fraud: bool = True, listing=None,
             ep=None, row_cap: int | None = None) -> VerifyResult:
    core = raw.claim.core
    lst = listing if listing is not None else listing_of(raw)
    cls = _cls_map(ledger, core.enc_class)
    kind_eta = cls.get(int(core.enc_class))
    vers = {"classes": [[int(core.enc_class), *kind_eta]] if kind_eta else [], "tier_t": None}
    res = VerifyResult(tier="E", listing=lst, versions=vers)
    sv = stage_b(raw, ledger, h, lst.header)
    if sv is not None:
        res.static_violation = tuple(sv)
        return res
    try:
        op = stage_c(raw, None, ledger, "E")
    except CutlineError as e:
        res.g0_reason, res.error = "openings", e.name
        return res
    res.openings, res.compiled = op, op.compiled
    facts = claim_facts(raw, op, "E")
    facts["enc_classes"] = [int(core.enc_class)]
    res.facts = facts
    if facts["tested_blocks"] == 0 or facts["sil_px"] == 0:
        res.g0_reason = NOTHING_CERTIFIED
        return res
    g, co, X, cert = lst.g, lst.coefs, op.compiled.X, op.compiled.m_cert
    # the verdict needs the first failing row only; a report keeps at most row_cap rows
    cap = (blocktest.ROW_CAP if row_cap is None else int(row_cap)) if all_rows else 1
    rows = blocktest.head_rows(blocktest.iter_class_rows(core.enc_class, X, co, g, cert, ep=ep, cls=cls), cap)
    res.verdict = "fail" if rows else "pass"
    res.rows = rows
    res.rows_capped = len(rows) >= cap
    facts.update(res_facts(g))
    facts.update(excess_facts(core.enc_class, X, co, g, cert, cls))
    # e0_hit is false by definition for class 1 and when the EXACT test fails on some block (first failure stops it)
    facts["e0_hit"] = bool(int(core.enc_class) != 1 and blocktest.e0_hit(X, co, g, cert))
    facts.update(tone_facts(raw, op))
    facts["background"] = background_class(op, core.enc_class, co, g, cls, ep)
    if rows and fraud:
        from cutline.verify.fraud import fraud_unit
        res.fraud = fraud_unit(raw, rows[0], lst)
    return res


def selfcheck(src, chain_id: int, registry: bytes, ledger=None, h=None) -> dict:
    """The seller's self-check: stage A, then ``verify_e`` / ``verify_t`` by the claim's tier. Returns
    ``{"ok": bool, "stage": "A" | "B" | "C" | "certified" | "test", "error" | "static_violation" | "fraud" ...}``."""
    from cutline.bundle import stage_a
    try:
        raw = stage_a(src, chain_id, registry)
    except CutlineError as e:
        return {"ok": False, "stage": "A", "error": e.name, "detail": e.detail}
    if raw.claim.core.tier == 2:
        from cutline.verify.verify_t import verify_t
        r = verify_t(raw, ledger, h=h)
    else:
        r = verify_e(raw, ledger, h=h)
    out = {"claim_id": "0x" + raw.claim_id.hex(), "tier": r.tier}
    if r.static_violation is not None:
        return {**out, "ok": False, "stage": "B", "static_violation": list(r.static_violation)}
    if r.g0_reason == "openings":
        return {**out, "ok": False, "stage": "C", "error": r.error}
    if r.nothing_certified:
        return {**out, "ok": False, "stage": "certified", "g0_reason": r.g0_reason,
                "mcert_count": r.facts.get("mcert_count"), "sil_px": r.facts.get("sil_px"),
                "tested_blocks": r.facts.get("tested_blocks")}
    out.update({"ok": r.verdict == "pass", "stage": "test", "verdict": r.verdict, "n_rows": len(r.rows),
                "rows_capped": bool(getattr(r, "rows_capped", False)),
                "tested_blocks": r.facts["tested_blocks"], "mcert_count": r.facts["mcert_count"]})
    if r.rows:
        out["first_row"] = [int(v) for v in r.rows[0]]
    if r.fraud is not None:
        out["fraud_unit"] = {"mcu": r.fraud.mcu, "comp": r.fraud.comp, "blk": r.fraud.blk, "k": r.fraud.k,
                             "provable": r.fraud.proof is not None}
    return out


__all__ = ["verify_e", "selfcheck", "excess_facts", "background_class", "VerifyResult", "NOTHING_CERTIFIED"]
