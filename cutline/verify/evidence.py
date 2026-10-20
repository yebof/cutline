"""cutline.verify.evidence - Evaluate: the evidence record, its level and label.

``evaluate(display_bytes, bundle_src, h, regime, ctx, ledger, fetch=None, *, urec=None, pubseg_runner=None, ...)
-> Evaluation`` runs steps 1-12 **in order**. It stops at the first failing step among 1-9 and 11 (the record then
has the short canonical form); step 10 records ``verdict = "fail"`` and continues with step 11.

| Step | What | Failure |
|---|---|---|
| 1 | ContainerOK v2.1 of the displayed file, rendition mode | ``"container"`` |
| 2 | a bundle; A1 (framing) and A2 (claim identity): ``bundle.read_bundle`` | ``"bundle"`` |
| 3 | R_Y v2 of the displayed file; chain assembly (``hop.assemble_chain``: the cap of 8 first, then connection, batch ids, inclusion) | ``"chain_too_long"`` / ``"unlinked"`` |
| 4 | presenter binding: ``core.marketplace = ctx.marketplace`` and ``core.listingRef = listing_ref(ctx)`` | ``"presenter"`` |
| 5 | ``ledger.claim_status(claimId, h)``: exists; ``level_cap``, ``cap``, ``claim``, ``receipt`` | ``"unregistered"`` |
| 6 | A3-A4 (``bundle.check_listing``, ``bundle.check_preimages``) | ``"bundle"`` |
| 7-10 | ``verify_e`` / ``verify_t`` by ``core.tier`` (static rules; openings; nothing certified; the tier test and Y0's fields) | ``static_violation`` / ``"openings"`` / ``"nothing_certified"`` |
| 10 | U on Y0: C_adj from S_pub and Sil; the S_pub source is a checked U record (``urec``) or the party's own PubSeg run (``pubseg_runner``) | - |
| 11 | ``hop.verify_chain`` over the assembled entries (fetch each parent, VerifyHop) | ``"ancestor"`` / ``"hop"`` (+ ``hop_index``) |
| 12 | chain fields, U carried to the displayed file, ``versions``, Level, L = min(Level, level_cap), label | - |

Inputs:
- ``display_bytes``: the displayed file (the listing itself, or the last rendition of the chain).
- ``bundle_src``: anything ``bundle.load_members`` takes (directory, ``.ustar`` path, archive bytes, member dict) or
  ``None`` (no bundle: ``"bundle"``).
- ``h``: the evaluation height; ``ledger``: a ``ledger_view.LedgerView`` (never a network).
- ``regime``: ``None`` or a genesis label regime (``labels.LABELS_V1``); ``label_version`` its table version.
- ``ctx``: ``types.Ctx(marketplace, listing_id, seller_account, salt_ref)`` of the page (``None``: no page context ->
  ``"presenter"``).
- ``fetch(locator) -> bytes | None``: the parent fetcher for step 11 (content-addressed by ``hop.fetch_parent``).
- ``urec``: a platform U record; used iff ``use_record`` accepts it with ``accepted_digests`` (the record's digest
  must be in ``accepted_digests``; when ``accepted_digests`` is None the party accepts only its own PubSeg build's
  digest, ``local_pubseg_digests()`` = ``[pubseg.digest("isnet", 3)]``); on any ``E.U.*`` error the party runs
  PubSeg itself (``notes["urec_error"]``). An accepted digest does not make S_pub verifiable: the party trusts the
  record's heuristic mask only under a digest it chose to accept.
- ``pubseg_runner(listing_bytes) -> (S_pub, digest)``: the party's own PubSeg (default ``default_pubseg_runner``:
  ``region.pubseg.s_pub`` with ISNet at 3 threads and ``pubseg.digest()``; loads ONNX Runtime).
- ``chain_id`` / ``registry``: default ``ledger.chain_id`` / ``ledger.registry`` (A2's claimant check).
- ``ep``: the local EP encoder override for class 7 (``verify.ep``); ``codec_kw``: passed to ``verify_chain``.
- ``evaluated``: False for a receipt-only party (labels only).

``Evaluation``: ``L`` (final level), ``level`` (Level of the record), ``label`` (``labels.label`` or None),
``evidence`` (``types.Evidence``), ``record`` (``verify.record.canonical`` bytes), ``notes`` (A5 anchor diagnostic,
U source, the step-10 rows count, the error name at the stopping step), ``verify`` (the ``VerifyResult``),
``chain`` (the ``hop.ChainOutcome``). ``tuple(evaluation) == (L, label, evidence)``.

``anchor_diagnostic(raw, ledger) -> dict``: A5's optional offline re-check of the inclusion of the capture's v2 leaf
with ``ledger.capture_inclusion``; a mismatch is a note and never changes the level.

Record fields that Level reads: ``geometry`` / ``cum_scale`` of a rendition = the chain's cumulative (Cw, Ch, Ow, Oh);
``uncovered_px`` / ``grid_wh`` = |C_adj(Y0)| on Y0's (W, H) for every evaluated file, the listing and each rendition
alike (U is a per-claim verdict decided once on Y0's grid and renditions do not re-test it, so a rendition never
exceeds its source level through U); ``s_pub_packbits`` / ``c_adj_packbits`` = Y0's masks; ``versions.classes`` = the
upload's class (tier E) and every hop's class. The C_adj pushed along the hops is reported only as information
(``notes["cadj_carried_px"]`` / ``notes["cadj_carried_grid"]``; never a level input).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cutline.errors import CutlineError
from cutline.types import Evidence
from cutline.verify import record as REC


@dataclass
class Evaluation:
    L: int
    level: int
    label: dict | None
    evidence: Evidence
    record: bytes
    notes: dict = field(default_factory=dict)
    verify: object = None
    chain: object = None

    def __iter__(self):
        return iter((self.L, self.label, self.evidence))


def default_pubseg_runner(listing_bytes: bytes):
    from cutline.region import pubseg
    return pubseg.s_pub(listing_bytes, "isnet", 3), pubseg.digest("isnet", 3)


def local_pubseg_digests() -> list:
    """The digests a party accepts when it names none: only its own PubSeg build's ``pubseg.digest("isnet", 3)``;
    an empty list when that build is not available here (the record is then rejected with ``E.U.RECORD_DIGEST`` and
    the party runs PubSeg itself)."""
    from cutline.region import pubseg
    try:
        return [pubseg.digest("isnet", 3)]
    except Exception:                       # builds differ / model file missing: accept nothing foreign
        return []


def anchor_diagnostic(raw, ledger) -> dict:
    from cutline.commit.anchor import anchor_leaf_v2
    from cutline.commit.rec import decode_rec_strict
    from cutline.commit.slots import slot_tag
    from cutline.crypto import H
    c = raw.claim
    try:
        rec = decode_rec_strict(c.rec)
        leaf = anchor_leaf_v2(H(c.rec), int.from_bytes(c.r, "big"), int.from_bytes(c.s, "big"),
                              slot_tag(rec.key_id, rec.epoch, rec.slot))
        ha = ledger.capture_inclusion(c.batch_id, leaf, c.leaf_index, list(c.anchor_path))
    except (CutlineError, ValueError, AttributeError) as e:
        return {"a5": "error", "detail": str(e)}
    return {"a5": "ok", "anchor_height": ha} if ha is not None else {"a5": "inclusion_mismatch"}


def _classes(ledger, enc_classes, tier_upload: str) -> list:
    from cutline.consts import CLASS_REGISTRY
    used = list(enc_classes[1:]) + ([enc_classes[0]] if tier_upload == "E" and enc_classes else [])
    out = []
    for c in sorted(set(int(x) for x in used)):
        ke = ledger.cls(c) if ledger is not None else CLASS_REGISTRY.get(c)
        if ke is not None:
            out.append([c, int(ke[0]), int(ke[1])])
    return out


def _short(e: Evidence, h: int, stop: str, notes: dict, regime, label_version, evaluated, vr=None, ch=None):
    from cutline import labels
    from cutline.verify.levels import final, level
    e.height = h
    e.versions = REC.versions()
    lv = level(e)
    L = final(e)
    notes.setdefault("stopped_at", stop)
    lab = labels.label(regime, L, e, label_version, evaluated) if regime else None
    return Evaluation(L, lv, lab, e, REC.canonical(e), notes, vr, ch)


def evaluate(display_bytes, bundle_src, h: int, regime, ctx, ledger, fetch=None, *, urec=None, pubseg_runner=None,
             accepted_digests=None, chain_id=None, registry=None, label_version: int = 1, ep=None,
             evaluated: bool = True, anchor_check: bool = True, **codec_kw) -> Evaluation:
    from cutline import bundle as B
    from cutline import labels
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    from cutline.jpeg.ry import r_y_v2
    from cutline.ledger.abi import listing_ref
    from cutline.region import uncovered as U
    from cutline.verify import hop
    from cutline.verify.levels import final, level

    e = Evidence()
    notes: dict = {}
    sh = lambda stop, vr=None, ch=None: _short(e, h, stop, notes, regime, label_version, evaluated, vr, ch)  # noqa

    # 1 ContainerOK v2.1, rendition mode
    try:
        disp = bytes(display_bytes)
        gd = container_ok_v2(disp, False)
        r_disp = r_y_v2(gd, read_coefs(disp, gd))
    except CutlineError as ex:
        e.g0_reason, notes["error"] = "container", ex.name
        return sh("1")
    # 2 bundle A1-A2
    if bundle_src is None:
        e.g0_reason, notes["error"] = "bundle", "no bundle"
        return sh("2")
    try:
        raw = B.read_bundle(bundle_src, ledger.chain_id if chain_id is None else chain_id,
                            ledger.registry if registry is None else registry)
    except CutlineError as ex:
        e.g0_reason, notes["error"] = "bundle", ex.name
        return sh("2")
    core = raw.claim.core
    notes["claim_id"] = raw.claim_id
    # 3 chain assembly
    try:
        entries = hop.assemble_chain(r_disp, core.r_y, raw.members.get("hops.bin", []), ledger, h)
    except CutlineError as ex:
        e.g0_reason, notes["error"] = hop.g0_reason_of(ex.name), ex.name
        return sh("3")
    # 4 presenter binding
    ok = ctx is not None and int(core.marketplace) == int(ctx.marketplace) and bytes(core.listing_ref) == listing_ref(
        int(ctx.marketplace), ctx.listing_id, ctx.seller_account, bytes(ctx.salt_ref))
    if not ok:
        e.g0_reason, notes["error"] = "presenter", "listingRef / marketplace"
        return sh("4")
    e.presenter = True
    # 5 claim registered at h
    st = ledger.claim_status(raw.claim_id, h)
    if not st.exists:
        e.g0_reason, notes["error"] = "unregistered", "claims[claimId] does not exist at h"
        return sh("5")
    e.level_cap, e.cap, e.claim, e.receipt = st.level_cap, st.cap, st.claim, st.receipt
    # 6 A3-A4
    try:
        B.check_listing(raw)
        B.check_preimages(raw)
    except CutlineError as ex:
        e.g0_reason, notes["error"] = "bundle", ex.name
        return sh("6")
    if anchor_check:
        notes["anchor"] = anchor_diagnostic(raw, ledger)
    # 7-10 the declared tier
    if int(core.tier) == 2:
        from cutline.verify.verify_t import verify_t
        vr = verify_t(raw, ledger, h=h)
    else:
        from cutline.verify.verify_e import verify_e
        vr = verify_e(raw, ledger, h=h, ep=ep)
    vr.evidence(e)
    if vr.static_violation is not None:
        return sh("7", vr)
    if vr.g0_reason == "openings":
        notes["error"] = vr.error
        return sh("8", vr)
    if vr.g0_reason is not None:
        return sh("9", vr)
    notes["rows"] = len(vr.rows)
    # 10 U on Y0
    listing = raw.members["listing.jpg"]
    sil = np.asarray(vr.compiled.sil, bool)
    S_pub = C = digest = None
    if urec is not None:
        from cutline.errors import UError
        try:
            acc = accepted_digests if accepted_digests is not None else local_pubseg_digests()
            C, rec = U.use_record_full(urec, core.r_y, acc, sil)       # one decode; grid checked before unpack
            S_pub, digest = rec.s_pub, rec.digest
            notes["u_source"] = "record"
        except UError as ex:
            notes["urec_error"] = ex.name
            C = None
    if C is None:
        S_pub, digest = (pubseg_runner or default_pubseg_runner)(listing)
        S_pub = np.asarray(S_pub, bool)
        C = U.c_adj(S_pub, sil)
        notes["u_source"] = "pubseg"
    e.heuristic_digests = dict(e.heuristic_digests or {}, pubseg=bytes(digest))
    # 11 the hops
    prog = vr.openings.program
    st0 = hop.initial_state(e, prog, sil=sil, c_adj=C, cert_mask=vr.compiled.cert_mask)
    if entries:
        ch = hop.verify_chain(entries, st0, fetch, disp, ledger, h, first_parent=listing, **codec_kw)
    else:
        ch = hop.ChainOutcome(st0)
    if ch.g0_reason is not None:
        e.g0_reason, e.hop_index, notes["error"] = ch.g0_reason, ch.hop_index, ch.error
        return sh("11", vr, ch)
    s = ch.state
    # 12 chain fields, U on Y0, versions, Level
    e.chain_class, e.lambda_chain_2p51 = s.chain_class, s.lambda_chain_2p51
    e.enc_classes = [int(c) for c in s.enc_classes]
    e.excess_sum, e.excess_max = s.excess_sum, s.excess_max
    e.partial = bool(s.partial)
    e.hop_depth, e.resampling_hops = s.hop_depth, s.resamples
    e.hop_ceilings = [[int(p), int(v)] for p, v in s.ceilings]
    if entries:
        e.geometry = [int(x) for x in s.cum]
        e.cum_scale = [int(x) for x in s.cum]
    # U is decided on Y0's grid (the U record / PubSeg on Y0); the rendition does not re-test it
    H0, W0 = sil.shape
    C0 = np.asarray(C, bool)
    e.__dict__.update(U.evidence_fields(int(C0.sum()), (W0, H0), digest, S_pub, C))
    if entries and s.c_adj is not None:            # informational only (never a level input)
        Hd, Wd = s.dims
        notes["cadj_carried_px"], notes["cadj_carried_grid"] = int(np.asarray(s.c_adj, bool).sum()), [int(Wd), int(Hd)]
    e.height = h
    e.versions = REC.versions(classes=_classes(ledger, e.enc_classes, vr.tier), tier_t=vr.versions.get("tier_t"),
                              hop_ceilings=e.hop_ceilings,
                              label_table={"regime": regime, "version": label_version} if regime else None)
    lv = level(e)
    L = final(e)
    lab = labels.label(regime, L, e, label_version, evaluated) if regime else None
    return Evaluation(L, lv, lab, e, REC.canonical(e), notes, vr, ch)


__all__ = ["Evaluation", "evaluate", "anchor_diagnostic", "default_pubseg_runner", "local_pubseg_digests"]
