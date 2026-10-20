"""cutline.verify.vote - a sidecar's vote on a path-(b) challenge target.

``vote(bundle_src, target, ledger, chain_id=None, registry=None, *, h=None, path_b_only=True)`` returns one of

- ``("abstain", reason)``: the sidecar casts no vote. ``reason`` is ``"not_verifying:<E.* name>"`` when the bundle
  is not a VerifyingBundle (stage A (A1-A4: framing, strict ustar and zlib, canonical ABI files with every exact
  length, the claimant check, the listing (ContainerOK v2.1 upload mode, R_Y = core.rY) and the preimages), stage C
  (occupancy, the completeness set, the cell multiproof, and for tier E the fill of every M_cert MCU with
  min alpha < 255)), ``"static_violation:<rule>,<i>,<j>"`` (the static proof strikes instead),
  ``"target"`` (``H(abi.encode(nf, claimant, core))`` ≠ the target) or ``"not_path_b"`` (a tier-E 4:4:4 claim: the
  block court's domain, never a challenge target; skipped with ``path_b_only=False`` to compute transcripts of any
  claim). A5 (the anchor data) is not part of the domain: a corrupted ``anchorPath`` changes nothing.
- ``("PASS", transcript)`` with ``transcript = H("pass" ‖ target)`` when the tested set has no failing row,
  including the vacuous case of an empty tested set (such a claim is G0 "nothing certified" anyway).
- ``("FAIL", rows, transcript)`` with ``transcript = H(abi.encode(TE[] rows))`` (tier E) or
  ``H(abi.encode(TT[] rows))`` (tier T) over **every** failing row, streamed (``ledger.abi.stream_fail_transcript``:
  one pass counts N, a second hashes the rows in order; the rows are never held in memory), and ``rows`` = the first
  ``row_cap`` rows (default ``blocktest.ROW_CAP`` = 1024) for the report. N ≥ 1 always, so ``EMPTY_ROWS`` is never
  emitted.
- ``("abstain", "no_bundle")`` when ``bundle_src`` is None.

``vote_any(sources, target, ledger, ...)``: the vote over the first verifying copy among any number of sources (the
platform's locator, the listing page, a watcher's, a cache); a platform that withholds its copy cannot force an
abstention when any other source has one.

**Tested set**: tier E: every real block of every MCU of M_cert (D excluded), every k, under the claim's class test on
Phi (h2v2 chroma at 4:2:0) (``blocktest.iter_class_row_chunks``); tier T: every aligned block inside the core (D's
MCUs excluded), every b, the claim's own table (``verify_t.core_mask`` / ``iter_tiert_row_chunks``). Stage B
precedes stage C (every failure abstains, so the order changes no vote).

``ledger`` (``LedgerView``) gives the tier-T table (tier T), the kernel families and the class registry; ``chain_id``
and ``registry`` default to ``ledger.chain_id`` / ``ledger.registry`` (the claimant check of A2).
"""
from __future__ import annotations

from cutline.errors import CutlineError
from cutline.ledger.abi import pass_transcript, stream_fail_transcript


def _tested(raw, ledger, h=None, listing=None) -> tuple:
    """(tier, chunks_factory) of the tested set for a bundle that passed stage A; raises ``_Abstain``.
    ``chunks_factory()`` returns a fresh generator of row chunks (one per band; int64 arrays or lists) in row order;
    None when the tested set has no failing row."""
    from cutline.dct import dec_star
    from cutline.verify import blocktest
    from cutline.verify.stages import listing_of, stage_b, stage_c
    from cutline.verify.verify_t import core_mask, iter_tiert_row_chunks
    core = raw.claim.core
    lst = listing if listing is not None else listing_of(raw)
    try:
        sv = stage_b(raw, ledger, h, lst.header)
    except CutlineError as e:
        raise _Abstain("not_verifying:" + e.name) from None
    if sv is not None:
        raise _Abstain("static_violation:%d,%d,%d" % tuple(sv))
    tier = "T" if core.tier == 2 else "E"
    try:
        op = stage_c(raw, None, ledger, tier)
    except CutlineError as e:
        raise _Abstain("not_verifying:" + e.name) from None
    comp = op.compiled
    if tier == "E":
        cls = None
        if ledger is not None:
            e = ledger.cls(core.enc_class)
            cls = {} if e is None else {int(core.enc_class): tuple(e)}
        fac = lambda: blocktest.iter_class_row_chunks(core.enc_class, comp.X, lst.coefs, lst.g,   # noqa: E731
                                                      comp.m_cert, cls=cls)
        try:
            first = blocktest.first_row(blocktest.rows_of_chunks(fac()))
        except CutlineError as e:
            raise _Abstain("not_verifying:" + e.name) from None
        return tier, (fac if first is not None else None)
    entry = ledger.tier_t(core.enc_class, h) if ledger is not None else None
    if entry is None or entry.record is None:
        raise _Abstain("not_verifying:E.CLAIM.RULE8")
    p = op.program
    Y = dec_star(raw.members["listing.jpg"])
    cm = core_mask(comp.alpha, p.D, p.sampling)
    fac = lambda: iter_tiert_row_chunks(Y, comp.X, cm, entry.table)    # noqa: E731
    return tier, (fac if blocktest.first_row(blocktest.rows_of_chunks(fac())) is not None else None)


def tested_rows(raw, ledger, h=None, listing=None) -> tuple:
    """(tier, rows): every row of the tested set as a list (unbounded; ``vote`` streams)."""
    from cutline.verify.blocktest import rows_of_chunks
    tier, fac = _tested(raw, ledger, h, listing)
    return tier, ([] if fac is None else list(rows_of_chunks(fac())))


class _Abstain(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def vote(bundle_src, target: bytes, ledger, chain_id: int | None = None, registry: bytes | None = None, *, h=None,
         path_b_only: bool = True, row_cap: int | None = None):
    from cutline.bundle import stage_a
    from cutline.verify.blocktest import ROW_CAP, head_rows
    from cutline.verify.stages import listing_of
    cid = ledger.chain_id if chain_id is None else chain_id
    reg = ledger.registry if registry is None else registry
    if bundle_src is None:
        return ("abstain", "no_bundle")
    try:
        raw = stage_a(bundle_src, cid, reg)
    except CutlineError as e:
        return ("abstain", "not_verifying:" + e.name)
    if raw.claim_id != bytes(target):
        return ("abstain", "target")
    lst = listing_of(raw)
    if path_b_only and raw.claim.core.tier != 2 and lst.g.sampling != 1:
        return ("abstain", "not_path_b")
    try:
        tier, fac = _tested(raw, ledger, h, lst)
    except _Abstain as a:
        return ("abstain", a.reason)
    if fac is None:
        return ("PASS", pass_transcript(bytes(target)))
    from cutline.verify.blocktest import rows_of_chunks
    _, transcript = stream_fail_transcript(fac, tier)
    return ("FAIL", head_rows(rows_of_chunks(fac()), ROW_CAP if row_cap is None else int(row_cap)), transcript)


# Reasons that belong to one copy of the bundle, not to the claim: another copy may still verify.
_COPY_REASONS = ("no_bundle", "target", "not_verifying:")


def vote_any(sources, target: bytes, ledger, chain_id: int | None = None, registry: bytes | None = None, *,
             h=None, path_b_only: bool = True, row_cap: int | None = None):
    """A sidecar may take the bundle from **any** source (the relaying platform, the listing page, a watcher's or
    another validator's locator, a local cache). A bundle is self-authenticating (stages A-C bind every member to the
    claim), so every verifying copy gives the same tested set
    and the same transcript. The sidecar tries the sources in order and votes on the first **verifying** copy for the
    target; a copy that is missing, of another claim or not verifying is skipped. It abstains only when no source
    yields a verifying copy (``"no_verifying_bundle"``), or for a reason of the claim itself (a static violation,
    a tier-E 4:4:4 target), which every copy shares. Returns ``vote``'s tuple (the abstain reason is the last copy's
    when every copy failed, prefixed ``no_verifying_bundle:``)."""
    last = "no_bundle"
    for src in sources:
        v = vote(src, target, ledger, chain_id, registry, h=h, path_b_only=path_b_only, row_cap=row_cap)
        if v[0] != "abstain" or not str(v[1]).startswith(_COPY_REASONS):
            return v
        last = v[1]
    return ("abstain", "no_verifying_bundle:" + last)


__all__ = ["vote", "vote_any", "tested_rows"]
