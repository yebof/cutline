"""cutline.verify.fraud - the fraud-proof unit of a failing tier-E block.

``cutline.ledger.proofin`` owns the ProofIn v2 encoder, the builder and the court reference model; this module only
connects VerifyE's first failing TE row to it.

- ``fraud_unit(raw, row, listing=None, prove=None) -> types.FraudUnit``: ``row`` = a TE row
  ``(mcu, comp, blk, k, c, w)``. For a 4:4:4 claim (the block court's domain) ``proof`` is
  ``proofin.build_proofin(proofin.parts_from_bundle(raw), mcu, comp, k)`` built from the bundle alone (the footprint
  cells from the cell multiproof, the fill block only when min alpha < 255, the MCU path from the listing);
  ``prove=False`` skips it. 4:2:0 claims are challenged through the sidecar vote (path b): ``proof`` stays None.
- ``prove_block(raw, mcu, comp, k) -> ProofIn`` and ``calldata(raw, mcu, comp, k) -> bytes`` (selector
  ``0x00aa6b4d`` ‖ abi.encode(ProofIn)).
- ``court_check(proof, raw) -> str``: ``proofin.outcome_of`` under a court state with the claim registered and the
  genesis class registry: ``"BlockFailure"`` for a provable unit, else the E.COURT.* name the chain would revert with
  (``"E.COURT.CONSISTENT"`` for a block that passes).
"""
from __future__ import annotations

from cutline.types import FraudUnit, claim_id


def prove_block(raw, mcu: int, comp: int = 0, k: int = 0):
    from cutline.ledger.proofin import build_proofin, parts_from_bundle
    return build_proofin(parts_from_bundle(raw), int(mcu), int(comp), int(k))


def calldata(raw, mcu: int, comp: int = 0, k: int = 0) -> bytes:
    from cutline.ledger.proofin import encode_proofin_calldata
    return encode_proofin_calldata(prove_block(raw, mcu, comp, k))


def fraud_unit(raw, row, listing=None, prove: bool | None = None) -> FraudUnit:
    mcu, comp, blk, k = (int(v) for v in row[:4])
    sampling = listing.g.sampling if listing is not None else None
    if sampling is None:
        from cutline.verify.stages import listing_of
        sampling = listing_of(raw).g.sampling
    proof = None
    if (prove is None or prove) and sampling == 0:
        proof = prove_block(raw, mcu, comp, k)
    return FraudUnit(mcu=mcu, comp=comp, k=k, blk=blk, row=tuple(int(v) for v in row), proof=proof)


def court_check(proof, raw=None) -> str:
    from cutline.ledger.proofin import CourtState, outcome_of
    cid = claim_id(proof.nf, proof.claimant, proof.core)
    return outcome_of(proof, CourtState(claims={cid: proof.nf}))


__all__ = ["fraud_unit", "prove_block", "calldata", "court_check"]
