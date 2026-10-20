"""cutline.verify.stages - stages B and C of the bundle reading order, and the verifier's Compile.

Stage A (A1-A4) is ``cutline.bundle.stage_a``; A5 is never an input. This module continues from a ``RawBundle`` that
passed stage A:

- ``Listing(g, coefs, header, r_mcu)``; ``listing_of(raw) -> Listing``: ContainerOK v2.1 (upload mode) of
  ``listing.jpg`` and its coefficients (A3 was checked by ``stage_a``; this re-reads the grammar the tests need).
- ``stage_b(raw, ledger=None, h=None, header=None) -> (rule, i, j) | None``: B1, the claim-wide static rules 1-9 in
  their search order (``static_rules.first_violation``) with the ledger's tier-T tables (``tier_t_map(h)``) and
  kernel families (``kfam(h)``); ``ledger`` None = genesis kfam and no tier-T table (a tier-T claim then raises
  ``E.CLAIM.RULE8``).
- ``stage_c(raw, program=None, ledger=None, tier="E") -> Openings``: C1-C3 in order, raising the first failure:
  C1 occupancy (``|occBits| = ceil(ny·nx/8)``, zero padding, ``H(occBits) = occHashes[l*]``: ``E.BUNDLE.PROOF``);
  C2 cells (opened set = the completeness set: ``E.VERIFYE.OPENING_SET``; ``MultiVerify`` against R_l*:
  ``E.MERKLE.MULTI`` / ``E.MERKLE.ROOT``); C3 fill (``bundle.read_fill``: ``E.BUNDLE.ZLIB`` / ``E.BUNDLE.ABI`` /
  ``E.BUNDLE.FILL`` / ``E.MERKLE.MULTI``; ``MultiVerify`` against core.rFill: ``E.MERKLE.*``; tier E only:
  ``mcus`` ⊇ every M_cert MCU with min alpha < 255: ``E.VERIFYE.FILL``). ``program`` defaults to the claim's enc(π);
  it must be well formed (stage B passed). Tier T needs no fill leaf.
- ``Openings``: ``program, lstar, hs, ws, ny, nx, occ, opened {(cy, cx): 1024 B}, fill (FillFile), fill_blocks
  {m: 3p² B}, n_mcu, bg, compiled, whole_fill, partial``, plus ``crop_planes() -> (C, alpha)`` (the level-l* crop
  planes, ``assemble_crop``), ``compile_with(lut=None, bg=None) -> Compiled`` (Compile again with another LUT
  or background, e.g. the identity LUT for the tone area or the constant-background test).
- ``VerifyResult`` (the return type of ``verify_e`` / ``verify_t``; ``.evidence(e=None) -> types.Evidence``),
  ``NOTHING_CERTIFIED``; ``res_facts(g)`` (resolution class and integers of Y0), ``claim_facts(raw, op, tier)``
  (mcert_count, sil_px, tested_blocks (tier E: 3 or 6 real blocks per M_cert MCU), geometry, cum_scale (the upload's
  (cw, ch, ow, oh)), partial, d_count, d_frac_ppm, d_gen, category, liveness), ``is_original(op)``,
  ``tone_facts(raw, op)`` (``verify.tone`` on the opened l* pixels; also ``heuristic_digests`` with
  ``tone`` = ``TONE_SHA256`` and ``pubseg`` None, which Evaluate fills from the U source).
- **Compile's background**: the opened fill block where one is opened and 0 elsewhere. A pixel whose MCU has no
  opened fill is either outside M_cert (never tested) or has alpha = 255 (the fill term is 0), so X is exact on every
  tested pixel.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from cutline.bundle import FillFile, RawBundle, read_fill
from cutline.commit.cells import CELL_BYTES, verify_cells
from cutline.commit.pyramid import level_dims
from cutline.compiler.compose import _compose, assemble_crop, opened_set, partial
from cutline.compiler.fill import mcu_grid, mcu_size, verify_fill
from cutline.compiler.program import decode_program
from cutline.compiler.static_rules import first_violation
from cutline.crypto import H
from cutline.enc import unpackbits_strict
from cutline.errors import BundleError, CutlineError, VerifyEError
from cutline.types import Compiled, Program


@dataclass(eq=False)
class Listing:
    g: object
    coefs: tuple
    header: bytes
    r_mcu: bytes


def listing_of(raw_or_bytes) -> Listing:
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    from cutline.jpeg.ry import r_y_parts
    b = raw_or_bytes.members["listing.jpg"] if isinstance(raw_or_bytes, RawBundle) else bytes(raw_or_bytes)
    g = container_ok_v2(b, True)
    co = read_coefs(b, g)
    header, r_mcu, _ = r_y_parts(g, co)
    return Listing(g, co, header, r_mcu)


# ------------------------------------------------------------------------------------------------ stage B
def stage_b(raw: RawBundle, ledger=None, h=None, header: bytes | None = None):
    if header is None:
        header = listing_of(raw).header
    tier_t = ledger.tier_t_map(h) if ledger is not None else None
    kfam = ledger.kfam(h) if ledger is not None else None
    return first_violation(raw.claim.prog, header, raw.proof.cut_pre, raw.claim.core, tier_t, kfam)


# ------------------------------------------------------------------------------------------------ stage C
@dataclass(eq=False)
class Openings:
    program: Program
    lstar: int
    hs: int
    ws: int
    ny: int
    nx: int
    occ: np.ndarray
    opened: dict
    fill: FillFile
    fill_blocks: dict
    n_mcu: int
    bg: np.ndarray
    compiled: Compiled
    whole_fill: bool
    partial: bool

    def crop_planes(self) -> tuple:
        return assemble_crop(self.opened, self.program.crop, self.hs, self.ws)

    def compile_with(self, lut=None, bg=None) -> Compiled:
        p = self.program if lut is None else dataclasses.replace(self.program, lut=bytes(lut))
        Cc, ac = self.crop_planes()
        return _compose(Cc, ac, p.out, p.place, p.canvas, p.kernel, p.lut, p.D, self.bg if bg is None else bg,
                        p.sampling)


def fill_background(canvas, sampling: int, fill_blocks: dict) -> np.ndarray:
    Hc, Wc = canvas
    p = mcu_size(sampling)
    nx = Wc // p
    bg = np.zeros((Hc, Wc, 3), np.uint8)
    for m, rgb in fill_blocks.items():
        my, mx = divmod(int(m), nx)
        bg[my * p:(my + 1) * p, mx * p:(mx + 1) * p] = np.frombuffer(bytes(rgb), np.uint8).reshape(p, p, 3)
    return bg


def stage_c(raw: RawBundle, program: Program | None = None, ledger=None, tier: str = "E") -> Openings:
    if program is None:
        program, ok = decode_program(raw.claim.prog)
        if not ok:
            raise ValueError("stage C needs a well-formed enc(pi) (stage B rule 5 sub-case 9 comes first)")
    cp = raw.proof.cut_pre
    H0, W0 = int.from_bytes(cp[0:2], "big"), int.from_bytes(cp[2:4], "big")
    ls = program.lstar
    hs, ws = level_dims(H0, W0, ls)
    ny, nx = -(-hs // 16), -(-ws // 16)
    # C1 occupancy
    ob = raw.proof.occ_bits
    try:
        occ = unpackbits_strict(ob, ny, nx, "E.BUNDLE.PROOF")
    except CutlineError as e:
        raise BundleError("E.BUNDLE.PROOF", f"occBits: {e.detail}") from None
    if H(ob) != raw.proof.occ_hashes[ls]:
        raise BundleError("E.BUNDLE.PROOF", "H(occBits) != occHashes[l*]")
    # C2 cells
    want = opened_set(occ, program, hs, ws)
    got = [(cy, cx) for cy, cx, _ in raw.cellidx]
    if got != want:
        raise VerifyEError("E.VERIFYE.OPENING_SET", f"{len(got)} cells opened, completeness set has {len(want)}")
    root = cp[5 + 32 * ls:5 + 32 * (ls + 1)]
    verify_cells(root, ls, ny, nx, raw.members["cellidx.bin"], raw.cells, raw.proof.cell_proof)
    opened = {(cy, cx): raw.cells[k * CELL_BYTES:(k + 1) * CELL_BYTES] for k, (cy, cx) in enumerate(got)}
    # C3 fill
    gy, gx = mcu_grid(*program.canvas, program.sampling)
    n_mcu = gy * gx
    f = read_fill(raw, n_mcu, program.sampling)
    verify_fill(n_mcu, f.mcus, f.rgb, f.fill_proof, raw.claim.core.r_fill, program.sampling)
    sz = 3 * mcu_size(program.sampling) ** 2
    blocks = {int(m): f.rgb[j * sz:(j + 1) * sz] for j, m in enumerate(f.mcus)}
    bg = fill_background(program.canvas, program.sampling, blocks)
    Cc, ac = assemble_crop(opened, program.crop, hs, ws)
    comp = _compose(Cc, ac, program.out, program.place, program.canvas, program.kernel, program.lut, program.D, bg,
                    program.sampling)
    if tier == "E":
        from cutline.compiler.fill import fill_open_set
        need = set(fill_open_set(comp.m_cert, comp.alpha, program.sampling))
        missing = need - set(blocks)
        if missing:
            raise VerifyEError("E.VERIFYE.FILL", f"{len(missing)} M_cert MCUs with min alpha < 255 lack a fill leaf")
    return Openings(program=program, lstar=ls, hs=hs, ws=ws, ny=ny, nx=nx, occ=occ, opened=opened, fill=f,
                    fill_blocks=blocks, n_mcu=n_mcu, bg=bg, compiled=comp, whole_fill=len(blocks) == n_mcu,
                    partial=partial(occ, program, hs, ws))


# ------------------------------------------------------------------------------------------------ the result
NOTHING_CERTIFIED = "nothing_certified"


@dataclass(eq=False)
class VerifyResult:
    """What VerifyE / VerifyT return (Evaluate steps 7-10 for the upload Y0).

    Exactly one of these holds: ``static_violation`` set (step 7, G1; ``verdict`` None); ``g0_reason`` =
    ``"openings"`` (step 8, with ``error`` = the E.* name) or ``"nothing_certified"`` (step 9); or ``verdict`` =
    ``"pass"`` / ``"fail"`` (step 10) with ``rows`` = the first failing TE (tier E) or TT (tier T) rows, ascending,
    at most ``blocktest.ROW_CAP`` (``rows_capped`` when the cap was reached).
    ``facts`` holds the Evidence fields this step computes; ``versions`` the ``classes`` and ``tier_t`` entries of
    the record's ``versions``; ``fraud`` the first failing unit (tier E; with its ProofIn for 4:4:4)."""
    tier: str
    verdict: str | None = None
    rows: list = dataclasses.field(default_factory=list)
    facts: dict = dataclasses.field(default_factory=dict)
    static_violation: tuple | None = None
    g0_reason: str | None = None
    error: str | None = None
    compiled: Compiled | None = None
    openings: Openings | None = None
    listing: Listing | None = None
    fraud: object = None
    versions: dict = dataclasses.field(default_factory=dict)
    rows_capped: bool = False

    @property
    def nothing_certified(self) -> bool:
        return self.g0_reason == NOTHING_CERTIFIED

    def evidence(self, e=None):
        """Copy ``facts`` (and ``static_violation`` / ``g0_reason`` / ``verdict``) into a ``types.Evidence``."""
        from cutline.types import Evidence
        e = Evidence() if e is None else e
        for k, v in self.facts.items():
            setattr(e, k, v)
        if self.static_violation is not None:
            r, i, j = self.static_violation
            e.static_violation = {"rule": r, "i": i, "j": j}
        e.g0_reason = self.g0_reason if self.g0_reason is not None else e.g0_reason
        e.verdict = self.verdict
        return e


def res_facts(g) -> dict:
    from cutline.verify.resclass import res_class
    c, ly, lc, my, mc = res_class(g.q3())
    return {"res_class": c, "lam_y_2p51": ly, "lam_c_2p51": lc, "ms_y_2p51": my, "ms_c_2p51": mc}


def claim_facts(raw: RawBundle, op: Openings, tier: str) -> dict:
    """The record fields that follow from the claim and Compile alone (both tiers)."""
    from cutline.commit.rec import decode_rec_strict
    p = op.program
    comp = op.compiled
    mc = len(comp.m_cert)
    nd = len(p.D)
    per_mcu = 3 if p.sampling == 0 else 6
    y0, x0, ch, cw = p.crop
    oh, ow = p.out
    return {"tier": tier, "mcert_count": mc, "sil_px": int(np.asarray(comp.sil).sum()),
            "tested_blocks": per_mcu * mc if tier == "E" else None,
            "geometry": [cw, ch, ow, oh], "cum_scale": [cw, ch, ow, oh], "partial": bool(op.partial),
            "d_count": nd, "d_frac_ppm": 0 if nd == 0 else (10 ** 6 * nd // mc if mc > 0 else 10 ** 6),
            "d_gen": bool((p.flags >> 1) & 1), "category": int(raw.claim.core.category),
            "liveness": int(decode_rec_strict(raw.claim.rec).liveness)}


def is_original(op: Openings) -> bool:
    """The listing is original: D = ∅ and every canvas pixel has alpha = 255."""
    return len(op.program.D) == 0 and bool((np.asarray(op.compiled.alpha) == 255).all())


def tone_facts(raw: RawBundle, op: Openings) -> dict:
    from cutline.verify.tone import TONE_SHA256, lut_delta, tone
    lut = op.program.lut
    Cc, ac = op.crop_planes()
    if lut_delta(lut) == 0:
        t = tone(lut, Cc, ac, category=raw.claim.core.category)
    else:
        from cutline.compiler.program import IDENTITY_LUT
        t = tone(lut, Cc, ac, compiled_nolut=op.compile_with(lut=IDENTITY_LUT), compiled=op.compiled,
                 category=raw.claim.core.category)
    return {"tone": t, "heuristic_digests": {"pubseg": None, "tone": TONE_SHA256}}


__all__ = ["Listing", "listing_of", "stage_b", "Openings", "fill_background", "stage_c", "NOTHING_CERTIFIED",
           "VerifyResult", "res_facts", "claim_facts", "is_original", "tone_facts"]
