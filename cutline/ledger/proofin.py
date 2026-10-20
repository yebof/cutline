"""cutline.ledger.proofin - ProofIn v2, its calldata, a builder, and the block-court reference model.

``court_reference`` is the Python twin of ``BlockCourt.proveBlockFailure`` / ``CourtHarness.evaluate`` (court steps
1-19, in order; the first failing check names the error), built on ``kernels.footprint`` / ``kernels.court_block``,
``phi.phi`` and ``dct.fdct_islow`` / ``quant_islow`` / ``d80`` / ``cell_fail`` (step 19). It is the counterpart of
``compiler.static_rules.prove_static_violation``; ``cutline.cli prove-block`` and ``types.FraudUnit`` use this module.

- ``Cell(cy, cx, salt, data, path)`` and ``ProofIn(nf, claimant, core, rec, cutPre, occHashes, occBits, prog, header,
  mcu, coefs, mcuPath, fill, fillPath, cells, comp, k)`` (ABI field order): ``.as_tuple()``, ``.from_tuple(t)``,
  ``.replace(**kw)``.
- ``PROOFIN_T`` (the ABI tuple type); ``encode_proofin_calldata(p) -> bytes`` = selector ``0x00aa6b4d`` ||
  abi.encode(ProofIn) (a single dynamic struct, so the encoding starts with the offset word 0x20);
  ``decode_proofin_calldata(b) -> ProofIn`` (canonical only; ``E.ABI.*``).
- ``CourtState(claims, cls)``: ``claims`` = {claimId: nf} (any status); ``cls`` = {encClass: (kind, eta20)} (default
  ``consts.CLASS_REGISTRY``, the genesis classes).
- ``court_reference(p, state, role=True) -> CourtResult(claim_id, X, w, c, fails)``, or raises ``CourtError``
  (``E.COURT.*``) / ``RoleError`` (``E.ROLE``). X is uint8[8, 8, 3]; w the islow output t (EXACT, scaled by 8) or D80_k
  (CELL); c the file's coefficient. A consistent block returns ``fails = False`` (the chain then reverts
  ``E.COURT.CONSISTENT``; ``outcome(result)`` gives that name).
- ``court_block_all(p, state) -> BlockResult(X, w[192], c[192], fail_mask0, fail_mask1)``: the twin of
  ``CourtHarness.evaluateBlock`` (steps 1-18 as above, then step 19 for every (comp, k); index 64 comp + k; mask1 bit
  = header entry 0).
- ``ProofParts`` (everything a prover holds for one claim) and ``build_proofin(parts, mcu, comp, k) -> ProofIn``: the
  footprint cells via ``kernels.footprint`` (occupied ones only, row-major, single Merkle paths in the level-l* tree),
  the R_mcu path, and the fill block with its rFill path only when min alpha < 255 (else ``b""`` and ``[]``).
  ``ProofParts.from_commitment(...)`` builds the parts from a ``types.Commitment`` and the listing data.
- ``parts_from_bundle(raw, listing_jpg=None) -> ProofParts``: the parts from a claim bundle alone, with single cell /
  fill paths recovered from the bundle's multiproofs (``sparse_levels``) and the MCU tree rebuilt from the listing
  (``listing_payloads``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .. import abi
from ..commit import merkle
from ..compiler import kernels
from ..compiler.program import decode_program
from ..consts import (CLAIM_CORE_T, CLASS_REGISTRY, SEL_PROVE_BLOCK, SIG_PROVE_BLOCK, TAG_CELL, TAG_FILL, TAG_LISTING_V2,
                      TAG_MCU_V2, TAG_OCC, TAG_PROG, TAG_ROOTS)
from ..crypto import H
from ..dct import cell_fail, d80, fdct_islow, quant_islow
from ..enc import u8, u16, u32
from ..errors import CourtError, CutlineError, MerkleError, error_for
from ..phi import phi
from ..types import ClaimCore, claim_id

CELL_T = "(uint16,uint16,bytes16,bytes,bytes32[])"
PROOFIN_T = ("(bytes32,address," + CLAIM_CORE_T + ",bytes,bytes,bytes32[],bytes,bytes,bytes,uint32,bytes,bytes32[],"
             "bytes,bytes32[]," + CELL_T + "[],uint8,uint8)")
assert SIG_PROVE_BLOCK == "proveBlockFailure(" + PROOFIN_T + ")"

REC_LEN, PROG_FIXED, HEADER_LEN, COEFS_LEN, CELL_LEN, FILL_LEN = 416, 794, 197, 384, 1024, 192
KIND_EXACT, KIND_CELL = 1, 2


# ================================================================================================ ProofIn
@dataclass(frozen=True)
class Cell:
    cy: int
    cx: int
    salt: bytes
    data: bytes
    path: tuple

    def as_tuple(self) -> tuple:
        return (self.cy, self.cx, self.salt, self.data, list(self.path))


@dataclass(frozen=True)
class ProofIn:
    nf: bytes
    claimant: bytes             # 20 bytes
    core: ClaimCore
    rec: bytes
    cutPre: bytes
    occHashes: tuple
    occBits: bytes
    prog: bytes
    header: bytes
    mcu: int
    coefs: bytes
    mcuPath: tuple
    fill: bytes
    fillPath: tuple
    cells: tuple                # of Cell
    comp: int
    k: int

    def as_tuple(self) -> tuple:
        return (self.nf, self.claimant, self.core.as_tuple(), self.rec, self.cutPre, list(self.occHashes), self.occBits,
                self.prog, self.header, self.mcu, self.coefs, list(self.mcuPath), self.fill, list(self.fillPath),
                [c.as_tuple() for c in self.cells], self.comp, self.k)

    @classmethod
    def from_tuple(cls, t) -> "ProofIn":
        (nf, claimant, core, rec, cut_pre, occ_hashes, occ_bits, prog, header, mcu, coefs, mcu_path, fill, fill_path,
         cells, comp, k) = t
        return cls(bytes(nf), bytes(claimant), ClaimCore.from_tuple(tuple(core)), bytes(rec), bytes(cut_pre),
                   tuple(bytes(h) for h in occ_hashes), bytes(occ_bits), bytes(prog), bytes(header), int(mcu),
                   bytes(coefs), tuple(bytes(h) for h in mcu_path), bytes(fill), tuple(bytes(h) for h in fill_path),
                   tuple(Cell(int(c[0]), int(c[1]), bytes(c[2]), bytes(c[3]), tuple(bytes(h) for h in c[4]))
                         for c in cells), int(comp), int(k))

    def replace(self, **kw) -> "ProofIn":
        return dataclasses.replace(self, **kw)

    def to_json(self) -> dict:
        """ethers-style JSON object (0x-hex bytes, ints) for Hardhat / ethers scripts."""
        hx = lambda b: "0x" + bytes(b).hex()  # noqa: E731
        c = self.core
        return {"nf": hx(self.nf), "claimant": hx(self.claimant),
                "core": {"recHash": hx(c.rec_hash), "rY": hx(c.r_y), "tHash": hx(c.t_hash), "rFill": hx(c.r_fill),
                         "encClass": c.enc_class, "tier": c.tier, "marketplace": c.marketplace, "category": c.category,
                         "listingRef": hx(c.listing_ref)},
                "rec": hx(self.rec), "cutPre": hx(self.cutPre), "occHashes": [hx(h) for h in self.occHashes],
                "occBits": hx(self.occBits), "prog": hx(self.prog), "header": hx(self.header), "mcu": self.mcu,
                "coefs": hx(self.coefs), "mcuPath": [hx(h) for h in self.mcuPath], "fill": hx(self.fill),
                "fillPath": [hx(h) for h in self.fillPath],
                "cells": [{"cy": x.cy, "cx": x.cx, "salt": hx(x.salt), "data": hx(x.data), "path": [hx(h) for h in x.path]}
                          for x in self.cells],
                "comp": self.comp, "k": self.k}


def encode_proofin_calldata(p: ProofIn) -> bytes:
    return SEL_PROVE_BLOCK + abi.encode_single(PROOFIN_T, p.as_tuple())


def decode_proofin_calldata(b: bytes) -> ProofIn:
    if bytes(b[:4]) != SEL_PROVE_BLOCK:
        raise error_for("E.ABI.NONCANONICAL", "selector")
    return ProofIn.from_tuple(abi.decode_single_canonical(PROOFIN_T, bytes(b[4:])))


# ================================================================================================ court reference
@dataclass
class CourtState:
    claims: dict = field(default_factory=dict)            # claimId -> nf (any status)
    cls: dict = field(default_factory=lambda: dict(CLASS_REGISTRY))   # encClass -> (kind, eta20)

    def seed(self, p: ProofIn) -> bytes:
        cid = claim_id(p.nf, p.claimant, p.core)
        self.claims[cid] = p.nf
        return cid


@dataclass(frozen=True)
class CourtResult:
    claim_id: bytes
    X: np.ndarray               # uint8[8, 8, 3]
    w: int
    c: int
    fails: bool

    @property
    def outcome(self) -> str:
        return "BlockFailure" if self.fails else "E.COURT.CONSISTENT"


@dataclass(frozen=True)
class BlockResult:
    claim_id: bytes
    X: np.ndarray
    w: tuple                    # 192, index 64 comp + k
    c: tuple
    fail_mask0: int
    fail_mask1: int


def _e(name: str, detail: str = ""):
    raise CourtError(name, detail)


def _be(b: bytes, o: int, n: int) -> int:
    return int.from_bytes(b[o:o + n], "big")


@dataclass
class _Ctx:
    claim_id: bytes = b""
    kind: int = 0
    eta20: int = 0
    program: object = None
    nmcu: int = 0
    bx: int = 0
    by: int = 0
    ny: int = 0
    nx: int = 0
    level_root: bytes = b""
    q: int = 0
    c: int = 0


def _prelude(p: ProofIn, state: CourtState) -> _Ctx:
    """Steps 1-13."""
    g = _Ctx()
    core = p.core
    # 1 claim, any status
    g.claim_id = claim_id(p.nf, p.claimant, core)
    if state.claims.get(g.claim_id) != p.nf:
        _e("E.COURT.CLAIM")
    # 2 tier E
    if core.tier != 1:
        _e("E.COURT.TIER")
    # 3 class
    kind, eta20 = state.cls.get(core.enc_class, (0, 0))
    if kind not in (KIND_EXACT, KIND_CELL):
        _e("E.COURT.CLASS")
    g.kind, g.eta20 = kind, eta20
    # 4 rec
    if len(p.rec) != REC_LEN or H(p.rec) != core.rec_hash:
        _e("E.COURT.REC")
    # 5 cutPre
    cp = p.cutPre
    if len(cp) < 5:
        _e("E.COURT.ROOTS")
    lmax = cp[4]
    if len(cp) != 5 + 32 * (lmax + 1) or H(TAG_ROOTS + cp) != p.rec[32:64]:
        _e("E.COURT.ROOTS")
    # 6 occHashes
    if len(p.occHashes) != lmax + 1 or H(TAG_OCC + b"".join(p.occHashes)) != p.rec[64:96]:
        _e("E.COURT.OCC", "occ root")
    # 7 program
    if H(TAG_PROG + p.prog) != core.t_hash:
        _e("E.COURT.PROGRAM")
    # 8 static preconditions, in order
    t = p.prog
    if len(t) < PROG_FIXED or len(t) != PROG_FIXED + 4 * _be(t, 792, 2):
        _e("E.COURT.STATIC", "rule 5(9)")
    if t[23] & 0xFD:
        _e("E.COURT.STATIC", "rule 7")
    if t[21] > 1 or t[22] > 1:
        _e("E.COURT.STATIC", "rules 5(0), 5(1)")
    l, y0, x0, ch, cw, oh, ow, oy, ox, Hc, Wc = (t[0],) + tuple(_be(t, 1 + 2 * i, 2) for i in range(10))
    if not (1 <= oh <= ch < 2 * oh and 1 <= ow <= cw < 2 * ow):
        _e("E.COURT.STATIC", "rule 3")
    if Hc % 16 or Wc % 16:
        _e("E.COURT.STATIC", "rule 5(3)")
    if oy + oh > Hc or ox + ow > Wc:
        _e("E.COURT.STATIC", "rule 5(5)")
    if l > lmax:
        _e("E.COURT.STATIC", "rule 5(7)")
    hl = -(-_be(cp, 0, 2) // (1 << l))
    wl = -(-_be(cp, 2, 2) // (1 << l))
    if y0 + ch > hl or x0 + cw > wl:
        _e("E.COURT.STATIC", "rule 5(6)")
    g.program, ok = decode_program(t)
    assert ok
    g.ny, g.nx = -(-hl // 16), -(-wl // 16)
    # 9 header
    h = p.header
    if len(h) != HEADER_LEN:
        _e("E.COURT.HEADER")
    if h[4] != t[22] or (_be(h, 0, 2), _be(h, 2, 2)) != (Wc, Hc):
        _e("E.COURT.STATIC", "rules 5(2), 5(4)")
    if h[4] != 0:
        _e("E.COURT.SAMPLING")
    # 10 index, q
    bpr = Wc // 8
    g.nmcu = bpr * (Hc // 8)
    if not (p.mcu < g.nmcu and p.comp < 3 and p.k < 64):
        _e("E.COURT.INDEX")
    g.q = h[5 + 64 * p.comp + p.k]
    if g.q < 1:
        _e("E.COURT.ZERO_Q")
    g.bx, g.by = p.mcu % bpr, p.mcu // bpr
    # 11 D membership (binary search over D as stored)
    lo, hi = 0, _be(t, 792, 2)
    while lo < hi:
        mid = (lo + hi) // 2
        v = _be(t, PROG_FIXED + 4 * mid, 4)
        if v == p.mcu:
            _e("E.COURT.DECLARED")
        if v < p.mcu:
            lo = mid + 1
        else:
            hi = mid
    # 12 coefs, R_mcu, R_Y
    if len(p.coefs) != COEFS_LEN:
        _e("E.COURT.COEFS")
    leaf = H(TAG_MCU_V2 + u32(p.mcu) + p.coefs)
    try:
        r_mcu = merkle.root_from_path(leaf, p.mcu, g.nmcu, list(p.mcuPath))
    except MerkleError:
        _e("E.COURT.MCU_PATH")
    if H(TAG_LISTING_V2 + h + r_mcu) != core.r_y:
        _e("E.COURT.RY")
    g.c = int.from_bytes(p.coefs[128 * p.comp + 2 * p.k:128 * p.comp + 2 * p.k + 2], "big", signed=True)
    # 13 occupancy of level l*
    g.level_root = cp[5 + 32 * l:37 + 32 * l]
    nbits = g.ny * g.nx
    ob = p.occBits
    if len(ob) != (nbits + 7) // 8 or (nbits % 8 and ob[-1] & ((1 << (8 - nbits % 8)) - 1)) \
            or H(ob) != p.occHashes[l]:
        _e("E.COURT.OCC", "occ")
    return g


def _occupied(ob: bytes, idx: int) -> bool:
    return (ob[idx >> 3] >> (7 - (idx & 7))) & 1 == 1


def _block_alpha(program, fp, opened) -> np.ndarray:
    """a[y][x] of court step 16: ResizeExact of alpha over the footprint inside the placed rectangle, 0 outside."""
    y0, x0 = program.crop[0], program.crop[1]
    a = np.zeros((8, 8), np.int64)

    def src(r, c):
        Y, X_ = y0 + r, x0 + c
        d = opened.get((Y // 16, X_ // 16))
        return 0 if d is None else d[((Y % 16) * 16 + X_ % 16) * 4 + 3]

    for dy, ty in enumerate(fp["ty"]):
        for dx, tx in enumerate(fp["tx"]):
            if ty is None or tx is None:
                continue
            (iy, wy), (ix, wx) = ty, tx
            u = sum(wyv * sum(wxv * src(iy + s, ix + t) for t, wxv in enumerate(wx)) for s, wyv in enumerate(wy))
            D = sum(wy) * sum(wx)
            a[dy, dx] = min((2 * u + D) // (2 * D), 255)
    return a


def _block(p: ProofIn, g: _Ctx) -> np.ndarray:
    """Steps 14-18; returns X uint8[8, 8, 3]."""
    prog = g.program
    # 14 taps and footprint
    fp = kernels.footprint(prog, g.bx, g.by)
    if fp is None:
        _e("E.COURT.NOT_CERTIFIED", "block outside the placed rectangle")
    # 15 exactly the occupied cells of the rectangle, row-major
    l = prog.lstar
    n = g.ny * g.nx
    opened = {}
    j = 0
    for cy, cx in fp["cells"]:
        idx = cy * g.nx + cx
        if not _occupied(p.occBits, idx):
            continue
        if j >= len(p.cells):
            _e("E.COURT.CELLS", "missing cell")
        c = p.cells[j]
        j += 1
        if (c.cy, c.cx) != (cy, cx) or len(c.data) != CELL_LEN:
            _e("E.COURT.CELLS", "cell idx")
        leaf = H(TAG_CELL + u8(l) + u16(c.cy) + u16(c.cx) + c.salt + c.data)
        try:
            r = merkle.root_from_path(leaf, idx, n, list(c.path))
        except MerkleError:
            _e("E.COURT.CELL_PATH")
        if r != g.level_root:
            _e("E.COURT.CELL_PATH")
        opened[(cy, cx)] = c.data
    if j != len(p.cells):
        _e("E.COURT.CELLS", "extra cell")
    # 16 alpha
    a = _block_alpha(prog, fp, opened)
    if a.max() == 0:
        _e("E.COURT.NOT_CERTIFIED")
    # 17 fill iff min alpha < 255
    if a.min() < 255:
        if len(p.fill) != FILL_LEN:
            _e("E.COURT.FILL", "fill length")
        try:
            r = merkle.root_from_path(H(TAG_FILL + u32(p.mcu) + p.fill), p.mcu, g.nmcu, list(p.fillPath))
        except MerkleError:
            _e("E.COURT.FILL", "fill path")
        if r != p.core.r_fill:
            _e("E.COURT.FILL", "fill root")
        bg = np.frombuffer(p.fill, np.uint8).reshape(8, 8, 3)
    else:
        if p.fill or p.fillPath:
            _e("E.COURT.FILL", "fill sent although min alpha = 255")
        bg = np.zeros((8, 8, 3), np.uint8)     # weight 0 everywhere (every a = 255)
    # 18 X
    X, certified = kernels.court_block(prog, g.bx, g.by, opened, bg)
    assert certified
    return X


def _plane(X: np.ndarray, comp: int) -> np.ndarray:
    return np.asarray(phi(X)[comp], np.int64) - 128


def court_reference(p: ProofIn, state: CourtState, role: bool = True) -> CourtResult:
    """Court steps 1-19 (the role check first). Raises CourtError / RoleError; returns the verdict otherwise."""
    if not role:
        raise error_for("E.ROLE", "caller is neither WATCHER nor validator")
    g = _prelude(p, state)
    X = _block(p, g)
    P = _plane(X, p.comp)
    if g.kind == KIND_EXACT:
        w = int(fdct_islow(P)[p.k // 8, p.k % 8])
        fails = int(quant_islow(np.array([[w]]), np.array([[g.q]]))[0, 0]) != g.c
    else:
        w = int(d80(P)[p.k])
        fails = cell_fail(g.c, g.q, w, g.eta20)
    return CourtResult(g.claim_id, X, w, g.c, bool(fails))


def court_block_all(p: ProofIn, state: CourtState) -> BlockResult:
    """Twin of CourtHarness.evaluateBlock."""
    g = _prelude(p, state)
    X = _block(p, g)
    ws, cs, m0, m1 = [], [], 0, 0
    for comp in range(3):
        P = _plane(X, comp)
        t = [int(v) for v in fdct_islow(P).reshape(64)] if g.kind == KIND_EXACT else [int(v) for v in d80(P)]
        for k in range(64):
            i = 64 * comp + k
            q = p.header[5 + i]
            c = int.from_bytes(p.coefs[2 * i:2 * i + 2], "big", signed=True)
            ws.append(t[k])
            cs.append(c)
            if q == 0:
                m1 |= 1 << i
            elif g.kind == KIND_EXACT:
                if int(quant_islow(np.array([[t[k]]]), np.array([[q]]))[0, 0]) != c:
                    m0 |= 1 << i
            elif cell_fail(c, q, t[k], g.eta20):
                m0 |= 1 << i
    return BlockResult(g.claim_id, X, tuple(ws), tuple(cs), m0, m1)


def outcome_of(p: ProofIn, state: CourtState, role: bool = True) -> str:
    """"BlockFailure", "E.COURT.CONSISTENT" or the revert name (the on-chain outcome of proveBlockFailure)."""
    try:
        return court_reference(p, state, role).outcome
    except CutlineError as e:
        return e.name


# ================================================================================================ builder
@dataclass
class ProofParts:
    """What the prover holds for one claim. ``cell(cy, cx) -> (salt16, data1024)`` opens a level-l* cell;
    ``level_leaves`` are the level-l* cell leaves (row-major); ``coefs[mcu]`` the listing's 384-byte MCU leaf payloads
    (all canvas MCUs); ``fill[mcu]`` the 192-byte fill blocks (all canvas MCUs, or None if the claim has no fill
    layer, rFill then unused)."""
    nf: bytes
    claimant: bytes
    core: ClaimCore
    rec: bytes
    cut_pre: bytes
    occ_hashes: tuple
    occ_bits: bytes               # packbits(O_l*)
    prog: bytes
    header: bytes
    coefs: list
    level_leaves: list
    cell: object                  # callable (cy, cx) -> (salt16, data1024)
    fill: list | None = None

    @cached_property
    def program(self):
        p, ok = decode_program(self.prog)
        if not ok:
            raise ValueError("enc(pi) length is not 794 + 4 nD")
        return p

    @cached_property
    def _mcu_levels(self):
        return merkle.merkle_levels([H(TAG_MCU_V2 + u32(i) + c) for i, c in enumerate(self.coefs)])

    @cached_property
    def _fill_levels(self):
        return merkle.merkle_levels([H(TAG_FILL + u32(i) + f) for i, f in enumerate(self.fill)])

    @cached_property
    def _cell_levels(self):
        return merkle.merkle_levels(self.level_leaves)

    @property
    def r_mcu(self) -> bytes:
        return self._mcu_levels[-1][0]

    @property
    def r_fill(self) -> bytes:
        return self._fill_levels[-1][0]

    @property
    def level_root(self) -> bytes:
        return self._cell_levels[-1][0]

    @classmethod
    def from_commitment(cls, cm, k_cap: bytes, *, nf, claimant, core, rec, prog, header, coefs, fill=None):
        """Parts from a types.Commitment (cells.commitment) and the listing data."""
        from ..commit import cells as cells_mod
        p, ok = decode_program(prog)
        if not ok:
            raise ValueError("enc(pi) length is not 794 + 4 nD")
        l = p.lstar
        C, al, leaves = cm.levels[l]
        return cls(nf=nf, claimant=claimant, core=core, rec=rec, cut_pre=cm.cut_pre, occ_hashes=tuple(cm.occ_hashes),
                   occ_bits=cm.occ_bits[l], prog=prog, header=header, coefs=list(coefs), level_leaves=list(leaves),
                   cell=lambda cy, cx: (cells_mod.salt(k_cap, l, cy, cx), cells_mod.cell_bytes(C, al, cy, cx)),
                   fill=None if fill is None else list(fill))


def build_proofin(parts: ProofParts, mcu: int, comp: int = 0, k: int = 0) -> ProofIn:
    """An honest ProofIn for (mcu, comp, k): the footprint's occupied cells (row-major), the MCU path, and the fill
    block with its path iff min alpha < 255 (else "" and [])."""
    prog = parts.program
    Hc, Wc = prog.canvas
    bpr = Wc // 8
    bx, by = mcu % bpr, mcu // bpr
    hl = -(-int.from_bytes(parts.cut_pre[0:2], "big") // (1 << prog.lstar))
    wl = -(-int.from_bytes(parts.cut_pre[2:4], "big") // (1 << prog.lstar))
    ny, nx = -(-hl // 16), -(-wl // 16)
    fp = kernels.footprint(prog, bx, by)
    cells, opened = [], {}
    if fp is not None:
        for cy, cx in fp["cells"]:
            idx = cy * nx + cx
            if not _occupied(parts.occ_bits, idx):
                continue
            salt, data = parts.cell(cy, cx)
            cells.append(Cell(cy, cx, bytes(salt), bytes(data), tuple(merkle.merkle_path(parts._cell_levels, idx))))
            opened[(cy, cx)] = data
    need_fill = fp is not None and _block_alpha(prog, fp, opened).min() < 255
    if need_fill and parts.fill is not None:
        fill, fill_path = bytes(parts.fill[mcu]), tuple(merkle.merkle_path(parts._fill_levels, mcu))
    else:
        fill, fill_path = b"", ()
    return ProofIn(nf=parts.nf, claimant=parts.claimant, core=parts.core, rec=parts.rec, cutPre=parts.cut_pre,
                   occHashes=tuple(parts.occ_hashes), occBits=parts.occ_bits, prog=parts.prog, header=parts.header,
                   mcu=mcu, coefs=bytes(parts.coefs[mcu]), mcuPath=tuple(merkle.merkle_path(parts._mcu_levels, mcu)),
                   fill=fill, fillPath=fill_path, cells=tuple(cells), comp=comp, k=k)


# ================================================================================================ from a bundle
class _SparseLevel(dict):
    """One level of a Merkle tree known only on the nodes a multiproof reconstruction touches; ``len`` is the level
    width, so ``merkle.merkle_path`` works on the known positions (KeyError elsewhere)."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def __len__(self):
        return self.width


def sparse_levels(n: int, I, leaves, proof) -> list:
    """The levels of an n-leaf tree reconstructed from a multiproof (every node MultiRoot computes or reads).
    Single paths (``merkle.merkle_path``) of every opened index I follow; raises MerkleError like ``multiroot``."""
    merkle.multiroot(n, list(I), list(leaves), list(proof))          # validates the proof (order, consumption)
    known, m, k = dict(zip(I, leaves)), n, 0
    out = []
    while True:
        lv = _SparseLevel(m)
        lv.update(known)
        out.append(lv)
        if m <= 1:
            break
        nxt = {}
        for i in sorted(known):
            p = i >> 1
            if p in nxt:
                continue
            s = i ^ 1
            if s >= m:
                sib = known[i]
            elif s in known:
                sib = known[s]
            else:
                sib = proof[k]
                k += 1
                lv[s] = sib
            nxt[p] = merkle.node(known[i], sib) if i % 2 == 0 else merkle.node(sib, known[i])
        known, m = nxt, (m + 1) >> 1
    return out


def listing_payloads(listing_jpg: bytes) -> tuple:
    """(header197, [384-byte MCU leaf payload per canvas MCU]) of a 4:4:4 upload (ContainerOK v2.1, read_coefs)."""
    from ..jpeg.coefs import read_coefs
    from ..jpeg.grammar import container_ok_v2
    from ..jpeg.ry import enc_block, mcu_grid, mcu_real_blocks, ry_header
    g = container_ok_v2(listing_jpg, True)
    if g.sampling != 0:
        raise ValueError("the block court takes 4:4:4 listings only")
    co = read_coefs(listing_jpg, g)
    my, mx = mcu_grid(g)
    return ry_header(g), [b"".join(enc_block(b) for _, _, b in mcu_real_blocks(g, co, m)) for m in range(my * mx)]


def parts_from_bundle(raw, listing_jpg: bytes | None = None) -> ProofParts:
    """ProofParts from a claim bundle alone (``raw`` = ``cutline.bundle.RawBundle`` after stage A): what a
    verifier holds is enough to prove any certified block. Cell and fill paths are single paths recovered from the
    bundle's level-l* cell multiproof and fill multiproof (``sparse_levels``); the MCU tree is rebuilt from the
    listing's coefficients. ``cell`` and ``fill`` answer only for opened cells / opened fill MCUs (a block that needs
    anything else is not provable from this bundle: KeyError)."""
    from ..bundle import read_fill
    from ..commit.cells import cell_leaf
    from ..compiler.fill import fill_leaf
    jpg = listing_jpg if listing_jpg is not None else raw.members["listing.jpg"]
    claim, proof = raw.claim, raw.proof
    prog, ok = decode_program(claim.prog)
    if not ok:
        raise ValueError("enc(pi) length is not 794 + 4 nD")
    header, payloads = listing_payloads(jpg)
    l = prog.lstar
    hl = -(-int.from_bytes(proof.cut_pre[0:2], "big") // (1 << l))
    wl = -(-int.from_bytes(proof.cut_pre[2:4], "big") // (1 << l))
    ny, nx = -(-hl // 16), -(-wl // 16)
    opened = {(cy, cx): (salt, raw.cells[j * CELL_LEN:(j + 1) * CELL_LEN]) for j, (cy, cx, salt) in enumerate(raw.cellidx)}
    I = [cy * nx + cx for cy, cx, _ in raw.cellidx]
    leaves = [cell_leaf(l, cy, cx, s, opened[(cy, cx)][1]) for cy, cx, s in raw.cellidx]
    Hc, Wc = prog.canvas
    nmcu = (Hc // 8) * (Wc // 8)
    f = read_fill(raw, nmcu, 0)
    fill = {m: f.rgb[j * FILL_LEN:(j + 1) * FILL_LEN] for j, m in enumerate(f.mcus)}
    parts = ProofParts(nf=raw.nf, claimant=bytes(claim.claimant), core=claim.core, rec=bytes(claim.rec),
                       cut_pre=bytes(proof.cut_pre), occ_hashes=tuple(proof.occ_hashes), occ_bits=bytes(proof.occ_bits),
                       prog=bytes(claim.prog), header=header, coefs=payloads, level_leaves=[],
                       cell=lambda cy, cx: opened[(cy, cx)], fill=fill)
    if I:
        parts.__dict__["_cell_levels"] = sparse_levels(ny * nx, I, leaves, proof.cell_proof)
        if parts.level_root != proof.cut_pre[5 + 32 * l:37 + 32 * l]:
            raise MerkleError("E.MERKLE.ROOT", "cell multiproof root != R_l*")
    if f.mcus:
        parts.__dict__["_fill_levels"] = sparse_levels(nmcu, f.mcus, [fill_leaf(m, fill[m]) for m in f.mcus],
                                                       f.fill_proof)
        if parts.r_fill != claim.core.r_fill:
            raise MerkleError("E.MERKLE.ROOT", "fill multiproof root != rFill")
    return parts


def honest_coefs(X: np.ndarray, header: bytes) -> bytes:
    """The E0 (islow) coefficients of one 4:4:4 block X under the header's tables: 3 x 64 i16 big-endian, natural
    order (the MCU leaf payload)."""
    out = b""
    for comp in range(3):
        t = fdct_islow(_plane(np.asarray(X, np.uint8).reshape(8, 8, 3), comp))
        q = np.frombuffer(header[5 + 64 * comp:69 + 64 * comp], np.uint8).astype(np.int64).reshape(8, 8)
        c = quant_islow(t, q).reshape(64)
        out += b"".join(int(v).to_bytes(2, "big", signed=True) for v in c)
    return out
