"""cutline.compiler.static_rules - claim-wide static rules 1-9, `proveStaticViolation` and its calldata.

Rule 8 is the v2 tier-T rule; rule 9 holds the size bounds: 9(0) master H0 * W0 <= 2^28, 9(1) crop ch * cw <= 2^25.
``prove_static_violation`` is the reference model of the on-chain check order (steps 1-11); ``first_violation`` is
the off-chain search order.

Rows: ``ROWS[(rule, i)] = Row(prog, header, rec, jmax)``; ``jmax`` = the constant j range (j <= jmax), or ``None`` for
rows 4 and 5(8), whose range depends on nD (checked at step 9).

- ``StaticIn`` (dataclass: nf, claimant (20 B), core (ClaimCore), rule, i, j, prog, header, rMcu, rec, cutPre):
  ``.as_tuple()``, ``StaticIn.from_tuple(t)``.
- ``encode_static_calldata(s) -> bytes``: selector ``0x8d02fe1f`` || abi.encode(StaticIn) (single struct, leading 0x20);
  ``decode_static_calldata(b) -> StaticIn`` (canonical; ``E.ABI.NONCANONICAL``).
- ``LedgerState(claims, tier_t, kfam)``: ``claims`` = registered claimIds (any status) or (nf, claimant, core)
  triples; ``tier_t[id] = TierT(kernel_family, sampling, codec, successor_height, set_height)``; ``kfam[f] = exact``
  (default genesis ``consts.KFAM_GENESIS``). ``.to_json()`` gives the ledger-state JSON object.
- ``violated(rule, i, j, P, prog, header, cut_pre, core, tier_t, kfam) -> bool``: the row predicate (O(1)); ``P`` =
  ``program.fields(prog)`` (None for 5(9)).
- ``prove_static_violation(s, state, role=True) -> "StaticViolation"`` or raises (``E.ROLE``, ``E.STATIC.*``) in the
  exact on-chain order.
- ``first_violation(prog, header, cut_pre, core, tier_t=None, kfam=None) -> (rule, i, j) | None``: the search
  order (5(9); rules 1-4; 5(0)-5(8); 6, 7, 8; 9; (i, j) ascending). Rows whose preimage is ``None`` are skipped.
  Raises ``E.STATIC.RY`` / ``E.STATIC.REC`` for a header / cutPre of the wrong length, and ``E.CLAIM.RULE8`` for a
  tier-T core whose table is absent from ``tier_t``.
- ``preimages(rule, i, prog, header, r_mcu, rec, cut_pre) -> (prog, header, rMcu, rec, cutPre)``: exactly the
  preimages the row reads (the others absent).
- ``rule8_at_registration(core, tier_t, height) -> None | "E.CLAIM.RULE8"`` (8a, 8b).
- ``aspect_ok(cw, ch, ow, oh) -> (ok, margin)``: rule 6 on Python big integers (also used by the hop chain rule);
  ``margin = max(10^6 cw oh - 1020202 ch ow, 10^6 ch ow - 1020202 cw oh)``, ok iff margin <= 0.
- ``tiert_record(...) -> bytes55``, ``parse_tiert_record(b) -> dict`` and ``TierT``: re-exported from
  ``cutline.registries`` (``tiert_record`` / ``parse_tiert_record`` / ``TierTEntry``; one tier-T record type).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import abi
from ..consts import (ASPECT_A, ASPECT_B, CLAIM_CORE_T, CROP_MAX_PIXELS, KFAM_GENESIS, MASTER_MAX_PIXELS, Q_POL,
                      SEL_PROVE_STATIC, SIG_PROVE_STATIC, TAG_LISTING_V2, TAG_ROOTS)
from ..crypto import H
from ..enc import u8, u16, u32
from ..errors import StaticError, error_for
from ..registries import TierTEntry, parse_tiert_record, tiert_record  # noqa: F401  (re-exports)
from ..types import ClaimCore, claim_id
from . import program as prog_mod

STATIC_T = "(bytes32,address," + CLAIM_CORE_T + ",uint8,uint8,uint16,bytes,bytes,bytes32,bytes,bytes)"
assert SIG_PROVE_STATIC == "proveStaticViolation(" + STATIC_T + ")"
ZERO32 = bytes(32)


@dataclass(frozen=True)
class Row:
    prog: bool
    header: bool
    rec: bool
    jmax: int | None           # constant j range [0, jmax]; None: rows 4 and 5(8)


ROWS: dict = {}
for _c in range(3):
    ROWS[(1, _c)] = Row(True, False, False, 254)
    ROWS[(2, _c)] = Row(False, True, False, 63)
ROWS[(3, 0)] = ROWS[(3, 1)] = Row(True, False, False, 0)
ROWS[(4, 0)] = Row(True, False, False, None)
for _i in range(10):
    ROWS[(5, _i)] = Row(True, _i in (2, 4), _i in (6, 7), None if _i == 8 else 0)
ROWS[(6, 0)] = ROWS[(6, 1)] = Row(True, False, False, 0)
for _b in (0, 2, 3, 4, 5, 6, 7):
    ROWS[(7, _b)] = Row(True, False, False, 0)
ROWS[(8, 0)] = Row(True, False, False, 0)
ROWS[(8, 1)] = Row(False, True, False, 0)
ROWS[(8, 2)] = Row(False, False, False, 0)
ROWS[(9, 0)] = Row(False, False, True, 0)        # 9(0): H0 * W0 > 2^28, read from cutPre (no prog)
ROWS[(9, 1)] = Row(True, False, False, 0)        # 9(1): ch * cw > 2^25, read from enc(pi)

# off-chain search order
SEARCH_ORDER = ([(5, 9)] + [(1, c) for c in range(3)] + [(2, c) for c in range(3)] + [(3, 0), (3, 1), (4, 0)]
                + [(5, i) for i in range(9)] + [(6, 0), (6, 1)] + [(7, b) for b in (0, 2, 3, 4, 5, 6, 7)]
                + [(8, 0), (8, 1), (8, 2)] + [(9, 0), (9, 1)])
assert sorted(SEARCH_ORDER) == sorted(ROWS)


# ------------------------------------------------------------------------------------------------ ledger state
# TierT is registries.TierTEntry (same fields and order).
TierT = TierTEntry


@dataclass
class LedgerState:
    claims: set = field(default_factory=set)           # claimIds (bytes32) of registered claims, any status
    tier_t: dict = field(default_factory=dict)         # id -> TierT
    kfam: dict = field(default_factory=lambda: dict(KFAM_GENESIS))
    claim_triples: list = field(default_factory=list)  # (nf, claimant, ClaimCore) for the state JSON

    def register(self, nf: bytes, claimant: bytes, core: ClaimCore) -> bytes:
        cid = claim_id(nf, claimant, core)
        self.claims.add(cid)
        self.claim_triples.append((nf, claimant, core))
        return cid

    def to_json(self) -> dict:
        return {"claims": [[nf, cl, list(core.as_tuple())] for nf, cl, core in self.claim_triples],
                "tierT": [{"id": k, "kernelFamily": t.kernel_family, "sampling": t.sampling, "codec": t.codec,
                           "record": t.record, "setHeight": t.set_height, "successorHeight": t.successor_height}
                          for k, t in sorted(self.tier_t.items())],
                "kfam": [{"f": f, "exact": e} for f, e in sorted(self.kfam.items())]}


# ------------------------------------------------------------------------------------------------ StaticIn
@dataclass(frozen=True)
class StaticIn:
    nf: bytes
    claimant: bytes
    core: ClaimCore
    rule: int
    i: int
    j: int
    prog: bytes = b""
    header: bytes = b""
    rMcu: bytes = ZERO32
    rec: bytes = b""
    cutPre: bytes = b""

    def as_tuple(self) -> tuple:
        return (self.nf, self.claimant, self.core.as_tuple(), self.rule, self.i, self.j, self.prog, self.header,
                self.rMcu, self.rec, self.cutPre)

    @classmethod
    def from_tuple(cls, t) -> "StaticIn":
        nf, cl, core, rule, i, j, prog, hdr, rm, rec, cp = t
        return cls(nf, cl, ClaimCore.from_tuple(core), rule, i, j, prog, hdr, rm, rec, cp)


def encode_static_calldata(s: StaticIn) -> bytes:
    return SEL_PROVE_STATIC + abi.encode_single(STATIC_T, s.as_tuple())


def decode_static_calldata(b: bytes) -> StaticIn:
    if b[:4] != SEL_PROVE_STATIC:
        raise error_for("E.ABI.NONCANONICAL", "selector")
    return StaticIn.from_tuple(abi.decode_single_canonical(STATIC_T, b[4:]))


# ------------------------------------------------------------------------------------------------ predicates
def _u16(b, o):
    return int.from_bytes(b[o:o + 2], "big")


def exact_of(f: int, kfam: dict | None) -> int | None:
    return (KFAM_GENESIS if kfam is None else kfam).get(f)


def aspect_ok(cw: int, ch: int, ow: int, oh: int) -> tuple:
    d0 = ASPECT_A * cw * oh - ASPECT_B * ch * ow
    d1 = ASPECT_A * ch * ow - ASPECT_B * cw * oh
    m = max(d0, d1)
    return m <= 0, m


def _tier_t_get(tier_t, enc_class):
    t = (tier_t or {}).get(enc_class)
    if isinstance(t, dict):
        t = TierT(t["kernelFamily"], t["sampling"], t.get("codec", 0), t.get("successorHeight", 0),
                  t.get("setHeight", 0))
    return t


def violated(rule, i, j, P, prog, header, cut_pre, core: ClaimCore, tier_t=None, kfam=None) -> bool:
    if rule == 1:
        return P["LUT"][i][j] > P["LUT"][i][j + 1]
    if rule == 2:
        q = header[5 + 64 * i + j]
        return q == 0 or q > Q_POL[64 * i + j]
    if rule == 3:
        if i == 0:
            return not (1 <= P["oh"] <= P["ch"] < 2 * P["oh"])
        return not (1 <= P["ow"] <= P["cw"] < 2 * P["ow"])
    if rule == 4:
        return P["D"][j] >= P["D"][j + 1]
    if rule == 5:
        if i == 9:
            return not prog_mod.prog_len_ok(prog)
        p = 8 if P["sampling"] == 0 else 16
        if i == 0:
            return P["kernel"] > 1
        if i == 1:
            return P["sampling"] > 1
        if i == 2:
            return P["sampling"] != header[4]
        if i == 3:
            return P["Hc"] % 16 != 0 or P["Wc"] % 16 != 0
        if i == 4:
            return (_u16(header, 0), _u16(header, 2)) != (P["Wc"], P["Hc"])
        if i == 5:
            return P["oy"] + P["oh"] > P["Hc"] or P["ox"] + P["ow"] > P["Wc"]
        if i in (6, 7):
            H0, W0, l = _u16(cut_pre, 0), _u16(cut_pre, 2), P["l"]
            if i == 7:
                return l > cut_pre[4]
            hs, ws = -(-H0 // (1 << l)), -(-W0 // (1 << l))
            return P["y0"] + P["ch"] > hs or P["x0"] + P["cw"] > ws
        if i == 8:
            return P["D"][j] >= (P["Hc"] // p) * (P["Wc"] // p)
    if rule == 6:
        a0 = ASPECT_A * P["cw"] * P["oh"] > ASPECT_B * P["ch"] * P["ow"]
        a1 = ASPECT_A * P["ch"] * P["ow"] > ASPECT_B * P["cw"] * P["oh"]
        return a0 if i == 0 else a1
    if rule == 7:
        return (P["flags"] >> i) & 1 == 1
    if rule == 9:
        if i == 0:  # 9(0): the master of the claim's capture (cutPre = u16 H0 || u16 W0 || ...) exceeds 2^28 px
            return _u16(cut_pre, 0) * _u16(cut_pre, 2) > MASTER_MAX_PIXELS
        return P["ch"] * P["cw"] > CROP_MAX_PIXELS          # 9(1): the level-l* crop exceeds 2^25 px
    if rule == 8:
        if core.tier != 2:
            return False
        t = _tier_t_get(tier_t, core.enc_class)
        if t is None:
            # A registered tier-T claim always names a defined table (rule 8b); a ledger state without it is
            # incomplete (or the claim is not registrable), so rule 8c cannot be decided and passing silently would
            # call a wrong profile static-clean. Fail with rule 8's registration name.
            raise error_for("E.CLAIM.RULE8", f"tierT[{core.enc_class}] is not defined in the given ledger state")
        if i == 0:
            return exact_of(t.kernel_family, kfam) != P["kernel"]
        if i == 1:
            return t.sampling != header[4]
        if i == 2:
            return t.codec != 0
    raise ValueError(f"no row ({rule}, {i})")


def _r(name, detail=""):
    return StaticError(name, detail) if name.startswith("E.STATIC.") else error_for(name, detail)


def prove_static_violation(s: StaticIn, state: LedgerState, role: bool = True) -> str:
    """On-chain check steps 1-11 (reference model; the strike itself is the ledger's)."""
    if not role:                                                                     # 1
        raise _r("E.ROLE")
    row = ROWS.get((s.rule, s.i))                                                    # 2
    if row is None or (row.jmax is not None and not 0 <= s.j <= row.jmax):
        raise _r("E.STATIC.RULE")
    if claim_id(s.nf, s.claimant, s.core) not in state.claims:                       # 3
        raise _r("E.STATIC.CLAIM")
    if ((not row.prog and s.prog != b"") or (not row.header and (s.header != b"" or s.rMcu != ZERO32))  # 4
            or (not row.rec and (s.rec != b"" or s.cutPre != b""))):
        raise _r("E.STATIC.ARGS")
    core = s.core
    if row.prog and prog_mod.t_hash(s.prog) != core.t_hash:                          # 5
        raise _r("E.STATIC.PROGRAM")
    is59 = (s.rule, s.i) == (5, 9)
    if row.prog and not is59 and not prog_mod.prog_len_ok(s.prog):                   # 6
        raise _r("E.STATIC.PROGRAM_LEN")
    if row.header and (len(s.header) != 197 or H(TAG_LISTING_V2 + s.header + s.rMcu) != core.r_y):   # 7
        raise _r("E.STATIC.RY")
    if row.rec:                                                                      # 8
        cp = s.cutPre
        if not (len(s.rec) == 416 and H(s.rec) == core.rec_hash and len(cp) >= 5 and len(cp) == 5 + 32 * (cp[4] + 1)
                and H(TAG_ROOTS + cp) == s.rec[32:64]):
            raise _r("E.STATIC.REC")
    P = prog_mod.fields(s.prog) if row.prog and not is59 else None
    if (s.rule, s.i) == (4, 0) and not s.j + 1 < P["nD"]:                            # 9
        raise _r("E.STATIC.RULE")
    if (s.rule, s.i) == (5, 8) and not s.j < P["nD"]:
        raise _r("E.STATIC.RULE")
    if not violated(s.rule, s.i, s.j, P, s.prog, s.header, s.cutPre, core, state.tier_t, state.kfam):   # 10
        raise _r("E.STATIC.NO_VIOLATION")
    return "StaticViolation"                                                         # 11


def preimages(rule: int, i: int, prog: bytes, header: bytes = b"", r_mcu: bytes = ZERO32, rec: bytes = b"",
              cut_pre: bytes = b"") -> tuple:
    row = ROWS[(rule, i)]
    return (prog if row.prog else b"", header if row.header else b"", r_mcu if row.header else ZERO32,
            rec if row.rec else b"", cut_pre if row.rec else b"")


def first_violation(prog: bytes, header: bytes | None, cut_pre: bytes | None, core: ClaimCore, tier_t=None,
                    kfam=None):
    """The first violated row in search order, or None. ``header`` / ``cut_pre`` None skip the rows that read them.

    Preimages outside their exact lengths raise the on-chain error names instead of an IndexError: a header that is
    not 197 bytes -> ``E.STATIC.RY``; a cutPre shorter than 5 or not 5 + 32 (Lmax + 1) -> ``E.STATIC.REC``.
    A tier-T core whose table is missing from ``tier_t`` raises ``E.CLAIM.RULE8`` when rule 8 is reached."""
    if header is not None and len(header) != 197:
        raise _r("E.STATIC.RY", f"header has {len(header)} bytes, not 197")
    if cut_pre is not None and (len(cut_pre) < 5 or len(cut_pre) != 5 + 32 * (cut_pre[4] + 1)):
        raise _r("E.STATIC.REC", "cutPre length is not 5 + 32 (Lmax + 1)")
    if not prog_mod.prog_len_ok(prog):
        return (5, 9, 0)
    P = prog_mod.fields(prog)
    for rule, i in SEARCH_ORDER[1:]:
        row = ROWS[(rule, i)]
        if (row.header and header is None) or (row.rec and cut_pre is None):
            continue
        if row.jmax is not None:
            js = range(row.jmax + 1)
        else:
            js = range(P["nD"] - 1) if rule == 4 else range(P["nD"])
        for j in js:
            if violated(rule, i, j, P, prog, header, cut_pre, core, tier_t, kfam):
                return (rule, i, j)
    return None


def rule8_at_registration(core: ClaimCore, tier_t, height: int):
    """Rule 8a and 8b: None, or "E.CLAIM.RULE8"."""
    if core.tier not in (1, 2):
        return "E.CLAIM.RULE8"
    if core.tier == 1:
        return None if core.enc_class == 1 else "E.CLAIM.RULE8"
    if core.enc_class < 1024:
        return "E.CLAIM.RULE8"
    t = _tier_t_get(tier_t, core.enc_class)
    if t is None or t.set_height > height or (t.successor_height != 0 and t.successor_height <= height):
        return "E.CLAIM.RULE8"
    return None
