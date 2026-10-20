"""cutline.types - the data types shared across the package.

Decoders with exact-length checks live in the owning packages: rec in ``cutline.commit.rec``, enc(pi) in
``cutline.compiler.program``.

- ``Rec``: rec v3, 13 fields in ABI word order after the domain word: ``R_cut, R_occ, R_M, h_beta, beta,
  pi_cap, H_m, key_id, owner, epoch, slot, liveness`` plus ``domain`` (last, defaults to ``REC_DOMAIN``, so negative
  test cases can set another). ``.encode() -> 416 B``; ``.strokes`` is an alias of ``H_m``.
  ``REC_TYPES`` is the 13-type ABI list.
- ``Program``: enc(pi): ``lstar, crop (y0, x0, ch, cw), out (oh, ow), place (oy, ox), canvas (Hc, Wc),
  kernel, sampling, flags, lut (768 bytes = LUT[0] || LUT[1] || LUT[2]), D (tuple of u32)``. ``.encode() -> 794 + 4 nD
  bytes`` (no validation beyond field widths: static rules are the verifier's job); ``.t_hash()`` =
  H(0x05 || "CUTLINE-prog-v1" || enc(pi)).
- ``ClaimCore``: 9 fields ``rec_hash, r_y, t_hash, r_fill, enc_class, tier, marketplace, category,
  listing_ref``; ``.encode() -> 288 B`` (static tuple, inline); ``.as_tuple()`` for ``cutline.abi``;
  ``ClaimCore.from_tuple(t)``. ``CLAIM_CORE_T`` is the ABI tuple type string.
- ``claim_id(nf, claimant, core) -> bytes32`` = H(abi.encode(bytes32 nf, address claimant, ClaimCore core)).
- ``Ctx``: presentation context ``(marketplace, listing_id, seller_account, salt_ref)``.
- ``Grammar``: the ContainerOK output record: ``W, H, sampling, tables: dict[int, bytes(64)] (natural order),
  tq: (int, int, int), progressive, restart_interval, scans``; ``.q3() -> (bytes64, bytes64, bytes64)`` (Y, Cb, Cr).
- ``AnchorProof(batch_id, leaf_index, path)``: the anchor inclusion proof.
- ``Commitment``: the capture commitment (produced by ``commit.cells.commitment``): ``H0, W0, Lmax,
  roots: list[bytes32], occ_bits: list[bytes] (packbits(O_l)), occ_hashes: list[bytes32], R_cut, R_occ, cut_pre
  (u16 H0 || u16 W0 || u8 Lmax || R_0 .. R_Lmax), levels`` with ``levels[l] = (C: uint8[h,w,3], alpha: uint8[h,w],
  leaves: list[bytes32])``.
- ``Compiled``: compiler output: ``X: uint8[Hc,Wc,3], alpha: uint8[Hc,Wc], m_cert: sorted list[int]
  (canvas MCU indices, D removed), sil: bool[Hc,Wc], cert_mask: bool[Hc,Wc]`` (pixels of the certified MCUs).
- ``Witness``: what the seller keeps: ``k_cap, rec (Rec), sig_dev ((r, s_low)), strokes (bool mask
  m or None), cells: dict[(l, cy, cx)] -> 1024 B of the occupied cells, commitment (Commitment | None)``.

A claim bundle is realised by the file types of ``cutline.bundle`` (``ClaimFile``, ``ProofFile``, ``FillFile``,
``RawBundle``).

- ``Evidence``: the evidence record, one attribute per field of the record (``EVIDENCE_FIELDS``, in the record's
  order) plus ``hop_index`` (the 0-based failing hop, printed only in the short form for ``"ancestor"`` /
  ``"hop"``). Every attribute defaults to ``None`` (not computed), so a verifier fills the record as evaluation
  proceeds and stops. Python values (``cutline.verify.record`` turns them into the canonical form):
  - ledger: ``cap`` ("valid" / "time-unverified" / "invalid"), ``claim`` ("authorised" / "none"), ``level_cap``
    (0, 1, 2, 4), ``receipt`` ("node" / "fresh" / "stale" / "none"), ``height`` (int), ``versions`` (the dict of
    exactly six keys built by ``record.versions(...)``);
  - ``presenter``, ``e0_hit``, ``partial``, ``d_gen``: bool (``e0_hit`` None for tier T);
  - ``tier``: "E" / "T" / "none"; ``enc_classes``: list of int (upload, then each hop);
  - ``g0_reason``: None or one of ``G0_REASONS``; ``static_violation``: None or ``{"rule": r, "i": i, "j": j}``;
    ``verdict``: None, "pass" or "fail";
  - ints: ``mcert_count, tested_blocks, sil_px, lam_y_2p51, lam_c_2p51, ms_y_2p51, ms_c_2p51, lambda_chain_2p51,
    excess_sum, excess_max, d_count, d_frac_ppm, category, uncovered_px, hop_depth, resampling_hops, liveness``
    (Python big ints where the values are big); ``res_class``, ``chain_class``: the class strings of
    ``cutline.verify.resclass``;
  - ``excess``: None (tier T) or ``{"map_2p30": [[mcu, comp, blk, value], ...], "p50": int, "p99": int, "max": int}``;
  - ``tone``: ``{"delta": int, "dE00_p50": float, "dE00_p95": float, "dE00_max": float, "area_dE6": int,
    "class": "identity" | "minor" | "moderate" | "major"}`` (the dE00 statistics are float64; the canonical form
    stores micro-units);
  - ``geometry``, ``cum_scale``: lists of ints (big); ``grid_wh``: [W, H]; ``hop_ceilings``: list of
    [platformId, v*];
  - ``pubseg_digest``, ``s_pub_packbits``, ``c_adj_packbits``: bytes; ``background``: "original" / "constant" /
    "generated-declared"; ``heuristic_digests``: ``{"pubseg": bytes32, "tone": bytes32}``.
  ``Evidence.is_short()`` is true when the short form applies (``g0_reason`` or ``static_violation`` set).
- ``FraudUnit(mcu, comp, k, blk=0, row=None, proof=None)``: the fraud-proof unit: the failing block
  (``mcu``, ``comp``, ``k``; ``blk`` the block inside the MCU as in the TE row) with ``row`` = the TE row
  ``(mcu, comp, blk, k, c, w)`` that failed and ``proof`` = the ``cutline.ledger.proofin.ProofIn`` holding exactly the
  fields of ``proveBlockFailure`` (the fill only if min alpha < 255). ``calldata()`` = the ``proveBlockFailure``
  calldata via ``ledger.proofin.encode_proofin_calldata``.
The ledger-state types of the static rules (``LedgerState``, ``StaticIn``) stay in ``cutline.compiler.static_rules``;
the tier-T entry (``static_rules.TierT``) is ``cutline.registries.TierTEntry`` (registry objects with codec logic stay
out of this module of plain dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import abi
from .consts import CLAIM_CORE_T, REC_DOMAIN, TAG_PROG
from .crypto import H
from .enc import u8, u16, u32

REC_TYPES = ["bytes32", "bytes32", "bytes32", "bytes32", "uint64", "bytes32", "bytes32", "bytes32", "bytes32",
             "address", "uint32", "uint16", "uint16"]


@dataclass(frozen=True)
class Rec:
    R_cut: bytes
    R_occ: bytes
    R_M: bytes
    h_beta: int
    beta: bytes
    pi_cap: bytes
    H_m: bytes
    key_id: bytes
    owner: bytes
    epoch: int
    slot: int
    liveness: int = 0
    domain: bytes = REC_DOMAIN

    @property
    def strokes(self) -> bytes:
        return self.H_m

    def words(self) -> list:
        return [self.domain, self.R_cut, self.R_occ, self.R_M, self.h_beta, self.beta, self.pi_cap, self.H_m,
                self.key_id, self.owner, self.epoch, self.slot, self.liveness]

    def encode(self) -> bytes:
        b = abi.encode(REC_TYPES, self.words())
        assert len(b) == 416
        return b


@dataclass(frozen=True)
class Program:
    lstar: int
    crop: tuple            # (y0, x0, ch, cw) at level l*
    out: tuple             # (oh, ow)
    place: tuple           # (oy, ox)
    canvas: tuple          # (Hc, Wc)
    kernel: int            # 0 area, 1 triangle
    sampling: int          # 0 = 4:4:4, 1 = 4:2:0
    flags: int             # bit 0 overlay (0 in v1), bit 1 dGen
    lut: bytes             # 768 bytes: LUT[0] || LUT[1] || LUT[2]
    D: tuple = ()          # declared-modified canvas MCU indices (u32, strictly ascending in a valid program)

    def encode(self) -> bytes:
        if len(self.lut) != 768:
            raise ValueError("lut must be 768 bytes")
        y0, x0, ch, cw = self.crop
        oh, ow = self.out
        oy, ox = self.place
        Hc, Wc = self.canvas
        b = (u8(self.lstar) + u16(y0) + u16(x0) + u16(ch) + u16(cw) + u16(oh) + u16(ow) + u16(oy) + u16(ox)
             + u16(Hc) + u16(Wc) + u8(self.kernel) + u8(self.sampling) + u8(self.flags) + bytes(self.lut)
             + u16(len(self.D)) + b"".join(u32(d) for d in self.D))
        assert len(b) == 794 + 4 * len(self.D)
        return b

    def t_hash(self) -> bytes:
        return H(TAG_PROG + self.encode())


@dataclass(frozen=True)
class ClaimCore:
    rec_hash: bytes
    r_y: bytes
    t_hash: bytes
    r_fill: bytes
    enc_class: int
    tier: int
    marketplace: int
    category: int
    listing_ref: bytes

    def as_tuple(self) -> tuple:
        return (self.rec_hash, self.r_y, self.t_hash, self.r_fill, self.enc_class, self.tier, self.marketplace,
                self.category, self.listing_ref)

    @classmethod
    def from_tuple(cls, t) -> "ClaimCore":
        return cls(*t)

    def encode(self) -> bytes:
        b = abi.encode_single(CLAIM_CORE_T, self.as_tuple())
        assert len(b) == 288
        return b


def claim_id(nf: bytes, claimant: bytes, core: ClaimCore) -> bytes:
    """claimId = H(abi.encode(bytes32 nf, address claimant, ClaimCore core)) (352-byte preimage)."""
    return H(abi.encode(["bytes32", "address", CLAIM_CORE_T], [nf, claimant, core.as_tuple()]))


@dataclass(frozen=True)
class Ctx:
    marketplace: int
    listing_id: str
    seller_account: str
    salt_ref: bytes


@dataclass(frozen=True)
class Grammar:
    W: int
    H: int
    sampling: int                      # 0 = 4:4:4, 1 = 4:2:0
    tables: dict                       # Tq -> 64 bytes, natural order
    tq: tuple                          # Tq of components 1, 2, 3
    progressive: bool
    restart_interval: int
    scans: int

    def q3(self) -> tuple:
        return tuple(bytes(self.tables[t]) for t in self.tq)

    __hash__ = None                    # tables is a dict


@dataclass(frozen=True)
class AnchorProof:
    batch_id: int
    leaf_index: int
    path: tuple                        # bytes32 siblings, leaves up


@dataclass(eq=False)
class Commitment:
    H0: int
    W0: int
    Lmax: int
    roots: list
    occ_bits: list
    occ_hashes: list
    R_cut: bytes
    R_occ: bytes
    cut_pre: bytes
    levels: list                       # levels[l] = (C uint8[h,w,3], alpha uint8[h,w], leaves list[bytes32])


@dataclass(eq=False)
class Compiled:
    X: Any                             # uint8[Hc, Wc, 3]
    alpha: Any                         # uint8[Hc, Wc]
    m_cert: list                       # sorted canvas MCU indices
    sil: Any                           # bool[Hc, Wc]
    cert_mask: Any                     # bool[Hc, Wc]


@dataclass(eq=False)
class Witness:
    k_cap: bytes
    rec: Rec
    sig_dev: tuple                     # (r, s_low)
    strokes: Any = None                # bool mask m (H0 x W0) or None
    cells: dict = field(default_factory=dict)      # (l, cy, cx) -> 1024 B, occupied cells
    commitment: Commitment | None = None


# ------------------------------------------------------------------------------------------------ evidence record
EVIDENCE_FIELDS = (
    "cap", "claim", "level_cap", "receipt", "presenter", "tier", "enc_classes", "e0_hit", "g0_reason",
    "static_violation", "verdict", "mcert_count", "tested_blocks", "sil_px", "res_class", "lam_y_2p51", "lam_c_2p51",
    "ms_y_2p51", "ms_c_2p51", "chain_class", "lambda_chain_2p51", "excess", "excess_sum", "excess_max", "tone",
    "geometry", "cum_scale", "partial", "d_count", "d_frac_ppm", "d_gen", "category", "uncovered_px", "grid_wh",
    "pubseg_digest", "s_pub_packbits", "c_adj_packbits", "background", "hop_depth", "resampling_hops", "hop_ceilings",
    "liveness", "height", "versions", "heuristic_digests")
G0_REASONS = ("container", "bundle", "chain_too_long", "unlinked", "presenter", "unregistered", "openings",
              "nothing_certified", "ancestor", "hop")


@dataclass
class Evidence:
    cap: str | None = None
    claim: str | None = None
    level_cap: int | None = None
    receipt: str | None = None
    presenter: bool | None = None
    tier: str | None = None
    enc_classes: list | None = None
    e0_hit: bool | None = None
    g0_reason: str | None = None
    static_violation: dict | None = None
    verdict: str | None = None
    mcert_count: int | None = None
    tested_blocks: int | None = None
    sil_px: int | None = None
    res_class: str | None = None
    lam_y_2p51: int | None = None
    lam_c_2p51: int | None = None
    ms_y_2p51: int | None = None
    ms_c_2p51: int | None = None
    chain_class: str | None = None
    lambda_chain_2p51: int | None = None
    excess: dict | None = None
    excess_sum: int | None = None
    excess_max: int | None = None
    tone: dict | None = None
    geometry: list | None = None
    cum_scale: list | None = None
    partial: bool | None = None
    d_count: int | None = None
    d_frac_ppm: int | None = None
    d_gen: bool | None = None
    category: int | None = None
    uncovered_px: int | None = None
    grid_wh: list | None = None
    pubseg_digest: bytes | None = None
    s_pub_packbits: bytes | None = None
    c_adj_packbits: bytes | None = None
    background: str | None = None
    hop_depth: int | None = None
    resampling_hops: int | None = None
    hop_ceilings: list | None = None
    liveness: int | None = None
    height: int | None = None
    versions: dict | None = None
    heuristic_digests: dict | None = None
    hop_index: int | None = None       # short form only ("ancestor", "hop")

    def is_short(self) -> bool:
        return self.g0_reason is not None or self.static_violation is not None


# ------------------------------------------------------------------------------------------------ fraud unit
@dataclass(frozen=True)
class FraudUnit:
    mcu: int
    comp: int
    k: int
    blk: int = 0
    row: tuple | None = None           # the failing TE row (mcu, comp, blk, k, c, w)
    proof: Any = None                  # cutline.ledger.proofin.ProofIn

    def calldata(self) -> bytes:
        from .ledger.proofin import encode_proofin_calldata     # local: ledger.proofin imports this module
        if self.proof is None:
            raise ValueError("FraudUnit has no ProofIn (build it with ledger.proofin.build_proofin)")
        return encode_proofin_calldata(self.proof)
