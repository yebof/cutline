"""cutline.registries - the registry records as Python objects: tier-T tables, kernel families, encoder classes, hop
ceilings and EP entries (the single source of the 55-byte tier-T record).

These are registry *records* with their codecs, not ledger state: ``cutline.verify.ledger_view`` holds them per
height. ``cutline.verify.verify_t`` (``TierTTable``, ``encode_table_record``, ``decode_table_record``,
``table_digest``) and ``cutline.compiler.static_rules`` (``TierT`` = ``TierTEntry``, ``tiert_record``,
``parse_tiert_record``) re-export from here.

Tier-T tables
- ``TierTTable(id, T, G4, G8, predecessor=0, kernel_family=1, sampling=0, codec=0, noise_milli=0, extra={})``:
  ``T = {2: (T_Y, T_Cb, T_Cr), 4: (...), 8: (...)}``; ``.get(key)`` gives dict-style access
  (``verify_t.tiert_statistics`` accepts a table or a dict). ``.profile`` = (kernel_family, sampling, codec);
  ``.record()`` = ``encode_table_record(self)``; ``.digest()`` = ``table_digest(self.record())``.
- ``encode_table_record(t) -> bytes55``: ``u16 id ‖ u16 predecessor ‖ u8 kernelFamily ‖ u8 sampling ‖ u8 codec ‖
  u32 T[b][c] (b = 2, 4, 8 outer; c = Y, Cb, Cr inner) ‖ u32 G4 ‖ u32 G8 ‖ u32 noiseMilli``.
- ``decode_table_record(b) -> TierTTable``: ``E.REG.TABLE_LEN`` unless exactly 55 bytes.
- ``table_digest(record) -> bytes32`` = H("CUTLINE-tierT-v1" ‖ record) (71-byte preimage).
- ``tiert_record(tid, predecessor, kernel_family, sampling, codec, T9, G4, G8, noise_milli) -> bytes55`` (T9 = the
  nine thresholds in record order) and ``parse_tiert_record(b) -> dict`` (keys ``id, predecessor, kernelFamily,
  sampling, codec, T (list of 9), G4, G8, noiseMilli``): the ``static_rules`` spellings over the codec above.
- ``TierTEntry(kernel_family, sampling, codec=0, successor_height=0, set_height=0, record=None)``: one ledger
  entry ``tierT[id] = {record, successorHeight}`` plus the height at which ``setTierTable`` stored it.
  ``TierTEntry.from_table(table, successor_height, set_height)`` builds an entry from a table and ``.table`` returns
  the decoded table, or None when only the profile is known.
  ``.current_at(h)``: ``set_height <= h`` and (``successor_height == 0`` or ``successor_height > h``) (static rule 8b,
  the reading of ``static_rules.rule8_at_registration``). ``.profile``, ``.table_id``, ``.noise_milli``.
- ``TierTRegistry``: the write-once tier-T registry. ``set_table(record, height, kfam=None) -> id`` runs
  ``setTierTable``'s checks in this order: ``E.REG.TABLE_LEN``; ``E.REG.DEFINED`` (id already set);
  ``E.REG.PROFILE`` (family not in kfam, sampling not in {0, 1} or codec != 0); ``E.REG.PREDECESSOR``
  (predecessor != ``current[profile]``, 0 when the profile has none); then stores it, sets the predecessor's
  successorHeight to ``height`` and ``current[profile] = id``. ``put(id, entry)`` stores an entry without checks
  (for test states). ``entries`` (id -> TierTEntry), ``current`` (profile -> id), ``get(id)``. The id range
  1,024-65,535 is not checked here (the on-chain ``setTierTable`` owns that revert).

Kernel families
- ``KFAM_GENESIS`` (f -> exact(f)); ``exact_of(f, kfam=None) -> int | None``.

Encoder classes
- ``ClassReg`` = ``consts.CLASS_REGISTRY`` (encClass -> (kind, eta20)); ``class_entry(c, cls=None) -> [c, kind,
  eta20]`` (the evidence ``versions.classes`` item); ``E.HOP.CLASS`` for an undefined class.

Hop ceilings
- ``HopCeilings()``: per-platform, versioned ceilings. ``versions(p) -> [(q192, height), ...]``, index = v, version 0
  = (IJG q75, 0) for every platform. ``set_ceiling(p, T192, height) -> v`` appends v = count(p) + 1 with
  ``q = elementwise max(previous q, T)``: ``E.REG.PLATFORM`` for p < 1 (0 is "not a platform"), ``E.REG.TABLE`` for
  a table that is not 192 entries in [1, 255], and a ``ValueError`` for a height below the previous version's (heights
  come from blocks). ``in_force(p, h) -> (v*, q192)`` with ``v* = max{v : height_v <= h}`` (a later version never
  changes an earlier hop). ``q_hop(p, v) -> q192``. ``copy()``.

EP entries
- ``EPEntry(build_sha256, params_sha256)`` (bytes32 each); ``EPRegistry``: ``set_entry(id, entry)`` (id >= 1, else
  ``ValueError``: no protocol error is defined for it; write-once, ``E.REG.DEFINED``), ``get(id) -> EPEntry | None``,
  ``entries``.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from .consts import CLASS_REGISTRY, IJG_Q75, KFAM_GENESIS, PFX_TIERT
from .errors import HopError, RegError

BS = (2, 4, 8)
ClassReg = CLASS_REGISTRY


# ------------------------------------------------------------------------------------------------ tier-T tables
@dataclass(frozen=True)
class TierTTable:
    id: int
    T: dict                              # b -> (T_Y, T_Cb, T_Cr)
    G4: int
    G8: int
    predecessor: int = 0
    kernel_family: int = 1
    sampling: int = 0
    codec: int = 0
    noise_milli: int = 0
    extra: dict = field(default_factory=dict)

    def get(self, key, default=None):   # dict-style access for tiert_statistics
        return getattr(self, key, default)

    @property
    def profile(self) -> tuple:
        return (self.kernel_family, self.sampling, self.codec)

    def record(self) -> bytes:
        return encode_table_record(self)

    def digest(self) -> bytes:
        return table_digest(self.record())


def encode_table_record(t: TierTTable) -> bytes:
    b = struct.pack(">HHBBB", t.id, t.predecessor, t.kernel_family, t.sampling, t.codec)
    for bb in BS:
        b += b"".join(struct.pack(">I", int(v)) for v in t.T[bb])
    b += struct.pack(">III", t.G4, t.G8, t.noise_milli)
    assert len(b) == 55
    return b


def decode_table_record(b: bytes) -> TierTTable:
    if len(b) != 55:
        raise RegError("E.REG.TABLE_LEN", f"{len(b)} bytes")
    i, pred, kf, smp, codec = struct.unpack(">HHBBB", b[:7])
    vals = struct.unpack(">" + "I" * 12, b[7:55])
    T = {bb: tuple(vals[3 * n:3 * n + 3]) for n, bb in enumerate(BS)}
    return TierTTable(id=i, T=T, G4=vals[9], G8=vals[10], predecessor=pred, kernel_family=kf, sampling=smp,
                      codec=codec, noise_milli=vals[11])


def table_digest(record: bytes) -> bytes:
    return hashlib.sha256(PFX_TIERT + bytes(record)).digest()


def tiert_record(tid: int, predecessor: int, kernel_family: int, sampling: int, codec: int, T9, G4: int, G8: int,
                 noise_milli: int) -> bytes:
    """u16 id || u16 predecessor || u8 kernelFamily || u8 sampling || u8 codec || u32 T[b][c] (9) || u32 G4 ||
    u32 G8 || u32 noiseMilli (55 bytes)."""
    if len(T9) != 9:
        raise ValueError("T needs 9 thresholds")
    T = {bb: tuple(int(v) for v in T9[3 * n:3 * n + 3]) for n, bb in enumerate(BS)}
    return encode_table_record(TierTTable(id=tid, T=T, G4=G4, G8=G8, predecessor=predecessor,
                                          kernel_family=kernel_family, sampling=sampling, codec=codec,
                                          noise_milli=noise_milli))


def parse_tiert_record(b: bytes) -> dict:
    t = decode_table_record(b)
    return {"id": t.id, "predecessor": t.predecessor, "kernelFamily": t.kernel_family, "sampling": t.sampling,
            "codec": t.codec, "T": [v for bb in BS for v in t.T[bb]], "G4": t.G4, "G8": t.G8,
            "noiseMilli": t.noise_milli}


@dataclass(frozen=True)
class TierTEntry:
    kernel_family: int
    sampling: int
    codec: int = 0
    successor_height: int = 0      # 0 = current
    set_height: int = 0            # height at which setTierTable stored it
    record: bytes | None = None    # the 55-byte record, if known

    @classmethod
    def from_table(cls, table: TierTTable, successor_height: int = 0, set_height: int = 0) -> "TierTEntry":
        return cls(table.kernel_family, table.sampling, table.codec, successor_height, set_height,
                   encode_table_record(table))

    @property
    def table(self) -> TierTTable | None:
        return None if self.record is None else decode_table_record(self.record)

    @property
    def table_id(self) -> int | None:
        return None if self.record is None else int.from_bytes(self.record[:2], "big")

    @property
    def noise_milli(self) -> int | None:
        return None if self.record is None else int.from_bytes(self.record[51:55], "big")

    @property
    def profile(self) -> tuple:
        return (self.kernel_family, self.sampling, self.codec)

    def current_at(self, h: int) -> bool:
        return self.set_height <= h and (self.successor_height == 0 or self.successor_height > h)


class TierTRegistry:
    """``tierT`` / ``currentTierT`` with ``setTierTable``'s checks (see the module docstring)."""

    def __init__(self):
        self.entries: dict = {}          # id -> TierTEntry
        self.current: dict = {}          # profile -> id

    def get(self, tid: int) -> TierTEntry | None:
        return self.entries.get(tid)

    def put(self, tid: int, entry: TierTEntry) -> None:
        self.entries[tid] = entry

    def set_table(self, record: bytes, height: int, kfam: dict | None = None) -> int:
        t = decode_table_record(record)                                         # E.REG.TABLE_LEN
        if t.id in self.entries:
            raise RegError("E.REG.DEFINED", f"tierT[{t.id}]")
        fams = KFAM_GENESIS if kfam is None else kfam
        if t.kernel_family not in fams or t.sampling not in (0, 1) or t.codec != 0:
            raise RegError("E.REG.PROFILE", f"profile {t.profile}")
        if t.predecessor != self.current.get(t.profile, 0):
            raise RegError("E.REG.PREDECESSOR", f"predecessor {t.predecessor}")
        if t.predecessor:
            p = self.entries[t.predecessor]
            self.entries[t.predecessor] = TierTEntry(p.kernel_family, p.sampling, p.codec, height, p.set_height,
                                                     p.record)
        self.entries[t.id] = TierTEntry.from_table(t, 0, height)
        self.current[t.profile] = t.id
        return t.id

    def copy(self) -> "TierTRegistry":
        r = TierTRegistry()
        r.entries, r.current = dict(self.entries), dict(self.current)
        return r


# ------------------------------------------------------------------------------------------------ kernel families
def exact_of(f: int, kfam: dict | None = None) -> int | None:
    return (KFAM_GENESIS if kfam is None else kfam).get(f)


# ------------------------------------------------------------------------------------------------ encoder classes
def class_entry(c: int, cls: dict | None = None) -> list:
    reg = ClassReg if cls is None else cls
    if c not in reg:
        raise HopError("E.HOP.CLASS", f"encClass {c} undefined")
    kind, eta20 = reg[c]
    return [c, kind, eta20]


# ------------------------------------------------------------------------------------------------ hop ceilings
class HopCeilings:
    """``hopCeiling[p][v] = {q[3][64], height}``; version 0 = (IJG q75, 0) implicitly for every platform."""

    def __init__(self):
        self._v: dict = {}               # p -> [(q192, height)] for v >= 1

    def versions(self, p: int) -> list:
        return [(IJG_Q75, 0)] + list(self._v.get(p, []))

    def set_ceiling(self, p: int, T, height: int) -> int:
        if p < 1:
            raise RegError("E.REG.PLATFORM", f"platform {p}")
        T = bytes(T)
        if len(T) != 192 or min(T) < 1:
            raise RegError("E.REG.TABLE", "a hop ceiling is 192 entries in [1, 255]")
        vs = self.versions(p)
        q_prev, h_prev = vs[-1]
        if height < h_prev:
            raise ValueError(f"height {height} below version {len(vs) - 1}'s {h_prev}")
        self._v.setdefault(p, []).append((bytes(max(a, b) for a, b in zip(q_prev, T)), int(height)))
        return len(vs)

    def in_force(self, p: int, h: int) -> tuple:
        vs = self.versions(p)
        v = max(i for i, (_, hv) in enumerate(vs) if hv <= h)
        return v, vs[v][0]

    def q_hop(self, p: int, v: int) -> bytes:
        return self.versions(p)[v][0]

    def copy(self) -> "HopCeilings":
        c = HopCeilings()
        c._v = {p: list(v) for p, v in self._v.items()}
        return c


# ------------------------------------------------------------------------------------------------ EP entries
@dataclass(frozen=True)
class EPEntry:
    build_sha256: bytes
    params_sha256: bytes


class EPRegistry:
    def __init__(self):
        self.entries: dict = {}

    def get(self, i: int) -> EPEntry | None:
        return self.entries.get(i)

    def set_entry(self, i: int, entry: EPEntry) -> None:
        if i < 1:
            raise ValueError("epEntry ids start at 1")
        if i in self.entries:
            raise RegError("E.REG.DEFINED", f"ep[{i}]")
        self.entries[i] = entry


__all__ = ["BS", "ClassReg", "TierTTable", "encode_table_record", "decode_table_record", "table_digest",
           "tiert_record", "parse_tiert_record", "TierTEntry", "TierTRegistry", "KFAM_GENESIS", "exact_of",
           "class_entry", "HopCeilings", "EPEntry", "EPRegistry"]
