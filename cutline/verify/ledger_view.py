"""cutline.verify.ledger_view - the one read-only ledger interface of the verifier.

Evaluate, VerifyE / VerifyT, VerifyHop and the vote function read the ledger only through a ``LedgerView``. Nothing
here touches a network. ``StateSnapshot`` is an in-memory implementation for tests and the emulated world; it is
**not normative** (the normative state is the contracts). Contract-backed readers implement the same protocol.

- ``ClaimStatus(exists, level_cap, cap, claim, receipt)``: the ledger fields of the evidence record at a height:
  ``exists`` bool; ``level_cap`` 0, 1, 2 or 4 (``None`` when ``exists`` is false); ``cap`` one of ``"valid"``,
  ``"time-unverified"``, ``"invalid"``; ``claim`` ``"authorised"`` or ``"none"``; ``receipt`` one of ``"node"``,
  ``"fresh"``, ``"stale"``, ``"none"``. ``ClaimStatus.MISSING`` = the status of an unknown claim.
- ``HopBatch(root, platform_id, height, count)``: ``HopRegistry.batches[id]``.
- ``LedgerView`` (``typing.Protocol``; heights are block numbers, ``h`` = the evaluation height):
  - attributes ``chain_id: int``, ``registry: bytes`` (the ClaimRegistry address, 20 B; ``read_bundle``'s claimant
    check), ``height: int`` (the latest height the view knows);
  - ``claim_status(claim_id, h) -> ClaimStatus``;
  - ``hop_batch_count(h) -> int``: the number of hop batches anchored at heights <= h;
  - ``hop_batch(batch_id, h) -> HopBatch``: ``E.HOP.UNKNOWN_BATCH`` if ``batch_id >= hop_batch_count(h)``;
  - ``hop_included(batch_id, leaf, leaf_index, path, h) -> bool``: ``hopIncluded`` (``MerkleVerify`` with the
    batch's count and root; False on any Merkle failure; ``E.HOP.UNKNOWN_BATCH`` for an unknown id);
  - ``hop_ceiling(p, height) -> (v*, q192)``: the version of platform p in force at ``height`` (callers pass
    ``hop_batch(b, h).height``);
  - ``tier_t(table_id, h=None) -> TierTEntry | None``: ``tierT[id]`` (with ``h``: None if stored after h);
    ``tier_t_map(h=None) -> dict`` (id -> TierTEntry; the ``tier_t`` argument of ``static_rules.first_violation``);
  - ``kfam(h=None) -> dict`` (f -> exact(f)); ``cls(enc_class) -> (kind, eta20) | None``;
    ``ep(ep_entry) -> EPEntry | None``;
  - ``capture_inclusion(batch_id, leaf, leaf_index, path) -> int | None``: the anchor height of a capture batch
    inclusion, None when it fails; only the A5 diagnostic re-check uses it (never a level input).
- ``StateSnapshot(chain_id=..., registry=..., height=0)``: the in-memory ``LedgerView``. Builders (each returns
  what the ledger would):
  - ``add_claim(claim_id, height=0, level_cap=4, cap="valid", claim="authorised", receipt="node") -> claim_id`` and
    ``register_claim(nf, claimant, core, **same) -> claim_id`` (claimId = H(abi.encode(nf, claimant, core));
    also remembered for ``static_state()``); ``set_claim_status(claim_id, height, **changes)`` records a later change
    (a strike, a lapsed licence: ``level_cap`` 1, ...), visible at heights >= ``height``;
  - ``add_hop_batch(leaves, platform_id, height) -> batch_id`` (root = MerkleRoot(leaves), count = len(leaves);
    ``E.REG.PLATFORM`` for platform 0) and ``hop_proof(batch_id, leaf_index) -> path``;
  - ``set_ceiling(p, T192, height) -> v`` (``HopCeilings.set_ceiling``);
  - ``add_tier_t(table_or_record, height) -> id`` (``TierTRegistry.set_table`` with its checks) and
    ``put_tier_t(id, entry)`` (no checks; for test states);
  - ``add_kfam(f, exact)``; ``add_ep(i, EPEntry)``; ``add_capture_batch(root, count, anchor_height) -> id``;
  - ``static_state() -> static_rules.LedgerState`` (the claims, tier-T entries and kfam).
- ``snapshot_from_ops(chain_id, registry, height, ops) -> StateSnapshot``: replays recorded builder calls (the form
  of ``cutline evaluate --ledger-ops``).
- ``from_chain_stub(stub, **snapshot_kw) -> StateSnapshot``: a snapshot with the stub's ``chain_id``, ``registry``,
  ``now`` and capture batches (``capture_emu.chain_stub.ChainStub`` is duck-typed: ``chain_id``, ``registry``,
  ``now``, ``batches[i].root / .count / .anchor_height``). The stub models no claims, hop batches, ceilings or tier-T
  tables; the caller adds them with the builders.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from cutline.commit.merkle import merkle_levels, merkle_path, merkle_root, merkle_verify
from cutline.consts import CLASS_REGISTRY, KFAM_GENESIS
from cutline.errors import HopError, MerkleError, RegError
from cutline.registries import (EPEntry, EPRegistry, HopCeilings, TierTEntry, TierTRegistry, TierTTable,
                                encode_table_record)
from cutline.types import ClaimCore, claim_id as _claim_id

CAP_VALUES = ("valid", "time-unverified", "invalid")
CLAIM_VALUES = ("authorised", "none")
RECEIPT_VALUES = ("node", "fresh", "stale", "none")
LEVEL_CAPS = (0, 1, 2, 4)


@dataclass(frozen=True)
class ClaimStatus:
    exists: bool
    level_cap: int | None = None
    cap: str | None = None
    claim: str | None = None
    receipt: str | None = None

    def __post_init__(self):
        if self.exists:
            if self.level_cap not in LEVEL_CAPS or self.cap not in CAP_VALUES or self.claim not in CLAIM_VALUES \
                    or self.receipt not in RECEIPT_VALUES:
                raise ValueError(f"bad claim status {self}")


ClaimStatus.MISSING = ClaimStatus(False)


@dataclass(frozen=True)
class HopBatch:
    root: bytes
    platform_id: int
    height: int
    count: int


@runtime_checkable
class LedgerView(Protocol):
    chain_id: int
    registry: bytes
    height: int

    def claim_status(self, claim_id: bytes, h: int) -> ClaimStatus: ...

    def hop_batch_count(self, h: int) -> int: ...

    def hop_batch(self, batch_id: int, h: int) -> HopBatch: ...

    def hop_included(self, batch_id: int, leaf: bytes, leaf_index: int, path, h: int) -> bool: ...

    def hop_ceiling(self, p: int, height: int) -> tuple: ...

    def tier_t(self, table_id: int, h: int | None = None) -> TierTEntry | None: ...

    def tier_t_map(self, h: int | None = None) -> dict: ...

    def kfam(self, h: int | None = None) -> dict: ...

    def cls(self, enc_class: int) -> tuple | None: ...

    def ep(self, ep_entry: int) -> EPEntry | None: ...

    def capture_inclusion(self, batch_id: int, leaf: bytes, leaf_index: int, path) -> int | None: ...


@dataclass(frozen=True)
class _CaptureBatch:
    root: bytes
    count: int
    anchor_height: int


class StateSnapshot:
    """In-memory ``LedgerView`` (not normative). See the module docstring for the builders."""

    def __init__(self, chain_id: int = 20260926, registry: bytes = bytes.fromhex(
            "5fbdb2315678afecb367f032d93f642f64180aa3"), height: int = 0):
        self.chain_id = int(chain_id)
        self.registry = bytes(registry)
        self.height = int(height)
        self._claims: dict = {}          # claimId -> [(height, ClaimStatus)] ascending
        self.claim_triples: list = []    # (nf, claimant, ClaimCore)
        self._hop_batches: list = []     # HopBatch, 0-based ids, heights non-decreasing
        self._hop_leaves: list = []      # the leaves of each hop batch (for hop_proof)
        self.ceilings = HopCeilings()
        self.tier_t_reg = TierTRegistry()
        self._kfam: dict = {f: (e, 0) for f, e in KFAM_GENESIS.items()}     # f -> (exact, height)
        self._cls: dict = dict(CLASS_REGISTRY)
        self.eps = EPRegistry()
        self._capture: list = []

    # ---------------------------------------------------------------------------------------------- builders
    def _bump(self, h: int) -> None:
        self.height = max(self.height, int(h))

    def add_claim(self, claim_id: bytes, height: int = 0, level_cap: int = 4, cap: str = "valid",
                  claim: str = "authorised", receipt: str = "node") -> bytes:
        cid = bytes(claim_id)
        if cid in self._claims:
            raise ValueError("claim already registered (E.CLAIM.* is the registry's job)")
        self._claims[cid] = [(int(height), ClaimStatus(True, level_cap, cap, claim, receipt))]
        self._bump(height)
        return cid

    def register_claim(self, nf: bytes, claimant: bytes, core: ClaimCore, height: int = 0, **status) -> bytes:
        cid = self.add_claim(_claim_id(nf, claimant, core), height, **status)
        self.claim_triples.append((bytes(nf), bytes(claimant), core))
        return cid

    def set_claim_status(self, claim_id: bytes, height: int, **changes) -> None:
        hist = self._claims[bytes(claim_id)]
        if height < hist[-1][0]:
            raise ValueError("status changes are recorded in height order")
        hist.append((int(height), replace(hist[-1][1], **changes)))
        self._bump(height)

    def add_hop_batch(self, leaves, platform_id: int, height: int) -> int:
        leaves = [bytes(x) for x in leaves]
        if platform_id < 1:
            raise RegError("E.REG.PLATFORM", "anchorHopBatch by a non-platform")
        if not leaves:
            raise ValueError("a hop batch has count >= 1")
        if self._hop_batches and height < self._hop_batches[-1].height:
            raise ValueError("hop batches are anchored in height order")
        self._hop_batches.append(HopBatch(merkle_root(leaves), int(platform_id), int(height), len(leaves)))
        self._hop_leaves.append(leaves)
        self._bump(height)
        return len(self._hop_batches) - 1

    def hop_proof(self, batch_id: int, leaf_index: int) -> list:
        return merkle_path(merkle_levels(self._hop_leaves[batch_id]), leaf_index)

    def set_ceiling(self, p: int, T, height: int) -> int:
        v = self.ceilings.set_ceiling(p, T, height)
        self._bump(height)
        return v

    def add_tier_t(self, table_or_record, height: int) -> int:
        rec = encode_table_record(table_or_record) if isinstance(table_or_record, TierTTable) else table_or_record
        tid = self.tier_t_reg.set_table(bytes(rec), height, self.kfam(height))
        self._bump(height)
        return tid

    def put_tier_t(self, tid: int, entry: TierTEntry) -> None:
        self.tier_t_reg.put(tid, entry)

    def add_kfam(self, f: int, exact: int, height: int = 0) -> None:
        if f in self._kfam:
            raise RegError("E.REG.DEFINED", f"kfam[{f}]")
        if exact not in (0, 1):
            raise RegError("E.REG.PROFILE", "exact must be 0 or 1")
        self._kfam[f] = (int(exact), int(height))

    def add_ep(self, i: int, entry: EPEntry) -> None:
        self.eps.set_entry(i, entry)

    def add_capture_batch(self, root: bytes, count: int, anchor_height: int) -> int:
        self._capture.append(_CaptureBatch(bytes(root), int(count), int(anchor_height)))
        return len(self._capture) - 1

    def static_state(self):
        from cutline.compiler.static_rules import LedgerState
        st = LedgerState(claims=set(self._claims), tier_t=dict(self.tier_t_reg.entries),
                         kfam={f: e for f, (e, _) in self._kfam.items()})
        st.claim_triples = list(self.claim_triples)
        return st

    # ---------------------------------------------------------------------------------------------- LedgerView
    def claim_status(self, claim_id: bytes, h: int) -> ClaimStatus:
        st = ClaimStatus.MISSING
        for hh, s in self._claims.get(bytes(claim_id), []):
            if hh <= h:
                st = s
        return st

    def hop_batch_count(self, h: int) -> int:
        return sum(1 for b in self._hop_batches if b.height <= h)

    def hop_batch(self, batch_id: int, h: int) -> HopBatch:
        if not 0 <= batch_id < self.hop_batch_count(h):
            raise HopError("E.HOP.UNKNOWN_BATCH", f"batch {batch_id}")
        return self._hop_batches[batch_id]

    def hop_included(self, batch_id: int, leaf: bytes, leaf_index: int, path, h: int) -> bool:
        b = self.hop_batch(batch_id, h)
        try:
            return merkle_verify(bytes(leaf), leaf_index, b.count, [bytes(p) for p in path], b.root)
        except MerkleError:
            return False

    def hop_ceiling(self, p: int, height: int) -> tuple:
        return self.ceilings.in_force(p, height)

    def tier_t(self, table_id: int, h: int | None = None) -> TierTEntry | None:
        e = self.tier_t_reg.get(table_id)
        if e is None or (h is not None and e.set_height > h):
            return None
        return e

    def tier_t_map(self, h: int | None = None) -> dict:
        return {i: e for i, e in self.tier_t_reg.entries.items() if h is None or e.set_height <= h}

    def kfam(self, h: int | None = None) -> dict:
        return {f: e for f, (e, hh) in self._kfam.items() if h is None or hh <= h}

    def cls(self, enc_class: int) -> tuple | None:
        return self._cls.get(enc_class)

    def ep(self, ep_entry: int) -> EPEntry | None:
        return self.eps.get(ep_entry)

    def capture_inclusion(self, batch_id: int, leaf: bytes, leaf_index: int, path) -> int | None:
        if not 0 <= batch_id < len(self._capture):
            return None
        b = self._capture[batch_id]
        try:
            merkle_verify(bytes(leaf), leaf_index, b.count, [bytes(p) for p in path], b.root)
        except MerkleError:
            return None
        return b.anchor_height


OP_BYTES = frozenset({"claim_id", "T", "table_or_record", "root", "build_sha256", "params_sha256"})


def snapshot_from_ops(chain_id: int, registry, height: int, ops) -> StateSnapshot:
    """A StateSnapshot rebuilt from recorded builder calls ``[[name, {kwargs}], ...]`` (the form
    ``cutline evaluate --ledger-ops`` uses): byte arguments as 0x hex (``OP_BYTES``; ``leaves`` a list of hex),
    ``add_ep`` with ``i``, ``build_sha256``, ``params_sha256``."""
    from cutline.enc import unhx
    s = StateSnapshot(chain_id=int(chain_id), registry=unhx(registry) if isinstance(registry, str) else registry,
                      height=int(height))
    for name, kw in ops:
        if name not in ("add_claim", "set_claim_status", "add_hop_batch", "set_ceiling", "add_tier_t", "add_ep",
                        "add_capture_batch", "add_kfam"):
            raise ValueError(f"unknown ledger op {name}")
        kw = {k: (unhx(v) if k in OP_BYTES else [unhx(x) for x in v] if k == "leaves" else v) for k, v in kw.items()}
        if name == "add_ep":
            s.add_ep(kw["i"], EPEntry(kw["build_sha256"], kw["params_sha256"]))
        else:
            getattr(s, name)(**kw)
    s.height = max(s.height, int(height))
    return s


def from_chain_stub(stub, **kw) -> StateSnapshot:
    """A StateSnapshot of a ``capture_emu.chain_stub.ChainStub`` (duck-typed; see the module docstring)."""
    s = StateSnapshot(chain_id=stub.chain_id, registry=stub.registry, height=stub.now, **kw)
    for b in stub.batches:
        s.add_capture_batch(b.root, b.count, b.anchor_height)
    return s


__all__ = ["ClaimStatus", "HopBatch", "LedgerView", "StateSnapshot", "from_chain_stub", "snapshot_from_ops", "CAP_VALUES", "CLAIM_VALUES",
           "RECEIPT_VALUES", "LEVEL_CAPS"]
