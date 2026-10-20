"""cutline.ledger.client - contract calls, events and ``RpcLedgerView`` (the ``LedgerView`` over JSON-RPC).

ABI encoding uses the project's own ``cutline.abi``; functions, events and custom errors come from the ABI export
``cutline_v1_abi.json`` (loaded by ``cutline.ledger.layout``); there is no web3 dependency.

- ``Contracts(addresses=None)`` (the frozen v1 addresses by default): ``fn(contract, sig)`` -> ``Fn`` (inputs,
  outputs, selector); ``calldata(contract, sig, args) -> bytes``; ``decode_output(contract, sig, data) -> tuple``.
  ``sig`` is the canonical signature (``"levelCap(bytes32)"``) or a unique function name (``"levelCap"``).
- ``abi_call(rpc, contract, sig, args, block="latest", contracts=None, sender=None) -> tuple``: ``eth_call`` +
  decoding; a revert raises ``CallRevert`` with ``.name`` = the protocol error name (``E.CLAIM.NOT_ACTIVE`` ...).
- ``decode_revert(data) -> str``: ``"E.X.Y"`` for the 109 v1 custom errors (``E_X_Y()`` <-> ``E.X.Y``, the first
  underscore after ``E_`` separates the area), ``"Error(msg)"``, ``"Panic(0x..)"``, ``"revert()"`` (empty) or
  ``"unknown:0x.."``.
- ``decode_log(log) -> Event | None``: ``Event(contract, name, args: dict, block, tx, index)`` for logs of the v1
  contracts (topic0 = keccak256 of the event signature; indexed args from topics, the rest from data).
  ``get_logs(rpc, contract, event, from_block, to_block, topics=None, step=500)`` decodes a range in chunks of
  <= ``step`` blocks (the sidecar's log scanner and the tier-T / kfam enumeration use it).
- ``LedgerHistoryError``: a height older than ``head - history_window`` (500, Besu Bonsai). A client exception, not
  an ``E.*`` name.
- ``RpcLedgerView(rpc, mode="node", contracts=None, pinned_validators=None, head_rpcs=None, clock=time.time,
  history_window=500, height=None, receipt_kw=None, on_receipt_error="raise")`` implements
  ``cutline.verify.ledger_view.LedgerView``:
  - ``chain_id`` (``eth_chainId``), ``registry`` (the ClaimRegistry address), ``height`` (the head at construction or
    ``refresh()``; the view reads un-heighted methods at this height, so one view gives consistent answers);
  - ``claim_status(cid, h)``: mode ``"node"`` reads ``levelCap`` plus the fields for ``cap`` (``validAt`` of the
    capture key at its anchor height; ``timeUnverified``) and ``claim`` (``principalOf`` = the claim's principal) at
    block h, ``receipt="node"``; mode ``"receipt"`` builds and verifies a receipt at h (``receipts``) against
    ``pinned_validators`` and the heads of ``head_rpcs`` (``receipt`` = ``"fresh"`` / ``"stale"``); a ReceiptError
    propagates (``on_receipt_error="raise"``) or gives ``receipt="none"`` with the status unknown
    (``on_receipt_error="none"``: exists, level_cap 4, valid, authorised, receipt "none", i.e. at most G2 and never a
    positive status);
  - ``hop_batch_count(h)``, ``hop_batch(b, h)`` (``E.HOP.UNKNOWN_BATCH`` for b >= count), ``hop_included`` (False on
    any Merkle failure), ``hop_ceiling(p, height)`` (the latest version with set height <= height);
  - ``tier_t(id, h=None)`` / ``tier_t_map(h=None)`` (the entries read at the view's height, then filtered by
    ``set_height <= h``, exactly the ``StateSnapshot`` semantics; ids enumerated from ``TierTableSet`` logs);
    ``kfam(h=None)`` (genesis families plus ``KernelFamilySet`` logs, read at h); ``cls(c)``; ``ep(i)``;
  - ``capture_inclusion(b, leaf, i, path)`` (``CaptureAnchor.checkInclusion``; None on a revert).
  Numeric heights below ``head - history_window`` raise ``LedgerHistoryError`` (``history_window=None`` disables
  the check, e.g. on EDR or an archive node).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from .. import abi
from ..consts import KFAM_GENESIS
from ..crypto import keccak256
from ..errors import ABIError, HopError
from ..registries import EPEntry, TierTEntry
from ..verify.ledger_view import ClaimStatus
from . import layout as L
from .rpc import RpcError, hx, unhx

HISTORY_WINDOW = 500


class LedgerHistoryError(Exception):
    """A read below the node's history window (Besu Bonsai keeps about 512 blocks)."""


class CallRevert(Exception):
    def __init__(self, name: str, data: bytes | None, what: str = ""):
        self.name, self.data = name, data
        super().__init__(f"{what}: {name}")


# ------------------------------------------------------------------------------------------------ ABI export
def _type_of(p: dict) -> str:
    t = p["type"]
    if t.startswith("tuple"):
        return "(" + ",".join(_type_of(c) for c in p["components"]) + ")" + t[len("tuple"):]
    return t


def spec_error_name(ident: str) -> str:
    if not ident.startswith("E_"):
        raise ValueError(ident)
    rest = ident[2:]
    area, _, name = rest.partition("_")
    return f"E.{area}.{name}" if name else f"E.{area}"


@dataclass(frozen=True)
class Fn:
    contract: str
    name: str
    sig: str
    inputs: tuple
    outputs: tuple
    selector: bytes
    mutability: str


@dataclass(frozen=True)
class Ev:
    contract: str
    name: str
    sig: str
    topic0: bytes
    inputs: tuple          # (name, type, indexed)


@lru_cache(maxsize=1)
def _index():
    fns, evs, errs = {}, {}, {}
    for c, spec in L.ABI_EXPORT["contracts"].items():
        for e in spec["abi"]:
            if e["type"] == "function":
                ins = tuple(_type_of(i) for i in e["inputs"])
                sig = f"{e['name']}({','.join(ins)})"
                f = Fn(c, e["name"], sig, ins, tuple(_type_of(o) for o in e["outputs"]), abi.selector(sig),
                       e.get("stateMutability", ""))
                fns.setdefault(c, {})[sig] = f
                fns[c].setdefault("#" + e["name"], []).append(f)
            elif e["type"] == "event":
                ins = tuple((i["name"], _type_of(i), bool(i.get("indexed"))) for i in e["inputs"])
                sig = f"{e['name']}({','.join(t for _, t, _ in ins)})"
                evs[keccak256(sig.encode())] = Ev(c, e["name"], sig, keccak256(sig.encode()), ins)
            elif e["type"] == "error":
                ins = ",".join(_type_of(i) for i in e["inputs"])
                errs[abi.selector(f"{e['name']}({ins})")] = spec_error_name(e["name"])
    return fns, evs, errs


class Contracts:
    def __init__(self, addresses: dict | None = None):
        self.addresses = dict(L.ADDRESSES if addresses is None else addresses)
        self._by_addr = {a: c for c, a in self.addresses.items()}

    def address(self, contract: str) -> bytes:
        return self.addresses[contract]

    def name_of(self, address: bytes) -> str | None:
        return self._by_addr.get(bytes(address))

    @staticmethod
    def fn(contract: str, sig: str) -> Fn:
        t = _index()[0][contract]
        if "(" in sig:
            return t[sig]
        fs = t["#" + sig]
        if len(fs) != 1:
            raise KeyError(f"{contract}.{sig} is overloaded; pass the signature")
        return fs[0]

    def calldata(self, contract: str, sig: str, args=()) -> bytes:
        f = self.fn(contract, sig)
        return f.selector + abi.encode(list(f.inputs), list(args))

    def decode_output(self, contract: str, sig: str, data: bytes) -> tuple:
        return abi.decode_canonical(list(self.fn(contract, sig).outputs), bytes(data))


def event(contract: str, name: str) -> Ev:
    for e in _index()[1].values():
        if e.contract == contract and e.name == name:
            return e
    raise KeyError(f"{contract}.{name}")


PANIC, ERROR_STRING = bytes.fromhex("4e487b71"), bytes.fromhex("08c379a0")


def decode_revert(data: bytes | None) -> str:
    if data is None:
        return "unknown:"
    data = bytes(data)
    if not data:
        return "revert()"
    sel = data[:4]
    errs = _index()[2]
    if sel in errs and len(data) == 4:
        return errs[sel]
    try:
        if sel == ERROR_STRING:
            return f"Error({abi.decode_canonical(['string'], data[4:])[0]})"
        if sel == PANIC:
            return f"Panic({hex(abi.decode_canonical(['uint256'], data[4:])[0])})"
    except ABIError:
        pass
    return "unknown:" + hx(data)


def abi_call(rpc, contract: str, sig: str, args=(), block="latest", contracts: Contracts | None = None,
             sender: bytes | None = None) -> tuple:
    cs = contracts or Contracts()
    f = cs.fn(contract, sig)
    try:
        out = rpc.eth_call(cs.address(contract), f.selector + abi.encode(list(f.inputs), list(args)), block, sender)
    except RpcError as e:
        d = e.revert_data
        if d is None and "revert" not in e.message.lower():
            raise
        raise CallRevert(decode_revert(d), d, f"{contract}.{f.sig}") from None
    return abi.decode_canonical(list(f.outputs), out)


# ------------------------------------------------------------------------------------------------ events
@dataclass(frozen=True)
class Event:
    contract: str
    name: str
    args: dict
    block: int
    tx: bytes
    index: int


def _decode_topic(t: str, topic: bytes):
    if t in ("string", "bytes") or t.endswith("]") or t.startswith("("):
        return topic                                   # indexed dynamic value: only its keccak is logged
    return abi.decode_canonical([t], topic)[0]


def decode_log(log: dict, contracts: Contracts | None = None) -> Event | None:
    cs = contracts or Contracts()
    topics = [unhx(t) for t in log.get("topics", [])]
    if not topics:
        return None
    ev = _index()[1].get(topics[0])
    if ev is None or cs.name_of(unhx(log["address"])) != ev.contract:
        return None
    args = {}
    it = iter(topics[1:])
    for name, t, indexed in ev.inputs:
        if indexed:
            args[name] = _decode_topic(t, next(it))
    rest = [(n, t) for n, t, ix in ev.inputs if not ix]
    vals = abi.decode_canonical([t for _, t in rest], unhx(log.get("data", "0x")))
    args.update({n: v for (n, _), v in zip(rest, vals)})
    return Event(ev.contract, ev.name, args, int(log["blockNumber"], 16), unhx(log.get("transactionHash") or "0x"),
                 int(log.get("logIndex", "0x0"), 16))


def get_logs(rpc, contract: str, name: str | None, from_block: int, to_block: int, topics=None, step: int = 500,
             contracts: Contracts | None = None) -> list:
    cs = contracts or Contracts()
    t0 = [hx(event(contract, name).topic0)] if name else [None]
    tp = t0 + list(topics or [])
    while tp and tp[-1] is None:
        tp.pop()
    out = []
    a = int(from_block)
    while a <= to_block:
        b = min(int(to_block), a + step - 1)
        for lg in rpc.call("eth_getLogs", {"address": hx(cs.address(contract)), "fromBlock": hex(a),
                                           "toBlock": hex(b), "topics": tp}):
            e = decode_log(lg, cs)
            if e is not None:
                out.append(e)
        a = b + 1
    return out


# ------------------------------------------------------------------------------------------------ RpcLedgerView
class RpcLedgerView:
    def __init__(self, rpc, mode: str = "node", contracts: Contracts | None = None, pinned_validators=None,
                 head_rpcs=None, clock=time.time, history_window: int | None = HISTORY_WINDOW,
                 height: int | None = None, receipt_kw: dict | None = None, on_receipt_error: str = "raise",
                 chain_id: int | None = None):
        if mode not in ("node", "receipt"):
            raise ValueError("mode is 'node' or 'receipt'")
        if mode == "receipt" and not pinned_validators:
            raise ValueError("receipt mode needs the pinned genesis validator set")
        if on_receipt_error not in ("raise", "none"):
            raise ValueError("on_receipt_error is 'raise' or 'none'")
        self.rpc, self.mode, self.cs = rpc, mode, contracts or Contracts()
        self.pinned = [bytes(v) for v in (pinned_validators or [])]
        self.head_rpcs = list(head_rpcs or [rpc])
        self.clock, self.history_window = clock, history_window
        self.receipt_kw = dict(receipt_kw or {})
        self.on_receipt_error = on_receipt_error
        self.chain_id = int(chain_id) if chain_id is not None else rpc.chain_id()
        self.registry = self.cs.address("ClaimRegistry")
        self.height = int(height) if height is not None else rpc.block_number()
        self._memo: dict = {}

    def refresh(self) -> int:
        self.height = self.rpc.block_number()
        return self.height

    # -- helpers
    def _check_h(self, h) -> int:
        h = int(h)
        if h < 0:
            raise ValueError("negative height")
        if self.history_window is not None and h < self.height - self.history_window:
            raise LedgerHistoryError(f"height {h} < head {self.height} - {self.history_window} (Bonsai window; "
                                     "use an archive node or history_window=None)")
        return h

    def _prefetch(self, items) -> None:
        """One JSON-RPC batch for the uncached (contract, sig, args, h) calls of ``items`` (results go to the memo)."""
        todo = []
        for contract, sig, args, h in items:
            key = (contract, sig, repr(args), h)
            if key not in self._memo and key not in [t[0] for t in todo]:
                f = self.cs.fn(contract, sig)
                data = f.selector + abi.encode(list(f.inputs), list(args))
                todo.append((key, f, ("eth_call", [self.rpc.call_obj(self.cs.address(contract), data), hex(h)])))
        if len(todo) < 2:
            return
        for (key, f, _), r in zip(todo, self.rpc.batch([t[2] for t in todo], raise_errors=False)):
            if isinstance(r, RpcError):
                d = r.revert_data
                if d is None and "revert" not in r.message.lower():
                    continue                                   # leave it to _call (raises the transport error)
                self._memo[key] = ("revert", CallRevert(decode_revert(d), d, f"{key[0]}.{f.sig}"))
            else:
                self._memo[key] = ("ok", abi.decode_canonical(list(f.outputs), unhx(r)))

    def _call(self, contract, sig, args, h):
        key = (contract, sig, repr(args), h)
        if key not in self._memo:
            try:
                self._memo[key] = ("ok", abi_call(self.rpc, contract, sig, args, h, self.cs))
            except CallRevert as e:
                self._memo[key] = ("revert", e)
        kind, v = self._memo[key]
        if kind == "revert":
            raise v
        return v

    # -- claims
    def claim_status(self, claim_id: bytes, h: int) -> ClaimStatus:
        h = self._check_h(h)
        cid = bytes(claim_id)
        if self.mode == "receipt":
            from .receipts import ReceiptError, build_receipt, verify_receipt
            from .rpc import heads
            try:
                r = build_receipt(self.rpc, cid, h, self.cs.addresses, **{k: v for k, v in self.receipt_kw.items()
                                                                            if k in ("hash_mode", "seal_mode",
                                                                                     "now_offset")})
                return verify_receipt(r, self.pinned, heads(self.head_rpcs), self.clock(), addresses=self.cs.addresses,
                                      **self.receipt_kw)
            except ReceiptError:
                if self.on_receipt_error == "raise":
                    raise
                return ClaimStatus(True, 4, "valid", "authorised", "none")
        self._prefetch([("ClaimRegistry", "claims(bytes32)", (cid,), h), ("ClaimRegistry", "levelCap(bytes32)", (cid,), h)])
        cl = self._call("ClaimRegistry", "claims(bytes32)", (cid,), h)
        claimant, _hgt, status, _tier, m, kind, nf, id_ = cl[:8]
        if claimant == L.ZERO20:
            return ClaimStatus.MISSING
        (lc,) = self._call("ClaimRegistry", "levelCap(bytes32)", (cid,), h)
        self._prefetch([("ClaimRegistry", "captures(bytes32)", (nf,), h),
                        ("ClaimRegistry", "principalOf(bytes32,address,uint8)", (nf, claimant, m), h)])
        cp = self._call("ClaimRegistry", "captures(bytes32)", (nf,), h)
        batch_id, time_unverified, key_id = cp[2], cp[3], cp[4]
        ha = self._call("CaptureAnchor", "batches(uint32)", (batch_id,), h)[3]
        (valid,) = self._call("DeviceRegistry", "validAt(bytes32,uint64)", (key_id, ha), h)
        try:
            p = self._call("ClaimRegistry", "principalOf(bytes32,address,uint8)", (nf, claimant, m), h)
        except CallRevert:
            p = None
        cap = "invalid" if not valid else "time-unverified" if time_unverified else "valid"
        claim = "authorised" if p is not None and tuple(p) == (kind, id_) else "none"
        return ClaimStatus(True, int(lc), cap, claim, "node")

    # -- hops
    def hop_batch_count(self, h: int) -> int:
        return int(self._call("HopRegistry", "batchCount()", (), self._check_h(h))[0])

    def hop_batch(self, batch_id: int, h: int):
        from ..verify.ledger_view import HopBatch
        h = self._check_h(h)
        if not 0 <= int(batch_id) < self.hop_batch_count(h):
            raise HopError("E.HOP.UNKNOWN_BATCH", f"batch {batch_id}")
        root, p, height, count = self._call("HopRegistry", "batches(uint32)", (int(batch_id),), h)
        return HopBatch(root, int(p), int(height), int(count))

    def hop_included(self, batch_id: int, leaf: bytes, leaf_index: int, path, h: int) -> bool:
        self.hop_batch(batch_id, h)                                # E.HOP.UNKNOWN_BATCH
        try:
            path = [bytes(x) for x in path]
            if not 0 <= int(leaf_index) < 1 << 32 or any(len(x) != 32 for x in path) or len(bytes(leaf)) != 32:
                return False
            return bool(self._call("HopRegistry", "hopIncluded(uint32,bytes32,uint32,bytes32[])",
                                   (int(batch_id), bytes(leaf), int(leaf_index), path), self._check_h(h))[0])
        except CallRevert:
            return False

    def hop_ceiling(self, p: int, height: int) -> tuple:
        blk = self.height
        n = int(self._call("HopRegistry", "hopCeilingCount(uint32)", (int(p),), blk)[0])
        best_v, best_q = 0, None
        for v in range(0, n + 1):
            q, hv = self._call("HopRegistry", "hopCeiling(uint32,uint32)", (int(p), v), blk)
            if int(hv) <= int(height):
                best_v, best_q = v, bytes(q)
        return best_v, best_q

    # -- registries
    def _tier_ids(self, blk: int) -> list:
        key = ("tier_ids", blk)
        if key not in self._memo:
            ev = get_logs(self.rpc, "Adjudicator", "TierTableSet", 0, blk, contracts=self.cs)
            self._memo[key] = sorted({int(e.args["id"]) for e in ev})
        return self._memo[key]

    def _tier_entry(self, tid: int, blk: int) -> TierTEntry | None:
        if not 0 <= int(tid) < 1 << 16:
            return None
        self._prefetch([("Adjudicator", s_, (int(tid),), blk) for s_ in ("tierTProfile(uint16)", "tierT(uint16)",
                                                                          "tierTSetHeight(uint16)")])
        defined, fam, sampling, codec = self._call("Adjudicator", "tierTProfile(uint16)", (int(tid),), blk)
        if not defined:
            return None
        record, succ = self._call("Adjudicator", "tierT(uint16)", (int(tid),), blk)
        (seth,) = self._call("Adjudicator", "tierTSetHeight(uint16)", (int(tid),), blk)
        return TierTEntry(int(fam), int(sampling), int(codec), int(succ), int(seth), bytes(record))

    def tier_t(self, table_id: int, h: int | None = None) -> TierTEntry | None:
        blk = self.height if h is None else max(self.height, self._check_h(h))
        e = self._tier_entry(table_id, blk)
        if e is None or (h is not None and e.set_height > h):
            return None
        return e

    def tier_t_map(self, h: int | None = None) -> dict:
        blk = self.height if h is None else max(self.height, self._check_h(h))
        out = {}
        for tid in self._tier_ids(blk):
            e = self._tier_entry(tid, blk)
            if e is not None and (h is None or e.set_height <= h):
                out[tid] = e
        return out

    def kfam(self, h: int | None = None) -> dict:
        blk = self.height if h is None else self._check_h(h)
        key = ("kfam_ids", blk)
        if key not in self._memo:
            ev = get_logs(self.rpc, "Adjudicator", "KernelFamilySet", 0, blk, contracts=self.cs)
            self._memo[key] = sorted(set(KFAM_GENESIS) | {int(e.args["f"]) for e in ev})
        out = {}
        for f in self._memo[key]:
            defined, exact = self._call("Adjudicator", "kfam(uint8)", (f,), blk)
            if defined:
                out[f] = int(exact)
        return out

    def cls(self, enc_class: int):
        if not 0 <= int(enc_class) < 1 << 16:
            return None
        defined, kind, eta20 = self._call("Adjudicator", "cls(uint16)", (int(enc_class),), self.height)
        return (int(kind), int(eta20)) if defined else None

    def ep(self, ep_entry: int):
        if not 0 <= int(ep_entry) < 1 << 32:
            return None
        b, p = self._call("Adjudicator", "ep(uint32)", (int(ep_entry),), self.height)
        if b == b"\x00" * 32 and p == b"\x00" * 32:
            return None
        return EPEntry(b, p)

    def capture_inclusion(self, batch_id: int, leaf: bytes, leaf_index: int, path):
        try:
            path = [bytes(x) for x in path]
            if not 0 <= int(batch_id) < 1 << 32 or not 0 <= int(leaf_index) < 1 << 32 or len(bytes(leaf)) != 32 \
                    or any(len(x) != 32 for x in path):
                return None
            (ha,) = self._call("CaptureAnchor", "checkInclusion(uint32,bytes32,uint32,bytes32[])",
                               (int(batch_id), bytes(leaf), int(leaf_index), path), self.height)
        except CallRevert:
            return None
        return int(ha)


__all__ = ["Contracts", "Fn", "Ev", "Event", "CallRevert", "LedgerHistoryError", "RpcLedgerView", "abi_call",
           "decode_revert", "decode_log", "get_logs", "event", "spec_error_name", "HISTORY_WINDOW"]
