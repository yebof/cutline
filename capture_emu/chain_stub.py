"""capture_emu.chain_stub - an in-memory Python model of the capture-side ledger (not normative; the Solidity
contracts are).

EMULATION / test double. It models just enough of DeviceRegistry and CaptureAnchor for the emulated app, the
aggregator and the back office to run end to end:

- Heights: ``now`` (the current block), ``advance(n)``; ``blockhash(h) = H(label || u64 h)`` for ``now - 256 <= h <
  now`` and 32 zero bytes otherwise (EVM semantics); ``latest_boundary()`` = the latest h_beta = 0 mod 15 below now.
- DeviceRegistry: ``enroll(key_id, pk_hash, holder, att_hash, rateN, nonce_E)`` (E.DEV.EXISTS, E.DEV.PK_ENROLLED,
  E.DEV.RATE, E.DEV.HOLDER_ZERO), ``confirm(key_id, validator)`` (E.DEV.STATE; ACTIVE at the quorum
  2 floor((n - 1)/3) + 1 of ``n_validators``), ``validAt(key_id, h)`` (forward-only), ``retire_key(key_id,
  signer)`` (E.DEV.NOT_HOLDER; tc := now once) and ``retire_key_relayed(key_id, signer, sig65)`` (relay digest
  with ``relay_nonce[signer]``; E.RELAY.SIG, E.RELAY.NONCE). ``devices[key_id]`` is a :class:`Device`.
- CaptureAnchor: ``anchor_batch(root, count, h_beta) -> id`` with the anchoring checks in order: count >= 1
  (E.ANCHOR.COUNT), h_beta = 0 mod 15 (E.ANCHOR.BEACON_BOUNDARY), h_beta < now <= h_beta + Delta
  (E.ANCHOR.STALE), blockhash(h_beta) != 0 (E.ANCHOR.BEACON); 0-based ids. ``check_inclusion(batch_id, leaf, i, path)
  -> anchorHeight`` (E.ANCHOR.UNKNOWN_BATCH, E.ANCHOR.INCLUSION).
- ``cap_valid(rec, pk64, r, s, batch_id, leaf_index, path, sig_hold, registry) -> h_a``: CapValid checks 1-8 in
  order (a convenience for integration tests; the normative check is the contract's).

Errors are :class:`cutline.errors.CutlineError` (subclasses by area).
"""
from __future__ import annotations

from dataclasses import dataclass

from cutline.commit.anchor import anchor_leaf_v2, check_inclusion as _merkle_incl
from cutline.commit.rec import decode_rec_strict, rec_hash, nf as _nf
from cutline.commit.slots import slot_tag
from cutline.consts import BEACON_EPOCH, DELTA, RATE_EPOCH, RATEN_MAX, RATEN_MIN
from cutline.crypto import H, N_P256, k1_recover, p256_verify
from cutline.enc import u64
from cutline.errors import AnchorError, CapError, DevError, RelayError
from cutline.ledger.abi import pk_hash as _pk_hash, relay_digest_for, vouch_digest

PENDING, ACTIVE = 1, 2
ZERO20 = b"\x00" * 20


def cap_step3(rec_hash_: bytes, r: int, s: int, pk64: bytes) -> None:
    """CapValid step 3: ``s <= n/2`` first (``E.CAP.HIGH_S``), then P256VERIFY (``E.CAP.DEVICE_SIG``). A high-s
    signature that is also invalid is ``E.CAP.HIGH_S`` (the order is fixed so every implementation reports the same
    error)."""
    if s > N_P256 // 2:
        raise CapError("E.CAP.HIGH_S")
    if not p256_verify(rec_hash_, r, s, pk64, require_low_s=True):
        raise CapError("E.CAP.DEVICE_SIG", "device sig")


@dataclass
class Device:
    pk_hash: bytes
    holder: bytes
    enrolled_at: int
    rateN: int
    state: int
    tc: int = 0
    att_hash: bytes = b""
    nonce_E: bytes = b""


@dataclass
class BatchRec:
    root: bytes
    beacon: bytes
    beacon_height: int
    anchor_height: int
    count: int


class ChainStub:
    def __init__(self, chain_id: int = 20260926, start_height: int = 1_800_020, label: bytes = b"cutline-chain-stub",
                 n_validators: int = 4, device_registry: bytes = bytes.fromhex("e7f1725e7734ce288f8367e1bb143e90bb3f0512"),
                 registry: bytes = bytes.fromhex("5fbdb2315678afecb367f032d93f642f64180aa3")):
        self.chain_id = chain_id
        self.now = start_height
        self.label = bytes(label)
        self.n_validators = n_validators
        self.device_registry = device_registry
        self.registry = registry
        self.devices: dict = {}
        self.enrolled_pk: dict = {}
        self.votes: dict = {}
        self.batches: list = []
        self.relay_nonce: dict = {}
        self.events: list = []

    # ---------------------------------------------------------------------------------------------- heights
    def advance(self, n: int = 1) -> int:
        self.now += n
        return self.now

    def blockhash(self, h: int) -> bytes:
        if not (self.now - 256 <= h < self.now) or h < 0:
            return b"\x00" * 32
        return H(self.label + u64(h))

    def latest_boundary(self) -> int:
        return ((self.now - 1) // BEACON_EPOCH) * BEACON_EPOCH

    def beacon(self) -> tuple:
        h = self.latest_boundary()
        return h, self.blockhash(h)

    # ---------------------------------------------------------------------------------------------- DeviceRegistry
    def quorum(self) -> int:
        return 2 * ((self.n_validators - 1) // 3) + 1

    def enroll(self, key_id: bytes, pk_hash: bytes, holder: bytes, att_hash: bytes, rateN: int, nonce_E: bytes):
        if key_id in self.devices:
            raise DevError("E.DEV.EXISTS")
        if pk_hash in self.enrolled_pk:
            raise DevError("E.DEV.PK_ENROLLED")
        if not RATEN_MIN <= rateN <= RATEN_MAX:
            raise DevError("E.DEV.RATE")
        if bytes(holder) == ZERO20:
            raise DevError("E.DEV.HOLDER_ZERO")
        self.devices[key_id] = Device(pk_hash, bytes(holder), self.now, rateN, PENDING, 0, bytes(att_hash),
                                      bytes(nonce_E))
        self.enrolled_pk[pk_hash] = key_id
        self.events.append(("Enrolled", key_id, holder, pk_hash, att_hash, rateN, nonce_E))

    def confirm(self, key_id: bytes, validator) -> None:
        d = self.devices.get(key_id)
        if d is None or d.state != PENDING:
            raise DevError("E.DEV.STATE")
        v = self.votes.setdefault(key_id, set())
        v.add(validator)
        if len(v) >= self.quorum():
            d.state = ACTIVE
            self.events.append(("Activated", key_id, self.now))

    def validAt(self, key_id: bytes, h: int) -> bool:
        d = self.devices.get(key_id)
        return d is not None and d.state == ACTIVE and (d.tc == 0 or h < d.tc)

    def _set_tc(self, key_id: bytes, source: int) -> None:
        d = self.devices[key_id]
        if d.tc == 0:
            d.tc = self.now
            self.events.append(("Revoked", key_id, d.tc, source))

    def retire_key(self, key_id: bytes, signer: bytes) -> None:
        d = self.devices.get(key_id)
        if d is None or bytes(signer) != d.holder:
            raise DevError("E.DEV.NOT_HOLDER")
        self._set_tc(key_id, 5)

    def retire_key_relayed(self, key_id: bytes, signer: bytes, sig65: bytes) -> None:
        nonce = self.relay_nonce.get(bytes(signer), 0)
        dg = relay_digest_for(self.chain_id, self.device_registry, signer, nonce, "retireKey", [key_id])
        rec = k1_recover(dg, sig65)
        if rec is None or rec != bytes(signer):
            raise RelayError("E.RELAY.SIG")
        self.retire_key(key_id, signer)
        self.relay_nonce[bytes(signer)] = nonce + 1

    # ---------------------------------------------------------------------------------------------- CaptureAnchor
    def anchor_batch(self, root: bytes, count: int, h_beta: int) -> int:
        if count < 1:
            raise AnchorError("E.ANCHOR.COUNT", "count")
        if h_beta % BEACON_EPOCH != 0:
            raise AnchorError("E.ANCHOR.BEACON_BOUNDARY")
        if not (h_beta < self.now <= h_beta + DELTA):
            raise AnchorError("E.ANCHOR.STALE", "stale beacon")
        bh = self.blockhash(h_beta)
        if bh == b"\x00" * 32:
            raise AnchorError("E.ANCHOR.BEACON", "beacon")
        bid = len(self.batches)
        self.batches.append(BatchRec(bytes(root), bh, h_beta, self.now, count))
        self.events.append(("Anchored", bid, root, count, h_beta))
        return bid

    def check_inclusion(self, batch_id: int, leaf: bytes, leaf_index: int, path) -> int:
        if not 0 <= batch_id < len(self.batches):
            raise AnchorError("E.ANCHOR.UNKNOWN_BATCH")
        b = self.batches[batch_id]
        _merkle_incl(b.root, b.count, leaf, leaf_index, path)
        return b.anchor_height

    # ---------------------------------------------------------------------------------------------- CapValid
    # step 3 is the module-level cap_step3 (high-s is checked before the signature)
    def cap_valid(self, rec: bytes, pk64: bytes, r: int, s: int, batch_id: int, leaf_index: int, path,
                  sig_hold: bytes, registry: bytes | None = None) -> int:
        rv = decode_rec_strict(rec)                                     # 1
        d = self.devices.get(rv.key_id)
        if d is None or d.pk_hash != _pk_hash(pk64):                    # 2
            raise CapError("E.CAP.PK", "pk")
        rh = rec_hash(rec)
        cap_step3(rh, r, s, pk64)                                       # 3
        if not (rv.slot < d.rateN and rv.epoch == rv.h_beta // RATE_EPOCH):   # 4
            raise CapError("E.CAP.SLOT", "slot")
        leaf = anchor_leaf_v2(rh, r, s, slot_tag(rv.key_id, rv.epoch, rv.slot))
        h_a = self.check_inclusion(batch_id, leaf, leaf_index, path)    # 5
        b = self.batches[batch_id]
        if b.beacon != rv.beta or b.beacon_height != rv.h_beta:         # 6
            raise CapError("E.CAP.BRACKET", "bracket")
        if not self.validAt(rv.key_id, h_a):                            # 7
            raise CapError("E.CAP.KEY_INVALID", "key invalid at anchor")
        vd = vouch_digest(self.chain_id, registry or self.registry, _nf(rh))
        if k1_recover(vd, sig_hold) != d.holder:                        # 8
            raise CapError("E.CAP.HOLDER", "holder")
        return h_a
