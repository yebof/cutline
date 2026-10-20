"""capture_emu.backoffice - the seller's back-office holder with the normative holder policy.

EMULATION of the back office (no HSM): the holder key is a Python int (secp256k1), the owner keys come from HKDF of
the seller master seed. The two procedures, step by step:

``ExportIntake(rec, sigma_dev)`` (``export_intake``), on every record arriving on the export channel:
 0 check 0 (silent): 416 canonical bytes; the key is enrolled on chain; the device public key is known to this back
   office with H(pk) = devices[keyId].pkHash; P256VERIFY with s <= n/2. Failure -> DROP, no signal.
 1 seen[H(rec)] exists -> no-op, no signal.
 2 keyId not a device this back office enrolled as holder, or owner not in Own_d -> DROP, no signal.
 3 (recommended, ``in_order=True``) i = index of owner in Own_d; i != exportIdx_d -> COMPROMISE SIGNAL; else
   exportIdx_d += 1.
 4 seen[H(rec)] = (rec, sigma_dev).

``CoSign(rec, sigma_dev, batchId, leafIndex, path)`` (``cosign``), on the seller's selection of a capture in seen:
 0 check 0 as above and leaf_v2(rec, sigma_dev) included in anchored batch batchId -> else DROP, no signal;
   then the precondition H(rec) in seen -> else REFUSE ("not in seen"), no signal.
 1 cosigned[H(rec)] exists -> RETURN it (idempotent retry), no signal.
 2 check 1: keyId is an own device -> else REFUSE, no signal.
 3 check 2a: owner in Own_d -> else REFUSE, no signal.
 4 check 2b: owner never co-signed -> else REFUSE + COMPROMISE SIGNAL.
 5 check 3: (keyId, epoch, slot) never co-signed -> else REFUSE + COMPROMISE SIGNAL.
 6 sigma_hold = Sign_holder(vouchDigest(nf(rec))); record owner and (keyId, epoch, slot) as used; cosigned[H] = sig.

A compromise signal calls ``retireKey(keyId)`` on the chain as a signed relay (holder retirement) and stops
co-signing for d: a CoSign for a stopped device after step 1 returns ``Refuse("device retired", False)`` and
ExportIntake after step 2 returns ``Intake("retired")`` (no further signal). Returns: ``export_intake`` ->
``Intake(status, signal)`` with status in {"accepted", "dropped", "noop", "signal", "retired"}; ``cosign`` ->
65-byte sigma_hold or ``Refuse(reason, signal)``.

Interface:

- ``BackOffice(holder_sk, chain, seed, chain_id=None, registry=None, in_order=True)``; ``.holder`` (address).
- ``provision(device_key_id, n, pk64) -> list[address]``: derives the next n valid owner keys (capIndex counter
  shared by all devices of this back office, invalid indices skipped), records Own_d and the device key.
- ``owner_sk(address) -> int``; ``export_intake(rec, sig64)``; ``cosign(rec, sig64, batch_id, leaf_index, path, require_seen=True)``;
  ``.signals`` (list of (key_id, recHash, reason)); ``.seen``, ``.cosigned``.
- ``owner_key(seed, cap_index) -> int | None`` (None for an invalid index).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cutline.commit.anchor import anchor_leaf_v2
from cutline.commit.rec import decode_rec_strict, nf, rec_hash
from cutline.commit.slots import slot_tag
from cutline.consts import HKDF_OWNER_INFO
from cutline.crypto import N_K1, address_of, hkdf_sha256, k1_sign, p256_verify
from cutline.enc import u32
from cutline.errors import CutlineError
from cutline.ledger.abi import pk_hash, relay_digest_for, sign_vouch, vouch_digest


def owner_key(seed: bytes, cap_index: int) -> int | None:
    okm = hkdf_sha256(seed, b"", HKDF_OWNER_INFO + u32(cap_index), 32)
    d = int.from_bytes(okm, "big")
    return None if d == 0 or d >= N_K1 else d


@dataclass(frozen=True)
class Refuse:
    reason: str
    signal: bool


@dataclass(frozen=True)
class Intake:
    status: str               # accepted | dropped | noop | signal | retired
    signal: bool = False


@dataclass
class _Dev:
    pk64: bytes
    own: list                                     # provisioned owner addresses, in provisioning order
    export_idx: int = 0
    retired: bool = False
    used_owner: set = field(default_factory=set)
    used_slot: set = field(default_factory=set)


class BackOffice:
    def __init__(self, holder_sk: int, chain, seed: bytes, chain_id: int | None = None, registry: bytes | None = None,
                 in_order: bool = True):
        self.holder_sk = int(holder_sk)
        self.holder = address_of(self.holder_sk)
        self.chain = chain
        self.seed = bytes(seed)
        self.chain_id = chain.chain_id if chain_id is None else chain_id
        self.registry = chain.registry if registry is None else bytes(registry)
        self.in_order = in_order
        self.next_cap_index = 0
        self.devs: dict = {}
        self.owner_keys: dict = {}
        self.owner_cap_index: dict = {}
        self.seen: dict = {}
        self.cosigned: dict = {}
        self.signals: list = []

    # ---------------------------------------------------------------------------------------------- provisioning
    def provision(self, device_key_id: bytes, n: int, pk64: bytes) -> list:
        d = self.devs.get(device_key_id)
        if d is None:
            d = self.devs[device_key_id] = _Dev(pk64=bytes(pk64), own=[])
        out = []
        while len(out) < n:
            i = self.next_cap_index
            self.next_cap_index += 1
            sk = owner_key(self.seed, i)
            if sk is None:
                continue                          # an invalid capIndex is skipped
            a = address_of(sk)
            self.owner_keys[a] = sk
            self.owner_cap_index[a] = i
            d.own.append(a)
            out.append(a)
        return out

    def owner_sk(self, address: bytes) -> int:
        return self.owner_keys[bytes(address)]

    # ---------------------------------------------------------------------------------------------- helpers
    def _check0(self, rec: bytes, sig64: bytes):
        """Check 0: returns (Rec, recHash, r, s) or None (silent drop)."""
        try:
            rv = decode_rec_strict(rec)
        except CutlineError:
            return None
        if len(sig64) != 64:
            return None
        dev_on_chain = self.chain.devices.get(rv.key_id)
        mine = self.devs.get(rv.key_id)
        if dev_on_chain is None or mine is None or pk_hash(mine.pk64) != dev_on_chain.pk_hash:
            return None                           # a key whose public key this back office does not hold
        rh = rec_hash(rec)
        r, s = int.from_bytes(sig64[:32], "big"), int.from_bytes(sig64[32:], "big")
        if not p256_verify(rh, r, s, mine.pk64, require_low_s=True):
            return None
        return rv, rh, r, s

    def _is_own(self, key_id: bytes) -> bool:
        d = self.chain.devices.get(key_id)
        return key_id in self.devs and d is not None and d.holder == self.holder

    def _signal(self, key_id: bytes, rh: bytes, reason: str) -> None:
        self.signals.append((key_id, rh, reason))
        dev = self.devs[key_id]
        if not dev.retired:
            dev.retired = True
            nonce = self.chain.relay_nonce.get(self.holder, 0)
            dg = relay_digest_for(self.chain.chain_id, self.chain.device_registry, self.holder, nonce, "retireKey",
                                  [key_id])
            self.chain.retire_key_relayed(key_id, self.holder, k1_sign(self.holder_sk, dg))

    # ---------------------------------------------------------------------------------------------- ExportIntake
    def export_intake(self, rec: bytes, sig64: bytes) -> Intake:
        rec, sig64 = bytes(rec), bytes(sig64)
        c0 = self._check0(rec, sig64)                                   # 0
        if c0 is None:
            return Intake("dropped")
        rv, rh, _, _ = c0
        if rh in self.seen:                                             # 1
            return Intake("noop")
        if not self._is_own(rv.key_id) or rv.owner not in self.devs[rv.key_id].own:     # 2
            return Intake("dropped")
        dev = self.devs[rv.key_id]
        if dev.retired:                                                 # d is retired: nothing more to do for d
            return Intake("retired")
        if self.in_order:                                               # 3
            i = dev.own.index(rv.owner)
            if i != dev.export_idx:
                self._signal(rv.key_id, rh, "export order")
                return Intake("signal", True)
            dev.export_idx += 1
        self.seen[rh] = (rec, sig64)                                    # 4
        return Intake("accepted")

    # ---------------------------------------------------------------------------------------------- CoSign
    def cosign(self, rec: bytes, sig64: bytes, batch_id: int, leaf_index: int, path, require_seen: bool = True):
        """CoSign. Its precondition, "the seller's selection of a capture in seen", is enforced: a record that never
        passed ExportIntake is refused silently (``Refuse("not in seen", False)``), so the in-order check at intake
        covers every co-signed record. ``require_seen=False`` is a test hook only, to reach
        CoSign's own checks 1-3 (defence in depth) with records that intake would have dropped."""
        rec, sig64 = bytes(rec), bytes(sig64)
        c0 = self._check0(rec, sig64)                                   # 0
        if c0 is None:
            return Refuse("check 0", False)
        rv, rh, r, s = c0
        leaf = anchor_leaf_v2(rh, r, s, slot_tag(rv.key_id, rv.epoch, rv.slot))
        try:
            self.chain.check_inclusion(batch_id, leaf, leaf_index, list(path))
        except CutlineError:
            return Refuse("check 0: not anchored", False)
        if require_seen and rh not in self.seen:                        # precondition: a capture in seen
            return Refuse("not in seen", False)
        if rh in self.cosigned:                                         # 1
            return self.cosigned[rh]
        if not self._is_own(rv.key_id):                                 # 2 (check 1)
            return Refuse("not an own device", False)
        dev = self.devs[rv.key_id]
        if dev.retired:                                                 # stopped co-signing for d
            return Refuse("device retired", False)
        if rv.owner not in dev.own:                                     # 3 (check 2a)
            return Refuse("foreign owner", False)
        if rv.owner in dev.used_owner:                                  # 4 (check 2b)
            self._signal(rv.key_id, rh, "owner reused")
            return Refuse("owner reused", True)
        if (rv.epoch, rv.slot) in dev.used_slot:                        # 5 (check 3)
            self._signal(rv.key_id, rh, "slot reused")
            return Refuse("slot reused", True)
        sig_hold = sign_vouch(self.holder_sk, vouch_digest(self.chain_id, self.registry, nf(rh)))     # 6
        dev.used_owner.add(rv.owner)
        dev.used_slot.add((rv.epoch, rv.slot))
        self.cosigned[rh] = sig_hold
        return sig_hold
