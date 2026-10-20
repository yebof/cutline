"""capture_emu.app - the emulated attested capture app.

EMULATION: a software P-256 key stands in for the StrongBox key, the attestation chain is the emulation profile
(:mod:`capture_emu.attestation`), k_cap and k_slot are derived from the app label instead of TEE randomness (so
emulated captures are reproducible). What it reproduces exactly is the protocol behaviour an honest app must have:
the enrolment challenge, the certified region, the commitment and rec v3, the slot PRP without replacement, the owner
index advanced durably BEFORE signing, low-s sigma_dev, and an export stream in signing order.

Interface:

- ``EmulatedApp(label, holder, rateN, chain_id, nonce_E, owner_addrs=None, k_slot=None, segmenter="birefnet_lite",
  app_id="cutline-capture-emu/1.0.0")``; ``.sk`` (int), ``.pk64``, ``.key_id`` (set by ``set_key_id`` after P_E
  assigns enrolSalt), ``.policy`` (the pi_cap record for ``segmenter``).
- ``.enrol() -> (pk64, chain_ders (leaf, batch, root), challenge)``.
- ``.set_key_id(key_id)``; ``.add_owner_addrs(addrs)`` (the back office's provisioned list, in order).
- ``.capture(master, strokes, soft_or_seg, beacon, liveness=0, compute_rm=False) -> (rec_bytes, sig64, leaf)``:
  ``soft_or_seg`` is a float soft map (thresholded >= 0.5), a bool mask S, or a registered segmenter name (live
  inference, ``cutline.region.dilate.seg_soft``); then A = ``certified_region(S, strokes)``, the commitment, rec v3.
  Raises ``SlotsExhausted`` ("rateN slots used") or ``RuntimeError("no owner address left")``. **Refuses an
  oversized master** (H0 * W0 > 2^28, or a side above the u16 range of cutPre) with ``E.VERIFYE.STATIC_RULE_9``
  before it segments, takes a slot or advances the owner index: every claim on such a capture violates static
  rule 9, so an honest app never creates one.
- ``.witnesses`` (recHash -> Witness), ``.export_stream() -> list[(rec_bytes, sig64)]`` in signing order;
  ``.owner_idx`` (the durable owner-address index).
"""
from __future__ import annotations

import numpy as np

from cutline.commit.capture import DeviceKey, capture as _capture
from cutline.commit.rec import pi_cap, rec_hash, sig_bytes
from cutline.consts import MASTER_MAX_PIXELS
from cutline.commit.slots import SlotAllocator
from cutline.crypto import H, p256_key_from_label, p256_pub
from cutline.enc import u32
from cutline.errors import error_for
from cutline.ledger.abi import enrol_challenge
from cutline.region.dilate import certified_region, seg_soft
from cutline.region.registry import policy_record

from .attestation import build_chain


def check_master_size(H0: int, W0: int) -> None:
    """Static rule 9: a conformant master has 1 <= H0, W0 <= 65535 (u16 in cutPre) and H0 * W0 <= 2^28; raises
    ``E.VERIFYE.STATIC_RULE_9`` otherwise."""
    if not (1 <= H0 <= 0xFFFF and 1 <= W0 <= 0xFFFF) or H0 * W0 > MASTER_MAX_PIXELS:
        raise error_for("E.VERIFYE.STATIC_RULE_9",
                        f"master {H0} x {W0} = {H0 * W0} px; the bound is H0 * W0 <= 2^28 = {MASTER_MAX_PIXELS}")


class EmulatedApp:
    def __init__(self, label: str, holder: bytes, rateN: int, chain_id: int, nonce_E: bytes, owner_addrs=None,
                 k_slot: bytes | None = None, segmenter: str = "birefnet_lite",
                 app_id: str = "cutline-capture-emu/1.0.0"):
        self.label = label
        self.sk = p256_key_from_label(label)
        self.pk64 = p256_pub(self.sk)
        self.holder, self.rateN, self.chain_id, self.nonce_E = bytes(holder), rateN, chain_id, bytes(nonce_E)
        self.owner_addrs = list(owner_addrs or [])
        self.owner_idx = 0
        self.k_slot = k_slot if k_slot is not None else H(("CUTLINE-emu k_slot " + label).encode())
        self.slots = SlotAllocator(self.k_slot, rateN)
        self.segmenter = segmenter
        self.policy = policy_record(segmenter, app_id)
        self.pi_cap = pi_cap(self.policy)
        self.key_id = None
        self.n_captures = 0
        self.witnesses: dict = {}
        self._stream: list = []

    def enrol(self):
        challenge = enrol_challenge(self.holder, self.rateN, self.chain_id, self.nonce_E)
        return self.pk64, build_chain(self.pk64, challenge), challenge

    def set_key_id(self, key_id: bytes) -> None:
        self.key_id = bytes(key_id)

    def add_owner_addrs(self, addrs) -> None:
        self.owner_addrs.extend(bytes(a) for a in addrs)

    def _k_cap(self) -> bytes:
        return H(("CUTLINE-emu k_cap " + self.label).encode() + u32(self.n_captures))

    def capture(self, master, strokes, soft_or_seg, beacon, liveness: int = 0, compute_rm: bool = False):
        if self.key_id is None:
            raise RuntimeError("device not enrolled (no keyId)")
        M = np.asarray(master)
        check_master_size(*M.shape[:2])
        if isinstance(soft_or_seg, str):
            S = seg_soft(soft_or_seg, M) >= 0.5
        else:
            a = np.asarray(soft_or_seg)
            S = a if a.dtype == bool else a >= 0.5
        A = certified_region(S, strokes)
        if self.owner_idx >= len(self.owner_addrs):
            raise RuntimeError("no owner address left")
        h_beta = int(beacon[0])
        slot = self.slots.next(h_beta // 1800)          # raises SlotsExhausted ("rateN slots used")
        owner = self.owner_addrs[self.owner_idx]
        self.owner_idx += 1                              # advanced durably before signing
        k_cap = self._k_cap()
        self.n_captures += 1
        rec, leaf, w = _capture(M, strokes, A, self.pi_cap, k_cap, beacon, DeviceKey(self.sk, self.key_id), owner,
                                slot, liveness=liveness, compute_rm=compute_rm)
        rb = rec.encode()
        sig = sig_bytes(*w.sig_dev)
        self.witnesses[rec_hash(rb)] = w
        self._stream.append((rb, sig))
        return rb, sig, leaf

    def export_stream(self) -> list:
        return list(self._stream)
