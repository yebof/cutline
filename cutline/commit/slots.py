"""cutline.commit.slots - rate slots: slotTag and the normative slot PRP.

- ``slot_tag(key_id, epoch, slot) = H("slot" || keyId || u32 epoch || u16 slot)`` (42-byte preimage).
- ``rate_epoch(h_beta) = h_beta div RATE_EPOCH`` (1,800).
- ``slot_prp(k_slot, epoch, N) -> list[int]``: the full permutation (HMAC-SHA256 stream
  ``HMAC(k_slot, "slot" || u32 e || u32 ctr)``, big-endian u16 draws, descending Durstenfeld with rejection sampling).
  ``slot_prp_blocks(k_slot, epoch, N, count) -> (first count slots, HMAC blocks consumed)`` also reports the number
  of HMAC blocks consumed by the whole shuffle.
- ``SlotAllocator(k_slot, rateN)``: ``.next(epoch) -> slot`` uses the epoch's permutation in order without
  replacement and raises ``SlotsExhausted`` ("rateN slots used") when all N are used; ``.used(epoch)``.
"""
from __future__ import annotations

from ..consts import HM_SLOT, PFX_SLOT, RATE_EPOCH
from ..crypto import H, hmac256
from ..enc import u16, u32


class SlotsExhausted(Exception):
    """The capture would need slot number N: "rateN slots used" (the app aborts the capture)."""


def slot_tag(key_id: bytes, epoch: int, slot: int) -> bytes:
    if len(key_id) != 32:
        raise ValueError("keyId must be 32 bytes")
    return H(PFX_SLOT + bytes(key_id) + u32(epoch) + u16(slot))


def rate_epoch(h_beta: int) -> int:
    return int(h_beta) // RATE_EPOCH


def _prp(k_slot: bytes, epoch: int, N: int):
    if not 1 <= N <= 65536:
        raise ValueError("N out of range")
    st = {"buf": b"", "ctr": 0, "pos": 0}

    def draw16() -> int:
        while st["pos"] + 2 > len(st["buf"]):
            st["buf"] += hmac256(k_slot, HM_SLOT + u32(epoch) + u32(st["ctr"]))
            st["ctr"] += 1
        v = int.from_bytes(st["buf"][st["pos"]:st["pos"] + 2], "big")
        st["pos"] += 2
        return v

    perm = list(range(N))
    for i in range(N - 1, 0, -1):
        m = i + 1
        lim = 65536 - (65536 % m)
        while True:
            v = draw16()
            if v < lim:
                break
        j = v % m
        perm[i], perm[j] = perm[j], perm[i]
    return perm, st["ctr"]


def slot_prp(k_slot: bytes, epoch: int, N: int) -> list:
    return _prp(k_slot, epoch, N)[0]


def slot_prp_blocks(k_slot: bytes, epoch: int, N: int, count: int):
    perm, blocks = _prp(k_slot, epoch, N)
    return perm[:count], blocks


class SlotAllocator:
    def __init__(self, k_slot: bytes, rateN: int):
        self.k_slot, self.rateN = bytes(k_slot), int(rateN)
        self._perm: dict = {}
        self._used: dict = {}

    def used(self, epoch: int) -> int:
        return self._used.get(epoch, 0)

    def next(self, epoch: int) -> int:
        n = self._used.get(epoch, 0)
        if n >= self.rateN:
            raise SlotsExhausted("rateN slots used")
        if epoch not in self._perm:
            self._perm[epoch] = slot_prp(self.k_slot, epoch, self.rateN)
        self._used[epoch] = n + 1
        return self._perm[epoch][n]
