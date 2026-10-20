"""cutline.commit.capture - ``commit.capture``: master + region -> (rec v3, anchor leaf v2, witness).

Runs, for one capture: the commitment (``cutline.commit.cells.commitment``), H_m and optionally R_M, the rate epoch,
rec v3, sigma_dev = low-s P-256 over recHash, slotTag and the v2 anchor leaf. The region A, the slot and the owner
address are inputs (the app computes A with ``cutline.region.dilate``, draws the slot from
``cutline.commit.slots.SlotAllocator`` and takes the owner from the back office's provisioned list; ``capture_emu.app``
wires them together).

- ``DeviceKey(sk: int, key_id: bytes32)``; ``.pk64``.
- ``capture(master, strokes, A, policy, k_cap, beacon, key, owner_addr, slot, liveness=0, compute_rm=False,
  keep_cells=False) -> (Rec, anchor_leaf, Witness)``:
  ``master`` uint8[H0, W0, 3]; ``strokes`` bool[H0, W0] or None (= the all-zero mask); ``A`` bool[H0, W0];
  ``policy`` the capture-policy dict (hashed to pi_cap) or a 32-byte pi_cap; ``beacon = (h_beta, beta)`` with h_beta a
  15-block boundary (``ValueError`` otherwise; the chain would reject the batch with E.ANCHOR.BEACON_BOUNDARY);
  ``owner_addr`` 20 bytes != 0. The Witness holds k_cap, the Rec, (r, s_low), the stroke mask, the Commitment and,
  with ``keep_cells``, the 1,024-byte cells of every occupied cell of every level.
- ``rec_bytes_sig(rec, witness) -> (416 B, 64 B r || s)`` convenience.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..consts import BEACON_EPOCH
from ..crypto import p256_pub
from ..types import Rec, Witness
from .anchor import anchor_leaf_v2
from .cells import cell_bytes, commitment
from .rec import R_M_NONE, h_m, h_m_none, pi_cap, r_m, rec_hash, sig_bytes, sign_rec
from .slots import rate_epoch, slot_tag


@dataclass(frozen=True)
class DeviceKey:
    sk: int
    key_id: bytes

    @property
    def pk64(self) -> bytes:
        return p256_pub(self.sk)


def capture(master, strokes, A, policy, k_cap: bytes, beacon, key: DeviceKey, owner_addr: bytes, slot: int,
            liveness: int = 0, compute_rm: bool = False, keep_cells: bool = False):
    M = np.asarray(master)
    if M.dtype != np.uint8 or M.ndim != 3 or M.shape[2] != 3:
        raise ValueError("master must be uint8[H0, W0, 3]")
    H0, W0 = M.shape[:2]
    A = np.asarray(A, bool)
    if A.shape != (H0, W0):
        raise ValueError("A must be H0 x W0")
    if len(k_cap) != 32:
        raise ValueError("k_cap must be 32 bytes")
    h_beta, beta = beacon
    if h_beta % BEACON_EPOCH != 0:
        raise ValueError("h_beta must be a beacon-epoch boundary (h_beta = 0 mod 15)")
    if len(owner_addr) != 20 or bytes(owner_addr) == b"\x00" * 20:
        raise ValueError("owner address must be 20 nonzero bytes")
    if strokes is not None:
        strokes = np.asarray(strokes, bool)
        if strokes.shape != (H0, W0):
            raise ValueError("stroke mask must be H0 x W0")
        Hm = h_m(k_cap, strokes)
    else:
        Hm = h_m_none(k_cap, H0, W0)
    pc = bytes(policy) if isinstance(policy, (bytes, bytearray)) else pi_cap(policy)
    cm = commitment(M, A, k_cap)
    epoch = rate_epoch(h_beta)
    rec = Rec(R_cut=cm.R_cut, R_occ=cm.R_occ, R_M=r_m(k_cap, M) if compute_rm else R_M_NONE, h_beta=int(h_beta),
              beta=bytes(beta), pi_cap=pc, H_m=Hm, key_id=bytes(key.key_id), owner=bytes(owner_addr), epoch=epoch,
              slot=int(slot), liveness=int(liveness))
    rb = rec.encode()
    rh = rec_hash(rb)
    r, s = sign_rec(key.sk, rh)
    leaf = anchor_leaf_v2(rh, r, s, slot_tag(rec.key_id, epoch, rec.slot))
    cells = {}
    if keep_cells:
        for l, (C, al, _) in enumerate(cm.levels):
            ny, nx = (al.shape[0] + 15) // 16, (al.shape[1] + 15) // 16
            for cy in range(ny):
                for cx in range(nx):
                    if al[16 * cy:16 * cy + 16, 16 * cx:16 * cx + 16].any():
                        cells[(l, cy, cx)] = cell_bytes(C, al, cy, cx)
    w = Witness(k_cap=bytes(k_cap), rec=rec, sig_dev=(r, s), strokes=strokes, cells=cells, commitment=cm)
    return rec, leaf, w


def rec_bytes_sig(rec: Rec, witness: Witness) -> tuple:
    r, s = witness.sig_dev
    return rec.encode(), sig_bytes(r, s)
