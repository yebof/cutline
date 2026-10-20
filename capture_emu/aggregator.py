"""capture_emu.aggregator - the emulated aggregator (role AGGREGATOR).

Collects (leaf_v2, rec, sigma_dev) submissions, groups them by the beacon (h_beta, beta) and anchors one batch per
beacon group and flush: ``root = MerkleRoot(leaves in submission order)``, ``anchorBatch(root, count, h_beta)``.

Interface:

- ``Aggregator()``; ``.submit(leaf, rec, sig64)`` (rec is the 416-byte record; its h_beta and beta select the group).
- ``.flush(chain) -> list[AnchoredBatch]``, one per beacon group in first-submission order; each carries
  ``batch_id, root, h_beta, entries: list[(recHash, leaf, leaf_index, path)]`` and ``proof(rec_hash) ->
  (batch_id, leaf_index, path)``. Groups the chain refuses (e.g. a stale beacon) raise the chain's AnchorError and
  stay pending.
"""
from __future__ import annotations

from dataclasses import dataclass

from cutline.commit.anchor import Batch
from cutline.commit.rec import decode_rec_strict, rec_hash


@dataclass
class AnchoredBatch:
    batch_id: int
    root: bytes
    h_beta: int
    entries: list

    def proof(self, rh: bytes):
        for r, _, i, p in self.entries:
            if r == rh:
                return self.batch_id, i, p
        raise KeyError("record not in this batch")


class Aggregator:
    def __init__(self):
        self.pending: dict = {}               # (h_beta, beta) -> [(recHash, leaf, rec, sig)]

    def submit(self, leaf: bytes, rec: bytes, sig64: bytes) -> None:
        rv = decode_rec_strict(rec)
        self.pending.setdefault((rv.h_beta, rv.beta), []).append((rec_hash(rec), bytes(leaf), bytes(rec),
                                                                  bytes(sig64)))

    def flush(self, chain) -> list:
        out = []
        for key in list(self.pending):
            items = self.pending[key]
            b = Batch([x[1] for x in items])
            bid = chain.anchor_batch(b.root, b.n, key[0])
            out.append(AnchoredBatch(bid, b.root, key[0], [(x[0], x[1], i, b.path(i)) for i, x in enumerate(items)]))
            del self.pending[key]
        return out
