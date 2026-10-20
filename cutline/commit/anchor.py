"""cutline.commit.anchor - anchor leaf v2 and anchor batches.

- ``slot_tag`` (re-exported from :mod:`cutline.commit.slots`).
- ``anchor_leaf_v2(rec_hash, r, s, slot_tag) = H(0x00 || "CUTLINE-anchor-v2" || recHash || H(r || s) || slotTag)``
  (114-byte preimage). The leaf commits to sigma_dev exactly as given: callers pass the low-s form; a high-s
  signature gives a different leaf, which is why no normalisation happens here.
- ``Batch(leaves)``: ``.root``, ``.n``, ``.path(i)`` (Merkle path, leaves up), ``.index(leaf)``.
- ``check_inclusion(root, n, leaf, i, path) -> True`` or raises ``AnchorError("E.ANCHOR.INCLUSION")`` (the
  Merkle-level ``MerkleError`` is remapped). ``check_batch_inclusion(batch, leaf, i, path)`` is the same on a Batch.
"""
from __future__ import annotations

from ..consts import TAG_ANCHOR_V2
from ..crypto import H
from ..errors import AnchorError, MerkleError
from .merkle import merkle_levels, merkle_path, merkle_verify
from .slots import slot_tag  # noqa: F401  (re-export)


def anchor_leaf_v2(rec_hash: bytes, r: int, s: int, slot_tag_: bytes) -> bytes:
    if len(rec_hash) != 32 or len(slot_tag_) != 32:
        raise ValueError("recHash and slotTag must be 32 bytes")
    sig_h = H(int(r).to_bytes(32, "big") + int(s).to_bytes(32, "big"))
    return H(TAG_ANCHOR_V2 + bytes(rec_hash) + sig_h + bytes(slot_tag_))


class Batch:
    def __init__(self, leaves):
        self.leaves = [bytes(x) for x in leaves]
        self._levels = merkle_levels(self.leaves)          # raises E.MERKLE.EMPTY for an empty batch
        self.root = self._levels[-1][0]
        self.n = len(self.leaves)

    def path(self, i: int) -> list:
        return merkle_path(self._levels, i)

    def index(self, leaf: bytes) -> int:
        return self.leaves.index(bytes(leaf))


def check_inclusion(root: bytes, n: int, leaf: bytes, i: int, path) -> bool:
    try:
        return merkle_verify(leaf, i, n, list(path), root)
    except MerkleError as e:
        raise AnchorError("E.ANCHOR.INCLUSION", e.name) from None


def check_batch_inclusion(batch: Batch, leaf: bytes, i: int, path) -> bool:
    return check_inclusion(batch.root, batch.n, leaf, i, path)
