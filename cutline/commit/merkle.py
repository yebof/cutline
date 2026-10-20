"""cutline.commit.merkle - level trees, paths and multiproofs.

SHA-256 Merkle trees: node = H(0x01 || left || right); an odd last node pairs with itself; a one-leaf tree's root is
the leaf; n = 0 is not a tree.

- ``node(a, b) -> bytes32``.
- ``merkle_levels(leaves) -> list[list[bytes32]]`` (level 0 = the leaves, last = [root]); ``E.MERKLE.EMPTY`` if empty.
- ``merkle_root(leaves) -> bytes32``.
- ``merkle_depth(n) -> int`` = ceil(log2 n) (0 for n = 1).
- ``merkle_path(levels, i) -> list[bytes32]``: siblings from the leaves up (a self-paired node is its own sibling).
- ``root_from_path(leaf, i, n, path) -> bytes32``: MerkleRootFromPath; raises ``E.MERKLE.PATH`` (i >= n or
  |path| != d(n)) or ``E.MERKLE.ODD`` (a self-paired position whose sibling is not the node itself).
- ``merkle_verify(leaf, i, n, path, root) -> True``: raises as above, or ``E.MERKLE.ROOT``.
- ``multiproof(leaves, I) -> list[bytes32]``: MultiProof; ``I`` strictly ascending, all < n (``E.MERKLE.MULTI``
  otherwise); ``I = []`` gives ``[]``.
- ``multiroot(n, I, leaves, proof) -> bytes32``: MultiRoot; ``E.MERKLE.MULTI`` on a bad index list, a missing or an
  unconsumed proof node.
- ``multi_verify(n, I, leaves, proof, root) -> True``: MultiVerify; an empty opening verifies iff n >= 1 and leaves and
  proof are empty (``E.MERKLE.MULTI``); otherwise ``E.MERKLE.ROOT`` when the root differs.
All errors are ``cutline.errors.MerkleError``; n always comes from the verifier, never from the prover.
"""
from __future__ import annotations

import hashlib

from ..errors import MerkleError

_NODE = b"\x01"


def node(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(_NODE + a + b).digest()


def merkle_levels(leaves) -> list:
    lv = list(leaves)
    if not lv:
        raise MerkleError("E.MERKLE.EMPTY")
    levels = [lv]
    while len(lv) > 1:
        ext = lv + [lv[-1]] if len(lv) % 2 else lv
        lv = [node(ext[i], ext[i + 1]) for i in range(0, len(ext), 2)]
        levels.append(lv)
    return levels


def merkle_root(leaves) -> bytes:
    return merkle_levels(leaves)[-1][0]


def merkle_depth(n: int) -> int:
    if n < 1:
        raise MerkleError("E.MERKLE.EMPTY")
    return (n - 1).bit_length()


def merkle_path(levels, i: int) -> list:
    if not 0 <= i < len(levels[0]):
        raise MerkleError("E.MERKLE.PATH", f"index {i} of {len(levels[0])}")
    out = []
    for lv in levels[:-1]:
        s = i ^ 1
        out.append(lv[s] if s < len(lv) else lv[i])
        i >>= 1
    return out


def root_from_path(leaf: bytes, i: int, n: int, path) -> bytes:
    if n < 1 or not (0 <= i < n) or len(path) != merkle_depth(n):
        raise MerkleError("E.MERKLE.PATH", f"i={i} n={n} |path|={len(path)}")
    h, m = leaf, n
    for s in path:
        if i % 2 == 0:
            if i == m - 1 and s != h:
                raise MerkleError("E.MERKLE.ODD")
            h = node(h, s)
        else:
            h = node(s, h)
        i >>= 1
        m = (m + 1) >> 1
    return h


def merkle_verify(leaf: bytes, i: int, n: int, path, root: bytes) -> bool:
    if root_from_path(leaf, i, n, path) != root:
        raise MerkleError("E.MERKLE.ROOT")
    return True


def _check_index(n: int, I) -> None:
    if n < 1 or not I or any(b <= a for a, b in zip(I, I[1:])) or I[0] < 0 or I[-1] >= n:
        raise MerkleError("E.MERKLE.MULTI", "index list not strictly ascending in [0, n)")


def multiproof(leaves, I) -> list:
    """MultiProof: the siblings the verifier cannot compute, leaves up, ascending within a level."""
    I = list(I)
    if not I:
        if not leaves:
            raise MerkleError("E.MERKLE.EMPTY")
        return []
    _check_index(len(leaves), I)
    levels = merkle_levels(leaves)
    known = {i: levels[0][i] for i in I}
    proof, lvl, m = [], 0, len(leaves)
    while m > 1:
        nxt = {}
        for i in sorted(known):
            p = i >> 1
            if p in nxt:
                continue
            s = i ^ 1
            if s >= m:
                sib = known[i]
            elif s in known:
                sib = known[s]
            else:
                sib = levels[lvl][s]
                proof.append(sib)
            nxt[p] = node(known[i], sib) if i % 2 == 0 else node(sib, known[i])
        known, lvl, m = nxt, lvl + 1, (m + 1) >> 1
    return proof


def multiroot(n: int, I, leaves, proof) -> bytes:
    """MultiRoot (nonempty I only)."""
    I = list(I)
    _check_index(n, I)
    if len(leaves) != len(I):
        raise MerkleError("E.MERKLE.MULTI", "|leaves| != |I|")
    known, m, k = dict(zip(I, leaves)), n, 0
    while m > 1:
        nxt = {}
        for i in sorted(known):
            p = i >> 1
            if p in nxt:
                continue
            s = i ^ 1
            if s >= m:
                sib = known[i]
            elif s in known:
                sib = known[s]
            else:
                if k >= len(proof):
                    raise MerkleError("E.MERKLE.MULTI", "proof too short")
                sib = proof[k]
                k += 1
            nxt[p] = node(known[i], sib) if i % 2 == 0 else node(sib, known[i])
        known, m = nxt, (m + 1) >> 1
    if k != len(proof):
        raise MerkleError("E.MERKLE.MULTI", "unconsumed proof nodes")
    return known[0]


def multi_verify(n: int, I, leaves, proof, root: bytes) -> bool:
    """MultiVerify: the empty opening binds nothing and verifies iff its proof is empty."""
    if not list(I):
        if n < 1 or list(leaves) or list(proof):
            raise MerkleError("E.MERKLE.MULTI", "empty opening with leaves or proof")
        return True
    if multiroot(n, I, leaves, proof) != root:
        raise MerkleError("E.MERKLE.ROOT")
    return True
