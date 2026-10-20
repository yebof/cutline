"""cutline.ledger.abi - the protocol digests of the ledger.

Enrolment and device identifiers, the holder vouch, listingRef / claim digest v2 / presenter digest, principal keys,
epoch-keyed delegation validity, signed relays and the path-(b) transcript encodings. Every function is a pure
function of its inputs; addresses are 20 raw bytes, keys are Python ints.

- ``pk_hash(pk64) = H(pk_x || pk_y)``; ``key_id(pk64, enrol_salt) = H(pk_x || pk_y || enrolSalt)``;
  ``att_hash(ders) = H(DER(leaf) || ... || DER(root))``.
- ``enrol_challenge(holder, rateN, chain_id, nonce_E) = H("CUTLINE-ENROL-v1" || holder || u16 rateN || u256 chainId
  || nonce_E)`` (86-byte preimage; nonce_E is 16 bytes).
- ``enrol_vote_subject(key_id, pk_hash) = keccak256(abi.encode(string "enrol", bytes32, bytes32))``.
- ``vouch_digest(chain_id, registry, nf)`` (192-byte encoding); ``sign_vouch(sk_holder, digest) -> 65 B``.
- ``listing_ref(m, l, s, salt_ref) = H("CUTLINE-lref" || u8 m || H(l) || H(s) || salt_ref)`` (l and s as UTF-8).
- ``claim_digest(chain_id, registry, core: ClaimCore)`` (448-byte encoding); ``claimant_of(digest, sig65)``.
- ``present_digest(m, l, s, nonce)``.
- ``principal_key(nf, kind, id) = H(nf || u8 kind || u32 id)`` (kind 0 = own, 1 = lic).
- ``Delegation(epoch, scope, expiry, revoked)``; ``delegation_valid(entry, captures_epoch, now, m) -> bool``
  (epoch equal, not revoked, expiry >= now, bit m of scope set; m < 64).
- ``args_hash(types, args) = keccak256(abi.encode(args))``; ``relay_digest(chain_id, target, signer, nonce, selector,
  args_hash)``; ``RELAY_SIGS`` (canonical signatures) and ``relay_selector(name)``.
- ``TE_T``, ``TT_T`` (row tuple types); ``pass_transcript(target) = H("pass" || target)``;
  ``fail_transcript_e(rows)``, ``fail_transcript_t(rows)`` = H(abi.encode(T[] rows)) with the rows sorted by their
  unique key (duplicate keys raise ValueError). An empty row list gives ``EMPTY_ROW_TRANSCRIPT``, the constant
  ``submitVerdict`` rejects (``is_rejected_fail_transcript``).
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import abi
from ..consts import (CLAIM_CORE_T, DOM_CLAIM, DOM_ENROL_VOTE, DOM_PRESENT, DOM_RELAY, DOM_VOUCH, EMPTY_ROW_TRANSCRIPT,
                      MARKETPLACE_LIMIT, PFX_ENROL, PFX_LREF, PFX_PASS, RATEN_MAX, RATEN_MIN)
from ..crypto import H, k1_recover, k1_sign, keccak256
from ..enc import u8, u16, u32, u256
from ..types import ClaimCore

TE_T = "(uint32,uint8,uint8,uint8,int16,int256)"
TT_T = "(uint16,uint8,uint32,uint8,uint8,int256)"
KIND_OWN, KIND_LIC = 0, 1

RELAY_SIGS = {
    "delegate": "delegate(bytes32,address,uint64,bool,uint64,bytes32[])",
    "revokeDelegation": "revokeDelegation(bytes32,address)",
    "offerTransfer": "offerTransfer(bytes32,address)",
    "acceptTransfer": "acceptTransfer(bytes32)",
    "withdraw": "withdraw(bytes32)",
    "retireKey": "retireKey(bytes32)",
}
RELAY_ARG_TYPES = {
    "delegate": ["bytes32", "address", "uint64", "bool", "uint64", "bytes32[]"],
    "revokeDelegation": ["bytes32", "address"],
    "offerTransfer": ["bytes32", "address"],
    "acceptTransfer": ["bytes32"],
    "withdraw": ["bytes32"],
    "retireKey": ["bytes32"],
}


def _b(x, n, what):
    x = bytes(x)
    if len(x) != n:
        raise ValueError(f"{what} must be {n} bytes")
    return x


# ------------------------------------------------------------------------------------------------ enrolment, vouch
def pk_hash(pk64: bytes) -> bytes:
    return H(_b(pk64, 64, "pk"))


def key_id(pk64: bytes, enrol_salt: bytes) -> bytes:
    return H(_b(pk64, 64, "pk") + _b(enrol_salt, 32, "enrolSalt"))


def att_hash(ders) -> bytes:
    return H(b"".join(bytes(d) for d in ders))


def enrol_challenge(holder: bytes, rateN: int, chain_id: int, nonce_E: bytes) -> bytes:
    if not RATEN_MIN <= rateN <= RATEN_MAX:
        raise ValueError("rateN outside 30..600")
    pre = PFX_ENROL + _b(holder, 20, "holder") + u16(rateN) + u256(chain_id) + _b(nonce_E, 16, "nonce_E")
    assert len(pre) == 86
    return H(pre)


def enrol_vote_subject(key_id_: bytes, pk_hash_: bytes) -> bytes:
    return keccak256(abi.encode(["string", "bytes32", "bytes32"], [DOM_ENROL_VOTE, key_id_, pk_hash_]))


def vouch_digest(chain_id: int, registry: bytes, nf: bytes) -> bytes:
    enc = abi.encode(["string", "uint256", "address", "bytes32"], [DOM_VOUCH, chain_id, _b(registry, 20, "registry"),
                                                                   _b(nf, 32, "nf")])
    assert len(enc) == 192
    return keccak256(enc)


def sign_vouch(sk_holder: int, digest: bytes) -> bytes:
    return k1_sign(sk_holder, digest)


# ------------------------------------------------------------------------------------------------ listing, claim, presenter
def listing_ref(m: int, listing_id: str, seller: str, salt_ref: bytes) -> bytes:
    pre = (PFX_LREF + u8(m) + H(listing_id.encode("utf-8")) + H(seller.encode("utf-8"))
           + _b(salt_ref, 32, "salt_ref"))
    assert len(pre) == 109
    return H(pre)


def claim_digest(chain_id: int, registry: bytes, core: ClaimCore) -> bytes:
    enc = abi.encode(["string", "uint256", "address", CLAIM_CORE_T],
                     [DOM_CLAIM, chain_id, _b(registry, 20, "registry"), core.as_tuple()])
    assert len(enc) == 448
    return keccak256(enc)


def claimant_of(digest: bytes, sig65: bytes) -> bytes | None:
    return k1_recover(digest, sig65)


def present_digest(m: int, listing_id: str, seller: str, nonce: bytes) -> bytes:
    return keccak256(abi.encode(["string", "uint8", "string", "string", "bytes32"],
                                [DOM_PRESENT, m, listing_id, seller, _b(nonce, 32, "nonce")]))


# ------------------------------------------------------------------------------------------------ principals, delegation, relays
def principal_key(nf: bytes, kind: int, id_: int) -> bytes:
    if kind not in (KIND_OWN, KIND_LIC):
        raise ValueError("kind is 0 (own) or 1 (lic)")
    return H(_b(nf, 32, "nf") + u8(kind) + u32(id_))


@dataclass(frozen=True)
class Delegation:
    epoch: int          # u32 ownership epoch the delegation was issued in
    scope: int          # u64 marketplace bitmask
    expiry: int         # u64 height
    revoked: bool = False


def delegation_valid(entry: Delegation | None, captures_epoch: int, now: int, m: int) -> bool:
    if entry is None or not 0 <= m < MARKETPLACE_LIMIT:
        return False
    return (entry.epoch == captures_epoch and not entry.revoked and entry.expiry >= now
            and (entry.scope >> m) & 1 == 1)


def args_hash(types, args) -> bytes:
    return keccak256(abi.encode(list(types), list(args)))


def relay_selector(name: str) -> bytes:
    return abi.selector(RELAY_SIGS[name])


def relay_digest(chain_id: int, target: bytes, signer: bytes, nonce: int, selector: bytes, args_hash_: bytes) -> bytes:
    return keccak256(abi.encode(["string", "uint256", "address", "address", "uint256", "bytes4", "bytes32"],
                                [DOM_RELAY, chain_id, _b(target, 20, "target"), _b(signer, 20, "signer"), nonce,
                                 _b(selector, 4, "selector"), _b(args_hash_, 32, "argsHash")]))


def relay_digest_for(chain_id: int, target: bytes, signer: bytes, nonce: int, name: str, args) -> bytes:
    return relay_digest(chain_id, target, signer, nonce, relay_selector(name), args_hash(RELAY_ARG_TYPES[name], args))


# ------------------------------------------------------------------------------------------------ transcripts
def pass_transcript(target: bytes) -> bytes:
    return H(PFX_PASS + _b(target, 32, "target"))


def _sorted_rows(rows, key_len):
    rows = sorted(tuple(int(v) for v in r) for r in rows)
    keys = [r[:key_len] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate row key")
    return rows


def encode_rows_e(rows) -> bytes:
    return abi.encode([TE_T + "[]"], [_sorted_rows(rows, 4)])


def encode_rows_t(rows) -> bytes:
    rs = sorted(tuple(int(v) for v in r) for r in rows)
    if len({r[1:5] for r in rs}) != len(rs):         # key (b, pos, channel, statId); tableId is constant per claim
        raise ValueError("duplicate row key")
    rs = sorted(rs, key=lambda r: r[1:5])
    return abi.encode([TT_T + "[]"], [rs])


_TE_RANGES = ((0, 1 << 32), (0, 1 << 8), (0, 1 << 8), (0, 1 << 8), (-(1 << 15), 1 << 15), (-(1 << 255), 1 << 255))
_TT_RANGES = ((0, 1 << 16), (0, 1 << 8), (0, 1 << 32), (0, 1 << 8), (0, 1 << 8), (-(1 << 255), 1 << 255))


def _row_bytes(r, ranges) -> bytes:
    """abi.encode of one static 6-field tuple: six 32-byte two's-complement words (equal to eth_abi's)."""
    if len(r) != 6:
        raise ValueError("a row has 6 fields")
    out = []
    for v, (lo, hi) in zip(r, ranges):
        v = int(v)
        if not lo <= v < hi:
            raise ValueError(f"row field {v} outside its ABI type")
        out.append(v.to_bytes(32, "big", signed=True))
    return b"".join(out)


def _array_bytes(A, ranges) -> bytes:
    """The same words for an int64 array [n, 6] of rows, vectorised."""
    import numpy as np
    A = np.asarray(A, np.int64)
    for j, (lo, hi) in enumerate(ranges):
        col = A[:, j]
        if len(col) and ((col < max(lo, -(1 << 63))).any() or (hi <= (1 << 63) - 1 and (col >= hi).any())):
            raise ValueError("row field outside its ABI type")
    n = len(A)
    W = np.zeros((n, 6, 32), np.uint8)
    W[:, :, :24] = np.where(A < 0, 0xFF, 0).astype(np.uint8)[:, :, None]
    W[:, :, 24:] = A.astype(">i8").view(np.uint8).reshape(n, 6, 8)
    return W.tobytes()


def _wide_bytes(A5, W, ranges) -> bytes:
    """Words of rows whose first five fields are int64 (A5 [n, 5]) and whose witness W (object [n]) may exceed int64
    (|W| < 2^127: split W = hi·2^64 + lo exactly, sign-extend hi)."""
    import numpy as np
    n = len(A5)
    head = _array_bytes(np.concatenate([A5, np.zeros((n, 1), np.int64)], axis=1), ranges[:5] + ((0, 1),))
    hi = np.asarray((W >> 64).astype(np.int64))                  # OverflowError beyond |W| < 2^127
    lo = np.asarray((W & ((1 << 64) - 1)).astype(np.uint64))
    w6 = np.zeros((n, 32), np.uint8)
    w6[:, :16] = np.where(hi < 0, 0xFF, 0).astype(np.uint8)[:, None]
    w6[:, 16:24] = hi.astype(">i8").view(np.uint8).reshape(n, 8)
    w6[:, 24:] = lo.astype(">u8").view(np.uint8).reshape(n, 8)
    H = np.frombuffer(head, np.uint8).reshape(n, 6, 32).copy()
    H[:, 5, :] = w6
    return H.tobytes()


def _chunk_keys(ch, tier):
    import numpy as np
    if isinstance(ch, np.ndarray):
        K = ch[:, :4] if tier == "E" else ch[:, 1:5]
        return K
    if hasattr(ch, "A") and hasattr(ch, "W"):
        return ch.A[:, :4]
    return [tuple(int(v) for v in (r[:4] if tier == "E" else r[1:5])) for r in ch]


def stream_fail_transcript(chunks_factory, tier: str = "E") -> tuple:
    """(N, H(abi.encode(T[] rows))) **without materialising the rows**.
    ``chunks_factory()`` returns a fresh iterator of row chunks in transcript order (ascending, unique keys); a chunk
    is an int64 array [n, 6] (encoded vectorised) or a list of row tuples. It is called twice: once to count N, once to
    hash ``u256 0x20 ‖ u256 N ‖ N × 6 words`` (a static tuple array has no per-element offsets). Equal to
    ``fail_transcript_e/t`` on the list. Order and key uniqueness are checked."""
    import hashlib

    import numpy as np
    ranges = _TE_RANGES if tier == "E" else _TT_RANGES
    n = 0
    for ch in chunks_factory():
        n += len(ch)
    h = hashlib.sha256()
    h.update((0x20).to_bytes(32, "big") + n.to_bytes(32, "big"))
    prev = None
    m = 0
    for ch in chunks_factory():
        if not len(ch):
            continue
        K = _chunk_keys(ch, tier)
        wide = hasattr(ch, "A") and hasattr(ch, "W")
        if isinstance(ch, np.ndarray) or wide:
            if len(K) > 1:
                d = np.diff(K, axis=0)
                nz = d != 0
                first = np.where(nz.any(axis=1), d[np.arange(len(d)), nz.argmax(axis=1)], 0)
                if (first <= 0).any():
                    raise ValueError("rows are not strictly ascending by the transcript key")
            k0, k1 = tuple(int(v) for v in K[0]), tuple(int(v) for v in K[-1])
            if prev is not None and not prev < k0:
                raise ValueError("rows are not strictly ascending by the transcript key")
            if wide:
                try:
                    h.update(_wide_bytes(ch.A, ch.W, ranges))
                except OverflowError:                       # a witness beyond 2^127: per-row words
                    for r in ch.rows():
                        h.update(_row_bytes(r, ranges))
            else:
                h.update(_array_bytes(ch, ranges))
            prev = k1
        else:
            for k, r in zip(K, ch):
                if prev is not None and not prev < k:
                    raise ValueError("rows are not strictly ascending by the transcript key")
                prev = k
                h.update(_row_bytes(r, ranges))
        m += len(ch)
    if m != n:
        raise ValueError("the row iterator is not deterministic")
    return n, h.digest()


def fail_transcript_e(rows) -> bytes:
    return H(encode_rows_e(rows))


def fail_transcript_t(rows) -> bytes:
    return H(encode_rows_t(rows))


def is_rejected_fail_transcript(t: bytes) -> bool:
    return bytes(t) == EMPTY_ROW_TRANSCRIPT
