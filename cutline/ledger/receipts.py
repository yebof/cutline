"""cutline.ledger.receipts - status receipts: a QBFT-sealed header plus eth_getProof of the levelCap read set.

A receipt-only party (one that runs no node) gets a header with 2f+1 commit seals and Merkle-Patricia proofs of
**every storage slot levelCap reads** at that header (``layout.read_plan``), recomputes levelCap in Python
(``levelcap_from_fields``, the twin of ``ClaimRegistry.levelCap``) and shows only negative status from it (the
evaluator caps a receipt).

- ``ReceiptError(Exception)``: any verification failure other than freshness (bad header / hash, < quorum seals, a
  bad or missing proof, a wrong codeHash, a malformed word). A client exception, not a protocol ``E.*`` name; the
  party treats the status as unknown (at most G2).
- ``mpt_get(root, key, proof) -> bytes | None``: Merkle-Patricia (hexary, keys as given) lookup of ``key`` under
  ``root`` using the RLP nodes of ``proof`` (a node is used only through its keccak256 reference, so extra nodes are
  harmless); returns the stored value, or None when the proof shows the key is absent; ReceiptError when the proof
  is incomplete or malformed. ``verify_account(state_root, address, proof)`` -> ``Account | None``;
  ``verify_storage(storage_root, slot, proof) -> int`` (an absent key is 0).
- ``Receipt(claim_id, header_rlp, block_hash, account_proofs, storage_proofs, table_versions)``: ``account_proofs``
  = {address20: [node, ...]}, ``storage_proofs`` = {address20: {slot int: [node, ...]}}; ``table_versions`` records
  the layout / ABI build and the preimage modes it was built for. ``to_json()`` / ``Receipt.from_json(d)``.
- ``build_receipt(rpc, claim_id, block, addresses=ADDRESSES) -> Receipt`` (rounds of ``eth_getProof``: the claim
  slots first, then the slots that depend on them).
- ``verify_receipt(r, pinned_validators, heads, now, k=30, t_wall=300, code_hashes=None, hash_mode=..., seal_mode=...,
  now_offset=0) -> ClaimStatus`` checks, in order: (1) the header decodes and its on-chain hash equals the
  receipt's ``block_hash`` (when present); (2) >= quorum distinct seals of ``pinned_validators`` (the pinned genesis
  validator set); (3) freshness: the reference head = the (f+1)-th highest of ``heads`` (``rpc.Head`` or ints),
  header.number >= reference − k, |header.timestamp − now| <= t_wall and, when the reports carry timestamps,
  |reference.timestamp − now| <= t_wall; (4) each account proof against ``stateRoot`` with codeHash = the pinned v1
  runtime hash (``layout.CODE_KECCAK_V1`` unless ``code_hashes`` is given); (5) each storage proof against the
  account's ``storageRoot``; (6) the fields decode by ``layout`` and levelCap is recomputed at now = header.number +
  ``now_offset``. A freshness failure returns ``receipt="stale"``; everything else failing raises ReceiptError. An
  unknown claim returns ``ClaimStatus.MISSING``.
- ``levelcap_from_fields(fields, now) -> (level_cap, cap, claim)`` and ``principal_of(fields, now)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import rlp

from ..crypto import keccak256
from ..verify.ledger_view import ClaimStatus
from . import layout as L
from . import qbft
from .rpc import Head, hx, reference_head, to_int, unhx

K_BLOCKS = 30
T_WALL_S = 300
EMPTY_TRIE_ROOT = keccak256(rlp.encode(b""))
EMPTY_CODE_HASH = keccak256(b"")
STATUS_WITHDRAWN = 3
DEV_ACTIVE = 2
KIND_OWN, KIND_LIC = 0, 1


class ReceiptError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ Merkle-Patricia
def _nibbles(b: bytes) -> list:
    out = []
    for x in b:
        out += [x >> 4, x & 15]
    return out


def _hp_decode(path: bytes):
    if not path:
        raise ReceiptError("empty hex-prefix path")
    flag = path[0] >> 4
    if flag > 3:
        raise ReceiptError("bad hex-prefix flag")
    nib = _nibbles(path)
    nib = nib[1:] if flag & 1 else nib[2:]
    if not flag & 1 and path[0] & 15:
        raise ReceiptError("bad hex-prefix padding")
    return nib, bool(flag & 2)


def mpt_get(root: bytes, key: bytes, proof) -> bytes | None:
    root = bytes(root)
    db = {}
    for n in proof:
        n = bytes(n)
        db[keccak256(n)] = n
    if root == EMPTY_TRIE_ROOT:
        return None
    want = _nibbles(bytes(key))
    ref = root
    for _ in range(2 * len(want) + 3):
        if isinstance(ref, bytes):
            if len(ref) == 32:
                raw = db.get(ref)
                if raw is None:
                    raise ReceiptError(f"proof node {ref.hex()[:16]}.. missing")
                try:
                    node = rlp.decode(raw)
                except Exception:
                    raise ReceiptError("proof node is not RLP") from None
            elif ref == b"":
                return None
            else:
                raise ReceiptError("a node reference must be a 32-byte hash or an embedded node")
        else:
            node = ref
        if not isinstance(node, list):
            raise ReceiptError("a trie node is an RLP list")
        if len(node) == 17:
            if not want:
                v = node[16]
                if not isinstance(v, bytes):
                    raise ReceiptError("branch value must be a byte string")
                return v or None
            ref, want = node[want[0]], want[1:]
            if ref == b"":
                return None
            continue
        if len(node) == 2:
            if not isinstance(node[0], bytes):
                raise ReceiptError("leaf / extension path must be a byte string")
            path, leaf = _hp_decode(node[0])
            if leaf:
                if not isinstance(node[1], bytes):
                    raise ReceiptError("leaf value must be a byte string")
                return node[1] if path == want else None
            if want[:len(path)] != path:
                return None
            if not path:
                raise ReceiptError("empty extension")
            ref, want = node[1], want[len(path):]
            continue
        raise ReceiptError(f"a trie node has 2 or 17 items, not {len(node)}")
    raise ReceiptError("proof too deep")


@dataclass(frozen=True)
class Account:
    nonce: int
    balance: int
    storage_root: bytes
    code_hash: bytes


def verify_account(state_root: bytes, address: bytes, proof) -> Account | None:
    v = mpt_get(state_root, keccak256(bytes(address)), proof)
    if v is None:
        return None
    try:
        a = rlp.decode(v)
    except Exception:
        raise ReceiptError("account value is not RLP") from None
    if not isinstance(a, list) or len(a) != 4 or not all(isinstance(x, bytes) for x in a) \
            or len(a[2]) != 32 or len(a[3]) != 32:
        raise ReceiptError("account = RLP[nonce, balance, storageRoot, codeHash]")
    return Account(int.from_bytes(a[0], "big"), int.from_bytes(a[1], "big"), a[2], a[3])


def verify_storage(storage_root: bytes, slot: int, proof) -> int:
    v = mpt_get(storage_root, keccak256(int(slot).to_bytes(32, "big")), proof)
    if v is None:
        return 0
    try:
        w = rlp.decode(v)
    except Exception:
        raise ReceiptError("storage value is not RLP") from None
    if not isinstance(w, bytes) or len(w) > 32 or (w and w[0] == 0) or not w:
        raise ReceiptError("storage value must be a minimal non-zero RLP integer")
    return int.from_bytes(w, "big")


# ------------------------------------------------------------------------------------------------ levelCap
def principal_of(fields: dict, now: int):
    """principal(cl.nf, cl.claimant, cl.marketplace) at ``now``: (kind, id), or None (it reverts)."""
    cl, cp, x = fields["claims"], fields["captures"], fields["exclusiveOf"]
    who, m = cl["claimant"], cl["marketplace"]
    if x["licensee"] != L.ZERO20 and x["expiry"] >= now:
        return (KIND_LIC, x["licId"]) if who == x["licensee"] else None
    if who == cp["owner"]:
        return (KIND_OWN, cp["epoch"])
    d = fields.get("delegations")
    if d is None:
        raise ReceiptError("the delegation entry is needed for principal() but was not proven")
    if d["epoch"] == cp["epoch"] and d["expiry"] != 0 and not d["revoked"] and d["expiry"] >= now \
            and (d["scope"] >> m) & 1:
        return (KIND_OWN, cp["epoch"])
    return None


def levelcap_from_fields(fields: dict, now: int) -> tuple:
    """(level_cap, cap, claim) from the decoded read set; the Python twin of ``ClaimRegistry.levelCap``
    (plus the evidence record's ``cap`` / ``claim`` fields). ``fields['claims']['claimant'] == 0`` means an
    unknown claim: (0, None, None)."""
    cl = fields["claims"]
    if cl["claimant"] == L.ZERO20:
        return 0, None, None
    cp, b, d, st = fields["captures"], fields["_batches"], fields["devices"], fields["strikes"]
    valid = d["state"] == DEV_ACTIVE and (d["tc"] == 0 or b["anchorHeight"] < d["tc"])
    cap = "invalid" if not valid else "time-unverified" if cp["timeUnverified"] else "valid"
    p = principal_of(fields, now)
    claim = "authorised" if p == (cl["kind"], cl["id"]) else "none"
    if not valid:
        return 0, cap, claim
    if st["courtStruck"] or st["quorumStrikes"] > 0 or cl["courtFailed"] or cl["quorumFailed"] \
            or cl["status"] == STATUS_WITHDRAWN:
        return 1, cap, claim
    if claim != "authorised":
        return 1, cap, claim
    return (2 if cp["timeUnverified"] else 4), cap, claim


# ------------------------------------------------------------------------------------------------ receipts
@dataclass
class Receipt:
    claim_id: bytes
    header_rlp: bytes
    block_hash: bytes | None
    account_proofs: dict = field(default_factory=dict)     # addr20 -> [node]
    storage_proofs: dict = field(default_factory=dict)     # addr20 -> {slot: [node]}
    table_versions: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"claim_id": hx(self.claim_id), "header_rlp": hx(self.header_rlp),
                "block_hash": None if self.block_hash is None else hx(self.block_hash),
                "account_proofs": {hx(a): [hx(n) for n in p] for a, p in self.account_proofs.items()},
                "storage_proofs": {hx(a): {hx(s.to_bytes(32, "big")): [hx(n) for n in p] for s, p in m.items()}
                                   for a, m in self.storage_proofs.items()},
                "table_versions": dict(self.table_versions)}

    @classmethod
    def from_json(cls, d: dict) -> "Receipt":
        return cls(unhx(d["claim_id"]), unhx(d["header_rlp"]),
                   None if d.get("block_hash") is None else unhx(d["block_hash"]),
                   {unhx(a): [unhx(n) for n in p] for a, p in d["account_proofs"].items()},
                   {unhx(a): {to_int(s): [unhx(n) for n in p] for s, p in m.items()}
                    for a, m in d["storage_proofs"].items()},
                   dict(d.get("table_versions", {})))


def table_versions(hash_mode=qbft.HASH_MODE, seal_mode=qbft.SEAL_MODE) -> dict:
    return {"layout": L.LAYOUT["version"], "buildInfoInputSha256": L.LAYOUT["buildInfoInputSha256"],
            "hash_mode": hash_mode, "seal_mode": seal_mode}


def _decode_known(claim_id, words_by_contract, now):
    """Decode every planned entry whose slots are all known; returns (fields, plan)."""
    fields: dict = {}
    for _ in range(4):
        plan = L.read_plan(claim_id, fields, now)
        new = {}
        for var, base in plan.items():
            words = words_by_contract.get(L.CONTRACT_OF[var], {})
            if all(s in words for s in L.struct_slots(var, base)):
                new[var] = L.decode_struct(var, base, words)
        if new.keys() == fields.keys():
            return fields, plan
        fields = new
    return fields, L.read_plan(claim_id, fields, now)


def build_receipt(rpc, claim_id: bytes, block, addresses=None, hash_mode=qbft.HASH_MODE,
                  seal_mode=qbft.SEAL_MODE, now_offset: int = 0) -> Receipt:
    addresses = L.ADDRESSES if addresses is None else addresses
    b = rpc.get_block(block)
    if b is None:
        raise ReceiptError(f"no block {block}")
    header = qbft.Header.from_json(b)
    h = header.number
    now = h + now_offset
    acc, sto, words = {}, {}, {}
    for _ in range(4):
        _, plan = _decode_known(claim_id, words, now)
        need: dict = {}
        for var, base in plan.items():
            c = L.CONTRACT_OF[var]
            for s in L.struct_slots(var, base):
                if s not in words.get(c, {}):
                    need.setdefault(c, []).append(s)
        if not need:
            break
        for c, slots in need.items():
            a = addresses[c]
            p = rpc.get_proof(a, sorted(set(slots)), h)
            acc[a] = [unhx(n) for n in p["accountProof"]]
            for sp in p["storageProof"]:
                s = to_int(sp["key"])
                sto.setdefault(a, {})[s] = [unhx(n) for n in sp["proof"]]
                words.setdefault(c, {})[s] = to_int(sp["value"])
    return Receipt(bytes(claim_id), header.rlp(), unhx(b["hash"]), acc, sto, table_versions(hash_mode, seal_mode))


def _as_head(x):
    if x is None or isinstance(x, Head):
        return x
    if isinstance(x, int):
        return Head("", x, b"", None)
    if isinstance(x, (tuple, list)):
        return Head("", int(x[0]), b"", None if len(x) < 2 else x[1])
    if isinstance(x, dict):
        return Head(x.get("url", ""), to_int(x["number"]), b"", None if x.get("timestamp") is None
                    else to_int(x["timestamp"]))
    raise TypeError(f"head report {x!r}")


def freshness(header: qbft.Header, n_validators: int, heads, now: float, k: int = K_BLOCKS,
              t_wall: int = T_WALL_S) -> tuple:
    """(fresh: bool, reason: str) of verification step 3."""
    f = (int(n_validators) - 1) // 3
    ref = reference_head([_as_head(x) for x in heads], f)
    if ref is None:
        return False, f"fewer than f+1 = {f + 1} head reports"
    if ref.timestamp is not None and abs(ref.timestamp - now) > t_wall:
        return False, f"reference head timestamp {ref.timestamp} off the clock {now:.0f} by > {t_wall} s"
    if header.number < ref.number - k:
        return False, f"header {header.number} < reference head {ref.number} - {k}"
    if abs(header.timestamp - now) > t_wall:
        return False, f"header timestamp {header.timestamp} off the clock {now:.0f} by > {t_wall} s"
    return True, "fresh"


def verify_receipt(r: Receipt, pinned_validators, heads, now: float, k: int = K_BLOCKS, t_wall: int = T_WALL_S,
                   code_hashes: dict | None = None, hash_mode: str = qbft.HASH_MODE, seal_mode: str = qbft.SEAL_MODE,
                   addresses: dict | None = None, now_offset: int = 0) -> ClaimStatus:
    addresses = L.ADDRESSES if addresses is None else addresses
    code_hashes = L.CODE_KECCAK_V1 if code_hashes is None else code_hashes
    # (1) header and hash
    try:
        header = qbft.Header.from_rlp(r.header_rlp)
        hsh = qbft.header_hash(header, hash_mode)
    except (qbft.HeaderError, ValueError) as e:
        raise ReceiptError(f"header: {e}") from None
    if r.block_hash is not None and hsh != r.block_hash:
        raise ReceiptError(f"header hash {hsh.hex()} != receipt block hash {r.block_hash.hex()}")
    # (2) 2f+1 seals of the pinned set
    try:
        qbft.verify_seals(header, pinned_validators, seal_mode, hash_mode)
    except (qbft.SealError, qbft.HeaderError) as e:
        raise ReceiptError(f"seals: {e}") from None
    # (3) freshness (a failure is "stale", not an error)
    fresh, _why = freshness(header, len(set(bytes(v) for v in pinned_validators)), heads, now, k, t_wall)
    # (4) + (5) proofs, iterating the read plan over verified words only
    now_b = header.number + now_offset
    accounts: dict = {}
    words: dict = {}

    def account(c):
        if c not in accounts:
            a = addresses[c]
            if a not in r.account_proofs:
                raise ReceiptError(f"no account proof for {c}")
            acct = verify_account(header.state_root, a, r.account_proofs[a])
            if acct is None:
                raise ReceiptError(f"{c} does not exist at this state root")
            want = code_hashes.get(c)
            if want is None:
                raise ReceiptError(f"no pinned code hash for {c}")
            if acct.code_hash != bytes(want):
                raise ReceiptError(f"{c} codeHash {acct.code_hash.hex()} is not the pinned v1 runtime")
            accounts[c] = acct
        return accounts[c]

    fields: dict = {}
    for _ in range(5):
        plan = L.read_plan(r.claim_id, fields, now_b)
        for var, base in plan.items():
            c = L.CONTRACT_OF[var]
            acct = account(c)
            a = addresses[c]
            for s in L.struct_slots(var, base):
                if s in words.setdefault(c, {}):
                    continue
                p = r.storage_proofs.get(a, {}).get(s)
                if p is None:
                    raise ReceiptError(f"no storage proof for {c}.{var} slot {s:#x}")
                words[c][s] = verify_storage(acct.storage_root, s, p)
        try:
            new = {var: L.decode_struct(var, base, words[L.CONTRACT_OF[var]]) for var, base in plan.items()}
        except ValueError as e:
            raise ReceiptError(str(e)) from None
        if new.keys() == fields.keys() and new == fields:
            break
        fields = new
    else:
        raise ReceiptError("read plan did not converge")
    # (6) levelCap
    lc, cap, claim = levelcap_from_fields(fields, now_b)
    if cap is None:
        return ClaimStatus.MISSING
    return ClaimStatus(True, lc, cap, claim, "fresh" if fresh else "stale")


__all__ = ["ReceiptError", "Receipt", "Account", "mpt_get", "verify_account", "verify_storage", "build_receipt",
           "verify_receipt", "freshness", "levelcap_from_fields", "principal_of", "table_versions", "K_BLOCKS",
           "T_WALL_S", "EMPTY_TRIE_ROOT", "EMPTY_CODE_HASH"]
