"""cutline.ledger.qbft - block headers, the QBFT extraData codec, block hashes and commit seals.

A status receipt carries one block header with 2f+1 QBFT commit seals. This module encodes and decodes the header,
the QBFT ``extraData`` = RLP[vanity32, validators, vote, round, seals] and recovers the seals.

**Preimages (determined empirically).** Besu's QBFT hashes a header whose ``extraData`` is re-encoded without the
commit seals (and, for the on-chain block hash, without the actual round number). The exact re-encoding is not
documented, so this module implements every plausible candidate (``EXTRA_MODES``) and
``determine_preimage(blocks, validators)`` finds the candidate that reproduces ``eth_getBlockByNumber.hash`` and whose
seals recover to the validator set, on real blocks. ``HASH_MODE`` / ``SEAL_MODE`` hold the adopted pair, determined
on Besu 26.8.1 (346 blocks of an n = 4, 2 s block-period devnet, 335 at round 0 and 11 at round 1, unique on both
counts):
  - on-chain block hash: ``round0_empty`` = extraData RLP[vanity, validators, vote, round 0 (0x80), seals []];
  - commit-seal digest: ``no_seals_empty`` = RLP[vanity, validators, vote, actual round, seals []] (equal to the
    block hash at round 0; different at round >= 1, which is what separates the candidates).
``PREIMAGE_STATUS`` records this provenance. Every function takes an explicit ``mode`` so the determination and the
tests can override them.

- ``Header``: the raw RLP items of a header (``items``, bytes each) with properties ``number``, ``timestamp``,
  ``state_root``, ``parent_hash``, ``extra_data``, ``gas_limit``, ``field(name)``; ``Header.from_json(block)`` (an
  ``eth_getBlockByNumber`` result; optional post-London fields are included iff present, in the Osaka order
  ``HEADER_FIELDS``), ``Header.from_rlp(b)``, ``.rlp()``, ``.eth_hash()`` = keccak(rlp(header)) (plain Ethereum).
- ``QbftExtra(vanity, validators, vote, round_raw, seals)``: ``decode_extra(b)``; ``encode_extra(x, mode)`` with
  mode in ``EXTRA_MODES`` = ``full`` (as is), ``no_seals`` (seals omitted, round kept), ``no_seals_empty`` (an empty
  seal list, round kept), ``no_round`` (seals and round omitted), ``round0`` (seals omitted, round 0),
  ``round0_empty`` (round 0 and an empty seal list; Besu's on-chain hash).
- ``header_hash(h, mode=HASH_MODE)``: the on-chain hash (``mode="full"`` is plain Ethereum, e.g. EDR).
- ``seal_digest(h, mode=SEAL_MODE)``: the digest each validator signs; ``SEAL_MODE="hash"`` means the on-chain hash.
- ``recover_seals(h, mode=SEAL_MODE, hash_mode=HASH_MODE) -> [address | None]`` (65-byte r || s || v, v in
  {0, 1, 27, 28}; low-s is not required, Besu signs low-s).
- ``quorum(n) = ceil(2n / 3)`` (= 2f+1 for n = 3f+1); ``verify_seals(h, validators, ...) -> set`` of distinct pinned
  signers; raises ``SealError`` below quorum (duplicates and foreign signers do not count).
- ``make_extra`` / ``seal_header`` build synthetic QBFT headers (tests and the fork / beacon tools).
- ``determine_preimage(blocks_json, validators) -> dict``: per candidate, how many blocks it reproduces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import rlp

from ..crypto import keccak256

# Osaka header field order (execution-specs); (json key, kind) with kind "b" bytes, "i" integer.
HEADER_FIELDS = (
    ("parentHash", "b"), ("sha3Uncles", "b"), ("miner", "b"), ("stateRoot", "b"), ("transactionsRoot", "b"),
    ("receiptsRoot", "b"), ("logsBloom", "b"), ("difficulty", "i"), ("number", "i"), ("gasLimit", "i"),
    ("gasUsed", "i"), ("timestamp", "i"), ("extraData", "b"), ("mixHash", "b"), ("nonce", "b"),
    ("baseFeePerGas", "i"), ("withdrawalsRoot", "b"), ("blobGasUsed", "i"), ("excessBlobGas", "i"),
    ("parentBeaconBlockRoot", "b"), ("requestsHash", "b"),
)
_IDX = {k: i for i, (k, _) in enumerate(HEADER_FIELDS)}
N_LEGACY = 15

EXTRA_MODES = ("full", "no_seals", "no_seals_empty", "no_round", "round0", "round0_empty")
SEAL_MODES = ("no_seals", "no_seals_empty", "no_round", "round0", "round0_empty", "hash")
HASH_MODE = "round0_empty"      # Besu 26.8.1, 346 blocks
SEAL_MODE = "no_seals_empty"    # Besu 26.8.1, 346 blocks incl. 11 at round 1
PREIMAGE_STATUS = "determined empirically on Besu 26.8.1 (346 blocks, rounds 0 and 1)"


class HeaderError(ValueError):
    pass


class SealError(ValueError):
    pass


def _int_bytes(v: int) -> bytes:
    v = int(v)
    if v < 0:
        raise HeaderError("negative header integer")
    return v.to_bytes((v.bit_length() + 7) // 8, "big") if v else b""


def _hexb(s) -> bytes:
    s = str(s)
    s = s[2:] if s[:2] in ("0x", "0X") else s
    return bytes.fromhex(s if len(s) % 2 == 0 else "0" + s)


@dataclass(frozen=True)
class Header:
    items: tuple            # raw RLP items (bytes), in HEADER_FIELDS order

    def __post_init__(self):
        if not N_LEGACY <= len(self.items) <= len(HEADER_FIELDS) or not all(isinstance(x, bytes) for x in self.items):
            raise HeaderError(f"a header has {N_LEGACY}..{len(HEADER_FIELDS)} byte-string fields")
        for (k, kind), x in zip(HEADER_FIELDS, self.items):
            if kind == "i" and x[:1] == b"\x00":
                raise HeaderError(f"non-minimal integer in header field {k}")

    @classmethod
    def from_json(cls, b: dict) -> "Header":
        items = []
        for i, (k, kind) in enumerate(HEADER_FIELDS):
            v = b.get(k)
            if v is None:
                if i < N_LEGACY:
                    raise HeaderError(f"missing header field {k}")
                break
            items.append(_int_bytes(int(v, 16)) if kind == "i" else _hexb(v))
        # an optional field after a missing one is an inconsistent header
        n = len(items)
        for k, _ in HEADER_FIELDS[n:]:
            if b.get(k) is not None:
                raise HeaderError(f"header field {k} present after a missing one")
        return cls(tuple(items))

    @classmethod
    def from_rlp(cls, data: bytes) -> "Header":
        try:
            items = rlp.decode(bytes(data))
        except Exception as e:                       # rlp raises DecodingError / DeserializationError
            raise HeaderError(f"header RLP: {e}") from None
        if not isinstance(items, list) or not all(isinstance(x, bytes) for x in items):
            raise HeaderError("a header is a flat RLP list of byte strings")
        h = cls(tuple(items))
        if h.rlp() != bytes(data):
            raise HeaderError("non-canonical header RLP")
        return h

    def rlp(self, extra: bytes | None = None) -> bytes:
        items = list(self.items)
        if extra is not None:
            items[_IDX["extraData"]] = bytes(extra)
        return rlp.encode(items)

    def eth_hash(self) -> bytes:
        return keccak256(self.rlp())

    def field(self, name: str) -> bytes | None:
        i = _IDX[name]
        return self.items[i] if i < len(self.items) else None

    def _int(self, name: str) -> int:
        return int.from_bytes(self.field(name) or b"", "big")

    @property
    def number(self) -> int:
        return self._int("number")

    @property
    def timestamp(self) -> int:
        return self._int("timestamp")

    @property
    def gas_limit(self) -> int:
        return self._int("gasLimit")

    @property
    def state_root(self) -> bytes:
        return self.field("stateRoot")

    @property
    def parent_hash(self) -> bytes:
        return self.field("parentHash")

    @property
    def extra_data(self) -> bytes:
        return self.field("extraData")


# ------------------------------------------------------------------------------------------------ QBFT extraData
@dataclass(frozen=True)
class QbftExtra:
    vanity: bytes
    validators: tuple                 # 20-byte addresses
    vote: tuple = ()                  # () or (recipient20, vote_type_bytes)
    round_raw: bytes = b""            # the round as encoded (RLP int scalar in QBFT)
    seals: tuple = field(default_factory=tuple)

    @property
    def round(self) -> int:
        return int.from_bytes(self.round_raw, "big")


def decode_extra(data: bytes) -> QbftExtra:
    try:
        x = rlp.decode(bytes(data))
    except Exception as e:
        raise HeaderError(f"extraData RLP: {e}") from None
    if not isinstance(x, list) or len(x) != 5:
        raise HeaderError("QBFT extraData is RLP[vanity, validators, vote, round, seals]")
    vanity, vals, vote, rnd, seals = x
    if not isinstance(vanity, bytes) or len(vanity) != 32:
        raise HeaderError("vanity must be 32 bytes")
    if not isinstance(vals, list) or not all(isinstance(v, bytes) and len(v) == 20 for v in vals):
        raise HeaderError("validators must be 20-byte addresses")
    if not isinstance(vote, list) or len(vote) not in (0, 2) or not all(isinstance(v, bytes) for v in vote):
        raise HeaderError("vote must be [] or [recipient, type]")
    if not isinstance(rnd, bytes) or len(rnd) > 4:
        raise HeaderError("round must be an int scalar of <= 4 bytes")
    if not isinstance(seals, list) or not all(isinstance(s, bytes) and len(s) == 65 for s in seals):
        raise HeaderError("seals must be 65-byte signatures")
    return QbftExtra(vanity, tuple(vals), tuple(vote), rnd, tuple(seals))


def encode_extra(x: QbftExtra, mode: str = "full") -> bytes:
    base = [x.vanity, list(x.validators), list(x.vote)]
    if mode == "full":
        return rlp.encode(base + [x.round_raw, list(x.seals)])
    if mode == "no_seals":
        return rlp.encode(base + [x.round_raw])
    if mode == "no_seals_empty":
        return rlp.encode(base + [x.round_raw, []])
    if mode == "no_round":
        return rlp.encode(base)
    if mode == "round0":
        return rlp.encode(base + [b""])
    if mode == "round0_empty":
        return rlp.encode(base + [b"", []])
    raise ValueError(f"unknown extraData mode {mode}")


def header_hash(h: Header, mode: str = HASH_MODE) -> bytes:
    """The on-chain block hash. ``mode="full"`` is plain Ethereum keccak(rlp(header))."""
    if mode == "full":
        return h.eth_hash()
    return keccak256(h.rlp(encode_extra(decode_extra(h.extra_data), mode)))


def seal_digest(h: Header, mode: str = SEAL_MODE, hash_mode: str = HASH_MODE) -> bytes:
    if mode == "hash":
        return header_hash(h, hash_mode)
    if mode == "full":
        raise ValueError("a seal cannot sign over itself")
    return keccak256(h.rlp(encode_extra(decode_extra(h.extra_data), mode)))


def _recover(digest: bytes, sig: bytes) -> bytes | None:
    from eth_keys import keys
    from eth_keys.exceptions import BadSignature, ValidationError
    if len(sig) != 65:
        return None
    v = sig[64]
    v = v - 27 if v in (27, 28) else v
    if v not in (0, 1):
        return None
    r, s = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:64], "big")
    try:
        pub = keys.Signature(vrs=(v, r, s)).recover_public_key_from_msg_hash(digest)
    except (BadSignature, ValidationError, ValueError):
        return None
    return pub.to_canonical_address()


def recover_seals(h: Header, mode: str = SEAL_MODE, hash_mode: str = HASH_MODE) -> list:
    d = seal_digest(h, mode, hash_mode)
    return [_recover(d, s) for s in decode_extra(h.extra_data).seals]


def quorum(n: int) -> int:
    """QBFT quorum ceil(2n/3); 2f+1 for n = 3f+1 (n = 4: 3, n = 7: 5)."""
    return -(-2 * int(n) // 3)


def verify_seals(h: Header, validators, mode: str = SEAL_MODE, hash_mode: str = HASH_MODE) -> set:
    """The distinct pinned validators whose commit seals are valid; SealError below ``quorum(len(validators))``."""
    pinned = {bytes(v) for v in validators}
    signers = {a for a in recover_seals(h, mode, hash_mode) if a is not None and a in pinned}
    need = quorum(len(pinned))
    if len(signers) < need:
        raise SealError(f"{len(signers)} valid distinct pinned seals < quorum {need}")
    return signers


# ------------------------------------------------------------------------------------------------ synthesis (tests)
def make_extra(validators, round_: int = 0, vanity: bytes = b"\x00" * 32, vote: tuple = ()) -> QbftExtra:
    return QbftExtra(bytes(vanity), tuple(bytes(v) for v in validators), tuple(vote), _int_bytes(round_), ())


def seal_header(h: Header, extra: QbftExtra, signer_keys, mode: str = SEAL_MODE, hash_mode: str = HASH_MODE,
                v_offset: int = 0) -> Header:
    """``h`` with extraData = ``extra`` plus the commit seals of ``signer_keys`` (ints) over ``seal_digest``."""
    from eth_keys import keys
    unsealed = Header(tuple(list(h.items[:_IDX["extraData"]]) + [encode_extra(extra, "full")]
                            + list(h.items[_IDX["extraData"] + 1:])))
    d = seal_digest(unsealed, mode, hash_mode)
    seals = []
    for sk in signer_keys:
        sig = keys.PrivateKey(int(sk).to_bytes(32, "big")).sign_msg_hash(d)
        seals.append(sig.r.to_bytes(32, "big") + sig.s.to_bytes(32, "big") + bytes([sig.v + v_offset]))
    full = QbftExtra(extra.vanity, extra.validators, extra.vote, extra.round_raw, tuple(seals))
    return Header(tuple(list(h.items[:_IDX["extraData"]]) + [encode_extra(full, "full")]
                        + list(h.items[_IDX["extraData"] + 1:])))


# ------------------------------------------------------------------------------------------------ determination
def determine_preimage(blocks, validators=None) -> dict:
    """For each hash candidate: the number of ``blocks`` (eth_getBlockByNumber results) whose ``hash`` it
    reproduces; for each seal candidate: the number of blocks on which every seal recovers to a member of
    ``validators`` (or of the block's own extraData validator list when None) and the quorum is met. Also the round
    histogram. The adopted pair must score len(blocks) on both, over rounds 0 and >= 1."""
    out = {"blocks": 0, "rounds": {}, "hash": {m: 0 for m in EXTRA_MODES}, "seal": {m: 0 for m in SEAL_MODES},
           "decode_errors": 0}
    for b in blocks:
        out["blocks"] += 1
        h = Header.from_json(b)
        want = _hexb(b["hash"])
        try:
            x = decode_extra(h.extra_data)
        except HeaderError:
            out["decode_errors"] += 1
            continue
        out["rounds"][x.round] = out["rounds"].get(x.round, 0) + 1
        good_hash = None
        for m in EXTRA_MODES:
            if header_hash(h, m) == want:
                out["hash"][m] += 1
                good_hash = good_hash or m
        vals = validators if validators is not None else x.validators
        for m in SEAL_MODES:
            if m == "hash" and good_hash is None:
                continue
            try:
                ok = all(a is not None and a in {bytes(v) for v in vals}
                         for a in recover_seals(h, m, good_hash or HASH_MODE))
                verify_seals(h, vals, m, good_hash or HASH_MODE)
            except SealError:
                ok = False
            if ok:
                out["seal"][m] += 1
    return out


__all__ = ["HEADER_FIELDS", "EXTRA_MODES", "SEAL_MODES", "HASH_MODE", "SEAL_MODE", "PREIMAGE_STATUS", "Header",
           "HeaderError", "SealError", "QbftExtra", "decode_extra", "encode_extra", "header_hash", "seal_digest",
           "recover_seals", "quorum", "verify_seals", "make_extra", "seal_header", "determine_preimage"]
