"""sidecar.chain - the one ledger boundary of the sidecar.

Every duty talks to the chain only through a ``ChainPort``; ``sidecar.rpcport.RpcPort`` implements it over JSON-RPC
to one Besu node (reads with ``cutline.abi`` against the frozen ABI export; transactions through
``cutline.ledger.relay.TxSender``). Any object with the same interface (e.g. an in-memory mock) can stand in.

Interface:

- ``Event(name, block, log_index, args, tx)``: one decoded log of ``EVENTS`` (``args`` keyed by the ABI input names;
  bytes32 / address as ``bytes``, integers as ``int``, strings as ``str``).
- ``Receipt(tx, block, ok, error, gas_used)``: ``error`` is the ``E.*`` name of a revert (``None`` when ``ok``).
- ``Revert(name)``: raised by ``view`` for a reverting read.
- ``ChainPort`` (``typing.Protocol``):
  - ``address: bytes`` (20 B, the sender of ``send``);
  - ``head() -> int``; ``block_hash(h) -> bytes | None``; ``parent_hash(h) -> bytes | None``;
  - ``logs(frm, to) -> list[Event]`` (inclusive range, ascending (block, log_index); only the events of ``EVENTS``);
  - ``view(fn, args, block=None) -> tuple`` (a view function of ``FUNCS`` by name; ``block`` None = latest);
  - ``simulate(fn, args, calldata=None) -> str | None``: an ``eth_call`` pre-check of a transaction at the latest
    block: None when it would succeed, else its ``E.*`` name;
  - ``send(fn, args, calldata=None) -> bytes``: submits a transaction, returns its hash (does not wait);
  - ``receipt(tx) -> Receipt | None`` (None while pending or unknown);
  - ``ledger() -> cutline.verify.ledger_view.LedgerView`` (tier-T tables, kfam, classes, claims at a height).

``FUNCS`` maps each function name the sidecar uses to ``(contract, canonical signature, output types)``; ``EVENTS``
maps each event name to ``(contract, canonical signature, [(name, type, indexed)])``. Both must match the ABI
export.

Codes (Types.sol ``Codes``): ``PASS = 1``, ``FAIL = 2``; ``KIND_CHALLENGE = 1``, ``KIND_APPEAL = 2``;
``OUTCOME_FAILED = 1``, ``OUTCOME_PASSED = 2``, ``OUTCOME_UNDECIDED = 3``; ``RVOTE_REVOKE = 1``, ``RVOTE_KEEP = 2``;
``DEV_UNKNOWN = 0``, ``DEV_PENDING = 1``, ``DEV_ACTIVE = 2``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

W_VOTE = 150            # vote window in blocks (Consts.W_VOTE)
BEACON_EPOCH = 15       # beacon epoch in blocks (Consts.BEACON_EPOCH)

PASS, FAIL = 1, 2
KIND_CHALLENGE, KIND_APPEAL = 1, 2
OUTCOME_FAILED, OUTCOME_PASSED, OUTCOME_UNDECIDED = 1, 2, 3
RVOTE_REVOKE, RVOTE_KEEP = 1, 2
DEV_UNKNOWN, DEV_PENDING, DEV_ACTIVE = 0, 1, 2

CORE_T = "(bytes32,bytes32,bytes32,bytes32,uint16,uint8,uint8,uint8,bytes32)"

FUNCS = {
    # Adjudicator
    "submitVerdict": ("Adjudicator", "submitVerdict(uint64,uint8,bytes32)", []),
    "finalizeChallenge": ("Adjudicator", "finalizeChallenge(uint64)", []),
    "finalizeAppeal": ("Adjudicator", "finalizeAppeal(uint64)", []),
    "openAppeal": ("Adjudicator", "openAppeal(bytes32)", ["uint64"]),
    "voted": ("Adjudicator", "voted(uint64,address)", ["bool"]),
    "challenges": ("Adjudicator", "challenges(uint64)", ["uint8", "bytes32", "bytes32", "uint64", "bool"]),
    "challengeCount": ("Adjudicator", "challengeCount()", ["uint64"]),
    "passed": ("Adjudicator", "passed(bytes32)", ["bool"]),
    # CaptureAnchor (beacon pulse)
    "pulse": ("CaptureAnchor", "pulse(bytes32)", []),
    # DeviceRegistry (enrolment, revocation)
    "confirm": ("DeviceRegistry", "confirm(bytes32)", []),
    "voteRevocation": ("DeviceRegistry", "voteRevocation(bytes32,uint8,bytes32)", []),
    "finalizeRevocation": ("DeviceRegistry", "finalizeRevocation(bytes32)", []),
    "devices": ("DeviceRegistry", "devices(bytes32)", ["bytes32", "address", "uint64", "uint16", "uint8", "uint64"]),
    "revocationWindow": ("DeviceRegistry", "revocationWindow(bytes32)", ["uint64", "uint32", "uint16", "uint16",
                                                                         "bool"]),
    "revocationVoted": ("DeviceRegistry", "revocationVoted(bytes32,uint32,address)", ["bool"]),
    # Governance
    "validatorCount": ("Governance", "validatorCount()", ["uint256"]),
    "f": ("Governance", "f()", ["uint256"]),
    "isValidator": ("Governance", "isValidator(address)", ["bool"]),
    # ClaimRegistry
    "claimInfo": ("ClaimRegistry", "claimInfo(bytes32)", ["bool", "bytes32", "address", "uint8", "uint8", "uint8",
                                                          "bytes32", "bool", "bool", "bool"]),
    "levelCap": ("ClaimRegistry", "levelCap(bytes32)", ["uint8"]),
    # the courts: arguments are given as ready calldata
    "proveStaticViolation": ("StaticRules", "proveStaticViolation((bytes32,address," + CORE_T
                             + ",uint8,uint8,uint16,bytes,bytes,bytes32,bytes,bytes))", ["bytes32"]),
    "proveBlockFailure": ("BlockCourt", "proveBlockFailure((bytes32,address," + CORE_T + ",bytes,bytes,bytes32[],"
                          "bytes,bytes,bytes,uint32,bytes,bytes32[],bytes,bytes32[],(uint16,uint16,bytes16,bytes,"
                          "bytes32[])[],uint8,uint8))", ["bytes32"]),
}

EVENTS = {
    "ChallengeOpened": ("Adjudicator", "ChallengeOpened(uint64,bytes32,string)",
                        [("id", "uint64", True), ("target", "bytes32", True), ("bundleLocator", "string", False)]),
    "AppealOpened": ("Adjudicator", "AppealOpened(uint64,bytes32)",
                     [("id", "uint64", True), ("target", "bytes32", True)]),
    "Verdict": ("Adjudicator", "Verdict(uint64,address,uint8,bytes32)",
                [("id", "uint64", True), ("validator", "address", True), ("status", "uint8", False),
                 ("transcript", "bytes32", False)]),
    "ChallengeFinalized": ("Adjudicator", "ChallengeFinalized(uint64,uint8)",
                           [("id", "uint64", True), ("outcome", "uint8", False)]),
    "AppealFinalized": ("Adjudicator", "AppealFinalized(uint64,bool)",
                        [("id", "uint64", True), ("cleared", "bool", False)]),
    "Pulse": ("CaptureAnchor", "Pulse(address,bytes32)", [("validator", "address", True), ("r", "bytes32", False)]),
    "Enrolled": ("DeviceRegistry", "Enrolled(bytes32,address,bytes32,bytes32,uint16,bytes16)",
                 [("keyId", "bytes32", True), ("holder", "address", True), ("pkHash", "bytes32", False),
                  ("attHash", "bytes32", False), ("rateN", "uint16", False), ("nonceE", "bytes16", False)]),
    "Activated": ("DeviceRegistry", "Activated(bytes32,uint64)", [("keyId", "bytes32", True),
                                                                   ("height", "uint64", False)]),
    "RevocationVoted": ("DeviceRegistry", "RevocationVoted(bytes32,address,uint32,uint8,bytes32)",
                        [("keyId", "bytes32", True), ("validator", "address", True), ("round", "uint32", False),
                         ("status", "uint8", False), ("entryHash", "bytes32", False)]),
    "RevocationFinalized": ("DeviceRegistry", "RevocationFinalized(bytes32,uint32,bool)",
                            [("keyId", "bytes32", True), ("round", "uint32", False), ("revoke", "bool", False)]),
    "Revoked": ("DeviceRegistry", "Revoked(bytes32,uint64,uint8)",
                [("keyId", "bytes32", True), ("tc", "uint64", False), ("source", "uint8", False)]),
    "ClaimRegistered": ("ClaimRegistry", "ClaimRegistered(bytes32,bytes32,address," + CORE_T + ",bytes32)",
                        [("claimId", "bytes32", True), ("nf", "bytes32", True), ("claimant", "address", True),
                         ("core", CORE_T, False), ("principalKey", "bytes32", False)]),
    "Struck": ("ClaimRegistry", "Struck(bytes32,bytes32,bytes32,bool)",
               [("nf", "bytes32", True), ("claimId", "bytes32", True), ("principalKey", "bytes32", False),
                ("byCourt", "bool", False)]),
    "StaticViolation": ("StaticRules", "StaticViolation(bytes32,uint8,uint8,uint16,address)",
                        [("claimId", "bytes32", True), ("rule", "uint8", False), ("i", "uint8", False),
                         ("j", "uint16", False), ("challenger", "address", False)]),
    "BlockFailure": ("BlockCourt", "BlockFailure(bytes32,uint32,uint8,uint8,int16,int256,address)",
                     [("claimId", "bytes32", True), ("mcu", "uint32", False), ("comp", "uint8", False),
                      ("k", "uint8", False), ("c", "int16", False), ("w", "int256", False),
                      ("challenger", "address", False)]),
}


@dataclass(frozen=True)
class Event:
    name: str
    block: int
    log_index: int
    args: dict = field(default_factory=dict)
    tx: bytes = b""


@dataclass(frozen=True)
class Receipt:
    tx: bytes
    block: int
    ok: bool
    error: str | None = None
    gas_used: int | None = None


class Revert(Exception):
    def __init__(self, name: str, detail: str = ""):
        super().__init__(f"{name} {detail}".strip())
        self.name = name


class ChainUnavailable(Exception):
    """The node did not answer (timeout, connection refused, a JSON-RPC transport error): the caller backs off."""


@runtime_checkable
class ChainPort(Protocol):
    address: bytes

    def head(self) -> int: ...

    def block_hash(self, h: int) -> bytes | None: ...

    def parent_hash(self, h: int) -> bytes | None: ...

    def logs(self, frm: int, to: int) -> list: ...

    def view(self, fn: str, args: tuple, block: int | None = None) -> tuple: ...

    def simulate(self, fn: str, args: tuple, calldata: bytes | None = None) -> str | None: ...

    def send(self, fn: str, args: tuple, calldata: bytes | None = None) -> bytes: ...

    def receipt(self, tx: bytes) -> Receipt | None: ...

    def ledger(self): ...


__all__ = ["ChainPort", "Event", "Receipt", "Revert", "ChainUnavailable", "FUNCS", "EVENTS", "W_VOTE", "BEACON_EPOCH",
           "PASS", "FAIL", "KIND_CHALLENGE", "KIND_APPEAL", "OUTCOME_FAILED", "OUTCOME_PASSED", "OUTCOME_UNDECIDED",
           "RVOTE_REVOKE", "RVOTE_KEEP", "DEV_UNKNOWN", "DEV_PENDING", "DEV_ACTIVE", "CORE_T"]
