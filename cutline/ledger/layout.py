"""cutline.ledger.layout - the v1 storage layout of the receipt read set.

Loads ``storage_layout_v1.json`` (exported from the solc ``storageLayout``) and the frozen addresses of
``cutline_v1_abi.json``, both from ``ABI_DIR`` (``<repo root>/abi`` unless the environment variable
``CUTLINE_ABI_DIR`` names another directory); computes the storage slots of the 7 read-set entries and decodes their
words. Nothing here touches the network.

- ``LAYOUT`` (the JSON), ``READ_SET`` (variable -> entry), ``ADDRESSES`` (contract name -> 20 bytes; the frozen v1
  deployment), ``CONTRACT_OF`` (variable -> contract name), ``CODE_KECCAK_V1`` (contract name -> keccak256 of the
  deployed v1 runtime, the ``codeHash`` a receipt's account proof must show; see below).
- slot functions (ints): ``slot_claims(claim_id)``, ``slot_captures(nf)``, ``slot_strikes(pk)``,
  ``slot_exclusive(nf, m)``, ``slot_delegation(nf, who)``, ``slot_batch(batch_id)``, ``slot_device(key_id)``;
  ``struct_slots(var, base) -> [slot]`` (the distinct slots the entry's fields occupy).
- ``decode_struct(var, base, words) -> dict`` (``words``: slot -> int; a missing slot is zero): field values as
  ``int`` / ``bool`` / 20-byte ``bytes`` (address) / 32-byte ``bytes`` (bytes32), per (slotOffset, byteOffset,
  bytes) with low-order packing.
- ``read_plan(claim_id, fields, now) -> {var: base}``: the read-set entries computable from the already decoded
  ``fields`` (``{var: dict}``); iterate to a fixed point (claims -> captures / strikes / exclusiveOf /
  delegations -> _batches / devices). ``delegations`` is included iff the claimant is not the owner and is not the
  licensee of a licence active at ``now`` (``expiry >= now``). This is the exact rule of ``principal()``, which is
  slightly wider than "only if the claimant is neither the owner nor the licensee": a claimant that is the licensee
  of a **lapsed** licence and a delegate of the current epoch reaches the delegation branch (a superset of that
  wording, never a subset).
- ``slots_for(claim_id, fields, now) -> {contract: [slot]}``: the same as storage slots per contract.

``CODE_KECCAK_V1``: the frozen v1 contracts carry immutables, so the deployed runtime (hence its keccak) is a
property of the frozen deployment (account 0, nonces 0..8, ``deploySystem``). The values below were read with
``eth_getProof`` / ``eth_getCode`` from a fresh EDR chain deployed by ``scripts/deploy.js`` (DEPLOY_N=4), and each
code's SHA-256 was checked against ``runtimeSha256`` of ``cutline_v1_abi.json``. A validator-count change does not
change them (the Governance constructor arguments are not in the other contracts' runtime).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import abi
from ..crypto import keccak256

ROOT = Path(__file__).resolve().parents[2]                     # repository root
ABI_DIR = Path(os.environ.get("CUTLINE_ABI_DIR") or ROOT / "abi")
LAYOUT_PATH = ABI_DIR / "storage_layout_v1.json"
ABI_PATH = ABI_DIR / "cutline_v1_abi.json"

LAYOUT = json.loads(LAYOUT_PATH.read_text())
ABI_EXPORT = json.loads(ABI_PATH.read_text())
READ_SET = {e["variable"]: e for e in LAYOUT["readSet"]}
CONTRACT_OF = {v: e["contract"] for v, e in READ_SET.items()}
ADDRESSES = {name: bytes.fromhex(c["address"][2:]) for name, c in ABI_EXPORT["contracts"].items()}
CHAIN_ID = int(ABI_EXPORT["chainId"])

# keccak256(runtime) of the frozen v1 deployment (see the module docstring).
CODE_KECCAK_V1: dict = {
    "ClaimRegistry": bytes.fromhex("cba0a5040937df1f7cbe119795e47e439ad914804ce2546cc4ca0c529dd4e18e"),
    "DeviceRegistry": bytes.fromhex("13376408fc4f4cf1d19245e4d646b979ed7a425b6323c9c0cc6804b858a54e7e"),
    "Governance": bytes.fromhex("7128012b7753b6ed153d337f85476d8f68e1cfe416573f5043d0f91df746bac6"),
    "CaptureAnchor": bytes.fromhex("b03225a10cf935300bc605ca1b12f86348dbc88036fdfe68fe2b4141e6086484"),
    "HopRegistry": bytes.fromhex("c2b6af1fe11b8c25f9e4a1d70ce3107898437dc23f676e4e66fdb6f4f55c8bcf"),
    "Adjudicator": bytes.fromhex("c1efd9fb9c97a9c787500ac4ab7bbdddeea4d545e5a058795ef9ff7910f7f205"),
    "StaticRules": bytes.fromhex("23a8b8712866365345cc537f47a1cfd1ac2688d955a0d2817a17462f0235fd45"),
    "BlockCourt": bytes.fromhex("8abb2592146074ed1e232b66cf082042799effa55803353e2b19cbb75b2c50c3"),
    "LabelPolicy": bytes.fromhex("f8f4d90e8bfe4c00d77eee7e9e7494087d6ceafc8e5a79a929379859178dcc23"),
}

# the layout's _batches entry is a dynamic array of 3-slot structs (Batch: root, beacon, packed heights / count)
_BATCH_STRUCT_SLOTS = 3


def _u(x) -> int:
    return int.from_bytes(bytes(x), "big") if isinstance(x, (bytes, bytearray)) else int(x)


def _map(key_type: str, key, slot: int) -> int:
    return int.from_bytes(keccak256(abi.encode([key_type, "uint256"], [key, int(slot)])), "big")


def slot_claims(claim_id: bytes) -> int:
    return _map("bytes32", bytes(claim_id), READ_SET["claims"]["baseSlot"])


def slot_captures(nf: bytes) -> int:
    return _map("bytes32", bytes(nf), READ_SET["captures"]["baseSlot"])


def slot_strikes(pk: bytes) -> int:
    return _map("bytes32", bytes(pk), READ_SET["strikes"]["baseSlot"])


def slot_exclusive(nf: bytes, m: int) -> int:
    return _map("uint8", int(m), _map("bytes32", bytes(nf), READ_SET["exclusiveOf"]["baseSlot"]))


def slot_delegation(nf: bytes, who: bytes) -> int:
    return _map("address", bytes(who), _map("bytes32", bytes(nf), READ_SET["delegations"]["baseSlot"]))


def slot_batch(batch_id: int) -> int:
    data = int.from_bytes(keccak256(abi.encode(["uint256"], [READ_SET["_batches"]["baseSlot"]])), "big")
    return data + _BATCH_STRUCT_SLOTS * int(batch_id)


def slot_device(key_id: bytes) -> int:
    return _map("bytes32", bytes(key_id), READ_SET["devices"]["baseSlot"])


def struct_slots(var: str, base: int) -> list:
    return sorted({(int(base) + f["slotOffset"]) % (1 << 256) for f in READ_SET[var]["fields"]})


def decode_struct(var: str, base: int, words: dict) -> dict:
    out = {}
    for f in READ_SET[var]["fields"]:
        w = int(words.get((int(base) + f["slotOffset"]) % (1 << 256), 0))
        v = (w >> (8 * f["byteOffset"])) & ((1 << (8 * f["bytes"])) - 1)
        t = f["type"]
        if t == "t_bool":
            if v > 1:
                raise ValueError(f"{var}.{f['field']}: bool word {v}")
            out[f["field"]] = bool(v)
        elif t == "t_address":
            out[f["field"]] = v.to_bytes(20, "big")
        elif t == "t_bytes32":
            out[f["field"]] = v.to_bytes(32, "big")
        else:
            out[f["field"]] = v
    return out


ZERO20 = b"\x00" * 20


def principal_key(nf: bytes, kind: int, id_: int) -> bytes:
    """principalKey = H(nf || u8 kind || u32 id); the same function as ``ledger.abi.principal_key``."""
    from ..ledger.abi import principal_key as pk
    return pk(nf, kind, id_)


def read_plan(claim_id: bytes, fields: dict, now: int) -> dict:
    """{var: base slot} of every read-set entry computable from ``fields`` (see the module docstring)."""
    plan = {"claims": slot_claims(claim_id)}
    cl = fields.get("claims")
    if cl is None or cl["claimant"] == ZERO20:
        return plan
    nf, m = cl["nf"], cl["marketplace"]
    plan["captures"] = slot_captures(nf)
    plan["strikes"] = slot_strikes(principal_key(nf, cl["kind"], cl["id"]))
    plan["exclusiveOf"] = slot_exclusive(nf, m)
    cp, x = fields.get("captures"), fields.get("exclusiveOf")
    if cp is not None and x is not None:
        active_lic = x["licensee"] == cl["claimant"] and x["licensee"] != ZERO20 and x["expiry"] >= int(now)
        if cl["claimant"] != cp["owner"] and not active_lic:
            plan["delegations"] = slot_delegation(nf, cl["claimant"])
    if cp is not None:
        plan["_batches"] = slot_batch(cp["batchId"])
        plan["devices"] = slot_device(cp["keyId"])
    return plan


def slots_for(claim_id: bytes, fields: dict, now: int) -> dict:
    out: dict = {}
    for var, base in read_plan(claim_id, fields, now).items():
        out.setdefault(CONTRACT_OF[var], []).extend(struct_slots(var, base))
    return {c: sorted(set(s)) for c, s in out.items()}


__all__ = ["LAYOUT", "READ_SET", "ADDRESSES", "CONTRACT_OF", "CHAIN_ID", "CODE_KECCAK_V1", "slot_claims",
           "slot_captures", "slot_strikes", "slot_exclusive", "slot_delegation", "slot_batch", "slot_device",
           "struct_slots", "decode_struct", "principal_key", "read_plan", "slots_for", "ZERO20"]
