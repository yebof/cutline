"""sidecar.enrol - the validators' enrolment check and ``confirm``.

- ``on_enrolled(ctx, ev)``: records ``Enrolled(keyId, holder, pkHash, attHash, rateN, nonceE)`` (status ``pending``).
- ``on_activated(ctx, ev)``: marks the key ``active``.
- ``fetch_chain(ctx, key_id) -> [der, der, der] | None``: ``<pe_url>/chain/<keyId 0x hex>.json`` =
  ``{"keyId": "0x..", "ders": ["0x<leaf>", "0x<batch>", "0x<root>"]}``, leaf first.
- ``check_chain(ders, rec, chain_id, pinned_root_sha, listed=None) -> None | reason``: ``H(chain) = attHash``
  and every attestation verifier check (``capture_emu.attestation.verify_chain``: the chain up to the pinned
  root, StrongBox levels, rootOfTrust, the application id, the leaf key's ``H(pk_x ‖ pk_y) = pkHash``) with the
  expected ``attestationChallenge = H("CUTLINE-ENROL-v1" ‖ holder ‖ u16 rateN ‖ u256 chainId ‖ nonce_E)``
  (``ledger.abi.enrol_challenge``); ``listed`` (a callable ``ders -> entry | None`` over the status list) refuses a
  listed chain ("not on the revocation status list").
- ``process(ctx, head)``: one pending key per call. The chain is fetched (retried with backoff, ``unavailable`` after
  8 attempts), checked, stored for the revocation duty, and ``confirm(keyId)`` is sent when the check passes (a
  refused key gets **no vote**; 2f+1 confirms activate it). ``E.DEV.STATE`` (already active) and ``E.VOTE.DUP``
  (already confirmed) in the pre-check mark the key without a transaction. Without ``pinned_root_sha256`` the
  emulation root of ``capture_emu.attestation`` is pinned.
"""
from __future__ import annotations

import json

from sidecar.sources import fetch

MAX_ATTEMPTS = 8


def _hx(b):
    return "0x" + bytes(b).hex()


def _unhx(s):
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def on_enrolled(ctx, ev) -> None:
    k = _hx(ev.args["keyId"])
    if k in ctx.state["keys"]:
        return
    a = ev.args
    ctx.state["keys"][k] = {"holder": _hx(a["holder"]), "pk_hash": _hx(a["pkHash"]), "att_hash": _hx(a["attHash"]),
                            "rate_n": int(a["rateN"]), "nonce_e": _hx(a["nonceE"]), "ders": None,
                            "status": "pending", "attempts": 0, "next_try": int(ev.block), "enrolled_at": int(ev.block)}


def on_activated(ctx, ev) -> None:
    rec = ctx.state["keys"].get(_hx(ev.args["keyId"]))
    if rec is not None:
        rec["active"] = True


def fetch_chain(ctx, key_id: str):
    if not ctx.cfg.pe_url:
        return None
    data, _ = fetch(ctx.cfg.pe_url.rstrip("/") + f"/chain/{key_id}.json", 1 << 20, ctx.cfg.fetch_timeout_s)
    if data is None:
        return None
    try:
        d = json.loads(data)
        ders = [_unhx(x) for x in d["ders"]]
    except (ValueError, KeyError, TypeError):
        return None
    return ders


def check_chain(ders, rec: dict, chain_id: int, pinned_root_sha: bytes, listed=None):
    from capture_emu.attestation import verify_chain
    from cutline.ledger.abi import att_hash, enrol_challenge
    if len(ders) != 3:
        return "chain length"
    if att_hash(ders) != _unhx(rec["att_hash"]):
        return "attHash"
    ch = enrol_challenge(_unhx(rec["holder"]), int(rec["rate_n"]), int(chain_id), _unhx(rec["nonce_e"]))
    why = verify_chain(ders, pinned_root_sha, {"pk_hash": _unhx(rec["pk_hash"]), "challenge": ch})
    if why is not None:
        return why
    if listed is not None and listed(ders) is not None:
        return "listed on the status list"
    return None


def process(ctx, head: int) -> int:
    for kid, rec in ctx.state["keys"].items():
        if rec["status"] != "pending" or int(rec.get("next_try", 0)) > head:
            continue
        ders = fetch_chain(ctx, kid)
        if ders is None:
            rec["attempts"] = int(rec["attempts"]) + 1
            rec["next_try"] = head + min(2 ** rec["attempts"], 64)
            if rec["attempts"] >= MAX_ATTEMPTS:
                rec["status"] = "unavailable"
                ctx.metrics.emit("enrol", key_id=kid, decision="refuse", reason="chain unavailable", tx=None)
            return 1
        rec["ders"] = [_hx(d) for d in ders]
        root = _unhx(ctx.cfg.pinned_root_sha256) if ctx.cfg.pinned_root_sha256 else None
        if root is None:
            from capture_emu.attestation import ROOT_SHA256
            root = ROOT_SHA256()
        from sidecar.revocation import listed_entry
        sl = ctx.status_list
        why = check_chain(ders, rec, ctx.chain_id(), root,
                          None if sl is None else (lambda ds: listed_entry(ds, sl)))
        if why is not None:
            rec["status"], rec["reason"] = "refused", why
            ctx.metrics.emit("enrol", key_id=kid, decision="refuse", reason=why, tx=None)
            return 1
        if ctx.silent(head) or not ctx.is_validator:    # a watcher checks but never confirms (E.ROLE)
            rec["status"] = "checked"
            return 1
        tx, err = ctx.send("confirm", kid, "confirm", (_unhx(kid),))
        if err is not None:
            rec["status"] = {"E.DEV.STATE": "active", "E.VOTE.DUP": "confirmed"}.get(err, "failed")
            rec["reason"] = err
        else:
            rec["status"] = "confirmed"
        ctx.metrics.emit("enrol", key_id=kid, decision="confirm", reason=err, tx=None if tx is None else _hx(tx))
        return 1
    return 0


__all__ = ["on_enrolled", "on_activated", "fetch_chain", "check_chain", "process"]
