"""sidecar.revocation - revocation-list polling and REVOKE / KEEP votes.

- ``entry_hash(serial, entry) -> bytes32``: ``H(canonical JSON {"serial", "status", "reason"?})``: the
  entry's dictionary key and its ``status`` verbatim, ``reason`` only if present; ``expires`` and ``comment`` never.
- ``keep_hash(key_id) = H("keep" ‖ keyId)`` (``CutlineLib.keepEntry``).
- ``cert_serials(ders) -> [serial]``: the lowercase-hex serials without leading zeros, chain order, leaf first.
- ``listed_entry(ders, status_list) -> (serial, entry) | None``: the entry of the **first listed certificate** in
  chain order.
- ``fetch_list(ctx) -> dict | None``: ``GET statuslist_url`` = ``{"entries": {serial: {...}}}`` (the format of
  Google's Android key attestation status list); cached as ``ctx.status_list``.
- ``on_revocation_voted(ctx, ev)``: records the window (keyId, round, openedAt = the first vote's block of the round)
  and marks it for a re-evaluation; ``on_revocation_finalized`` / ``on_revoked``.
- ``process(ctx, head)``: every ``revocation_poll_s`` seconds (or when a new window appeared) re-reads the list and,
  for each key whose attestation chain is known (enrolment duty), and whose ``tc`` is 0:
  - listed → ``voteRevocation(keyId, REVOKE, entry_hash)`` unless it already voted in the open window
    (``revocationVoted(keyId, round, me)``); a vote outside an open window opens a new one;
  - not listed and a window is open → ``voteRevocation(keyId, KEEP, H("keep" ‖ keyId))`` once.
  Every vote is simulated first; one vote per (keyId, round) is recorded in ``rev_votes``.
"""
from __future__ import annotations

import json
import time

from cutline.crypto import H
from cutline.enc import jcs
from sidecar.chain import RVOTE_KEEP, RVOTE_REVOKE, W_VOTE
from sidecar.sources import fetch


def _hx(b):
    return "0x" + bytes(b).hex()


def _unhx(s):
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def entry_hash(serial: str, entry: dict) -> bytes:
    obj = {"serial": serial, "status": entry["status"]}
    if "reason" in entry:
        obj["reason"] = entry["reason"]
    return H(jcs(obj))


def keep_hash(key_id: bytes) -> bytes:
    return H(b"keep" + bytes(key_id))


def cert_serials(ders) -> list:
    from cryptography import x509
    return [format(x509.load_der_x509_certificate(bytes(d)).serial_number, "x") for d in ders]


def listed_entry(ders, status_list: dict):
    entries = (status_list or {}).get("entries") or {}
    for s in cert_serials(ders):
        if s in entries:
            return s, entries[s]
    return None


def fetch_list(ctx):
    if not ctx.cfg.statuslist_url:
        return None
    data, _ = fetch(ctx.cfg.statuslist_url, 64 << 20, ctx.cfg.fetch_timeout_s)
    if data is None:
        return None
    try:
        d = json.loads(data)
    except ValueError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("entries", {}), dict) else None


def on_revocation_voted(ctx, ev) -> None:
    kid, rnd = _hx(ev.args["keyId"]), int(ev.args["round"])
    r = ctx.state["rev"].get(kid)
    if r is None or int(r["round"]) < rnd:
        ctx.state["rev"][kid] = {"round": rnd, "opened_at": int(ev.block), "finalised": False}
        ctx.rev_dirty = True
    if bytes(ev.args["validator"]) == ctx.me:
        ctx.state["rev_votes"].setdefault(f"{kid}:{rnd}", {})["included"] = int(ev.block)


def on_revocation_finalized(ctx, ev) -> None:
    kid, rnd = _hx(ev.args["keyId"]), int(ev.args["round"])
    r = ctx.state["rev"].get(kid)
    if r is not None and int(r["round"]) == rnd:
        r["finalised"] = True
        r["revoke"] = bool(ev.args["revoke"])


def on_revoked(ctx, ev) -> None:
    rec = ctx.state["keys"].get(_hx(ev.args["keyId"]))
    if rec is not None:
        rec["tc"] = int(ev.args["tc"])


def process(ctx, head: int) -> int:
    now = time.time()
    due = now - float(ctx.state["last_revocation_poll"]) >= ctx.cfg.revocation_poll_s
    if not (due or ctx.rev_dirty):
        return 0
    if due or ctx.status_list is None:
        sl = fetch_list(ctx)
        if sl is not None:
            ctx.status_list = sl
        ctx.state["last_revocation_poll"] = now
    ctx.rev_dirty = False
    if ctx.status_list is None or ctx.silent(head):
        return 0
    n = 0
    for kid, rec in ctx.state["keys"].items():
        if not rec.get("ders") or rec.get("tc"):
            continue
        key = _unhx(kid)
        dev = ctx.port.view("devices", (key,))
        if int(dev[4]) == 0 or int(dev[5]) != 0:          # unknown key, or tc already set
            if int(dev[5]) != 0:
                rec["tc"] = int(dev[5])
            continue
        opened_at, rnd = (int(x) for x in ctx.port.view("revocationWindow", (key,))[:2])
        is_open = opened_at != 0 and head < opened_at + W_VOTE
        le = listed_entry([_unhx(d) for d in rec["ders"]], ctx.status_list)
        if le is None and not is_open:
            continue
        # the round this vote lands in: the open one, else a new one (the vote opens it)
        vr = rnd if is_open else rnd + 1
        slot = f"{kid}:{vr}"
        if slot in ctx.state["rev_votes"]:
            continue
        if is_open and ctx.port.view("revocationVoted", (key, rnd, ctx.me))[0]:
            ctx.state["rev_votes"][slot] = {"status": "seen"}
            continue
        if le is not None:
            status, eh = RVOTE_REVOKE, entry_hash(*le)
        else:
            status, eh = RVOTE_KEEP, keep_hash(key)
        tx, err = ctx.send("revocation", slot, "voteRevocation", (key, status, eh))
        ctx.state["rev_votes"][slot] = {"status": "REVOKE" if status == RVOTE_REVOKE else "KEEP",
                                        "entry_hash": _hx(eh), "tx": None if tx is None else _hx(tx), "error": err}
        ctx.metrics.emit("revocation", key_id=kid, status="REVOKE" if status == RVOTE_REVOKE else "KEEP",
                         entry_hash=_hx(eh), round=vr, tx=None if tx is None else _hx(tx), error=err)
        n += 1
    return n


__all__ = ["entry_hash", "keep_hash", "cert_serials", "listed_entry", "fetch_list", "on_revocation_voted",
           "on_revocation_finalized", "on_revoked", "process"]
