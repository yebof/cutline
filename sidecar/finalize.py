"""sidecar.finalize - finalisation calls after the windows end.

``process(ctx, head)``: for every challenge / appeal window and revocation window this sidecar has seen that is not
finalised, once ``head ≥ opened_at + W_vote + (index − 1) · finalize_stagger``, it pre-checks the call with
``eth_call`` (``simulate``) and sends ``finalizeChallenge(id)`` / ``finalizeAppeal(id)`` /
``finalizeRevocation(keyId)`` if it would succeed. The first sender wins; later sidecars see ``E.CHAL.WINDOW`` /
``E.VOTE.WINDOW`` in the pre-check and only mark the window finalised. Anyone may finalise, so a Byzantine
or silent sidecar still finalises (liveness does not depend on honest votes). One attempt per window per sidecar
(``finalize_sent``); a lost transaction clears it.
"""
from __future__ import annotations

from sidecar.chain import KIND_APPEAL, W_VOTE

MAX_RETRY = 20
_DONE = ("E.CHAL.WINDOW", "E.VOTE.WINDOW", "E.CHAL.UNKNOWN")


def _skip(rec: dict, head: int, wait: int) -> bool:
    return bool(rec.get("finalised") or rec.get("finalize_sent") or rec.get("finalize_retry", 0) >= MAX_RETRY
                or head < int(rec["opened_at"]) + W_VOTE + wait)


def process(ctx, head: int) -> int:
    if not ctx.cfg.finalize:
        return 0
    wait = (ctx.cfg.index - 1) * ctx.cfg.finalize_stagger
    n = 0
    for wid, w in ctx.state["windows"].items():
        if _skip(w, head, wait):
            continue
        fn = "finalizeAppeal" if int(w["kind"]) == KIND_APPEAL else "finalizeChallenge"
        n += _one(ctx, w, fn, wid, (int(wid),))
    for kid, r in ctx.state["rev"].items():
        if _skip(r, head, wait):
            continue
        n += _one(ctx, r, "finalizeRevocation", kid, (bytes.fromhex(kid[2:]),))
    return n


def _one(ctx, rec, fn, key, args) -> int:
    tx, err = ctx.send("finalize", f"{fn}:{key}", fn, args)
    if err is not None:
        if err in _DONE:
            rec["finalised"] = True
        ctx.metrics.emit("finalize_skip", fn=fn, key=key, reason=err)
        rec["finalize_retry"] = int(rec.get("finalize_retry", 0)) + 1
        return 0
    rec["finalize_sent"] = True
    ctx.metrics.emit("finalize", fn=fn, key=key, tx="0x" + tx.hex())
    return 1


def on_lost(ctx, key: str) -> None:
    fn, k = key.split(":", 1)
    rec = ctx.state["rev"].get(k) if fn == "finalizeRevocation" else ctx.state["windows"].get(k)
    if rec is not None:
        rec["finalize_sent"] = False


__all__ = ["process", "on_lost"]
