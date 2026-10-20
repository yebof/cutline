"""sidecar.verdict - path-(b) challenge votes and appeal votes.

Event handlers (idempotent; ``ctx`` is the ``service.Sidecar``):

- ``on_challenge_opened(ctx, ev)``: records the locator under the target (every ``ChallengeOpened`` on a target is a
  bundle source) and queues window ``id`` (kind CHALLENGE, ``opened_at`` = the event's block).
- ``on_appeal_opened(ctx, ev)``: queues window ``id`` (kind APPEAL).
- ``on_verdict(ctx, ev)`` (own votes: status ``included``), ``on_finalized(ctx, ev)`` (``finalised``, ``outcome``).

``process_queue(ctx, head)``: the earliest-deadline window first; at most one vote computation per call.

1. Overload: with more than ``max_queue`` queued windows, the latest-deadline ones abstain ``overload``.
2. A window at or past ``opened_at + W_vote − deadline_margin`` abstains ``deadline`` (a vote would not land).
3. ``byzantine = "silent"`` (before ``silent_until``) keeps the window queued without voting.
4. ``voted(id, me)`` on-chain, or a pending transaction for the window: nothing to do (restart idempotence).
5. The vote: ``runner.run_vote`` over ``sources.source_list`` at **h = opened_at**, i.e. the claim's own tier-T
   table and the kfam / class registry of that block; ``vote_any`` tries every source.
6. Challenge: PASS → ``submitVerdict(id, 1, H("pass" ‖ target))``; FAIL → ``submitVerdict(id, 2, H(abi.encode(rows)))``;
   abstain → no transaction (a ``static_violation:*`` abstention hands the claim to the court duty when ``watcher``).
   Appeal: PASS → ``submitVerdict(id, 1, …)``; FAIL or abstain → no vote (FAIL votes revert in an appeal).
7. Byzantine fault-injection modes (tests): ``fail-all`` votes FAIL with the fixed transcript
   ``H("byzantine-fail" ‖ target)`` on every challenge (identical across colluders, so they reach f+1) and nothing on appeals; ``pass-all`` votes PASS on both.

Every vote is simulated (``eth_call``) before it is sent; ``E.VOTE.DUP`` marks the window voted.
"""
from __future__ import annotations

from cutline.crypto import H
from sidecar.chain import FAIL, KIND_APPEAL, KIND_CHALLENGE, PASS, W_VOTE

BYZ_FAIL_TAG = b"byzantine-fail"


def _hx(b: bytes) -> str:
    return "0x" + bytes(b).hex()


def _unhx(s: str) -> bytes:
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def on_challenge_opened(ctx, ev) -> None:
    wid, target, loc = str(int(ev.args["id"])), _hx(ev.args["target"]), ev.args.get("bundleLocator", "")
    locs = ctx.state["locators"].setdefault(target, [])
    if loc and loc not in locs:
        locs.append(loc)
    w = ctx.state["windows"]
    if wid not in w:
        w[wid] = {"kind": KIND_CHALLENGE, "target": target, "opened_at": int(ev.block), "locator": loc,
                  "status": "queued", "finalised": False}
        ctx.state["queue"].append(wid)


def on_appeal_opened(ctx, ev) -> None:
    wid, target = str(int(ev.args["id"])), _hx(ev.args["target"])
    w = ctx.state["windows"]
    if wid not in w:
        w[wid] = {"kind": KIND_APPEAL, "target": target, "opened_at": int(ev.block), "locator": "",
                  "status": "queued", "finalised": False}
        ctx.state["queue"].append(wid)


def on_verdict(ctx, ev) -> None:
    if bytes(ev.args["validator"]) != ctx.me:
        return
    w = ctx.state["windows"].get(str(int(ev.args["id"])))
    if w is not None:
        w["status"] = "included"
        w["included_at"] = int(ev.block)


def on_finalized(ctx, ev) -> None:
    w = ctx.state["windows"].get(str(int(ev.args["id"])))
    if w is not None:
        w["finalised"] = True
        if "outcome" in ev.args:
            w["outcome"] = int(ev.args["outcome"])
        if "cleared" in ev.args:
            w["cleared"] = bool(ev.args["cleared"])


def _deadline(w) -> int:
    return int(w["opened_at"]) + W_VOTE


def _drop(ctx, wid, status, reason, **extra) -> None:
    w = ctx.state["windows"][wid]
    w["status"], w["reason"] = status, reason
    if wid in ctx.state["queue"]:
        ctx.state["queue"].remove(wid)
    ctx.metrics.emit("abstain", id=int(wid), target=w["target"], reason=reason, **extra)


def _pending_for(ctx, wid) -> bool:
    return any(p.get("purpose") == "vote" and p.get("key") == wid for p in ctx.state["pending"].values())


MAX_INTERRUPTS = 3                     # after that a repeatedly signalled child counts as a local failure (abstain)


def process_queue(ctx, head: int) -> int:
    """Returns the number of vote computations done (0 or 1)."""
    cfg, st = ctx.cfg, ctx.state
    q = st["queue"]
    q.sort(key=lambda wid: (_deadline(st["windows"][wid]), int(wid)))
    for wid in list(q):
        if head >= _deadline(st["windows"][wid]) - cfg.deadline_margin:
            _drop(ctx, wid, "abstained", "deadline", head=head)
    while len(q) > cfg.max_queue:
        _drop(ctx, q[-1], "abstained", "overload")
    for wid in list(q):
        w = st["windows"][wid]
        if ctx.silent(head):
            continue
        if _pending_for(ctx, wid):
            continue
        if ctx.port.view("voted", (int(wid), ctx.me))[0]:
            w["status"] = "voted"
            q.remove(wid)
            continue
        _vote_window(ctx, wid, w, head)
        return 1
    return 0


def _vote_window(ctx, wid: str, w: dict, head: int) -> None:
    cfg = ctx.cfg
    target = _unhx(w["target"])
    kind = int(w["kind"])
    if cfg.byzantine == "fail-all":
        res = {"decision": "FAIL", "transcript": _hx(H(BYZ_FAIL_TAG + target)), "reason": "byzantine", "compute_s": 0}
    elif cfg.byzantine == "pass-all":
        res = {"decision": "PASS", "transcript": _hx(H(b"pass" + target)), "reason": "byzantine", "compute_s": 0}
    else:
        locs = list(ctx.state["locators"].get(w["target"], []))
        if w.get("locator") in locs:                    # the triggering event's locator first
            locs.remove(w["locator"])
            locs.insert(0, w["locator"])
        from sidecar.sources import source_list
        budget = max(1.0, min(cfg.vote_timeout_s, (_deadline(w) - cfg.deadline_margin - head) * cfg.block_period_s))
        job = {"target": target, "descriptors": source_list(cfg, target, locs, ctx.cache), "ledger": ctx.ledger(),
               "h": int(w["opened_at"]), "limit": cfg.fetch_limit, "timeout": cfg.fetch_timeout_s,
               "cache_dir": cfg.cache_dir, "chain_id": cfg.chain_id,
               "registry": None if cfg.registry is None else _unhx(cfg.registry)}
        res = ctx.run_vote(job, budget)
        if res.get("interrupted"):             # a shutdown killed the child: keep the window queued
            w["interrupts"] = int(w.get("interrupts", 0)) + 1
            ctx.metrics.emit("vote_interrupted", id=int(wid), target=w["target"], reason=res.get("reason"),
                             interrupts=w["interrupts"])
            if w["interrupts"] <= MAX_INTERRUPTS:
                return
    w["decision"], w["transcript"], w["reason"] = res["decision"], res.get("transcript"), res.get("reason")
    w["compute_s"] = res.get("compute_s")
    rec = dict(id=int(wid), target=w["target"], window="appeal" if kind == KIND_APPEAL else "challenge",
               opened_at=w["opened_at"], h=w["opened_at"], decision=res["decision"], reason=res.get("reason"),
               transcript=res.get("transcript"), compute_s=res.get("compute_s"), peak_rss_mb=res.get("peak_rss_mb"),
               source=res.get("source"), fetch_log=res.get("fetch_log"), sent_head=head)
    status = None
    if not ctx.is_validator:                         # a watcher computes votes only to find static violations
        pass
    elif res["decision"] == "PASS":
        status = PASS
    elif res["decision"] == "FAIL" and kind == KIND_CHALLENGE:
        status = FAIL
    if status is None:
        if not ctx.is_validator and res["decision"] != "abstain":
            rec["reason"] = "watch_only"
        elif res["decision"] == "FAIL":
            rec["reason"] = "no_vote_appeal_fail"
        ctx.metrics.emit("vote", tx=None, **rec)
        _drop(ctx, wid, "abstained", rec["reason"] or "abstain")
        reason = str(res.get("reason") or "")
        if kind == KIND_CHALLENGE and reason.startswith("static_violation:") and cfg.watcher:
            from sidecar.court import enqueue_static
            enqueue_static(ctx, target, reason)
        return
    tx, err = ctx.send("vote", wid, "submitVerdict", (int(wid), status, _unhx(res["transcript"])))
    rec["tx"] = None if tx is None else _hx(tx)
    ctx.metrics.emit("vote", **rec)
    if err is not None:
        _drop(ctx, wid, "voted" if err == "E.VOTE.DUP" else "failed", err)
        return
    w["status"], w["tx"] = "sent", _hx(tx)
    ctx.state["queue"].remove(wid)


def on_vote_receipt(ctx, key: str, r) -> None:
    w = ctx.state["windows"].get(key)
    if w is None:
        return
    ctx.metrics.emit("vote_included", id=int(key), tx=_hx(r.tx), block=r.block, ok=r.ok, error=r.error,
                     latency_blocks=r.block - int(w["opened_at"]), gas_used=r.gas_used)
    if r.ok or r.error == "E.VOTE.DUP":
        w["status"] = "included" if r.ok else "voted"
    else:
        w["status"], w["reason"] = "failed", r.error


def on_vote_lost(ctx, key: str, head: int) -> None:
    """A vote transaction never appeared (dropped from the pool): re-queue it while the window is still open; the
    ``voted`` view before the next send prevents a duplicate."""
    w = ctx.state["windows"].get(key)
    if w is not None and head < _deadline(w) - ctx.cfg.deadline_margin and key not in ctx.state["queue"]:
        w["status"] = "queued"
        ctx.state["queue"].append(key)


__all__ = ["on_challenge_opened", "on_appeal_opened", "on_verdict", "on_finalized", "process_queue",
           "on_vote_receipt", "on_vote_lost", "BYZ_FAIL_TAG"]
