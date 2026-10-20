"""sidecar.court - the watcher duty: static proofs (path a') and block-failure proofs (path a).

Items enter ``state.court[claimId]`` as

- ``"static"``: a challenge vote abstained ``static_violation:r,i,j`` (the operator, or any WATCHER, then submits
  ``proveStaticViolation``); ``enqueue_static(ctx, target, reason)``;
- ``"feed"``: the watcher feed of tier-E claims: ``ClaimRegistered`` with ``core.tier = 1`` when ``watch_claims``
  (``on_claim_registered``), or ``report(ctx, claim_id)``.

``compute_court(job) -> dict`` (run through ``runner.run_task("court", …)``, i.e. isolated like a vote): the first
copy from the sources that passes stage A for the target, then

- ``static``: the first violated static-rule row at height ``h`` (``static_rules.first_violation`` with the
  ledger's tier-T map and kfam), ``StaticIn`` with exactly the row's preimages, **checked with the reference model**
  ``static_rules.prove_static_violation`` (all of the contract's steps) before any calldata is returned;
- ``feed``: a tier-T or 4:2:0 claim is ``path_b`` (challenges, not the court); otherwise stage B (a static violation
  goes the static way), then ``verify_e`` to the first failing TE row, the fraud unit
  ``verify.fraud.prove_block`` → ``ProofIn`` (``proofin.build_proofin(parts_from_bundle(raw), mcu, comp, k)``),
  **checked with** ``proofin.court_reference`` (``fraud.court_check`` = ``"BlockFailure"``).

Result: ``{"result": "static" | "block" | "pass" | "path_b" | "no_violation" | "no_bundle" | "not_verifying:<E>" |
"check:<outcome>", "calldata": hex | None, "detail": ..., "check": ...}``.

``process(ctx, head)``: one queued item per call; ``claimInfo`` first (unknown claim → ``unknown``; already
court-failed → ``done``); the calldata is sent as ``proveStaticViolation`` / ``proveBlockFailure`` after an
``eth_call`` pre-check. Receipts give the ``court_included`` metric with the gas used.
"""
from __future__ import annotations

from sidecar.sources import BundleCache, Copies, source_list


def _hx(b):
    return "0x" + bytes(b).hex()


def _unhx(s):
    return bytes.fromhex(s[2:] if s.startswith("0x") else s)


def enqueue_static(ctx, target: bytes, reason: str = "") -> None:
    k = _hx(target)
    c = ctx.state["court"].get(k)
    if c is None or c.get("status") in ("no_bundle",):
        ctx.state["court"][k] = {"kind": "static", "status": "queued", "reason": reason}


def report(ctx, claim_id: bytes) -> None:
    k = _hx(claim_id)
    if k not in ctx.state["court"]:
        ctx.state["court"][k] = {"kind": "feed", "status": "queued"}


def on_claim_registered(ctx, ev) -> None:
    if not (ctx.cfg.watcher and ctx.cfg.watch_claims):
        return
    core = ev.args.get("core")
    tier = int(core[5]) if core is not None else None          # ClaimCore field 6 = tier
    if tier == 1:
        report(ctx, ev.args["claimId"])


# ------------------------------------------------------------------------------------------------ child side
def _raw_for(job):
    from cutline import bundle as B
    from cutline.errors import CutlineError
    led = job["ledger"]
    cid = led.chain_id if job.get("chain_id") is None else job["chain_id"]
    reg = led.registry if job.get("registry") is None else job["registry"]
    copies = Copies(job["descriptors"], job.get("limit", 256 << 20), job.get("timeout", 20.0),
                    BundleCache(job.get("cache_dir")), bytes(job["target"]))
    last = "no_bundle"
    for data in copies:
        try:
            raw = B.stage_a(data, cid, reg)
        except CutlineError as e:
            last = "not_verifying:" + e.name
            continue
        if raw.claim_id != bytes(job["target"]):
            last = "target"
            continue
        return raw, copies
    return None, last


def _static(raw, led, h):
    from cutline.compiler.static_rules import (LedgerState, StaticIn, encode_static_calldata, preimages,
                                               prove_static_violation)
    from cutline.errors import CutlineError
    from cutline.verify.stages import listing_of, stage_b
    lst = listing_of(raw)
    fv = stage_b(raw, led, h, lst.header)
    if fv is None:
        return {"result": "no_violation", "calldata": None, "detail": None, "check": None}
    rule, i, j = fv
    c = raw.claim
    s = StaticIn(raw.nf, c.claimant, c.core, rule, i, j,
                 *preimages(rule, i, c.prog, lst.header, lst.r_mcu, c.rec, raw.proof.cut_pre))
    st = LedgerState(claims={raw.claim_id}, tier_t=dict(led.tier_t_map(h)), kfam=dict(led.kfam(h)))
    try:
        chk = prove_static_violation(s, st)
    except CutlineError as e:
        chk = e.name
    if chk != "StaticViolation":
        return {"result": "check:" + chk, "calldata": None, "detail": [rule, i, j], "check": chk}
    return {"result": "static", "calldata": _hx(encode_static_calldata(s)), "detail": [rule, i, j], "check": chk}


def compute_court(job: dict) -> dict:
    import time
    t0 = time.perf_counter()
    raw, info = _raw_for(job)
    if raw is None:
        return {"result": info, "calldata": None, "detail": None, "check": None, "compute_s": 0.0}
    led, h = job["ledger"], job.get("h")
    if job["kind"] == "static":
        out = _static(raw, led, h)
    else:
        out = _feed(raw, led, h)
    out["compute_s"] = round(time.perf_counter() - t0, 4)
    out["source"] = info.last
    if out["calldata"] is not None and info.last != "cache:":
        BundleCache(job.get("cache_dir")).put(raw.claim_id, info.last_bytes)
    return out


def _feed(raw, led, h):
    from cutline.ledger.proofin import encode_proofin_calldata
    from cutline.verify.fraud import court_check
    from cutline.verify.stages import listing_of
    from cutline.verify.verify_e import verify_e
    lst = listing_of(raw)
    if raw.claim.core.tier == 2 or lst.g.sampling != 0:
        return {"result": "path_b", "calldata": None, "detail": None, "check": None}
    st = _static(raw, led, h)
    if st["result"] != "no_violation":
        return st
    r = verify_e(raw, led, h=h, all_rows=False, fraud=True, listing=lst)
    if r.g0_reason is not None and not r.rows:
        return {"result": "not_verifying:" + str(r.error or r.g0_reason), "calldata": None, "detail": None,
                "check": None}
    if not r.rows:
        return {"result": "pass", "calldata": None, "detail": None, "check": None}
    fu = r.fraud
    chk = court_check(fu.proof)
    detail = [int(v) for v in fu.row]
    if chk != "BlockFailure":
        return {"result": "check:" + chk, "calldata": None, "detail": detail, "check": chk}
    return {"result": "block", "calldata": _hx(encode_proofin_calldata(fu.proof)), "detail": detail, "check": chk}


# ------------------------------------------------------------------------------------------------ sidecar side
def process(ctx, head: int) -> int:
    if not ctx.cfg.watcher:
        return 0
    for k, c in ctx.state["court"].items():
        if c.get("status") != "queued":
            continue
        target = _unhx(k)
        info = ctx.port.view("claimInfo", (target,))
        if not info[0]:
            c["status"] = "unknown"
            return 1
        if info[7]:
            c["status"] = "done"
            return 1
        locs = ctx.state["locators"].get(k, [])
        desc = source_list(ctx.cfg, target, locs, ctx.cache)
        if "cache:" in desc:                          # the copy the vote used is the cheapest source
            desc.remove("cache:")
            desc.insert(0, "cache:")
        job = {"kind": c["kind"], "target": target, "descriptors": desc, "ledger": ctx.ledger(), "h": head,
               "limit": ctx.cfg.fetch_limit, "timeout": ctx.cfg.fetch_timeout_s, "cache_dir": ctx.cfg.cache_dir,
               "chain_id": ctx.cfg.chain_id, "registry": None if ctx.cfg.registry is None else _unhx(ctx.cfg.registry)}
        res = ctx.run_task("court", job, ctx.cfg.vote_timeout_s)
        if res.get("interrupted") and int(c.get("interrupts", 0)) < 3:   # a shutdown: retry after the restart
            c["interrupts"] = int(c.get("interrupts", 0)) + 1
            return 1
        c["result"], c["detail"], c["check"] = res.get("result"), res.get("detail"), res.get("check")
        if res.get("decision") == "abstain":                   # a local failure of the child (timeout, rss)
            c["status"], c["result"] = "failed", res.get("reason")
            return 1
        if not res.get("calldata"):
            c["status"] = res.get("result") or "none"
            ctx.metrics.emit("court", claim_id=k, proof=c["kind"], detail=res.get("detail"), check=res.get("check"),
                             result=res.get("result"), tx=None)
            return 1
        fn = "proveStaticViolation" if res["result"] == "static" else "proveBlockFailure"
        tx, err = ctx.send("court", k, fn, (), calldata=_unhx(res["calldata"]))
        c["status"] = "sent" if err is None else "rejected"
        c["tx"], c["error"], c["fn"] = None if tx is None else _hx(tx), err, fn
        ctx.metrics.emit("court", claim_id=k, proof="static" if fn == "proveStaticViolation" else "block",
                         detail=res.get("detail"), check=res.get("check"), result=res.get("result"),
                         tx=c["tx"], error=err, calldata_len=len(res["calldata"]) // 2 - 1)
        return 1
    return 0


def on_receipt(ctx, key: str, r) -> None:
    c = ctx.state["court"].get(key)
    if c is not None:
        c["status"] = "included" if r.ok else "reverted"
        c["gas_used"], c["error"] = r.gas_used, r.error
    ctx.metrics.emit("court_included", claim_id=key, tx=_hx(r.tx), block=r.block, ok=r.ok, error=r.error,
                     gas_used=r.gas_used)


__all__ = ["enqueue_static", "report", "on_claim_registered", "compute_court", "process", "on_receipt"]
