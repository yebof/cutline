"""sidecar.runner - one vote computation, inline or in an isolated child.

``compute_vote(job) -> dict`` runs ``cutline.verify.vote.vote_any`` over the lazily fetched bundle copies
(``sources.Copies``) at the height ``job["h"]`` (the ChallengeOpened / AppealOpened block, so the tier-T table, kfam
and class reads are identical across validators). ``job`` keys: ``target`` (bytes32), ``descriptors``
(``sources.source_list``), ``ledger`` (a picklable ``LedgerView``), ``h``, ``limit``, ``timeout``, ``cache_dir``,
``chain_id``/``registry`` (None: the ledger's). The result:

``{"decision": "PASS" | "FAIL" | "abstain", "transcript": hex | None, "reason": str | None, "rows_head": [[...]],
"source": descriptor | None, "fetch_log": [[descriptor, reason]], "compute_s": float}``

A copy that produced a vote or a claim-wide abstention (``static_violation:*``, ``not_path_b``) is stored in the local
cache (the peer cache server and the court duty read it there); a ``no_verifying_bundle:*`` abstention stores
nothing.

``run_vote(job, isolation, rss_mb, timeout_s) -> dict``: ``"inline"`` calls ``compute_vote`` directly; ``"process"``
runs it in a ``spawn`` child whose RSS the parent polls every 100 ms (``/proc/<pid>/status`` VmRSS; killed above
``rss_mb``: ``reason "rss_limit"``) and whose wall time is capped (``"timeout"``); a child that dies gives
``"child_died:<exitcode>"``. These are local failures: the sidecar abstains (it holds no computed vote), and the
reason is logged. Exception: a child ended by a shutdown signal (SIGTERM / SIGINT / SIGHUP, e.g. a stop of the
supervisor) returns ``"interrupted": True``; the duty keeps its item queued and retries it after the restart (restart
idempotence) instead of recording a final abstention. The reference verifier needs about 1.7 GB at the bundle size
bounds, hence the 1.8 GB default.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time


def _row(r) -> list:
    return [int(v) for v in r]


def compute_vote(job: dict) -> dict:
    from cutline.verify.vote import vote_any
    from sidecar.sources import BundleCache, Copies
    t0 = time.perf_counter()
    cache = BundleCache(job.get("cache_dir"))
    target = bytes(job["target"])
    copies = Copies(job["descriptors"], job.get("limit", 256 << 20), job.get("timeout", 20.0), cache, target)
    v = vote_any(copies, target, job["ledger"], job.get("chain_id"), job.get("registry"), h=job.get("h"))
    out = {"decision": v[0], "transcript": None, "reason": None, "rows_head": [], "source": None,
           "fetch_log": [list(x) for x in copies.log]}
    if v[0] == "PASS":
        out["transcript"] = "0x" + v[1].hex()
    elif v[0] == "FAIL":
        out["transcript"] = "0x" + v[2].hex()
        out["rows_head"] = [_row(r) for r in list(v[1])[:8]]
    else:
        out["reason"] = str(v[1])
    claim_wide = v[0] != "abstain" or not str(v[1]).startswith("no_verifying_bundle")
    if claim_wide and copies.last_bytes is not None:
        out["source"] = copies.last
        if copies.last != "cache:":
            cache.put(target, copies.last_bytes)
    out["compute_s"] = round(time.perf_counter() - t0, 4)
    return out


TASKS = {"vote": ("sidecar.runner", "compute_vote"), "court": ("sidecar.court", "compute_court")}


def _task(name: str):
    import importlib
    mod, fn = TASKS[name]
    return getattr(importlib.import_module(mod), fn)


def _child(conn, task, job):
    try:
        conn.send(("ok", _task(task)(job)))
    except BaseException as e:           # noqa: BLE001 (report anything to the parent)
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _rss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


SHUTDOWN_EXIT = (-15, -2, -1)        # the child was ended by SIGTERM / SIGINT / SIGHUP (a supervisor stop)


def _abstain(reason: str, t0: float, peak: float = 0.0) -> dict:
    return {"decision": "abstain", "transcript": None, "reason": reason, "rows_head": [], "source": None,
            "fetch_log": [], "compute_s": round(time.perf_counter() - t0, 4), "peak_rss_mb": round(peak, 1)}


def run_vote(job: dict, isolation: str = "process", rss_mb: int = 1800, timeout_s: float = 240.0) -> dict:
    return run_task("vote", job, isolation, rss_mb, timeout_s)


def run_task(task: str, job: dict, isolation: str = "process", rss_mb: int = 1800, timeout_s: float = 240.0) -> dict:
    """``run_vote`` for any task of ``TASKS`` (``"vote"``, ``"court"``); failures give ``decision "abstain"``."""
    t0 = time.perf_counter()
    if isolation == "inline":
        try:
            return _task(task)(job)
        except MemoryError:
            return _abstain("rss_limit", t0)
        except Exception as e:                  # noqa: BLE001 (a local failure: abstain and log)
            return _abstain(f"error:{type(e).__name__}: {e}"[:200], t0)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, task, job), daemon=True)
    p.start()
    child.close()
    peak, res = 0.0, None
    try:
        while True:
            if parent.poll(0.1):
                try:
                    res = parent.recv()
                except EOFError:
                    res = None
                break
            if not p.is_alive():
                if parent.poll(0.05):
                    try:
                        res = parent.recv()
                    except EOFError:
                        res = None
                break
            peak = max(peak, _rss_mb(p.pid))
            if peak > rss_mb:
                p.kill()
                p.join(5)
                return _abstain("rss_limit", t0, peak)
            if time.perf_counter() - t0 > timeout_s:
                p.kill()
                p.join(5)
                return _abstain("timeout", t0, peak)
    finally:
        parent.close()
    p.join(10)
    if res is None:
        out = _abstain(f"child_died:{p.exitcode}", t0, peak)
        if p.exitcode in SHUTDOWN_EXIT:
            out["interrupted"] = True
        return out
    if res[0] == "err":
        return _abstain("error:" + res[1][:200], t0, peak)
    out = res[1]
    out["peak_rss_mb"] = round(peak, 1)
    out["wall_s"] = round(time.perf_counter() - t0, 4)
    return out


__all__ = ["compute_vote", "run_vote", "run_task", "TASKS"]

if os.name != "posix":           # pragma: no cover
    raise ImportError("the sidecar runner needs /proc (Linux)")
