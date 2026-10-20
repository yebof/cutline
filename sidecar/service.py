"""sidecar.service - the per-validator loop and the supervisor.

``Sidecar(cfg, port, peers=(), state=None, metrics=None)``: one validator's sidecar over a ``ChainPort``.

- ``tick() -> dict``: one poll iteration, in this order: the head follower (continuity, cross-node alarm), the log
  scanner (``ChallengeOpened``, ``AppealOpened``, ``Verdict``, ``ChallengeFinalized``, ``AppealFinalized``,
  ``Pulse``, ``Enrolled``, ``Activated``, ``RevocationVoted``, ``RevocationFinalized``, ``Revoked``,
  ``ClaimRegistered``), receipts of pending transactions, the pulse, **at most one** vote computation, finalisation,
  one enrolment check, the revocation poll, one court item; then the state is saved. Returns counters.
- ``run(stop=None, max_ticks=None)``: ``tick()`` every ``poll_s``; a ``ChainUnavailable`` / ``OSError`` backs off
  exponentially (0.5 s doubling to 30 s, ``backoff`` metric) without advancing any state, so a node restart or an
  overloaded RPC only delays the duties.
- helpers used by the duties: ``send(purpose, key, fn, args, calldata=None) -> (tx | None, error | None)`` (an
  ``eth_call`` pre-check, then the transaction; recorded in ``state.pending``), ``silent(head)``, ``ledger()``,
  ``chain_id()``, ``run_vote(job, budget_s)``, ``run_task(task, job, budget_s)``.
- Pending transactions: a receipt is routed to its duty (vote → ``vote_included``; court → ``court_included``;
  failures → ``tx_failed``); a transaction unseen for ``TX_LOST_BLOCKS`` (30) blocks is dropped and its duty may
  retry (a vote only after the ``voted`` view says it has not landed).

``Supervisor(configs, python=sys.executable)``: runs one child process per validator config
(``python -m sidecar one --config <file>``), restarts a child that exits (backoff 1 s doubling to 60 s), and stops
them all on ``stop()`` / SIGTERM. The vote children are spawned by each sidecar, so a memory cap on the
supervisor's cgroup covers them too.

``write_configs(base, validators, out_dir, overrides=None) -> [path]`` writes the per-sidecar JSON configs.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from sidecar import court, enrol, finalize, pulse, revocation, verdict
from sidecar.chain import ChainUnavailable, Revert
from sidecar.metrics import Metrics
from sidecar.runner import run_task, run_vote
from sidecar.sources import BundleCache
from sidecar.state import DurableState
from sidecar.watch import HeadFollower, LogScanner

TX_LOST_BLOCKS = 30


def _hx(b):
    return "0x" + bytes(b).hex()


class Sidecar:
    def __init__(self, cfg, port, peers=(), state=None, metrics=None):
        self.cfg, self.port = cfg, port
        cfg.ensure_dirs()
        self.state = state if state is not None else DurableState(cfg.state_path)
        self.metrics = metrics if metrics is not None else Metrics(cfg.metrics_path, cfg.index)
        self.me = bytes(port.address)
        self.cache = BundleCache(cfg.cache_dir)
        self.status_list = None
        self.rev_dirty = False
        self._f = None
        self.peers = list(peers)
        self.follower = None
        self.scanner = LogScanner(port, self.state, cfg.log_range, cfg.start_block)
        self.head = None
        self.handlers = {
            "ChallengeOpened": verdict.on_challenge_opened, "AppealOpened": verdict.on_appeal_opened,
            "Verdict": verdict.on_verdict, "ChallengeFinalized": verdict.on_finalized,
            "AppealFinalized": verdict.on_finalized, "Pulse": pulse.on_pulse, "Enrolled": enrol.on_enrolled,
            "Activated": enrol.on_activated, "RevocationVoted": revocation.on_revocation_voted,
            "RevocationFinalized": revocation.on_revocation_finalized, "Revoked": revocation.on_revoked,
            "ClaimRegistered": court.on_claim_registered,
        }

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def f(self) -> int:
        if self._f is None:
            self._f = int(self.port.view("f", ())[0])
        return self._f

    @property
    def is_validator(self) -> bool:
        if getattr(self, "_is_val", None) is None:
            self._is_val = bool(self.port.view("isValidator", (self.me,))[0])
        return self._is_val

    def ledger(self):
        return self.port.ledger()

    def chain_id(self) -> int:
        return int(self.cfg.chain_id) if self.cfg.chain_id is not None else int(self.ledger().chain_id)

    def silent(self, head: int) -> bool:
        return self.cfg.byzantine == "silent" and (self.cfg.silent_until is None or head < self.cfg.silent_until)

    def send(self, purpose: str, key: str, fn: str, args: tuple, calldata: bytes | None = None):
        err = self.port.simulate(fn, args, calldata)
        if err is not None:
            return None, err
        try:
            tx = self.port.send(fn, args, calldata)
        except Revert as e:                              # a gas estimate that reverts, a node rejection
            return None, e.name
        self.state["pending"][_hx(tx)] = {"purpose": purpose, "key": key, "fn": fn,
                                          "sent_head": self.head if self.head is not None else self.port.head()}
        return tx, None

    def run_vote(self, job: dict, budget_s: float) -> dict:
        return run_vote(job, self.cfg.isolation, self.cfg.vote_rss_mb, budget_s)

    def run_task(self, task: str, job: dict, budget_s: float) -> dict:
        return run_task(task, job, self.cfg.isolation, self.cfg.vote_rss_mb, budget_s)

    def dispatch(self, ev) -> None:
        h = self.handlers.get(ev.name)
        if h is not None:
            h(self, ev)

    # ------------------------------------------------------------------------------------------ receipts
    def _receipts(self, head: int) -> None:
        for txh, p in list(self.state["pending"].items()):
            r = self.port.receipt(bytes.fromhex(txh[2:]))
            if r is None:
                if head - int(p["sent_head"]) > TX_LOST_BLOCKS:
                    del self.state["pending"][txh]
                    self.metrics.emit("tx_failed", purpose=p["purpose"], key=p["key"], error="lost", tx=txh)
                    self._lost(p, head)
                continue
            del self.state["pending"][txh]
            if p["purpose"] == "vote":
                verdict.on_vote_receipt(self, p["key"], r)
            elif p["purpose"] == "court":
                court.on_receipt(self, p["key"], r)
            if not r.ok and p["purpose"] != "court":
                self.metrics.emit("tx_failed", purpose=p["purpose"], key=p["key"], error=r.error, tx=txh)

    def _lost(self, p: dict, head: int) -> None:
        if p["purpose"] == "vote":
            verdict.on_vote_lost(self, p["key"], head)
        elif p["purpose"] == "finalize":
            finalize.on_lost(self, p["key"])
        elif p["purpose"] == "pulse":
            self.state["pulses"].pop(p["key"], None)
        elif p["purpose"] == "revocation":
            self.state["rev_votes"].pop(p["key"], None)
        elif p["purpose"] == "confirm":
            rec = self.state["keys"].get(p["key"])
            if rec is not None:
                rec["status"] = "pending"
        elif p["purpose"] == "court":
            c = self.state["court"].get(p["key"])
            if c is not None:
                c["status"] = "queued"

    # ------------------------------------------------------------------------------------------ loop
    def tick(self) -> dict:
        if self.follower is None:
            self.follower = HeadFollower(self.port, self.peers, self.state, self.metrics, self.f, self.cfg.alarm_lag)
        head = self.follower.step()
        self.head = head
        n_ev = self.scanner.step(head, self.dispatch)
        self._receipts(head)
        pulse.on_head(self, head)
        n_vote = verdict.process_queue(self, head)
        n_fin = finalize.process(self, head)
        n_enr = enrol.process(self, head)
        n_rev = revocation.process(self, head)
        n_court = court.process(self, head)
        self.state.save()
        return {"head": head, "events": n_ev, "votes": n_vote, "finalize": n_fin, "enrol": n_enr,
                "revocation": n_rev, "court": n_court, "queue": len(self.state["queue"])}

    def run(self, stop=None, max_ticks: int | None = None) -> None:
        delay, ticks = 0.5, 0

        def pause(s):          # interruptible by the stop event (a supervisor stop must not wait out a 30-s backoff)
            if stop is not None:
                stop.wait(s)
            else:
                time.sleep(s)
        while not (stop is not None and stop.is_set()) and (max_ticks is None or ticks < max_ticks):
            try:
                self.tick()
                delay = 0.5
                pause(self.cfg.poll_s)
            except (ChainUnavailable, OSError) as e:
                self.metrics.emit("backoff", error=f"{type(e).__name__}: {e}"[:200], delay_s=delay)
                pause(delay)
                delay = min(delay * 2, 30.0)
            ticks += 1


class Supervisor:
    def __init__(self, config_paths, python: str = sys.executable, log_dir: str | None = None):
        self.paths, self.python, self.log_dir = list(config_paths), python, log_dir
        self.procs, self.backoff = {}, {}
        self._stop = False

    def _start(self, i: int) -> None:
        env = dict(os.environ)
        code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = code + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        out = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            out = open(os.path.join(self.log_dir, f"sidecar{i}.log"), "a")
        self.procs[i] = subprocess.Popen([self.python, "-m", "sidecar", "one", "--config", self.paths[i]], env=env,
                                         stdout=out, stderr=subprocess.STDOUT if out else None)

    def stop(self, *_):
        self._stop = True
        for p in self.procs.values():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        end = time.time() + 20.0            # one shared deadline for all children
        for p in self.procs.values():
            try:
                p.wait(max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        for i in range(len(self.paths)):
            self._start(i)
            self.backoff[i] = 1.0
        next_try = {}
        while not self._stop:
            for i, p in list(self.procs.items()):
                if p.poll() is not None and not self._stop:
                    t = next_try.get(i)
                    if t is None:
                        next_try[i] = time.time() + self.backoff[i]
                        self.backoff[i] = min(self.backoff[i] * 2, 60.0)
                    elif time.time() >= t:
                        del next_try[i]
                        self._start(i)
            time.sleep(0.5)


def write_configs(base: dict, validators: list, out_dir: str, overrides: dict | None = None) -> list:
    """One JSON config per validator: ``base`` (shared fields) + ``validators[k]`` (index, key, rpc, ...) +
    ``overrides[index]``; ``state_path`` / ``metrics_path`` / ``cache_dir`` default under ``out_dir``."""
    from sidecar.config import SidecarConfig
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for v in validators:
        d = dict(base)
        d.update(v)
        d.update((overrides or {}).get(int(v["index"]), {}))
        i = int(d["index"])
        d.setdefault("state_path", os.path.join(out_dir, "state", f"v{i}.json"))
        d.setdefault("metrics_path", os.path.join(out_dir, "metrics", f"v{i}.jsonl"))
        d.setdefault("cache_dir", os.path.join(out_dir, "cache", f"v{i}"))
        SidecarConfig.from_json(d)                      # validate
        p = os.path.join(out_dir, f"sidecar_v{i}.json")
        with open(p, "w") as f:
            json.dump(d, f, indent=1, sort_keys=True)
        paths.append(p)
    return paths


__all__ = ["Sidecar", "Supervisor", "write_configs", "TX_LOST_BLOCKS"]
