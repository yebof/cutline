"""sidecar.state - the durable per-validator state file.

``DurableState(path)`` keeps one JSON object, written atomically (temp file in the same directory, flush, fsync,
rename, directory fsync), so a kill at any point leaves either the old or the new file. ``path`` None keeps the state
in memory only (unit tests). Restart idempotence comes from this file **plus** the on-chain views checked before
every send (``voted(id, me)``, ``revocationVoted(keyId, round, me)``, ``devices(keyId).state``, ``claimInfo``): a
vote whose transaction was sent but not yet seen is recorded in ``pending`` and is never re-sent while it may still
land.

Top-level keys (``SCHEMA``; hex strings for bytes):

- ``last_scanned``: the last block whose logs were dispatched; ``heads``: {height: hash} of the recent heads
  (continuity check), ``reorgs``: count;
- ``pulses``: {epoch: {"tx", "sent_head"}} (one per epoch; epoch = boundary height div 15);
- ``windows``: {id: {"kind", "target", "opened_at", "status", "decision", "reason", "tx", "finalised",
  "compute_s"}}: challenge (kind 1) and appeal (kind 2) windows seen in the logs;
- ``locators``: {target: [locator, ...]} of every ``ChallengeOpened`` on a target (bundle sources);
- ``queue``: [window id, ...] still to be voted on (earliest deadline first);
- ``keys``: {keyId: {"holder", "pk_hash", "att_hash", "rate_n", "nonce_e", "ders", "status"}} (enrolments);
- ``rev``: {keyId: {"round", "opened_at", "finalised"}} (revocation windows seen); ``rev_votes``: {"keyId:round":
  {"status", "entry_hash", "tx"}};
- ``court``: {claimId: {"kind", "tx", "status", "detail"}}; ``feed``: {claimId: "done" | reason} (watcher);
- ``pending``: {tx: {"purpose", "key", "sent_head", "fn"}} transactions sent and not yet seen in a receipt;
- ``alarms``: [{"height", "own", "others", "t"}] (cross-node alarm); ``last_revocation_poll``: unix time.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile

SCHEMA_VERSION = 1
SCHEMA = {"version": SCHEMA_VERSION, "last_scanned": None, "heads": {}, "reorgs": 0, "pulses": {}, "windows": {},
          "locators": {}, "queue": [], "keys": {}, "rev": {}, "rev_votes": {}, "court": {}, "feed": {},
          "pending": {}, "alarms": [], "last_revocation_poll": 0.0}
HEADS_KEPT = 128
PULSES_KEPT = 64


class DurableState:
    def __init__(self, path: str | None):
        self.path = path
        self.data = copy.deepcopy(SCHEMA)
        if path and os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            if d.get("version") != SCHEMA_VERSION:
                raise ValueError(f"state file {path} has version {d.get('version')}, not {SCHEMA_VERSION}")
            for k, v in SCHEMA.items():
                d.setdefault(k, copy.deepcopy(v))
            self.data = d

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v

    def prune(self) -> None:
        hs = self.data["heads"]
        if len(hs) > HEADS_KEPT:
            for k in sorted(hs, key=int)[:-HEADS_KEPT]:
                del hs[k]
        ps = self.data["pulses"]
        if len(ps) > PULSES_KEPT:
            for k in sorted(ps, key=int)[:-PULSES_KEPT]:
                del ps[k]

    def save(self) -> None:
        self.prune()
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, sort_keys=True, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


__all__ = ["DurableState", "SCHEMA", "SCHEMA_VERSION"]
