"""sidecar.metrics - the per-sidecar metrics JSONL.

``Metrics(path)``; ``emit(kind, **fields)`` appends one line ``{"t": unix time (ms precision), "v": validator index,
"kind": kind, ...}`` and flushes (``path`` None keeps the lines in ``.lines`` only). Kinds and their fields:

| kind | fields |
|---|---|
| ``vote`` | ``id, target, window ("challenge" / "appeal"), opened_at, h, decision, reason, transcript, compute_s, peak_rss_mb, source, fetch_log, sent_head, tx`` |
| ``vote_included`` | ``id, tx, block, ok, error, latency_blocks`` (from the window's opening block) |
| ``abstain`` | ``id, reason`` (also ``deadline``, ``overload``, ``silent``, ``no_vote_appeal``) |
| ``pulse`` | ``epoch, boundary, sent_head, tx``; ``pulse_included``: ``epoch, block, offset`` (block − boundary) |
| ``finalize`` | ``fn, key, tx`` / ``finalize_skip``: ``fn, key, reason`` |
| ``enrol`` | ``key_id, decision ("confirm" / "refuse"), reason, tx`` |
| ``revocation`` | ``key_id, status ("REVOKE" / "KEEP"), entry_hash, round, tx`` |
| ``court`` | ``claim_id, proof ("static" / "block" / "feed"), detail, check, result, tx``; ``court_included``: ``tx, ok, error, gas_used`` |
| ``alarm`` | ``height, own, others`` (cross-node hash disagreement); ``reorg``: ``height, expected, got`` |
| ``backoff`` | ``error, delay_s`` |
| ``tx_failed`` | ``purpose, key, error`` |
"""
from __future__ import annotations

import json
import os
import time


class Metrics:
    def __init__(self, path: str | None, index: int = 0):
        self.path, self.index, self.lines = path, int(index), []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def emit(self, kind: str, **fields) -> dict:
        rec = {"t": round(time.time(), 3), "v": self.index, "kind": kind}
        rec.update(fields)
        self.lines.append(rec)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, sort_keys=True, default=_default) + "\n")
        return rec

    def of(self, kind: str) -> list:
        return [x for x in self.lines if x["kind"] == kind]


def _default(o):
    if isinstance(o, (bytes, bytearray)):
        return "0x" + bytes(o).hex()
    raise TypeError(type(o).__name__)


__all__ = ["Metrics"]
