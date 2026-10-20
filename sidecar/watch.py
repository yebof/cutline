"""sidecar.watch - head follower, cross-node alarm and log scanner.

- ``cross_node_alarm(own, others, f) -> bool``: the status-equivocation detector: True iff the own node's hash at a
  height differs from the reports of **at least f+1 other nodes** (``others``: the peers' hashes at that height,
  ``None`` for a peer that has no block there or did not answer; those never count).
- ``HeadFollower(port, peers, state, metrics, f, lag)``: ``step() -> head``. Records the recent heads, checks
  parent-hash continuity (a mismatch is a ``reorg`` metric; QBFT has immediate finality, so on a correct chain it
  never fires) and runs the cross-node alarm at ``head − lag`` once per height (``alarm`` metric and ``state.alarms``).
  ``peers`` are objects with ``block_hash(h)`` (other nodes' ``ChainPort`` / RPC readers).
- ``LogScanner(port, state, log_range, start_block)``: ``step(head, dispatch)`` reads ``logs`` from
  ``last_scanned + 1`` to ``head`` in ranges of at most ``log_range`` blocks, calls ``dispatch(event)`` in chain order
  and saves ``last_scanned`` after every range (so a restart re-dispatches at most one range; the handlers are
  idempotent).
"""
from __future__ import annotations

from sidecar.chain import W_VOTE

HEAD_CHECK = 16          # heights re-read per step at most (continuity)


def cross_node_alarm(own: bytes | None, others, f: int) -> bool:
    if own is None:
        return False
    diff = sum(1 for x in others if x is not None and bytes(x) != bytes(own))
    return diff >= f + 1


def _hx(b):
    return None if b is None else "0x" + bytes(b).hex()


class HeadFollower:
    def __init__(self, port, peers, state, metrics, f: int, lag: int = 2):
        self.port, self.peers, self.state, self.metrics = port, list(peers), state, metrics
        self.f, self.lag = int(f), int(lag)

    def step(self) -> int:
        head = self.port.head()
        heads = self.state["heads"]
        known = [int(k) for k in heads]
        start = max(max(known) + 1 if known else head, head - HEAD_CHECK + 1, 0)
        for h in range(start, head + 1):
            hh = self.port.block_hash(h)
            if hh is None:
                break
            prev = heads.get(str(h - 1))
            if prev is not None:
                par = self.port.parent_hash(h)
                if par is not None and _hx(par) != prev:
                    self.state["reorgs"] += 1
                    self.metrics.emit("reorg", height=h, expected=prev, got=_hx(par))
            heads[str(h)] = _hx(hh)
        self._alarm(head)
        return head

    def _alarm(self, head: int) -> None:
        if not self.peers:
            return
        h = head - self.lag
        last = self.state.data.setdefault("alarm_checked", None)
        if h < 0 or (last is not None and h <= last):
            return
        for hc in range(max(h - HEAD_CHECK + 1, (last + 1) if last is not None else h), h + 1):
            own = self.port.block_hash(hc)
            others = []
            for p in self.peers:
                try:
                    others.append(p.block_hash(hc))
                except Exception:                   # noqa: BLE001 (a silent peer reports nothing)
                    others.append(None)
            if cross_node_alarm(own, others, self.f):
                rec = {"height": hc, "own": _hx(own), "others": [_hx(x) for x in others]}
                self.state["alarms"].append(rec)
                self.metrics.emit("alarm", **rec)
        self.state["alarm_checked"] = h


class LogScanner:
    def __init__(self, port, state, log_range: int = 500, start_block: int | None = None):
        self.port, self.state, self.range, self.start_block = port, state, int(log_range), start_block

    def step(self, head: int, dispatch) -> int:
        last = self.state["last_scanned"]
        if last is None:
            last = (self.start_block - 1) if self.start_block is not None else max(-1, head - 2 * W_VOTE - 1)
            self.state["last_scanned"] = last
        n = 0
        while last < head:
            a, b = last + 1, min(head, last + self.range)
            for ev in self.port.logs(a, b):
                dispatch(ev)
                n += 1
            last = b
            self.state["last_scanned"] = b
            self.state.save()
        return n


__all__ = ["cross_node_alarm", "HeadFollower", "LogScanner"]
