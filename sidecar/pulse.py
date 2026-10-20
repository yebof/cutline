"""sidecar.pulse - the beacon pulse.

``on_head(ctx, head)``: with ``boundary = head − head mod 15`` (the latest h ≡ 0 (mod 15) at or below the head) and
``epoch = boundary div 15``, the sidecar sends ``CaptureAnchor.pulse(r)`` with 32 fresh random bytes
(``os.urandom``) once per epoch, i.e. for the epoch whose first block is boundary + 1 (it influences
``blockhash(boundary + 15)``). A head observed later than ``boundary + PULSE_LATEST`` (13) skips the epoch: the pulse
could not land before the next boundary. ``pulse = False`` (``--no-pulse``) and ``byzantine = "silent"`` send
nothing.

``on_pulse(ctx, ev)``: the own ``Pulse`` event gives the ``pulse_included`` metric (offset = block − boundary; the
expected bound is ≤ 3 at f = 2 for an honest proposer turn).
"""
from __future__ import annotations

import os

from sidecar.chain import BEACON_EPOCH

PULSE_LATEST = 13


def on_head(ctx, head: int) -> None:
    if not ctx.cfg.pulse or ctx.silent(head):
        return
    boundary = head - head % BEACON_EPOCH
    epoch = str(boundary // BEACON_EPOCH)
    if epoch in ctx.state["pulses"] or head - boundary > PULSE_LATEST:
        return
    if any(p.get("purpose") == "pulse" and p.get("key") == epoch for p in ctx.state["pending"].values()):
        return
    tx, err = ctx.send("pulse", epoch, "pulse", (os.urandom(32),))
    txh = None if tx is None else "0x" + tx.hex()
    ctx.state["pulses"][epoch] = {"boundary": boundary, "sent_head": head, "tx": txh, "error": err}
    ctx.metrics.emit("pulse", epoch=int(epoch), boundary=boundary, sent_head=head, tx=txh, error=err)


def on_pulse(ctx, ev) -> None:
    if bytes(ev.args["validator"]) != ctx.me:
        return
    b = int(ev.block)
    boundary = b - 1 - (b - 1) % BEACON_EPOCH           # the epoch of block b starts at boundary + 1
    p = ctx.state["pulses"].get(str(boundary // BEACON_EPOCH))
    if p is not None:
        p["included"] = b
    ctx.metrics.emit("pulse_included", epoch=boundary // BEACON_EPOCH, block=b, offset=b - boundary)


__all__ = ["on_head", "on_pulse", "PULSE_LATEST"]
