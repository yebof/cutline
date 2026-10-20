"""sidecar.rpcport - the ``ChainPort`` over JSON-RPC to one Besu node.

``RpcPort(url, sk=None, *, rpc=None, contracts=None, sender=None, timeout=10.0, ledger_kw=None,
chain_id=None)`` (the ``TxSender`` is built on the first send):

- reads with ``cutline.ledger.client`` (``abi_call`` against the frozen ABI export and addresses; ``decode_log``),
  transactions with ``cutline.ledger.relay.TxSender`` (EIP-155 legacy, gas price 0, local nonces);
- ``head`` = ``eth_blockNumber``; ``block_hash`` / ``parent_hash`` from ``eth_getBlockByNumber`` (a small cache);
- ``logs(frm, to)`` = one ``eth_getLogs`` over the addresses of the contracts that emit ``chain.EVENTS``, decoded and
  filtered to those events (the caller keeps ranges ≤ 500 blocks);
- ``view`` raises ``chain.Revert(name)`` for a reverting read; ``simulate`` = ``eth_call`` from the sidecar's
  address at ``latest`` (``None`` or the decoded ``E.*`` name);
- ``send`` returns the hash; a gas estimate that reverts raises ``chain.Revert`` (no nonce consumed); a node
  rejection raises ``chain.Revert("rejected:<msg>")`` after ``resync()`` of the nonce;
- ``receipt`` = ``eth_getTransactionReceipt`` → ``chain.Receipt`` (a failed one is named by replaying it at the
  parent block, ``TxSender.failure_name``);
- ``ledger()`` = a fresh ``client.RpcLedgerView(rpc, mode="node")`` (read at the current head; ``tier_t`` /
  ``kfam`` filtered by the vote height as ``StateSnapshot`` does);
- transport failures (``RpcTransportError``) become ``chain.ChainUnavailable`` (the service backs off).

``sk=None`` gives a read-only port (the peers of the cross-node alarm).
"""
from __future__ import annotations

import functools

from sidecar.chain import EVENTS, FUNCS, ChainUnavailable, Event, Receipt, Revert


def _wrap(fn):
    @functools.wraps(fn)
    def w(self, *a, **kw):
        from cutline.ledger.rpc import RpcTransportError
        try:
            return fn(self, *a, **kw)
        except RpcTransportError as e:
            raise ChainUnavailable(str(e)) from None
    return w


class RpcPort:
    BLOCK_CACHE = 256

    def __init__(self, url: str | None, sk: int | None = None, *, rpc=None, contracts=None, sender=None,
                 timeout: float = 10.0, ledger_kw: dict | None = None, chain_id: int | None = None):
        from cutline.ledger.client import Contracts
        from cutline.ledger.rpc import Rpc
        self.rpc = rpc if rpc is not None else Rpc(url, timeout=timeout)
        self.cs = contracts or Contracts()
        self._sk, self._sender, self.chain_id_hint = sk, sender, chain_id
        if sender is not None:
            self.address = bytes(sender.address)
        elif sk is not None:
            from cutline.crypto import address_of
            self.address = address_of(int(sk))
        else:
            self.address = bytes(20)
        self.ledger_kw = dict(ledger_kw or {})
        self._blocks = {}
        self._txs = {}
        self._emitters = sorted({c for c, _, _ in EVENTS.values()})

    @property
    def sender(self):
        """The ``TxSender``, built on first use (its constructor asks the node for the chain id and the nonce, so a
        port can be created while the node is still down)."""
        if self._sender is None and self._sk is not None:
            from cutline.ledger.relay import TxSender
            self._sender = TxSender(self.rpc, int(self._sk), chain_id=self.chain_id_hint, contracts=self.cs)
        return self._sender

    def _to(self, fn):
        c, sig, _ = FUNCS[fn]
        return c, sig, self.cs.address(c)

    @_wrap
    def head(self) -> int:
        return int(self.rpc.block_number())

    def _block(self, h: int):
        b = self._blocks.get(h)
        if b is None:
            b = self.rpc.get_block(int(h))
            if b is None:
                return None
            if len(self._blocks) >= self.BLOCK_CACHE:
                for k in sorted(self._blocks)[: self.BLOCK_CACHE // 2]:
                    del self._blocks[k]
            self._blocks[h] = b
        return b

    @_wrap
    def block_hash(self, h: int):
        from cutline.ledger.rpc import unhx
        b = self._block(int(h))
        return None if b is None else unhx(b["hash"])

    @_wrap
    def parent_hash(self, h: int):
        from cutline.ledger.rpc import unhx
        b = self._block(int(h))
        return None if b is None else unhx(b["parentHash"])

    @_wrap
    def logs(self, frm: int, to: int) -> list:
        from cutline.ledger.client import decode_log
        from cutline.ledger.rpc import hx
        out = []
        raw = self.rpc.call("eth_getLogs", {"address": [hx(self.cs.address(c)) for c in self._emitters],
                                            "fromBlock": hex(int(frm)), "toBlock": hex(int(to))})
        for lg in raw:
            e = decode_log(lg, self.cs)
            if e is None or e.name not in EVENTS or EVENTS[e.name][0] != e.contract:
                continue
            out.append(Event(e.name, int(e.block), int(e.index), dict(e.args), bytes(e.tx)))
        out.sort(key=lambda e: (e.block, e.log_index))
        return out

    @_wrap
    def view(self, fn: str, args: tuple, block=None) -> tuple:
        from cutline.ledger.client import CallRevert, abi_call
        c, sig, _ = self._to(fn)
        try:
            return abi_call(self.rpc, c, sig, list(args), "latest" if block is None else block, self.cs,
                            self.address)
        except CallRevert as e:
            raise Revert(e.name) from None

    def _data(self, fn, args, calldata):
        c, sig, to = self._to(fn)
        return to, (bytes(calldata) if calldata is not None else self.cs.calldata(c, sig, list(args)))

    @_wrap
    def simulate(self, fn: str, args: tuple, calldata: bytes | None = None) -> str | None:
        from cutline.ledger.client import decode_revert
        from cutline.ledger.rpc import RpcError
        to, data = self._data(fn, args, calldata)
        try:
            self.rpc.eth_call(to, data, "latest", self.address)
        except RpcError as e:
            if e.revert_data is None and "revert" not in e.message.lower():
                raise
            return decode_revert(e.revert_data)
        return None

    @_wrap
    def send(self, fn: str, args: tuple, calldata: bytes | None = None) -> bytes:
        from cutline.ledger.relay import TxError
        if self.sender is None:
            raise Revert("E.SIDECAR.READ_ONLY")
        to, data = self._data(fn, args, calldata)
        try:
            h = self.sender.send(to, data)
        except TxError as e:
            if e.name.startswith("rejected:"):
                self.sender.resync()
            raise Revert(e.name) from None
        self._txs[bytes(h)] = (to, data)
        return bytes(h)

    @_wrap
    def receipt(self, tx: bytes):
        from cutline.ledger.rpc import hx, to_int
        r = self.rpc.call("eth_getTransactionReceipt", hx(tx))
        if r is None:
            return None
        ok = to_int(r["status"]) == 1
        err = None
        if not ok:
            td = self._txs.get(bytes(tx))
            err = self.sender.failure_name(td[0], td[1], r) if (td and self.sender) else "revert()"
        self._txs.pop(bytes(tx), None)
        return Receipt(bytes(tx), to_int(r["blockNumber"]), ok, err, to_int(r.get("gasUsed", "0x0")))

    def ledger(self):
        from cutline.ledger.client import RpcLedgerView
        kw = dict(self.ledger_kw)
        if "chain_id" not in kw:
            if getattr(self, "_chain_id", None) is None:
                self._chain_id = int(self.rpc.chain_id())
            kw["chain_id"] = self._chain_id
        return RpcLedgerView(self.rpc, mode="node", contracts=self.cs, **kw)


__all__ = ["RpcPort"]
