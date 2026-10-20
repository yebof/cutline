"""cutline.ledger.rpc - a small JSON-RPC 2.0 client over stdlib ``urllib``.

No web3 dependency. Used by ``client.RpcLedgerView``, ``receipts.build_receipt``, ``relay.TxSender`` and the
validator sidecars.

- ``Rpc(url, timeout=10.0, retries=3, backoff=0.25)``:
  - ``call(method, *params)`` -> the ``result`` member; a JSON-RPC ``error`` raises ``RpcError`` (not retried);
    transport failures (connection refused / reset, timeouts, HTTP 5xx, malformed JSON) are retried ``retries``
    times with exponential backoff, then raise ``RpcTransportError``;
  - ``batch([(method, params), ...], raise_errors=True)`` -> results in request order (one HTTP request; servers that
    reject batches are handled by falling back to sequential calls). With ``raise_errors=False`` a failed item is
    returned as its ``RpcError`` instance;
  - convenience: ``block_number()``, ``chain_id()``, ``get_block(tag, full=False)``, ``get_proof(addr, slots, tag)``,
    ``eth_call(to, data, tag, sender=None, gas=None)`` (returns bytes; a revert raises ``RpcError`` whose
    ``revert_data`` holds the returned bytes when the node reports them).
- ``RpcError(code, message, data)``: ``revert_data`` = the revert bytes (``bytes`` or None).
- ``tag(h)``: an int height -> ``"0x.."``; strings (``"latest"``, ``"0x.."``) pass through.
- hex helpers ``hx(b) -> "0x.."``, ``unhx(s) -> bytes``, ``qty(n) -> "0x.."`` (minimal quantity), ``to_int(s)``.
- ``Head(url, number, hash, timestamp)``; ``heads(rpcs) -> [Head | None]`` (one ``eth_getBlockByNumber("latest")``
  per endpoint; an unreachable endpoint gives None) and ``reference_head(heads, f) -> Head | None``: the (f+1)-th
  highest reported head (an "f+1-reported head"; None when fewer than f+1 endpoints answered).
"""
from __future__ import annotations

import itertools
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class RpcTransportError(Exception):
    """The endpoint could not be reached or returned something that is not JSON-RPC (after retries)."""


class RpcError(Exception):
    """A JSON-RPC error object returned by the node."""

    def __init__(self, code: int, message: str, data=None, method: str = ""):
        self.code, self.message, self.data, self.method = code, message, data, method
        super().__init__(f"{method}: [{code}] {message}" + (f" data={data!r}"[:300] if data is not None else ""))

    @property
    def revert_data(self) -> bytes | None:
        """The revert bytes of a failed eth_call / eth_estimateGas, when the node reports them (Besu and EDR put the
        0x-hex return data into ``error.data``; some nodes nest it one level deeper)."""
        d = self.data
        if isinstance(d, dict):
            d = d.get("data", d.get("result"))
        if isinstance(d, str) and d.startswith("0x"):
            try:
                return unhx(d)
            except ValueError:
                return None
        return None


# ------------------------------------------------------------------------------------------------ hex helpers
def hx(b: bytes) -> str:
    return "0x" + bytes(b).hex()


def unhx(s) -> bytes:
    if isinstance(s, (bytes, bytearray)):
        return bytes(s)
    s = str(s)
    s = s[2:] if s[:2] in ("0x", "0X") else s
    if len(s) % 2:
        s = "0" + s
    return bytes.fromhex(s)


def qty(n: int) -> str:
    if n < 0:
        raise ValueError("negative quantity")
    return hex(int(n))


def to_int(s) -> int:
    if isinstance(s, int):
        return s
    return int(s, 16) if isinstance(s, str) and s[:2] in ("0x", "0X") else int(s)


def tag(h) -> str:
    """A block tag: an int height -> quantity; a string passes through; None -> "latest"."""
    if h is None:
        return "latest"
    if isinstance(h, bool):
        raise TypeError("block tag")
    if isinstance(h, int):
        return qty(h)
    return str(h)


# ------------------------------------------------------------------------------------------------ client
class Rpc:
    def __init__(self, url: str, timeout: float = 10.0, retries: int = 3, backoff: float = 0.25):
        self.url, self.timeout, self.retries, self.backoff = url, float(timeout), int(retries), float(backoff)
        self._ids = itertools.count(1)
        self._batch_ok = True
        self.calls = 0                       # number of HTTP requests sent (tests and metrics)

    def __repr__(self):
        return f"Rpc({self.url!r})"

    # -- transport
    def _post(self, payload):
        body = json.dumps(payload).encode()
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.backoff * (2 ** (attempt - 1)))
            req = urllib.request.Request(self.url, data=body, headers={"content-type": "application/json"})
            try:
                self.calls += 1
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    raw = r.read()
                return json.loads(raw)
            except urllib.error.HTTPError as e:          # 4xx is final (bad request); 5xx is retried
                raw = e.read()
                if e.code < 500:
                    try:
                        return json.loads(raw)
                    except ValueError:
                        raise RpcTransportError(f"{self.url}: HTTP {e.code}") from None
                last = e
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError, ValueError) as e:
                last = e
        raise RpcTransportError(f"{self.url}: {last!r}")

    @staticmethod
    def _result(resp, method):
        if not isinstance(resp, dict) or ("result" not in resp and "error" not in resp):
            raise RpcTransportError(f"not a JSON-RPC response: {str(resp)[:200]}")
        if "error" in resp and resp["error"] is not None:
            e = resp["error"]
            raise RpcError(int(e.get("code", 0)), str(e.get("message", "")), e.get("data"), method)
        return resp["result"]

    def call(self, method: str, *params):
        resp = self._post({"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": list(params)})
        return self._result(resp, method)

    def batch(self, calls, raise_errors: bool = True) -> list:
        calls = [(m, list(p)) for m, p in calls]
        if not calls:
            return []
        out = None
        if self._batch_ok and len(calls) > 1:
            ids = [next(self._ids) for _ in calls]
            resp = self._post([{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in zip(ids, calls)])
            if isinstance(resp, list):
                by_id = {r.get("id"): r for r in resp if isinstance(r, dict)}
                if all(i in by_id for i in ids):
                    out = []
                    for i, (m, _) in zip(ids, calls):
                        try:
                            out.append(self._result(by_id[i], m))
                        except RpcError as e:
                            out.append(e)
            if out is None:                               # the server does not batch: sequential from now on
                self._batch_ok = False
        if out is None:
            out = []
            for m, p in calls:
                try:
                    out.append(self.call(m, *p))
                except RpcError as e:
                    out.append(e)
        if raise_errors:
            for r in out:
                if isinstance(r, RpcError):
                    raise r
        return out

    # -- convenience
    def block_number(self) -> int:
        return to_int(self.call("eth_blockNumber"))

    def chain_id(self) -> int:
        return to_int(self.call("eth_chainId"))

    def get_block(self, h="latest", full: bool = False) -> dict | None:
        return self.call("eth_getBlockByNumber", tag(h), bool(full))

    def get_proof(self, address: bytes, slots, h="latest") -> dict:
        return self.call("eth_getProof", hx(address), ["0x" + int(s).to_bytes(32, "big").hex() for s in slots], tag(h))

    @staticmethod
    def call_obj(to: bytes, data: bytes, sender: bytes | None = None, gas: int | None = None) -> dict:
        o = {"to": hx(to), "data": hx(data)}
        if sender is not None:
            o["from"] = hx(sender)
        if gas is not None:
            o["gas"] = qty(gas)
        return o

    def eth_call(self, to: bytes, data: bytes, h="latest", sender: bytes | None = None, gas: int | None = None) -> bytes:
        return unhx(self.call("eth_call", self.call_obj(to, data, sender, gas), tag(h)))


# ------------------------------------------------------------------------------------------------ heads (f+1)
@dataclass(frozen=True)
class Head:
    url: str
    number: int
    hash: bytes
    timestamp: int


def heads(rpcs) -> list:
    """The latest head reported by each endpoint (None for an endpoint that did not answer)."""
    out = []
    for r in rpcs:
        try:
            b = r.get_block("latest")
            out.append(Head(r.url, to_int(b["number"]), unhx(b["hash"]), to_int(b["timestamp"])))
        except (RpcError, RpcTransportError, KeyError, TypeError, ValueError):
            out.append(None)
    return out


def reference_head(hs, f: int):
    """The (f+1)-th highest reported head: at least f+1 reports are at or above it, so at least one honest validator
    has reached its height. None when fewer than f+1 reports are available."""
    hs = sorted((h for h in hs if h is not None), key=lambda h: h.number, reverse=True)
    return hs[f] if len(hs) >= f + 1 else None


__all__ = ["Rpc", "RpcError", "RpcTransportError", "Head", "heads", "reference_head", "hx", "unhx", "qty", "to_int",
           "tag"]
