"""cutline.ledger.relay - transactions and signed relays.

- EIP-155 legacy transactions (gasPrice 0 on the v1 devnet: ``--min-gas-price=0``, ``zeroBaseFee``), signed with
  eth-keys; a per-account nonce manager (``eth_getTransactionCount(.., "pending")`` once, then local).
- Relay payloads: the signer's secp256k1 signature over ``ledger.abi.relay_digest`` at the target's current
  ``relayNonce[signer]``, submitted by an allowlisted relayer as ``F(args..., address signer, bytes sig)``.
- A receipt waiter; a failed transaction is decoded to its protocol error name (``client.decode_revert`` over the
  ABI export's custom errors) by replaying it with ``eth_call`` at the parent block.

- ``sign_legacy_tx(sk, nonce, gas_price, gas, to, value, data, chain_id) -> (raw: bytes, tx_hash: bytes)``;
  ``signing_hash_legacy(...)``; ``tx_sender(raw, chain_id) -> address`` (recovers the sender; tests).
- ``TxError(name, receipt=None)``: ``.name`` = ``"E.X.Y"`` / ``"revert()"`` / ``"Error(..)"`` / ``"timeout"`` /
  ``"rejected:<node message>"``.
- ``TxSender(rpc, sk, chain_id=None, gas_price=0, gas_margin=1.25, poll=0.25, timeout=60, nonce=None)``:
  ``.address``; ``send(to, data, gas=None, value=0) -> tx_hash`` (``eth_estimateGas`` × margin when ``gas`` is None;
  an estimate that reverts raises ``TxError`` with the decoded name and consumes no nonce); ``wait(tx_hash) ->
  receipt dict`` (``TxError("timeout")``); ``transact(to, data, gas=None) -> receipt`` (send + wait; status 0 raises
  ``TxError`` with the name); ``call(contract, sig, args)`` / ``tx(contract, sig, args, gas=None)`` against the v1
  addresses; ``resync()`` re-reads the pending nonce (after a rejected send).
- Relays: ``RELAY_TARGET`` (function name -> contract); ``relay_nonce(rpc, contract, signer)``;
  ``relay_calldata(name, args, signer_sk, nonce, chain_id, contracts=None) -> bytes`` (the on-chain entry point
  ``name(args..., signer, sig)`` of the ABI export); ``relay(sender, name, args, signer_sk, gas=None)`` reads the
  nonce, signs and transacts. Builders: ``delegate_args(nf, to, scope, exclusive, expiry, withdraw)`` etc. are
  plain argument lists in the canonical relay argument order (``ledger.abi.RELAY_ARG_TYPES``).
"""
from __future__ import annotations

import time

import rlp

from ..crypto import address_of, k1_sign, keccak256
from ..ledger.abi import RELAY_ARG_TYPES, relay_digest_for
from . import layout as L
from .client import CallRevert, Contracts, abi_call, decode_revert
from .rpc import RpcError, hx, to_int, unhx

RELAY_TARGET = {"delegate": "ClaimRegistry", "revokeDelegation": "ClaimRegistry", "offerTransfer": "ClaimRegistry",
                "acceptTransfer": "ClaimRegistry", "withdraw": "ClaimRegistry", "retireKey": "DeviceRegistry"}
N_K1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class TxError(Exception):
    def __init__(self, name: str, receipt: dict | None = None, tx_hash: bytes | None = None):
        self.name, self.receipt, self.tx_hash = name, receipt, tx_hash
        super().__init__(name)


# ------------------------------------------------------------------------------------------------ EIP-155
def _i(x: int) -> bytes:
    x = int(x)
    return x.to_bytes((x.bit_length() + 7) // 8, "big") if x else b""


def signing_hash_legacy(nonce, gas_price, gas, to, value, data, chain_id) -> bytes:
    return keccak256(rlp.encode([_i(nonce), _i(gas_price), _i(gas), bytes(to or b""), _i(value), bytes(data),
                                 _i(chain_id), b"", b""]))


def sign_legacy_tx(sk: int, nonce: int, gas_price: int, gas: int, to: bytes | None, value: int, data: bytes,
                   chain_id: int) -> tuple:
    from eth_keys import keys
    d = signing_hash_legacy(nonce, gas_price, gas, to, value, data, chain_id)
    sig = keys.PrivateKey(int(sk).to_bytes(32, "big")).sign_msg_hash(d)
    r, s, rec = sig.r, sig.s, sig.v
    if s > N_K1 // 2:                                  # eth-keys signs low-s already; kept for safety
        s, rec = N_K1 - s, rec ^ 1
    v = rec + 35 + 2 * int(chain_id)
    raw = rlp.encode([_i(nonce), _i(gas_price), _i(gas), bytes(to or b""), _i(value), bytes(data), _i(v), _i(r),
                      _i(s)])
    return raw, keccak256(raw)


def tx_sender(raw: bytes, chain_id: int) -> bytes:
    from eth_keys import keys
    n, gp, g, to, val, data, v, r, s = rlp.decode(bytes(raw))
    v = int.from_bytes(v, "big")
    rec = v - 35 - 2 * int(chain_id)
    if rec not in (0, 1):
        raise ValueError("not an EIP-155 signature for this chain")
    d = keccak256(rlp.encode([n, gp, g, to, val, data, _i(chain_id), b"", b""]))
    pub = keys.Signature(vrs=(rec, int.from_bytes(r, "big"), int.from_bytes(s, "big"))).recover_public_key_from_msg_hash(d)
    return pub.to_canonical_address()


# ------------------------------------------------------------------------------------------------ sender
class TxSender:
    def __init__(self, rpc, sk: int, chain_id: int | None = None, gas_price: int = 0, gas_margin: float = 1.25,
                 poll: float = 0.25, timeout: float = 60.0, nonce: int | None = None,
                 contracts: Contracts | None = None):
        self.rpc, self.sk = rpc, int(sk)
        self.address = address_of(self.sk)
        self.chain_id = int(chain_id) if chain_id is not None else rpc.chain_id()
        self.gas_price, self.gas_margin, self.poll, self.timeout = int(gas_price), float(gas_margin), poll, timeout
        self.cs = contracts or Contracts()
        self._nonce = nonce

    def resync(self) -> int:
        self._nonce = to_int(self.rpc.call("eth_getTransactionCount", hx(self.address), "pending"))
        return self._nonce

    @property
    def nonce(self) -> int:
        if self._nonce is None:
            self.resync()
        return self._nonce

    def estimate(self, to: bytes, data: bytes, value: int = 0) -> int:
        o = {"from": hx(self.address), "to": hx(to), "data": hx(data)}
        if value:
            o["value"] = hex(value)
        try:
            g = to_int(self.rpc.call("eth_estimateGas", o))
        except RpcError as e:
            d = e.revert_data
            if d is not None or "revert" in e.message.lower():
                raise TxError(decode_revert(d)) from None
            raise
        return int(g * self.gas_margin) + 1

    def send(self, to: bytes, data: bytes, gas: int | None = None, value: int = 0) -> bytes:
        if gas is None:
            gas = self.estimate(to, data, value)
        n = self.nonce
        raw, h = sign_legacy_tx(self.sk, n, self.gas_price, gas, to, value, data, self.chain_id)
        try:
            got = self.rpc.call("eth_sendRawTransaction", hx(raw))
        except RpcError as e:
            raise TxError(f"rejected:{e.message}") from None
        self._nonce = n + 1
        if unhx(got) != h:
            raise TxError(f"rejected:node returned hash {got}, expected {hx(h)}")
        return h

    def send_raw_presigned(self, raw: bytes) -> bytes:
        return unhx(self.rpc.call("eth_sendRawTransaction", hx(raw)))

    def wait(self, tx_hash: bytes, timeout: float | None = None) -> dict:
        end = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            r = self.rpc.call("eth_getTransactionReceipt", hx(tx_hash))
            if r is not None:
                return r
            if time.monotonic() > end:
                raise TxError("timeout", None, tx_hash)
            time.sleep(self.poll)

    def failure_name(self, to: bytes, data: bytes, receipt: dict) -> str:
        blk = to_int(receipt["blockNumber"])
        try:
            self.rpc.call("eth_call", {"from": hx(self.address), "to": hx(to), "data": hx(data),
                                       "gas": receipt.get("gasUsed", hex(30_000_000))}, hex(max(0, blk - 1)))
        except RpcError as e:
            return decode_revert(e.revert_data)
        return "revert()"

    def transact(self, to: bytes, data: bytes, gas: int | None = None, value: int = 0) -> dict:
        h = self.send(to, data, gas, value)
        r = self.wait(h)
        if to_int(r["status"]) != 1:
            raise TxError(self.failure_name(to, data, r), r, h)
        return r

    def call(self, contract: str, sig: str, args=(), block="latest") -> tuple:
        return abi_call(self.rpc, contract, sig, args, block, self.cs, self.address)

    def tx(self, contract: str, sig: str, args=(), gas: int | None = None) -> dict:
        return self.transact(self.cs.address(contract), self.cs.calldata(contract, sig, args), gas)


# ------------------------------------------------------------------------------------------------ relays
def relay_nonce(rpc, contract: str, signer: bytes, block="latest", contracts: Contracts | None = None) -> int:
    return int(abi_call(rpc, contract, "relayNonce(address)", (bytes(signer),), block, contracts)[0])


def relay_signature(name: str, args, signer_sk: int, nonce: int, chain_id: int, contracts: Contracts | None = None
                    ) -> bytes:
    cs = contracts or Contracts()
    target = cs.address(RELAY_TARGET[name])
    return k1_sign(signer_sk, relay_digest_for(chain_id, target, address_of(signer_sk), nonce, name, list(args)))


def relay_calldata(name: str, args, signer_sk: int, nonce: int, chain_id: int = L.CHAIN_ID,
                   contracts: Contracts | None = None) -> bytes:
    cs = contracts or Contracts()
    sig = relay_signature(name, args, signer_sk, nonce, chain_id, cs)
    types = RELAY_ARG_TYPES[name] + ["address", "bytes"]
    return cs.calldata(RELAY_TARGET[name], f"{name}({','.join(types)})", list(args) + [address_of(signer_sk), sig])


def relay(sender: TxSender, name: str, args, signer_sk: int, gas: int | None = None) -> dict:
    c = RELAY_TARGET[name]
    n = relay_nonce(sender.rpc, c, address_of(signer_sk), "pending", sender.cs)
    data = relay_calldata(name, args, signer_sk, n, sender.chain_id, sender.cs)
    return sender.transact(sender.cs.address(c), data, gas)


def delegate_args(nf: bytes, to: bytes, scope: int, exclusive: bool, expiry: int, withdraw=()) -> list:
    return [bytes(nf), bytes(to), int(scope), bool(exclusive), int(expiry), [bytes(w) for w in withdraw]]


def withdraw_args(claim_id: bytes) -> list:
    return [bytes(claim_id)]


def retire_key_args(key_id: bytes) -> list:
    return [bytes(key_id)]


def transfer_args(nf: bytes, to: bytes | None = None) -> list:
    return [bytes(nf)] if to is None else [bytes(nf), bytes(to)]


__all__ = ["TxSender", "TxError", "sign_legacy_tx", "signing_hash_legacy", "tx_sender", "RELAY_TARGET",
           "relay_nonce", "relay_signature", "relay_calldata", "relay", "delegate_args", "withdraw_args",
           "retire_key_args", "transfer_args", "CallRevert"]
