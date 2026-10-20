"""cutline.verify.hopcodec - the op record, opHash, hop leaves and ``Bundle.hops``.

Byte codecs only: no ledger access and no image work (VerifyHop and the chain rules are ``cutline.verify.hop``).

- ``Op(kind, crop, out, kernel_id, decoder_id, q, enc_class, ep_entry, platform_id)``: ``crop = (y, x, h, w)`` in
  the parent's pixels, ``out = (h, w)`` (= the child's SOF Y, X), ``q`` = 192 bytes (the child's three tables,
  natural order, as in its R_Y header). ``Op.q3() -> (bytes64, bytes64, bytes64)``;
  ``Op.resampling`` = ``out != (crop.h, crop.w)`` (counted in ``resampling_hops``).
- ``encode_op(op) -> bytes`` (exactly 217): ``u8 kind ‖ u16 crop.y, crop.x, crop.h, crop.w ‖ u16 out.h, out.w ‖
  u8 kernelId ‖ u8 decoderId ‖ q[3][64] ‖ u16 encClass ‖ u32 epEntry ‖ u32 platformId`` (big-endian). It checks field
  widths only (``ValueError``); every semantic check is VerifyHop's.
- ``decode_op(b) -> Op``: ``E.HOP.OP`` if ``len(b) != 217`` or ``kind != 1`` (VerifyHop's second require). The other
  fields are returned as read, whatever their values (``E.HOP.DECODER`` / ``KERNEL`` / ``CLASS`` / ... are
  ``hop.hop_checks``' job, in the protocol's order).
- ``op_hash(op) -> bytes32`` = H(op) (217-byte preimage, no tag; ``op`` = an ``Op`` or its 217 bytes).
- ``hop_leaf(ry_parent, ry_child, op_hash) -> bytes32`` = H(0x00 ‖ "CUTLINE-hop-v1" ‖ R_Y(parent) ‖ R_Y(child) ‖
  opHash) (111-byte preimage).
- ``HopEntry(op, r_y_parent, batch_id, leaf_index, path, parent_locator)``: one element of ``Bundle.hops``;
  ``op`` is the raw bytes carried in the bundle (so a malformed op survives decoding and fails in VerifyHop with
  ``E.HOP.OP``, not in the framing); ``.decoded_op()`` = ``decode_op(self.op)``; ``.as_tuple()`` / ``from_tuple``.
- ``encode_hops(entries) -> bytes`` = ``abi.encode(HopEntry[])`` with the leading ``0x20`` word
  (``bundle.HOPS_T``), the ``hops.bin`` member; ``decode_hops(b) -> list[HopEntry]``: the canonical decoding of
  ``bundle.read_bundle`` (``E.BUNDLE.ABI``).
- ``child_ry(entries, i, ry_displayed) -> bytes32``: the child R_Y of entry i (``entries[i+1].r_y_parent``, or
  ``ry_displayed`` for the last entry).
"""
from __future__ import annotations

from dataclasses import dataclass

from cutline import abi
from cutline.bundle import HOPS_T, _single
from cutline.consts import TAG_HOP
from cutline.crypto import H
from cutline.enc import u8, u16, u32
from cutline.errors import HopError

OP_LEN = 217
KIND_TRANSCODE = 1


@dataclass(frozen=True)
class Op:
    kind: int
    crop: tuple            # (y, x, h, w)
    out: tuple             # (h, w)
    kernel_id: int
    decoder_id: int
    q: bytes               # 192 bytes: q[0] || q[1] || q[2], natural order
    enc_class: int
    ep_entry: int = 0
    platform_id: int = 0

    def q3(self) -> tuple:
        q = bytes(self.q)
        return (q[0:64], q[64:128], q[128:192])

    @property
    def resampling(self) -> bool:
        return tuple(self.out) != (self.crop[2], self.crop[3])


def encode_op(op: Op) -> bytes:
    if len(op.q) != 192:
        raise ValueError("op.q must be 192 bytes")
    y, x, h, w = op.crop
    oh, ow = op.out
    b = (u8(op.kind) + u16(y) + u16(x) + u16(h) + u16(w) + u16(oh) + u16(ow) + u8(op.kernel_id) + u8(op.decoder_id)
         + bytes(op.q) + u16(op.enc_class) + u32(op.ep_entry) + u32(op.platform_id))
    assert len(b) == OP_LEN
    return b


def decode_op(b: bytes) -> Op:
    b = bytes(b)
    if len(b) != OP_LEN:
        raise HopError("E.HOP.OP", f"op has {len(b)} bytes, not 217")
    if b[0] != KIND_TRANSCODE:
        raise HopError("E.HOP.OP", f"op.kind {b[0]} != 1")
    r = lambda o, n: int.from_bytes(b[o:o + n], "big")  # noqa: E731
    return Op(kind=b[0], crop=(r(1, 2), r(3, 2), r(5, 2), r(7, 2)), out=(r(9, 2), r(11, 2)), kernel_id=b[13],
              decoder_id=b[14], q=b[15:207], enc_class=r(207, 2), ep_entry=r(209, 4), platform_id=r(213, 4))


def op_hash(op) -> bytes:
    return H(encode_op(op) if isinstance(op, Op) else bytes(op))


def hop_leaf(ry_parent: bytes, ry_child: bytes, op_hash_: bytes) -> bytes:
    if len(ry_parent) != 32 or len(ry_child) != 32 or len(op_hash_) != 32:
        raise ValueError("hop_leaf takes three bytes32 values")
    return H(TAG_HOP + bytes(ry_parent) + bytes(ry_child) + bytes(op_hash_))


@dataclass(frozen=True)
class HopEntry:
    op: bytes              # raw op bytes (217 in a well-formed entry)
    r_y_parent: bytes
    batch_id: int
    leaf_index: int
    path: tuple            # bytes32 siblings, leaves up
    parent_locator: str

    def decoded_op(self) -> Op:
        return decode_op(self.op)

    def as_tuple(self) -> tuple:
        return (bytes(self.op), bytes(self.r_y_parent), self.batch_id, self.leaf_index,
                [bytes(p) for p in self.path], self.parent_locator)

    @classmethod
    def from_tuple(cls, t) -> "HopEntry":
        op, ry, bid, li, path, loc = t
        return cls(bytes(op), bytes(ry), int(bid), int(li), tuple(bytes(p) for p in path), loc)


def encode_hops(entries) -> bytes:
    return abi.encode_single(HOPS_T, [e.as_tuple() if isinstance(e, HopEntry) else e for e in entries])


def decode_hops(b: bytes) -> list:
    return [HopEntry.from_tuple(t) for t in _single(HOPS_T, bytes(b), "hops.bin")]


def child_ry(entries, i: int, ry_displayed: bytes) -> bytes:
    return bytes(entries[i + 1].r_y_parent) if i + 1 < len(entries) else bytes(ry_displayed)


__all__ = ["OP_LEN", "Op", "encode_op", "decode_op", "op_hash", "hop_leaf", "HopEntry", "encode_hops",
           "decode_hops", "child_ry"]
