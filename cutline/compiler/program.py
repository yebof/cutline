"""cutline.compiler.program - enc(pi), its strict decoder, and tHash.

enc(pi) is 794 + 4 nD bytes: u8 l*, u16 y0 x0 ch cw oh ow oy ox Hc Wc, u8 kernel, u8 sampling, u8 flags
(bit 0 overlay = 0 in v1, bit 1 dGen, bits 2-7 reserved), LUT[3][256], u16 nD, u32 D[nD].
tHash = H(0x05 || "CUTLINE-prog-v1" || enc(pi)).

- ``encode_program(p: types.Program) -> bytes`` (no static-rule validation; that is the verifier's job).
- ``decode_program(b) -> (Program | None, len_ok)``: never lenient; any length other than exactly 794 + 4 nD (or
  < 794) gives ``(None, False)``, which is static rule 5 sub-case 9. Field values are not validated here.
- ``prog_len_ok(b) -> bool``: the exact-length predicate.
- ``t_hash(prog_bytes) -> bytes32``.
- ``fields(b) -> dict``: the raw fields of a length-valid enc(pi) (``l, y0, x0, ch, cw, oh, ow, oy, ox, Hc, Wc,
  kernel, sampling, flags, LUT (3 x bytes256), nD, D``), used by the static rules.
- ``lut_bytes(lut) -> bytes768`` from a (3, 256) array / three sequences / 768 bytes; ``IDENTITY_LUT``.
"""
from __future__ import annotations

import struct

import numpy as np

from ..consts import TAG_PROG
from ..crypto import H
from ..types import Program

PROG_FIXED = 794
_HEAD = struct.Struct(">BHHHHHHHHHHBBB")          # 24 bytes
_KEYS = ("l", "y0", "x0", "ch", "cw", "oh", "ow", "oy", "ox", "Hc", "Wc", "kernel", "sampling", "flags")
IDENTITY_LUT = bytes(range(256)) * 3


def lut_bytes(lut) -> bytes:
    if isinstance(lut, (bytes, bytearray, memoryview)):
        b = bytes(lut)
    else:
        a = np.asarray(lut)
        if a.shape != (3, 256) or a.min() < 0 or a.max() > 255:
            raise ValueError("LUT must be (3, 256) values in [0, 255]")
        b = a.astype(np.uint8).tobytes()
    if len(b) != 768:
        raise ValueError("LUT must be 768 bytes")
    return b


def encode_program(p: Program) -> bytes:
    return p.encode()


def prog_len_ok(b: bytes) -> bool:
    return len(b) >= PROG_FIXED and len(b) == PROG_FIXED + 4 * int.from_bytes(b[792:794], "big")


def fields(b: bytes) -> dict:
    if not prog_len_ok(b):
        raise ValueError("enc(pi) length is not 794 + 4 nD (static rule 5 sub-case 9)")
    d = dict(zip(_KEYS, _HEAD.unpack_from(b, 0)))
    d["LUT"] = (bytes(b[24:280]), bytes(b[280:536]), bytes(b[536:792]))
    d["nD"] = int.from_bytes(b[792:794], "big")
    d["D"] = tuple(int.from_bytes(b[794 + 4 * k:798 + 4 * k], "big") for k in range(d["nD"]))
    return d


def decode_program(b: bytes) -> tuple:
    b = bytes(b)
    if not prog_len_ok(b):
        return None, False
    f = fields(b)
    p = Program(lstar=f["l"], crop=(f["y0"], f["x0"], f["ch"], f["cw"]), out=(f["oh"], f["ow"]),
                place=(f["oy"], f["ox"]), canvas=(f["Hc"], f["Wc"]), kernel=f["kernel"], sampling=f["sampling"],
                flags=f["flags"], lut=b"".join(f["LUT"]), D=f["D"])
    assert p.encode() == b
    return p, True


def t_hash(prog_bytes: bytes) -> bytes:
    return H(TAG_PROG + bytes(prog_bytes))
