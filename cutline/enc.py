"""cutline.enc - fixed-width integers, packbits, canonical JSON and hex.

- ``u8 / u16 / u32 / u64 / u256(x) -> bytes``: unsigned big-endian of exactly 1, 2, 4, 8, 32 bytes (range-checked,
  ``ValueError`` otherwise). ``i16(x)``, ``i256(x)``: two's complement big-endian.
- ``rd_u8 / rd_u16 / rd_u32 / rd_u64(b, off) -> int``: read at a byte offset (``ValueError`` past the end).
- ``packbits(mask) -> bytes``: row-major, MSB first, zero padding bits, length ``ceil(h*w/8)``.
- ``unpackbits_strict(b, h, w, err_name) -> np.ndarray[bool] (h, w)``: the inverse; raises
  ``CutlineError(err_name)`` when ``len(b) != ceil(h*w/8)`` or a padding bit is 1.
- ``jcs(obj) -> bytes``: RFC 8785 for a restricted subset (objects with ASCII keys, arrays, str, bool, None, ints with
  ``|x| < 2^53``); floats, larger ints, bytes and non-ASCII keys raise ``TypeError`` / ``ValueError``.
- ``hx(b) -> "0x..."`` (lowercase, ``"0x"`` for empty); ``unhx(s) -> bytes`` (accepts with or without ``0x``).
"""
from __future__ import annotations

import json

import numpy as np

from .errors import CutlineError

INT53 = 1 << 53


def _u(x: int, n: int) -> bytes:
    x = int(x)
    if not 0 <= x < (1 << (8 * n)):
        raise ValueError(f"u{8 * n} out of range: {x}")
    return x.to_bytes(n, "big")


def u8(x: int) -> bytes:
    return _u(x, 1)


def u16(x: int) -> bytes:
    return _u(x, 2)


def u32(x: int) -> bytes:
    return _u(x, 4)


def u64(x: int) -> bytes:
    return _u(x, 8)


def u256(x: int) -> bytes:
    return _u(x, 32)


def _i(x: int, n: int) -> bytes:
    x = int(x)
    if not -(1 << (8 * n - 1)) <= x < (1 << (8 * n - 1)):
        raise ValueError(f"i{8 * n} out of range: {x}")
    return x.to_bytes(n, "big", signed=True)


def i16(x: int) -> bytes:
    return _i(x, 2)


def i256(x: int) -> bytes:
    return _i(x, 32)


def _rd(b: bytes, off: int, n: int) -> int:
    if off < 0 or off + n > len(b):
        raise ValueError(f"read of {n} bytes at {off} past the end ({len(b)})")
    return int.from_bytes(b[off:off + n], "big")


def rd_u8(b: bytes, off: int) -> int:
    return _rd(b, off, 1)


def rd_u16(b: bytes, off: int) -> int:
    return _rd(b, off, 2)


def rd_u32(b: bytes, off: int) -> int:
    return _rd(b, off, 4)


def rd_u64(b: bytes, off: int) -> int:
    return _rd(b, off, 8)


# ------------------------------------------------------------------------------------------------ packbits
def packbits(mask) -> bytes:
    """Row-major, MSB-first bit string of a binary mask; the last byte is padded with zero bits."""
    m = np.asarray(mask)
    return np.packbits((m != 0).astype(np.uint8).reshape(-1)).tobytes()


def unpackbits_strict(b: bytes, h: int, w: int, err_name: str) -> np.ndarray:
    """Inverse of packbits with the exact-length and zero-padding checks; raises CutlineError(err_name)."""
    n = h * w
    need = (n + 7) // 8
    if len(b) != need:
        raise CutlineError(err_name, f"packbits length {len(b)} != ceil({h}*{w}/8) = {need}")
    bits = np.unpackbits(np.frombuffer(bytes(b), np.uint8))
    if bits[n:].any():
        raise CutlineError(err_name, "nonzero padding bit")
    return bits[:n].astype(bool).reshape(h, w)


# ------------------------------------------------------------------------------------------------ JCS
def _jcs_check(o):
    if o is None or isinstance(o, (bool, str)):
        return
    if isinstance(o, int):
        if not -INT53 < o < INT53:
            raise ValueError(f"integer {o} outside |x| < 2^53: encode it as a decimal string")
        return
    if isinstance(o, float):
        raise TypeError("floats are not allowed in canonical JSON")
    if isinstance(o, (list, tuple)):
        for x in o:
            _jcs_check(x)
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if not isinstance(k, str) or not k.isascii():
                raise ValueError(f"canonical JSON keys must be ASCII strings: {k!r}")
            _jcs_check(v)
        return
    raise TypeError(f"type {type(o).__name__} not allowed in canonical JSON")


def jcs(obj) -> bytes:
    """RFC 8785 canonical JSON for the supported subset (equals sorted-key compact json.dumps, UTF-8)."""
    _jcs_check(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ------------------------------------------------------------------------------------------------ hex
def hx(b: bytes) -> str:
    return "0x" + bytes(b).hex()


def unhx(s: str) -> bytes:
    if s.startswith(("0x", "0X")):
        s = s[2:]
    return bytes.fromhex(s)
