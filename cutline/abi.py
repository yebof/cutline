"""cutline.abi - the Solidity contract ABI encoder and the canonical decoder.

A self-contained coder (no eth-abi dependency), byte-compatible with eth_abi for every type below.

Types: ``uintN`` / ``intN`` (N = 8..256, multiple of 8), ``address``, ``bool``, ``bytesN`` (N = 1..32), ``bytes``,
``string``, ``T[]``, tuples ``(T1,T2,...)`` (nested freely). No spaces inside type strings are required (they are
stripped). Fixed-size arrays ``T[k]`` are not used by the protocol and are rejected.

Python values: ints for ``uintN``/``intN``; ``bytes`` (20) for ``address`` (a ``"0x..."`` string is also accepted
when encoding); ``bool``; ``bytes`` for ``bytesN``/``bytes``; ``str`` for ``string``; list/tuple for ``T[]`` and
tuples. Decoding returns ints, bytes (address = 20 bytes), bools, str, **lists** for ``T[]`` and **tuples** for tuple
types.

- ``encode(types: list[str], values: list) -> bytes``: ``abi.encode(a, b, ...)``, the field-list (argument-list) form.
- ``encode_single(type: str, value) -> bytes``: Solidity ``abi.encode(x)`` of one value; for a dynamic type this is
  the head word ``u256 0x20`` followed by x's encoding; a static value is inline.
  (It equals ``encode([type], [value])``.)
- ``decode_canonical(types, data) -> tuple``: strict decode, then re-encode and require identical bytes; raises
  ``ABIError("E.ABI.NONCANONICAL")`` on any malformed or non-canonical input (bad offsets, padding, high bytes,
  non-minimal sign extension, bool not 0/1, invalid UTF-8, trailing bytes). Single pass: each offset must equal
  the canonical tail position before it is followed, so decoding is O(len(data)) (aliased offsets are rejected
  before anything is materialised). It never raises anything else on arbitrary input bytes. Callers remap the name
  (e.g. to ``E.BUNDLE.ABI``).
- ``decode_single_canonical(type, data)``: the inverse of ``encode_single`` (same error).
- ``selector(sig: str) -> bytes`` (4 bytes): ``keccak256(sig)[:4]``.
- ``is_dynamic(type) -> bool``; ``parse_type(type)`` / ``type_str(parsed)``.
Encoding a value that does not fit its type raises ``ABIError("E.ABI.VALUE")``; an unknown type raises ``ValueError``.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .crypto import keccak256
from .errors import ABIError

_UINT = re.compile(r"uint(\d*)")
_INT = re.compile(r"int(\d*)")
_BYTESN = re.compile(r"bytes(\d+)")


# ------------------------------------------------------------------------------------------------ types
def _split(inner: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            raise ValueError("unbalanced tuple type")
        cur += ch
    if depth:
        raise ValueError("unbalanced tuple type")
    if cur or parts:
        parts.append(cur)
    return parts


def _bits(m, default=256) -> int:
    n = int(m.group(1)) if m.group(1) else default
    if not (8 <= n <= 256 and n % 8 == 0):
        raise ValueError(f"bad integer width {n}")
    return n


@lru_cache(maxsize=None)
def parse_type(t: str):
    """Parse a type string into ``"elementary"`` | ``("array", T)`` | ``("tuple", (T, ...))``."""
    t = t.replace(" ", "")
    if t.endswith("[]"):
        return ("array", parse_type(t[:-2]))
    if t.endswith("]"):
        raise ValueError(f"fixed-size arrays are not supported: {t}")
    if t.startswith("("):
        if not t.endswith(")"):
            raise ValueError(f"bad tuple type {t}")
        parts = _split(t[1:-1])
        if not parts or any(p == "" for p in parts):
            raise ValueError(f"empty tuple component in {t}")
        return ("tuple", tuple(parse_type(p) for p in parts))
    if t in ("address", "bool", "bytes", "string"):
        return t
    m = _UINT.fullmatch(t)
    if m:
        return f"uint{_bits(m)}"
    m = _INT.fullmatch(t)
    if m:
        return f"int{_bits(m)}"
    m = _BYTESN.fullmatch(t)
    if m and 1 <= int(m.group(1)) <= 32:
        return t
    raise ValueError(f"unsupported ABI type {t!r}")


def type_str(pt) -> str:
    if isinstance(pt, tuple):
        if pt[0] == "array":
            return type_str(pt[1]) + "[]"
        return "(" + ",".join(type_str(x) for x in pt[1]) + ")"
    return pt


def _dyn(pt) -> bool:
    if pt in ("bytes", "string"):
        return True
    if isinstance(pt, tuple):
        return pt[0] == "array" or any(_dyn(x) for x in pt[1])
    return False


def is_dynamic(t: str) -> bool:
    return _dyn(parse_type(t))


def _static_size(pt) -> int:
    if isinstance(pt, tuple) and pt[0] == "tuple":
        return sum(_static_size(x) for x in pt[1])
    return 32


# ------------------------------------------------------------------------------------------------ encoder
def _bad(t, v, why=""):
    raise ABIError("E.ABI.VALUE", f"{type_str(t)}: {why or repr(v)[:80]}")


def _enc(pt, v) -> bytes:
    if isinstance(pt, tuple):
        if pt[0] == "array":
            if not isinstance(v, (list, tuple)):
                _bad(pt, v, "array value must be a list or tuple")
            return len(v).to_bytes(32, "big") + _enc_tuple([pt[1]] * len(v), v)
        if not isinstance(v, (list, tuple)) or len(v) != len(pt[1]):
            _bad(pt, v, "tuple arity")
        return _enc_tuple(pt[1], v)
    if pt == "bytes" or pt == "string":
        if pt == "string":
            if not isinstance(v, str):
                _bad(pt, v, "string value must be str")
            b = v.encode("utf-8")
        else:
            if not isinstance(v, (bytes, bytearray, memoryview)):
                _bad(pt, v, "bytes value must be bytes")
            b = bytes(v)
        return len(b).to_bytes(32, "big") + b + bytes((-len(b)) % 32)
    if pt == "address":
        if isinstance(v, str):
            try:
                v = bytes.fromhex(v[2:] if v.startswith("0x") else v)
            except ValueError:
                _bad(pt, v, "bad hex")
        if not isinstance(v, (bytes, bytearray)) or len(v) != 20:
            _bad(pt, v, "address must be 20 bytes")
        return bytes(12) + bytes(v)
    if pt == "bool":
        if not isinstance(v, bool) and v not in (0, 1):
            _bad(pt, v, "bool must be True/False or 0/1")
        return (1 if v else 0).to_bytes(32, "big")
    if pt.startswith("uint"):
        if isinstance(v, bool) or not isinstance(v, int):
            v = _as_int(pt, v)
        if not 0 <= v < (1 << int(pt[4:])):
            _bad(pt, v, "out of range")
        return int(v).to_bytes(32, "big")
    if pt.startswith("int"):
        if isinstance(v, bool) or not isinstance(v, int):
            v = _as_int(pt, v)
        n = int(pt[3:])
        if not -(1 << (n - 1)) <= v < (1 << (n - 1)):
            _bad(pt, v, "out of range")
        return int(v).to_bytes(32, "big", signed=True)
    # bytesN
    n = int(pt[5:])
    if not isinstance(v, (bytes, bytearray, memoryview)) or len(v) != n:
        _bad(pt, v, f"must be exactly {n} bytes")
    return bytes(v) + bytes(32 - n)


def _as_int(pt, v) -> int:
    try:                                  # numpy integers
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
    except ImportError:                   # pragma: no cover
        pass
    _bad(pt, v, "integer value required")


def _enc_tuple(pts, vals) -> bytes:
    if len(pts) != len(vals):
        raise ABIError("E.ABI.VALUE", f"{len(vals)} values for {len(pts)} types")
    head_len = sum(32 if _dyn(t) else _static_size(t) for t in pts)
    head, tail = [], []
    tail_len = 0
    for t, v in zip(pts, vals):
        if _dyn(t):
            head.append((head_len + tail_len).to_bytes(32, "big"))
            e = _enc(t, v)
            tail.append(e)
            tail_len += len(e)
        else:
            head.append(_enc(t, v))
    return b"".join(head) + b"".join(tail)


def encode(types: list[str], values: list) -> bytes:
    """``abi.encode(v1, v2, ...)`` (field-list form, no leading offset word for the list itself)."""
    return _enc_tuple([parse_type(t) for t in types], list(values))


def encode_single(type: str, value) -> bytes:  # noqa: A002  (Solidity's word)
    """Solidity ``abi.encode(x)`` of one value: dynamic types get the leading ``u256 0x20`` word."""
    return encode([type], [value])


# ------------------------------------------------------------------------------------------------ decoder
# Canonical single-pass decoder. Every dynamic head word must equal the running canonical tail position *before* it
# is followed, every padding / high byte is checked in place, and every region of the input is read exactly once.
# Decoding is therefore O(len(data)) in time and in the size of the decoded values: aliased offsets (several heads
# pointing at one tail, a quadratic memory amplification) are rejected at the first aliased head. The re-encoding
# comparison is kept as a second, linear-cost, check.
class _Malformed(Exception):
    pass


def _word(buf, pos) -> bytes:
    if pos < 0 or pos + 32 > len(buf):
        raise _Malformed("short word")
    return buf[pos:pos + 32]


def _cdec(pt, buf, pos):
    """Canonical decode of ``pt`` whose encoding starts at ``buf[pos]``; returns ``(value, end)``."""
    if isinstance(pt, tuple):
        if pt[0] == "array":
            n = int.from_bytes(_word(buf, pos), "big")
            el = pt[1]
            per = 32 if _dyn(el) else _static_size(el)
            if n * per > len(buf) - pos - 32:
                raise _Malformed("array length past end")
            vals, end = _cdec_tuple([el] * n, buf, pos + 32)
            return list(vals), end
        return _cdec_tuple(pt[1], buf, pos)
    if pt in ("bytes", "string"):
        n = int.from_bytes(_word(buf, pos), "big")
        padded = n + (-n) % 32
        if padded > len(buf) - pos - 32:
            raise _Malformed("bytes past end")
        b = bytes(buf[pos + 32:pos + 32 + n])
        if any(buf[pos + 32 + n:pos + 32 + padded]):
            raise _Malformed("nonzero bytes padding")
        if pt == "string":
            try:
                b = b.decode("utf-8")
            except UnicodeDecodeError:
                raise _Malformed("invalid UTF-8") from None
        return b, pos + 32 + padded
    w = _word(buf, pos)
    end = pos + 32
    if pt == "address":
        if any(w[:12]):
            raise _Malformed("address high bytes")
        return bytes(w[12:]), end
    if pt == "bool":
        x = int.from_bytes(w, "big")
        if x > 1:
            raise _Malformed("bool word not 0/1")
        return bool(x), end
    if pt.startswith("uint"):
        x = int.from_bytes(w, "big")
        if x >> int(pt[4:]):
            raise _Malformed("uint high bits")
        return x, end
    if pt.startswith("int"):
        x = int.from_bytes(w, "big", signed=True)
        nb = int(pt[3:])
        if not -(1 << (nb - 1)) <= x < (1 << (nb - 1)):
            raise _Malformed("int sign extension")
        return x, end
    nb = int(pt[5:])
    if any(w[nb:]):
        raise _Malformed("bytesN padding")
    return bytes(w[:nb]), end


def _cdec_tuple(pts, buf, pos):
    """Canonical decode of a tuple / field list at ``buf[pos]``; offsets are relative to ``pos``. Returns
    ``(values, end)`` where ``end`` is the end of the tuple's tail (its canonical encoding is ``buf[pos:end]``)."""
    head_len = sum(32 if _dyn(t) else _static_size(t) for t in pts)
    if head_len > len(buf) - pos:
        raise _Malformed("head past end")
    out, off = [], pos
    tail = pos + head_len                     # the canonical position of the next dynamic tail
    for t in pts:
        if _dyn(t):
            o = int.from_bytes(_word(buf, off), "big")
            if o != tail - pos:               # checked before following: no aliasing, no gaps, no reordering
                raise _Malformed("offset is not the canonical tail position")
            v, tail = _cdec(t, buf, tail)
            out.append(v)
            off += 32
        else:
            v, _ = _cdec(t, buf, off)
            out.append(v)
            off += _static_size(t)
    return tuple(out), tail


def decode_canonical(types: list[str], data: bytes) -> tuple:
    """Decode the field list ``types`` canonically: single pass, offsets checked against the canonical tail
    position before they are followed, no trailing bytes; then re-encode and require identical bytes."""
    pts = [parse_type(t) for t in types]
    buf = bytes(data)
    try:
        vals, end = _cdec_tuple(pts, buf, 0)
    except (_Malformed, RecursionError) as e:
        raise ABIError("E.ABI.NONCANONICAL", str(e)) from None
    if end != len(buf):
        raise ABIError("E.ABI.NONCANONICAL", "trailing data")
    try:
        re_enc = _enc_tuple(pts, vals)
    except ABIError as e:
        raise ABIError("E.ABI.NONCANONICAL", f"value outside its type: {e.detail}") from None
    if re_enc != buf:
        raise ABIError("E.ABI.NONCANONICAL", "re-encoding differs (offsets, padding, high bytes or trailing data)")
    return vals


def decode_single_canonical(type: str, data: bytes):  # noqa: A002
    """Canonical inverse of ``encode_single`` (a dynamic value must start with the ``0x20`` word)."""
    return decode_canonical([type], data)[0]


def selector(sig: str) -> bytes:
    """The 4-byte function selector ``keccak256(sig)[:4]`` of a canonical signature string."""
    return keccak256(sig.encode("ascii"))[:4]
