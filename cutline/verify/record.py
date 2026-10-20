"""cutline.verify.record - the canonical form of the evidence record.

The canonical form is how two verifiers compare records: canonical JSON (``cutline.enc.jcs``) of the record with
every integer a decimal string, every byte string ``0x``-prefixed lowercase hex, the dE00 statistics in micro-units,
and booleans and ``null`` as themselves.

- ``micro(x) -> int``: ``round_half_even(x·10^6)`` evaluated **exactly** on the float64 value x
  (``round(Fraction(x) * 10**6)``; 18.0340635 → 18,034,063, where the float64 product gives 18,034,064).
- ``Micro(int)``: an int already in micro-units (what ``from_canonical`` returns for the dE00 statistics, so a
  record read back re-encodes to the same bytes).
- ``versions(classes=(), tier_t=None, hop_ceilings=(), label_table=None) -> dict``: the ``versions`` object with
  exactly six keys (``spec`` = ``consts.SPEC_VERSION``, ``u_is_level_criterion`` = ``consts.U_IS_LEVEL_CRITERION``);
  ``classes`` = ``[[encClass, kind, eta20], ...]`` ascending without repeats (``registries.class_entry``),
  ``tier_t`` = None or ``{"table", "family", "exact", "noise_milli"}``, ``hop_ceilings`` = ``[[platformId, v*], ...]``
  in chain order, ``label_table`` = None or ``{"regime": str, "version": int}``. ``short_versions(v=None)``:
  ``spec`` and ``u_is_level_criterion`` kept, the other four keys null.
- ``full_form(e) -> dict``: every field of ``types.EVIDENCE_FIELDS`` (not ``hop_index``), converted.
- ``short_form(e) -> dict``: ``{g0_reason, static_violation, level_cap, height, versions}`` with
  ``short_versions``; plus ``mcert_count``, ``tested_blocks``, ``sil_px`` for ``"nothing_certified"``; plus
  ``hop_index`` for ``"ancestor"`` and ``"hop"``. ``level_cap`` is printed as the record holds it: the verifier leaves
  it None unless the claim-status step passed.
- ``to_obj(e)``: ``short_form`` if ``e.is_short()`` else ``full_form``; ``canonical(e) -> bytes`` = ``jcs(to_obj(e))``.
- ``from_canonical(b) -> Evidence``: the inverse (``ValueError`` unless ``b`` is canonical JSON of a full or a short
  record); ``canonical(from_canonical(b)) == b``. Decimal strings become ints, ``0x`` strings bytes, the dE00
  statistics ``Micro``; the string-valued fields (``STRING_FIELDS``, ``tone.class``, ``versions.spec``,
  ``versions.label_table.regime``) stay strings.

Conversion rules (``full_form`` / ``short_form``): ``None`` → null; ``bool`` (and ``numpy.bool_``) → bool; any
other integral value (Python or numpy) → its decimal string; ``Micro`` → its decimal string; a float is allowed
only as a ``tone`` dE00 statistic (→ ``micro``), anything else raises ``TypeError``; ``bytes`` → ``0x`` hex; tuples
→ lists; dict keys must be ASCII strings.
"""
from __future__ import annotations

import json
import numbers
import re
from fractions import Fraction

import numpy as np

from cutline.consts import SPEC_VERSION, U_IS_LEVEL_CRITERION
from cutline.enc import jcs
from cutline.types import EVIDENCE_FIELDS, G0_REASONS, Evidence

VERSION_KEYS = ("spec", "classes", "tier_t", "hop_ceilings", "label_table", "u_is_level_criterion")
DE00_KEYS = ("dE00_p50", "dE00_p95", "dE00_max")
STRING_FIELDS = frozenset({"cap", "claim", "receipt", "tier", "g0_reason", "verdict", "res_class", "chain_class",
                           "background"})
SHORT_KEYS = ("g0_reason", "static_violation", "level_cap", "height", "versions")
_INT_RE = re.compile(r"^(0|-?[1-9][0-9]*)$")
_HEX_RE = re.compile(r"^0x([0-9a-f]{2})*$")


class Micro(int):
    """An integer already in micro-units (the dE00 statistics)."""


def micro(x) -> int:
    if isinstance(x, Micro):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(Fraction(float(x)) * 10 ** 6)
    raise TypeError(f"micro() takes a float64 dE00 statistic, not {type(x).__name__}")


def versions(classes=(), tier_t=None, hop_ceilings=(), label_table=None) -> dict:
    cl = [list(c) for c in classes]
    if [c[0] for c in cl] != sorted({c[0] for c in cl}):
        raise ValueError("versions.classes must be ascending by encClass without repeats")
    return {"spec": SPEC_VERSION, "classes": cl, "tier_t": None if tier_t is None else dict(tier_t),
            "hop_ceilings": [list(x) for x in hop_ceilings],
            "label_table": None if label_table is None else dict(label_table),
            "u_is_level_criterion": U_IS_LEVEL_CRITERION}


def short_versions(v: dict | None = None) -> dict:
    v = v or {}
    return {"spec": v.get("spec", SPEC_VERSION), "classes": None, "tier_t": None, "hop_ceilings": None,
            "label_table": None, "u_is_level_criterion": v.get("u_is_level_criterion", U_IS_LEVEL_CRITERION)}


# ------------------------------------------------------------------------------------------------ to canonical
def _cv(v, float_ok: bool = False):
    if v is None or isinstance(v, (bool, np.bool_)):
        return None if v is None else bool(v)
    if isinstance(v, Micro):
        return str(int(v))
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float_ok:
            return str(micro(v))
        raise TypeError("floats are not allowed in the canonical evidence form (only tone dE00 statistics)")
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "0x" + bytes(v).hex()
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return [_cv(x) for x in v]
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str) or not k.isascii():
                raise ValueError(f"record keys must be ASCII strings: {k!r}")
            out[k] = _cv(x)
        return out
    raise TypeError(f"type {type(v).__name__} not allowed in the evidence record")


def _tone(t):
    if t is None:
        return None
    return {k: _cv(x, float_ok=k in DE00_KEYS) for k, x in t.items()}


def _check_versions(v, short: bool) -> None:
    if v is None or tuple(sorted(v)) != tuple(sorted(VERSION_KEYS)):
        raise ValueError(f"versions must have exactly the keys {VERSION_KEYS}")


def _check_common(e: Evidence) -> None:
    if e.g0_reason is not None and e.g0_reason not in G0_REASONS:
        raise ValueError(f"unknown g0_reason {e.g0_reason!r}")
    if e.static_violation is not None and set(e.static_violation) != {"rule", "i", "j"}:
        raise ValueError("static_violation is {rule, i, j}")


def full_form(e: Evidence) -> dict:
    _check_common(e)
    _check_versions(e.versions, False)
    out = {}
    for f in EVIDENCE_FIELDS:
        v = getattr(e, f)
        out[f] = _tone(v) if f == "tone" else _cv(v)
    return out


def short_form(e: Evidence) -> dict:
    _check_common(e)
    out = {"g0_reason": e.g0_reason, "static_violation": _cv(e.static_violation), "level_cap": _cv(e.level_cap),
           "height": _cv(e.height), "versions": _cv(short_versions(e.versions))}
    if e.g0_reason == "nothing_certified":
        out.update(mcert_count=_cv(e.mcert_count), tested_blocks=_cv(e.tested_blocks), sil_px=_cv(e.sil_px))
    if e.g0_reason in ("ancestor", "hop"):
        out["hop_index"] = _cv(e.hop_index)
    return out


def to_obj(e: Evidence) -> dict:
    return short_form(e) if e.is_short() else full_form(e)


def canonical(e: Evidence) -> bytes:
    return jcs(to_obj(e))


# ------------------------------------------------------------------------------------------------ from canonical
def _dv(v):
    if isinstance(v, str):
        if _INT_RE.match(v):
            return int(v)
        if _HEX_RE.match(v):
            return bytes.fromhex(v[2:])
        return v
    if isinstance(v, list):
        return [_dv(x) for x in v]
    if isinstance(v, dict):
        return {k: _dv(x) for k, x in v.items()}
    return v


def _dversions(v):
    if v is None:
        return None
    out = {k: (x if k in ("spec", "u_is_level_criterion") else _dv(x)) for k, x in v.items()}
    lt = v.get("label_table")
    if lt is not None:
        out["label_table"] = {k: (x if k == "regime" else _dv(x)) for k, x in lt.items()}
    return out


def _dtone(t):
    if t is None:
        return None
    return {k: (x if k == "class" else Micro(int(x)) if k in DE00_KEYS else _dv(x)) for k, x in t.items()}


def from_canonical(b: bytes) -> Evidence:
    obj = json.loads(bytes(b).decode("utf-8"))
    if not isinstance(obj, dict) or jcs(obj) != bytes(b):
        raise ValueError("not canonical JSON of an evidence record")
    keys = set(obj)
    extra = {"nothing_certified": {"mcert_count", "tested_blocks", "sil_px"}, "ancestor": {"hop_index"},
             "hop": {"hop_index"}}
    is_short = keys != set(EVIDENCE_FIELDS)
    if is_short and keys != set(SHORT_KEYS) | extra.get(obj.get("g0_reason"), set()):
        raise ValueError(f"record keys match neither the full nor the short form: {sorted(keys)}")
    e = Evidence()
    for k, v in obj.items():
        if k in STRING_FIELDS:
            setattr(e, k, v)
        elif k == "versions":
            e.versions = _dversions(v)
        elif k == "tone":
            e.tone = _dtone(v)
        else:
            setattr(e, k, _dv(v))
    if is_short != e.is_short():
        raise ValueError("the short form is used iff g0_reason or static_violation is set")
    return e


__all__ = ["Micro", "micro", "versions", "short_versions", "full_form", "short_form", "to_obj", "canonical",
           "from_canonical", "VERSION_KEYS", "DE00_KEYS", "STRING_FIELDS", "SHORT_KEYS"]
