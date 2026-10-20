"""cutline.errors - the typed error model.

Every protocol check that can fail has a name ``E.<AREA>.<NAME>`` (for example ``"E.CONTAINER.NO_SOI"``). The
reference raises a :class:`CutlineError` subclass carrying that name in ``.name`` (and a free-text ``.detail`` that is
never normative); the name, not the class, is what other implementations must agree on.

- ``CutlineError(name, detail="")``: base class; ``str(e)`` is ``name`` or ``"name: detail"``.
- Area subclasses: ``ContainerError`` (E.CONTAINER), ``ABIError`` (E.ABI, reference-internal, callers remap it, e.g. to
  E.BUNDLE.ABI), ``MerkleError`` (E.MERKLE), ``RecError`` (E.REC), ``BundleError`` (E.BUNDLE), ``StaticError``
  (E.STATIC), ``HopError`` (E.HOP), ``ProgramError`` (enc(pi) handling; E.STATIC.PROGRAM* / rule 5(9)),
  ``RYError`` (E.RY), ``AnchorError`` (E.ANCHOR), ``CapError`` (E.CAP), ``ClaimError`` (E.CLAIM), ``RegError``
  (E.REG), ``CourtError`` (E.COURT), ``VerifyEError`` (E.VERIFYE), ``DevError`` (E.DEV, DeviceRegistry),
  ``RelayError`` (E.RELAY, signed relays), ``RoleError`` (E.ROLE, role checks such as ``proveStaticViolation``
  step 1), ``ChainError`` (E.CHAIN: ``E.CHAIN.TOO_LONG``, ``E.CHAIN.UNLINKED``, ``E.CHAIN.ANCESTOR_UNAVAILABLE``),
  ``UError`` (E.U: ``E.U.RECORD_RY``, ``E.U.RECORD_DIGEST``, ``E.U.RECORD_MISMATCH``, ``E.U.MASK_SIZE``) and
  ``EvalError`` (E.EVAL, **reference-internal**: a misuse of the Evaluate API such as a missing ledger input or an
  inconsistent call, never a protocol outcome; Evaluate maps every protocol failure to a level and a ``g0_reason``
  and never lets an ``EvalError`` stand for one).
- ``error_for(name, detail="")``: builds the subclass that matches the name's area (``CutlineError`` otherwise), so a
  generic caller can write ``raise error_for("E.CAP.HIGH_S")``.
- ``area_of(name)``: the ``<AREA>`` part.
"""
from __future__ import annotations

import re

_NAME_RE = re.compile(r"^E\.[A-Z0-9]+(\.[A-Z0-9_]+)?$")


class CutlineError(Exception):
    """A protocol error. ``name`` is the ``E.*`` string; ``detail`` is free text (not normative)."""

    area: str | None = None          # set by subclasses; informational

    def __init__(self, name: str, detail: str = ""):
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"not an E.* error name: {name!r}")
        self.name = name
        self.detail = detail
        super().__init__(f"{name}: {detail}" if detail else name)

    def __reduce__(self):              # picklable across process pools
        return (self.__class__, (self.name, self.detail))


class ContainerError(CutlineError):
    area = "CONTAINER"


class ABIError(CutlineError):
    area = "ABI"


class MerkleError(CutlineError):
    area = "MERKLE"


class RecError(CutlineError):
    area = "REC"


class BundleError(CutlineError):
    area = "BUNDLE"


class StaticError(CutlineError):
    area = "STATIC"


class HopError(CutlineError):
    area = "HOP"


class ProgramError(CutlineError):
    area = "PROGRAM"


class RYError(CutlineError):
    area = "RY"


class AnchorError(CutlineError):
    area = "ANCHOR"


class CapError(CutlineError):
    area = "CAP"


class ClaimError(CutlineError):
    area = "CLAIM"


class RegError(CutlineError):
    area = "REG"


class CourtError(CutlineError):
    area = "COURT"


class VerifyEError(CutlineError):
    area = "VERIFYE"


class DevError(CutlineError):
    area = "DEV"


class RelayError(CutlineError):
    area = "RELAY"


class RoleError(CutlineError):
    area = "ROLE"


class ChainError(CutlineError):
    area = "CHAIN"


class UError(CutlineError):
    area = "U"


class EvalError(CutlineError):
    area = "EVAL"


_BY_AREA = {c.area: c for c in (ContainerError, ABIError, MerkleError, RecError, BundleError, StaticError, HopError,
                                ProgramError, RYError, AnchorError, CapError, ClaimError, RegError, CourtError,
                                VerifyEError, DevError, RelayError, RoleError, ChainError, UError,
                                EvalError)}


def area_of(name: str) -> str:
    """``"E.CONTAINER.NO_SOI"`` -> ``"CONTAINER"``."""
    return name.split(".")[1]


def error_for(name: str, detail: str = "") -> CutlineError:
    """The CutlineError subclass instance for ``name``'s area (CutlineError when the area has no subclass)."""
    return _BY_AREA.get(area_of(name), CutlineError)(name, detail)
