"""cutline.labels - (level, evidence fields) -> label per regime and table version.

The pinned label vocabulary and the per-regime label tables as data. ``labels_v1.json`` next to this module
(canonical JSON) is the exported form; the on-chain LabelPolicy contract pins the same vocabulary in bytecode.

Data:
- ``VOCABULARY``: name -> minimum level (7 names at level 0, ``PARTIALLY_VERIFIED`` at 2, 12 names at 3). No
  AI-positive term below G2; absent at every level: "AI-generated", "no AI", ``digitalCapture`` / ``humanEdits``
  (v1.1), the seller's ``TrainedAlgorithmicMedia`` tags (``ABSENT``).
- ``LABELS_V1 = {regime: {"no_provenance_code", "version": 1, "code": [5 ints], "name": [5 names],
  "terms": [[condition, name, min level], ...]}}`` for the genesis regimes ``EU``, ``CN``, ``IPTC``, ``GMC``: the base
  table (one code and one name per level G0..G4, what ``LabelPolicy.setTable`` stores) and the field-derived terms
  that a relying party composes for G3/G4 (labels are a function of the level and the evidence fields; the on-chain
  table fixes only each level's base code and name).
- Term conditions (``TERM_CONDITIONS``), evaluated on the evidence record: ``background_generated``
  (background = generated-declared), ``d_nonempty`` (d_count > 0), ``d_gen`` (d_count > 0 and d_gen), ``tone_moderate``,
  ``hop_e1`` (a hop of class 3 or 5), ``upload_420`` (``levels.upload_420``).

API:
- ``label(regime, L, e=None, version=1, evaluated=True) -> dict``: ``{"regime", "version", "level", "code", "name",
  "terms", "overlay"}``. ``L`` is the final level (``levels.final``). Terms only at L >= 3 and only those whose minimum
  level is <= L; ``overlay`` (the certified silhouette overlay, mandatory with every G3/G4 label) iff L >= 3. A party
  that does not run VerifyE (``evaluated=False``, receipt-only) shows only negative status or "not evaluated":
  ``code`` = the no-provenance code, ``level`` = L for L <= 1 (invalid / failed) else None with the name
  ``NOT_EVALUATED``, and ``status`` = ``receipt_status(L, e)`` (invalid, failed, time-unverified, not_evaluated);
  never a positive code, term or overlay. The on-chain LabelPolicy is unaffected: it stores the base table of
  evaluating parties; NOT_EVALUATED is a level-0 vocabulary name there.
- ``table(regime, version=1) -> (code[5], name[5])``; ``check_table(code, name, no_provenance_code)`` = the
  ``setTable`` checks 2-4 of the LabelPolicy contract (monotone codes; G0 = G1 = the no-provenance code; every name in
  the vocabulary at a level >= its minimum), raising ``ValueError`` with the contract's error name.
- ``export_json() -> bytes`` (canonical JSON of the whole table set; equals ``labels_v1.json``).
"""
from __future__ import annotations

import os

VOCABULARY = {
    "NO_PROVENANCE_INFO": 0, "NOT_EVALUATED": 0, "REGIME_DEFAULT": 0, "PLATFORM_DEFAULT": 0,
    "PLATFORM_OWN_NOTICE": 0, "SELLER_TAGS_KEPT": 0, "NONE": 0,
    "PARTIALLY_VERIFIED": 2,
    "CAPTURE_REGION_VERIFIED": 3, "CAPTURE_VERIFIED_ENC_TOLERANCE": 3, "CAPTURE_VERIFIED_PENDING": 3,
    "PROVENANCE_NOTE_REGION_VERIFIED": 3, "AI_BACKGROUND_DECLARED": 3, "RETOUCHED_AREAS_MARKED": 3,
    "GENERATIVE_RETOUCH_DECLARED": 3, "COLOUR_ADJUSTED": 3, "AI_ELEMENT_NOTICE": 3, "IPTC_COMPOSITE_SYNTHETIC": 3,
    "IPTC_COMPOSITE_TRAINED_ALGO": 3, "GMC_COMPOSITE_SYNTHETIC": 3,
}
ABSENT = ("AI_GENERATED", "NO_AI", "DIGITAL_CAPTURE", "HUMAN_EDITS", "TRAINED_ALGORITHMIC_MEDIA")
TERM_CONDITIONS = ("background_generated", "d_nonempty", "d_gen", "tone_moderate", "hop_e1", "upload_420")

LABELS_V1 = {
    "EU": {"no_provenance_code": 0, "version": 1, "code": [0, 0, 1, 2, 3],
           "name": ["NO_PROVENANCE_INFO", "NO_PROVENANCE_INFO", "PARTIALLY_VERIFIED", "CAPTURE_REGION_VERIFIED",
                    "CAPTURE_REGION_VERIFIED"],
           "terms": [["background_generated", "AI_BACKGROUND_DECLARED", 3],
                     ["d_nonempty", "RETOUCHED_AREAS_MARKED", 3], ["d_gen", "GENERATIVE_RETOUCH_DECLARED", 3],
                     ["tone_moderate", "COLOUR_ADJUSTED", 3], ["hop_e1", "CAPTURE_VERIFIED_ENC_TOLERANCE", 3],
                     ["upload_420", "CAPTURE_VERIFIED_PENDING", 3]]},
    "CN": {"no_provenance_code": 0, "version": 1, "code": [0, 0, 1, 2, 3],
           "name": ["PLATFORM_OWN_NOTICE", "PLATFORM_OWN_NOTICE", "PLATFORM_DEFAULT",
                    "PROVENANCE_NOTE_REGION_VERIFIED", "PROVENANCE_NOTE_REGION_VERIFIED"],
           "terms": [["background_generated", "AI_ELEMENT_NOTICE", 3], ["d_gen", "AI_ELEMENT_NOTICE", 3]]},
    "IPTC": {"no_provenance_code": 0, "version": 1, "code": [0, 0, 0, 1, 1],
             "name": ["NONE", "NONE", "NONE", "NONE", "NONE"],
             "terms": [["background_generated", "IPTC_COMPOSITE_SYNTHETIC", 3],
                       ["d_gen", "IPTC_COMPOSITE_TRAINED_ALGO", 3]]},
    "GMC": {"no_provenance_code": 0, "version": 1, "code": [0, 0, 0, 1, 1],
            "name": ["SELLER_TAGS_KEPT", "SELLER_TAGS_KEPT", "SELLER_TAGS_KEPT", "SELLER_TAGS_KEPT",
                     "SELLER_TAGS_KEPT"],
            "terms": [["background_generated", "GMC_COMPOSITE_SYNTHETIC", 3]]},
}
JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_v1.json")


def check_table(code, name, no_provenance_code: int) -> None:
    code, name = list(code), list(name)
    if len(code) != 5 or len(name) != 5:
        raise ValueError("E.LABEL.VOCAB: a table has five levels")
    if any(code[i] > code[i + 1] for i in range(4)):
        raise ValueError("E.LABEL.MONOTONE")
    if code[0] != no_provenance_code or code[1] != no_provenance_code:
        raise ValueError("E.LABEL.NO_PROVENANCE")
    for lv, n in enumerate(name):
        if n not in VOCABULARY or lv < VOCABULARY[n]:
            raise ValueError(f"E.LABEL.VOCAB: {n!r} at G{lv}")


def table(regime: str, version: int = 1) -> tuple:
    r = LABELS_V1[regime]
    if version != r["version"]:
        raise KeyError(f"no table version {version} for regime {regime}")
    return list(r["code"]), list(r["name"])


def _g(e, k, default=None):
    return e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)


def term_holds(cond: str, e) -> bool:
    from cutline.verify.levels import upload_420
    if e is None:
        return False
    if cond == "background_generated":
        return _g(e, "background") == "generated-declared"
    if cond == "d_nonempty":
        return int(_g(e, "d_count") or 0) > 0
    if cond == "d_gen":
        return int(_g(e, "d_count") or 0) > 0 and bool(_g(e, "d_gen"))
    if cond == "tone_moderate":
        return ((_g(e, "tone") or {}).get("class")) == "moderate"
    if cond == "hop_e1":
        return any(int(c) in (3, 5) for c in list(_g(e, "enc_classes") or [])[1:])
    if cond == "upload_420":
        return upload_420(e)
    raise ValueError(f"unknown term condition {cond}")


def label(regime: str, L: int, e=None, version: int = 1, evaluated: bool = True) -> dict:
    if not 0 <= int(L) <= 4:
        raise ValueError("E.REG.TABLE: level above 4")
    code, name = table(regime, version)
    if not evaluated:
        return _receipt_only(regime, version, int(L), e, name)
    L = int(L)
    out = {"regime": regime, "version": version, "level": L, "code": code[L], "name": name[L], "terms": [],
           "overlay": L >= 3}
    if L >= 3:
        seen = []
        for cond, n, ml in LABELS_V1[regime]["terms"]:
            if L >= ml and term_holds(cond, e) and n not in seen:
                seen.append(n)
        out["terms"] = seen
    return out


def receipt_status(L: int, e=None) -> str:
    """The negative ledger status a receipt-only party may show: ``"invalid"`` (L = 0),
    ``"failed"`` (L = 1: court- or quorum-FAILED, WITHDRAWN, a lapsed licence, a struck principal),
    ``"time-unverified"`` (cap = time-unverified), else ``"not_evaluated"``."""
    if L <= 0:
        return "invalid"
    if L == 1:
        return "failed"
    if e is not None and _g(e, "cap") == "time-unverified":
        return "time-unverified"
    return "not_evaluated"


def _receipt_only(regime, version, L, e, name) -> dict:
    """A party that does not run VerifyE shows only negative status or "not evaluated", never a positive
    level or code: the code is always the regime's no-provenance code; ``level`` is 0 / 1 for the negative statuses
    and None otherwise (``name`` NOT_EVALUATED); no terms, no overlay."""
    npc = LABELS_V1[regime]["no_provenance_code"]
    status = receipt_status(L, e)
    lv = L if L <= 1 else None
    return {"regime": regime, "version": version, "level": lv, "code": npc,
            "name": name[L] if lv is not None else "NOT_EVALUATED", "terms": [], "overlay": False,
            "status": status}


def export_obj() -> dict:
    return {"v": "CUTLINE-labels-v1", "vocabulary": dict(VOCABULARY), "absent": list(ABSENT),
            "term_conditions": list(TERM_CONDITIONS), "regimes": LABELS_V1}


def export_json() -> bytes:
    from cutline.enc import jcs
    return jcs(export_obj())


def load_json(path: str = JSON_PATH) -> dict:
    import json
    return json.loads(open(path, "rb").read())


for _r, _t in LABELS_V1.items():
    check_table(_t["code"], _t["name"], _t["no_provenance_code"])
    for _c, _n, _ml in _t["terms"]:
        assert _c in TERM_CONDITIONS and VOCABULARY[_n] <= _ml, (_r, _c, _n)

__all__ = ["VOCABULARY", "ABSENT", "TERM_CONDITIONS", "LABELS_V1", "JSON_PATH", "check_table", "table", "term_holds",
           "label", "receipt_status", "export_obj", "export_json", "load_json"]
