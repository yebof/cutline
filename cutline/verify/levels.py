"""cutline.verify.levels - Level L_off of the evidence record.

``level(e)`` is a function of the evidence record alone (the record carries every input of Level), as a cascade:

1. ``g0_reason`` in the stage-A set (``container``, ``bundle``, ``chain_too_long``, ``unlinked``, ``presenter``,
   ``unregistered``) -> G0;
2. ``static_violation`` set -> G1 (a claim-wide static rule; before the openings);
3. any other ``g0_reason`` (``openings``, ``nothing_certified``, ``ancestor``, ``hop``) -> G0;
4. ``verdict == "fail"`` -> G1 (the declared tier's test; no fallback);
5. G2 if any of ``g2_reasons(e)``; 6. G3 if any of ``g3_reasons(e)``; 7. else G4.

``final(e) = min(level(e), e.level_cap)`` (``level_cap`` None only in a short record that stopped before the claim
status was read, where L = Level).

How the conditions are read from the record:
- "the upload is 4:2:0" (G3): the record has no sampling field; for tier E ``tested_blocks`` counts the real blocks of
  M_cert's MCUs (3 per MCU at 4:4:4, 6 at 4:2:0), so the upload is 4:2:0 iff ``tested_blocks == 6 * mcert_count``
  (``mcert_count > 0`` whenever the record reaches Level). Tier T is G2 anyway.
- "some hop has an E1 class (3, 5)": ``enc_classes[1:]`` (upload first, then each hop).
- "U > tau_U": ``uncovered_px`` on ``grid_wh = (W, H)`` with the integer rule ``region.uncovered.u_exceeds``,
  applied iff ``versions.u_is_level_criterion`` (the release constant stamped in the record; ``consts`` when absent).
  Both fields are Y0's (|C_adj(Y0)| on Y0's grid) in every record, a rendition's included: U is a per-claim verdict
  decided on Y0 and is not re-tested on a rendition's grid, so a rendition's Level never exceeds its source level
  through U (monotonicity).
- "chain_class = coarse" includes a hop table above Q_hop(p, v*) (``resclass.class_hop`` already makes it coarse).

``g2_reasons(e)`` / ``g3_reasons(e)`` return the names of the triggered conditions (for reports), in order:
G2 ``tier_t``, ``coarse``, ``tone_major``, ``d_frac``, ``partial``, ``u``, ``hop_depth``, ``resampling``,
``cap_time_unverified``, ``receipt``; G3 ``background_generated``, ``tone_moderate``, ``d_nonempty``, ``hop_e1``,
``upload_420``.
"""
from __future__ import annotations

from cutline.consts import D_HOP, TAU_U, U_IS_LEVEL_CRITERION

STAGE_A_G0 = frozenset({"container", "bundle", "chain_too_long", "unlinked", "presenter", "unregistered"})
LATE_G0 = frozenset({"openings", "nothing_certified", "ancestor", "hop"})
E1_CLASSES = (3, 5)


def _g(e, k, default=None):
    return e.get(k, default) if isinstance(e, dict) else getattr(e, k, default)


def u_criterion(e) -> bool:
    v = _g(e, "versions") or {}
    x = v.get("u_is_level_criterion")
    return U_IS_LEVEL_CRITERION if x is None else bool(x)


def u_over(e) -> bool:
    from cutline.region.uncovered import u_exceeds
    px, wh = _g(e, "uncovered_px"), _g(e, "grid_wh")
    if px is None or wh is None:
        raise ValueError("U is a level criterion but the record has no uncovered_px / grid_wh")
    return u_exceeds(int(px), int(wh[0]), int(wh[1]), TAU_U)


def upload_420(e) -> bool:
    tb, mc = _g(e, "tested_blocks"), _g(e, "mcert_count")
    return _g(e, "tier") == "E" and mc is not None and tb is not None and int(mc) > 0 and int(tb) == 6 * int(mc)


def _tone_class(e):
    t = _g(e, "tone")
    return None if t is None else t.get("class")


def g2_reasons(e) -> list:
    d, m = int(_g(e, "d_count") or 0), int(_g(e, "mcert_count") or 0)
    checks = [
        ("tier_t", _g(e, "tier") == "T"),
        ("coarse", _g(e, "chain_class") == "coarse"),
        ("tone_major", _tone_class(e) == "major"),
        ("d_frac", 50 * d > m),
        ("partial", bool(_g(e, "partial"))),
        ("u", u_criterion(e) and u_over(e)),
        ("hop_depth", int(_g(e, "hop_depth") or 0) > D_HOP),
        ("resampling", int(_g(e, "resampling_hops") or 0) > 1),
        ("cap_time_unverified", _g(e, "cap") == "time-unverified"),
        ("receipt", _g(e, "receipt") in ("stale", "none")),
    ]
    return [k for k, v in checks if v]


def g3_reasons(e) -> list:
    ecs = list(_g(e, "enc_classes") or [])
    checks = [
        ("background_generated", _g(e, "background") == "generated-declared"),
        ("tone_moderate", _tone_class(e) == "moderate"),
        ("d_nonempty", int(_g(e, "d_count") or 0) > 0),
        ("hop_e1", any(int(c) in E1_CLASSES for c in ecs[1:])),
        ("upload_420", upload_420(e)),
    ]
    return [k for k, v in checks if v]


def level(e) -> int:
    g0 = _g(e, "g0_reason")
    if g0 in STAGE_A_G0:
        return 0
    if _g(e, "static_violation") is not None:
        return 1
    if g0 is not None:
        if g0 not in LATE_G0:
            raise ValueError(f"unknown g0_reason {g0!r}")
        return 0
    if _g(e, "verdict") == "fail":
        return 1
    if _g(e, "verdict") != "pass":
        raise ValueError("a record without g0_reason / static_violation must carry verdict pass or fail")
    if g2_reasons(e):
        return 2
    if g3_reasons(e):
        return 3
    return 4


def final(e) -> int:
    lv = level(e)
    cap = _g(e, "level_cap")
    return lv if cap is None else min(lv, int(cap))


def explain(e) -> dict:
    """``{"level", "final", "g2", "g3"}`` (the triggered conditions, for reports)."""
    lv = level(e)
    full = lv >= 2
    return {"level": lv, "final": final(e), "g2": g2_reasons(e) if full else [], "g3": g3_reasons(e) if full else []}


__all__ = ["level", "final", "explain", "g2_reasons", "g3_reasons", "upload_420", "u_over", "u_criterion",
           "STAGE_A_G0", "LATE_G0"]
