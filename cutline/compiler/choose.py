"""cutline.compiler.choose - ChooseLevelAndCrop, the exact-aspect compiler.

``ChooseLevelAndCrop(bbox(O), target (oh, ow))``: l* is the coarsest level whose product bbox is at least the target
size; the crop is then enlarged by at most 2% per axis (enlargement only, so the bbox stays inside) and oh chosen so
that static rules 3 and 6 hold and ``cw·oh = ch·ow`` exactly when possible, else with the smallest aspect error.

This is the listing-geometry helper of ``cli compile``; it is not used by ``capture_emu.session.fit_program``.

- ``exact_aspect(cw0, ch0, ow, max_w=None, max_h=None) -> (cw, ch, ow, oh) | None``: the search loop order is cw
  ascending from cw0 to ⌊1.02·cw0⌋, then ch ascending from ch0 to ⌊1.02·ch0⌋, then oh = ⌊ow·ch/cw⌋ then ⌈ow·ch/cw⌉;
  the first exact candidate is returned, else the first candidate with the smallest error; candidates must satisfy
  rule 3 (``ow <= cw < 2·ow``, ``oh <= ch < 2·oh``, oh >= 1) and fit (``cw <= max_w``, ``ch <= max_h`` when given).
  The error is compared **exactly**: ``|log(cw·oh / (ch·ow))|`` is ordered like ``max(r, 1/r)`` with r the rational
  ``cw·oh / (ch·ow)``. Rule 6 is then asserted (``static_rules.aspect_ok``); None if no candidate.
- ``aspect_err(cw, ch, ow, oh) -> float``: ``|log(cw·oh / (ch·ow))|`` (reporting only).
- ``place_crop(bbox, size, dims) -> (y, x)``: where an enlarged crop of ``size = (ch, cw)`` goes: centred on the bbox
  (the extra split as ⌊extra/2⌋ before, the rest after), then shifted into ``[0, dims − size]``; the result always
  contains the bbox when ``size >= bbox`` and ``size <= dims``.
- ``choose_level_and_crop(occ_bbox_per_level, dims_per_level, target_ow) -> (lstar, crop, out)``: ``occ_bbox`` per
  level l = (y, x, h, w) of O_l = 1 in level-l pixels (level 0 = finest), ``dims`` = (h_l, w_l). l* = the
  largest l with ``w >= target_ow`` for which ``exact_aspect`` finds a candidate (rule 3 needs ``cw < 2·ow``, so a
  finer level is never better); ``crop = (y, x, ch, cw)`` (enc(π) order), ``out = (oh, ow)``. ``ValueError`` if no
  level fits: the target is larger than the product at level 0 (no upscaling), or narrower than
  ``min_listing_width`` = ⌊bw_Lmax/2⌋ + 1 (even the coarsest level would be downscaled by 2x or more, which rule 3
  forbids; the message names the minimum).
"""
from __future__ import annotations

import math
from fractions import Fraction

from cutline.compiler.static_rules import aspect_ok


def aspect_err(cw: int, ch: int, ow: int, oh: int) -> float:
    return abs(math.log((cw * oh) / (ch * ow)))


def _key(cw: int, ch: int, ow: int, oh: int) -> Fraction:
    r = Fraction(cw * oh, ch * ow)
    return r if r >= 1 else 1 / r


def exact_aspect(cw0: int, ch0: int, ow: int, max_w: int | None = None, max_h: int | None = None):
    best, best_key = None, None
    for cw in range(cw0, int(cw0 * 1.02) + 1):
        if not (ow <= cw < 2 * ow) or (max_w is not None and cw > max_w):
            continue
        for ch in range(ch0, int(ch0 * 1.02) + 1):
            if max_h is not None and ch > max_h:
                break
            lo, hi = (ow * ch) // cw, -(-(ow * ch) // cw)
            for oh in ((lo,) if lo == hi else (lo, hi)):
                if oh < 1 or not (oh <= ch < 2 * oh):
                    continue
                k = _key(cw, ch, ow, oh)
                if best is None or k < best_key:
                    best, best_key = (cw, ch, ow, oh), k
                if k == 1:
                    assert aspect_ok(*best)[0]
                    return best
    if best is not None:
        assert aspect_ok(*best)[0], best
    return best


def place_crop(bbox, size, dims) -> tuple:
    by, bx, bh, bw = bbox
    ch, cw = size
    H, W = dims
    y = min(max(0, by - (ch - bh) // 2), H - ch)
    x = min(max(0, bx - (cw - bw) // 2), W - cw)
    assert y <= by and x <= bx and y + ch >= by + bh and x + cw >= bx + bw, (bbox, size, dims)
    return y, x


def choose_level_and_crop(occ_bbox_per_level, dims_per_level, target_ow: int) -> tuple:
    if len(occ_bbox_per_level) != len(dims_per_level):
        raise ValueError("one bbox and one dims per level")
    for l in range(len(occ_bbox_per_level) - 1, -1, -1):
        by, bx, bh, bw = occ_bbox_per_level[l]
        H, W = dims_per_level[l]
        if bw < target_ow or bh < 1:
            continue
        r = exact_aspect(bw, bh, target_ow, W, H)
        if r is None:
            continue
        cw, ch, ow, oh = r
        y, x = place_crop((by, bx, bh, bw), (ch, cw), (H, W))
        return l, (y, x, ch, cw), (oh, ow)
    lo = min_listing_width(occ_bbox_per_level)
    if lo is not None and target_ow < lo:
        raise ValueError(f"a listing of width {target_ow} is narrower than the minimum {lo}: even the coarsest level's "
                         f"product bbox ({occ_bbox_per_level[-1][3]} px) would be downscaled by 2x or more, which "
                         "static rule 3 forbids (cw < 2·ow); list the product at least that wide")
    raise ValueError(f"no level fits a listing of width {target_ow} (no upscaling)")


def min_listing_width(occ_bbox_per_level):
    """The narrowest listing any level can serve, ``⌊bw_Lmax / 2⌋ + 1`` (rule 3 needs ``cw < 2·ow`` and the crop
    contains the product bbox, whose width at the coarsest level Lmax is bw_Lmax, at most 256 px); None without a
    nonempty bbox. Narrower listings are refused with that minimum in the message."""
    for by, bx, bh, bw in reversed(list(occ_bbox_per_level)):
        if bw >= 1 and bh >= 1:
            return bw // 2 + 1
    return None


__all__ = ["exact_aspect", "aspect_err", "place_crop", "choose_level_and_crop", "min_listing_width"]
