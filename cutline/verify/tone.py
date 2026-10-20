"""cutline.verify.tone - tone evidence (a heuristic field of the evidence record).

The pinned float64 CIEDE2000 reference and the tone class with its 10^-6 guard band. Tone is a **heuristic** field
(like U): two verifiers agree unless their float results straddle the guard band, and ``heuristic_digests.tone`` =
``TONE_SHA256`` (SHA-256 of this file's bytes) names the implementation. The class only ever lowers a level.

- **Pixel set:** the opened level-l* pixels inside the crop with alpha = 255 (``Openings.crop_planes()``; at
  alpha = 255 the premultiplied colour is the colour).
- **Statistics:** dE00 between p and LUT(p) per pixel with ``de2000(srgb_to_lab(p), srgb_to_lab(LUT(p)))`` (sRGB
  D65, Sharma-Wu-Dalal 2005, kL = kC = kH = 1); ``dE00_p50`` and ``dE00_p95`` by nearest rank (the value at 1-based
  rank ⌈P/100·N⌉ of the ascending values) and ``dE00_max``. The values are computed once per distinct colour (same
  float64 results).
- **``area_dE6``:** the size of the largest 8-connected region of listing pixels with dE00 > 6 between the composite
  without and with the LUT, over the listing pixels with alpha = 255 (in listing pixels).
- **``delta``** = max_{c, v} |LUT[c][v] − v| (integer).
- **Class:** ``identity`` iff the LUT is the identity (delta = 0); else ``minor`` iff p95 ≤ 2.0 and max ≤ 4.0;
  ``moderate`` iff p95 ≤ 4.0 and max ≤ 10.0; else ``major``. Guard band: a comparison x ≤ t holds only if
  x < t − 10^-6 (a statistic within 10^-6 of a threshold resolves to the worse class). **Category caps**
  (the claim's ``category``): 1 = cosmetics, 2 = apparel need p95 ≤ 2.0 (with the guard band), else ``major``; 0 and
  3-255 are general.
- **Empty pixel set** (no opened alpha-255 pixel in the crop) with a non-identity LUT: the statistics are 0.0,
  ``area_dE6`` = 0 and the class is ``major`` (nothing to measure the LUT on resolves to the worse class).
- The canonical record stores each dE00 statistic in micro-units ``record.micro(x)`` (exact product).

API:
- ``srgb_to_lab(rgb)``, ``de2000(lab1, lab2)``: the pinned float64 reference.
- ``lut_planes(lut) -> int64[3, 256]``; ``lut_delta(lut) -> int``; ``apply_lut(lut, P) -> uint8[..., 3]``.
- ``de00_pixels(P, lut) -> float64[N]`` for uint8 pixels P [N, 3] (one value per pixel).
- ``nearest_rank_f(sorted_values, P) -> float``.
- ``tone_class(p95, dmax, identity, category=0) -> str``.
- ``tone(lut, C_lstar, alpha_lstar, crop=None, compiled_nolut=None, compiled=None, category=0) -> dict``: the record's
  ``tone`` object ``{delta, dE00_p50, dE00_p95, dE00_max, area_dE6, class}``. ``C_lstar`` / ``alpha_lstar`` are the
  crop planes (or the whole level-l* planes with ``crop = (y0, x0, ch, cw)``). ``compiled`` / ``compiled_nolut``
  (``types.Compiled`` with and without the LUT) give ``area_dE6``; without them the area is 0.
- ``TONE_SHA256`` (bytes32).
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np

GUARD = 1e-6
TONE_SHA256 = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).digest()
CLASSES = ("identity", "minor", "moderate", "major")


# ------------------------------------------------------------------------------------------------ pinned reference
def srgb_to_lab(rgb):
    """sRGB uint8 (...,3) -> CIELAB (D65), float64."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de2000(lab1, lab2):
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005 formulation), kL=kC=kH=1."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0, np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# ------------------------------------------------------------------------------------------------ helpers
def lut_planes(lut) -> np.ndarray:
    if isinstance(lut, (bytes, bytearray, memoryview)):
        b = bytes(lut)
        if len(b) != 768:
            raise ValueError("a LUT is 768 bytes")
        return np.frombuffer(b, np.uint8).astype(np.int64).reshape(3, 256)
    a = np.asarray(lut).astype(np.int64)
    if a.shape != (3, 256):
        raise ValueError("a LUT is (3, 256)")
    return a


def lut_delta(lut) -> int:
    return int(np.abs(lut_planes(lut) - np.arange(256)[None, :]).max())


def apply_lut(lut, P) -> np.ndarray:
    L = lut_planes(lut)
    P = np.asarray(P)
    return np.stack([L[c][P[..., c].astype(np.int64)] for c in range(3)], -1).astype(np.uint8)


def de00_pixels(P, lut) -> np.ndarray:
    """dE00(p, LUT(p)) for uint8 pixels P [N, 3], evaluated once per distinct colour."""
    P = np.asarray(P, np.uint8).reshape(-1, 3)
    if P.shape[0] == 0:
        return np.zeros(0, np.float64)
    key = (P[:, 0].astype(np.int64) << 16) | (P[:, 1].astype(np.int64) << 8) | P[:, 2].astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    U = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8)
    d = de2000(srgb_to_lab(U), srgb_to_lab(apply_lut(lut, U)))
    return d[inv.reshape(-1)]


def nearest_rank_f(sorted_values, P) -> float:
    n = len(sorted_values)
    if n == 0:
        return 0.0
    r = max(1, math.ceil(P * n / 100)) if not isinstance(P, int) else max(1, -(-P * n // 100))
    return float(sorted_values[r - 1])


def _le(x: float, t: float) -> bool:
    return x < t - GUARD


def tone_class(p95: float, dmax: float, identity: bool, category: int = 0, empty: bool = False) -> str:
    if identity:
        return "identity"
    if empty:
        return "major"
    if int(category) in (1, 2) and not _le(p95, 2.0):
        return "major"
    if _le(p95, 2.0) and _le(dmax, 4.0):
        return "minor"
    if _le(p95, 4.0) and _le(dmax, 10.0):
        return "moderate"
    return "major"


def largest_region(mask) -> int:
    from scipy import ndimage
    m = np.asarray(mask, bool)
    if not m.any():
        return 0
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    return int(np.bincount(lab.reshape(-1))[1:].max())


def area_de6(compiled_nolut, compiled, lut) -> int:
    a = np.asarray(compiled.alpha) == 255
    if not a.any():
        return 0
    X0 = np.asarray(compiled_nolut.X)
    X1 = np.asarray(compiled.X)
    d = np.zeros(a.shape, np.float64)
    ys, xs = np.nonzero(a)
    P0, P1 = X0[ys, xs], X1[ys, xs]
    if not np.array_equal(apply_lut(lut, P0), P1):
        raise ValueError("compiled.X != LUT(compiled_nolut.X) on alpha-255 pixels (not the same program)")
    d[ys, xs] = de00_pixels(P0, lut)
    return largest_region(a & (d > 6.0))


def tone(lut, C_lstar, alpha_lstar, crop=None, compiled_nolut=None, compiled=None, category: int = 0) -> dict:
    C = np.asarray(C_lstar)
    A = np.asarray(alpha_lstar)
    if crop is not None:
        y0, x0, ch, cw = crop
        C, A = C[y0:y0 + ch, x0:x0 + cw], A[y0:y0 + ch, x0:x0 + cw]
    delta = lut_delta(lut)
    identity = delta == 0
    P = C[A == 255]
    empty = P.shape[0] == 0
    if identity or empty:
        p50 = p95 = dmax = 0.0
        area = 0
    else:
        d = np.sort(de00_pixels(P, lut))
        p50, p95, dmax = nearest_rank_f(d, 50), nearest_rank_f(d, 95), float(d[-1])
        area = area_de6(compiled_nolut, compiled, lut) if compiled is not None and compiled_nolut is not None else 0
    return {"delta": delta, "dE00_p50": p50, "dE00_p95": p95, "dE00_max": dmax, "area_dE6": area,
            "class": tone_class(p95, dmax, identity, category, empty)}


__all__ = ["srgb_to_lab", "de2000", "lut_planes", "lut_delta", "apply_lut", "de00_pixels", "nearest_rank_f",
           "tone_class", "largest_region", "area_de6", "tone", "TONE_SHA256", "GUARD", "CLASSES"]
