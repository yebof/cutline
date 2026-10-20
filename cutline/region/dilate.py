"""cutline.region.dilate - the certified capture region A.

A = A0 u {4-connected holes C of not-A0 with 10^6 |C| <= a_max * H0 * W0}, A0 = Dilate_ellipse(S u m, d),
d = ceil(0.01 * diag) in exact integers, a_max = 2048 px per master MP; enlarge-only. No verifier recomputes A (it is
committed through the cells' alpha); this fixes what an honest (emulated) app certifies.

- ``session(name, threads=3)``: the pinned ONNX Runtime session (CPUExecutionProvider, ORT_ENABLE_ALL, ``threads``
  intra-op and 1 inter-op thread); cached per (name, threads).
- ``seg_soft(name, M, threads=3) -> float32[H0, W0]``: step 1 (Pillow LANCZOS to s x s, float32 division by the
  maximum, float32 mean/std, NCHW float32, output [0, 0] as float64, sigmoid (biref) or min-max (u2), cast to
  float32, ``cv2.resize`` INTER_LINEAR to (W0, H0)).
- ``seg_mask(name, M, threads=3) -> bool`` = ``seg_soft(...) >= 0.5``.
- ``d_ceil(H0, W0)`` (step 2): the least integer d with 10^4 d^2 >= H0^2 + W0^2. ``d_round(H0, W0)`` is the float
  ``round(0.01 * hypot)`` variant (not normative).
- ``dilate(S, d) -> bool`` (step 3: ``cv2.dilate`` with the (2d+1) ellipse, default border).
- ``holes(A0) -> (labels int32[H0, W0], areas list[int] per hole id 1..n)`` (step 4: ``scipy.ndimage.label`` default,
  4-connected, structure on not-A0; components touching the border are dropped and the rest renumbered 1..n in
  label order).
- ``certified_region(S, strokes=None, a_max_ppm=2048, d=None, stats=False) -> A`` (steps 2-5; ``d`` overrides
  ``d_ceil``; with ``stats=True`` returns ``(A, {"d", "holes": sorted areas, "filled": sorted areas})``).
- ``person_policy(M, A, psi=None) -> bool`` (step 6): True = certify. In v1 no psi is pinned, so it always returns
  True; a pinned psi raises NotImplementedError.
- ``region_from_master(name, M, strokes=None, threads=3) -> A`` = steps 1-6.
"""
from __future__ import annotations

import math

import numpy as np

from ..consts import A_MAX_PPM
from .registry import SEGMENTERS, model_path

_MEAN_STD = {"biref": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             "u2": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             "isnet": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))}
_SESSIONS: dict = {}


def session(name: str, threads: int = 3):
    import onnxruntime as ort
    key = (name, threads)
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[key] = ort.InferenceSession(model_path(name), so, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def seg_soft(name: str, M, threads: int = 3) -> np.ndarray:
    import cv2
    from PIL import Image
    _, _, size, kind = SEGMENTERS[name]
    M = np.asarray(M)
    sess = session(name, threads)
    im = Image.fromarray(M).resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    a = a / max(a.max(), 1e-6)
    mean, std = _MEAN_STD[kind]
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    inp = {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)}
    o = sess.run(None, inp)[0][0, 0].astype(np.float64)
    if kind == "biref":
        o = 1 / (1 + np.exp(-o))
    else:
        o = (o - o.min()) / max(o.max() - o.min(), 1e-12)
    h, w = M.shape[:2]
    return cv2.resize(o.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def seg_mask(name: str, M, threads: int = 3) -> np.ndarray:
    return seg_soft(name, M, threads) >= 0.5


def d_ceil(H0: int, W0: int) -> int:
    s = H0 * H0 + W0 * W0
    d = math.isqrt(s) // 100
    while 10 ** 4 * d * d < s:
        d += 1
    return d


def d_round(H0: int, W0: int) -> int:
    return int(round(0.01 * float(np.hypot(H0, W0))))


def dilate(S, d: int) -> np.ndarray:
    import cv2
    S = np.asarray(S)
    if d <= 0:
        return S.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    return cv2.dilate(S.astype(np.uint8), k) > 0


def holes(A0) -> tuple:
    from scipy import ndimage
    A0 = np.asarray(A0, bool)
    lab, n = ndimage.label(~A0)
    if n == 0:
        return np.zeros(A0.shape, np.int32), []
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep = np.ones(n + 1, bool)
    keep[0] = False
    keep[border] = False
    remap = np.zeros(n + 1, np.int32)
    ids = np.nonzero(keep)[0]
    remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    out = remap[lab]
    areas = np.bincount(out.ravel(), minlength=len(ids) + 1)[1:]
    return out, [int(a) for a in areas]


def certified_region(S, strokes=None, a_max_ppm: int = A_MAX_PPM, d: int | None = None, stats: bool = False):
    S = np.asarray(S, bool)
    H0, W0 = S.shape
    if strokes is not None:
        S = S | np.asarray(strokes, bool)
    dd = d_ceil(H0, W0) if d is None else int(d)
    A0 = dilate(S, dd)
    lab, areas = holes(A0)
    fill_ids = [i + 1 for i, a in enumerate(areas) if 10 ** 6 * a <= a_max_ppm * H0 * W0]
    A = A0.copy()
    if fill_ids:
        sel = np.zeros(len(areas) + 1, bool)
        sel[fill_ids] = True
        A |= sel[lab]
    if not stats:
        return A
    return A, {"d": dd, "holes": sorted(areas), "filled": sorted(areas[i - 1] for i in fill_ids)}


def person_policy(M, A, psi: str | None = None) -> bool:
    if psi is None or psi == "00" * 32:
        return True
    raise NotImplementedError("person detector psi is not pinned in v1")


def region_from_master(name: str, M, strokes=None, threads: int = 3, psi: str | None = None):
    S = seg_mask(name, M, threads)
    A = certified_region(S, strokes)
    if not person_policy(M, A, psi):
        return None
    return A
