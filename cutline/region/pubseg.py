"""cutline.region.pubseg - PubSeg, the pinned public segmenter of the listing half.

S_pub = PubSeg(Dec*(Y0)) >= 0.5 with ISNet-general-use (normative) and a pinned pre/post-processing pipeline;
BiRefNet-lite is an alternative arm (its own model, sigmoid post-processing and digest). S_pub is a heuristic field;
everything computed from it is exact.

- ``session(name="isnet", threads=3, arena=True)``: the ONNX Runtime 1.30.0 session (CPUExecutionProvider,
  ORT_ENABLE_ALL, ``threads`` intra-op, 1 inter-op); ``arena=False`` turns the CPU arena and memory pattern off
  (allocation only, numerics unchanged). Cached per (name, threads, arena).
- ``soft_from_rgb(rgb, name="isnet", threads=3) -> float32[H, W]``: pipeline steps 2-7 on an H x W x 3 uint8 image.
- ``s_pub_from_rgb(rgb, name="isnet", threads=3) -> bool[H, W]``: + step 8 (``s >= 0.5``).
- ``s_pub(listing_bytes, name="isnet", threads=3) -> bool[H, W]``: step 1 is ``cutline.dct.dec_star`` (Dec*).
- ``quantise8(soft) -> uint8``: ``rint(255 * clip(s, 0, 1))``, an 8-bit mask cache (``>= 128 <=> s >= 0.5`` for
  float32 s).
- ``BUILDS``: the pinned library builds; ``builds()`` the installed ones (``RuntimeError`` in ``manifest`` when they
  differ, since the digest names the pinned builds).
- ``SOURCES`` / ``sources() -> {repo-relative path: sha256 hex}``: the files on the S_pub computation path, listed
  explicitly: ``region/pubseg.py`` (steps 2-8), ``dct.py`` and ``jpeg/_io.py`` (Dec*, step 1) and
  ``region/registry.py`` (the model pins). Listing them explicitly (rather than the transitive import closure) keeps
  the digest stable under edits to package files that S_pub never executes.
- ``manifest(threads=3, name="isnet") -> dict``: the PubSeg manifest (the BiRefNet-lite arm uses ``norm``
  "div_max;(x-mean)/std imagenet" and ``post`` "sigmoid").
- ``container_manifest(oci_digest_hex) -> dict``: the container form of the manifest.
- ``digest(name="isnet", threads=3, manifest=None) -> bytes32`` =
  ``H("CUTLINE-pubseg-v1" || H(model file bytes) || H(JCS(manifest)))`` (81-byte preimage).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from ..consts import PFX_PUBSEG, PUBSEG_MANIFEST_V
from ..enc import jcs
from .registry import PUBSEGS, pubseg_model_path

_MEAN_STD = {"isnet": ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
             "biref": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}
_SESSIONS: dict = {}
BUILDS = {"pillow": "12.3.0", "pillow_libjpeg_turbo": "3.1.4.1", "opencv": "5.0.0", "onnxruntime": "1.30.0",
          "numpy": "2.5.3"}
THRESHOLD_PPM = 500_000
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the cutline package
_REPO = os.path.abspath(os.path.join(_PKG_DIR, ".."))                            # repository root


# ------------------------------------------------------------------------------------------------ pipeline
def session(name: str = "isnet", threads: int = 3, arena: bool = True):
    import onnxruntime as ort
    key = (name, threads, arena)
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        if not arena:
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[key] = ort.InferenceSession(pubseg_model_path(name), so, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def soft_from_rgb(rgb, name: str = "isnet", threads: int = 3, arena: bool = True) -> np.ndarray:
    import cv2
    from PIL import Image
    _, _, size, kind = PUBSEGS[name]
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("PubSeg input must be an H x W x 3 uint8 image")
    sess = session(name, threads, arena)
    im = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)                       # step 2
    a = np.asarray(im).astype(np.float32)
    a = a / max(a.max(), 1e-6)                                                          # step 3
    mean, std = _MEAN_STD[kind]
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)                   # step 4
    inp = {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)}
    o = sess.run(None, inp)[0][0, 0].astype(np.float64)                                # step 5
    if kind == "biref":
        o = 1 / (1 + np.exp(-o))
    else:
        o = (o - o.min()) / max(o.max() - o.min(), 1e-12)                               # step 6
    h, w = rgb.shape[:2]
    return cv2.resize(o.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)     # step 7


def s_pub_from_rgb(rgb, name: str = "isnet", threads: int = 3, arena: bool = True) -> np.ndarray:
    return soft_from_rgb(rgb, name, threads, arena) >= 0.5                              # step 8


def s_pub(listing_bytes: bytes, name: str = "isnet", threads: int = 3, arena: bool = True) -> np.ndarray:
    from ..dct import dec_star
    return s_pub_from_rgb(dec_star(listing_bytes), name, threads, arena)                # step 1


def quantise8(soft) -> np.ndarray:
    return np.rint(np.clip(np.asarray(soft), 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ digest
def builds() -> dict:
    import cv2
    import numpy
    import onnxruntime
    import PIL
    from PIL import features
    return {"pillow": PIL.__version__, "pillow_libjpeg_turbo": features.version("libjpeg_turbo"),
            "opencv": cv2.__version__, "onnxruntime": onnxruntime.__version__, "numpy": numpy.__version__}


# The S_pub computation path: this module (steps 2-8), Dec* (step 1: ``cutline.dct.dec_star`` and its temporary-file
# helper ``cutline.jpeg._io``) and the model pins (``cutline.region.registry``). Package files that S_pub never
# executes (bundle, claim, ledger code ...) are not sources, so editing them does not change the PubSeg digest.
SOURCES = ("cutline/region/pubseg.py", "cutline/dct.py", "cutline/jpeg/_io.py", "cutline/region/registry.py")


def sources() -> dict:
    return {rel: hashlib.sha256(open(os.path.join(_REPO, rel), "rb").read()).hexdigest() for rel in SOURCES}


def manifest(threads: int = 3, name: str = "isnet") -> dict:
    got = builds()
    if got != BUILDS:
        raise RuntimeError(f"PubSeg builds {got} differ from the pinned {BUILDS}")
    _, _, size, kind = PUBSEGS[name]
    params = {"size": size, "resize_in": "PIL.LANCZOS",
              "norm": "div_max;(x-0.5)/1.0" if kind == "isnet" else "div_max;(x-mean)/std imagenet",
              "post": "minmax;1e-12" if kind == "isnet" else "sigmoid",
              "resize_out": "cv2.INTER_LINEAR", "threshold_ppm": THRESHOLD_PPM, "ort_threads": int(threads),
              "ort_opt": "ORT_ENABLE_ALL"}
    return {"v": PUBSEG_MANIFEST_V, "sources": sources(), "builds": dict(BUILDS), "params": params}


def container_manifest(oci_digest_hex: str) -> dict:
    h = oci_digest_hex.lower().removeprefix("sha256:")
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("OCI image digest must be 64 hex digits")
    return {"v": PUBSEG_MANIFEST_V, "container": "sha256:" + h}


def digest(name: str = "isnet", threads: int = 3, manifest: dict | None = None) -> bytes:
    m = globals()["manifest"](threads, name) if manifest is None else manifest
    model_h = bytes.fromhex(PUBSEGS[name][1])
    pubseg_model_path(name)                    # checks the file on disk against the pinned hash
    pre = PFX_PUBSEG + model_h + hashlib.sha256(jcs(m)).digest()
    assert len(pre) == 81
    return hashlib.sha256(pre).digest()
