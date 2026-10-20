"""cutline.region.registry - the segmenter registry.

Hash-pinned ONNX segmenters, read from ``MODELS_DIR``: BiRefNet-lite (the capture default of the emulated app, input
1,024, BiRefNet kind) and u2netp (registered alternative, input 320, u2net kind). The person-policy hook psi is
unpinned in v1.

- ``SEGMENTERS = {name: (file name, sha256 hex, input size, kind)}``; kinds ``"biref"``, ``"u2"``.
- ``MODELS_DIR``: ``<repo root>/models`` by default (override with the environment variable ``CUTLINE_MODELS_DIR``).
- ``model_path(name) -> str``: the model file after checking its SHA-256 (``ValueError`` on a mismatch; the digest
  is cached per (path, size, mtime)).
- ``policy_record(name, app, a_max=2048, psi=None, thr_ppm=500000) -> dict``: the capture-policy record naming the
  model by its SHA-256 and input size (feed it to ``cutline.commit.rec.pi_cap``).
- ``PUBSEGS = {name: (file name, sha256 hex, input size, kind)}``: the public segmenters of the listing half.
  ``"isnet"`` is PubSeg (ISNet-general-use, normative; kind ``"isnet"``: mean 0.5, std 1, min-max post-processing);
  ``"birefnet_lite"`` is the alternative arm (the capture model's file, kind ``"biref"``: sigmoid post-processing).
- ``DEFAULT_PUBSEG = "isnet"``.
- ``pubseg_model_path(name) -> str``: as ``model_path`` for ``PUBSEGS`` (SHA-256 checked, ``ValueError`` on a mismatch).
"""
from __future__ import annotations

import hashlib
import os

from ..commit.rec import pi_cap_record

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root
MODELS_DIR = os.environ.get("CUTLINE_MODELS_DIR") or os.path.join(_ROOT, "models")

SEGMENTERS = {
    "birefnet_lite": ("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
                      "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333", 1024, "biref"),
    "u2netp": ("u2netp.onnx", "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8", 320, "u2"),
}
DEFAULT_CAPTURE_SEGMENTER = "birefnet_lite"
PUBSEGS = {
    "isnet": ("isnet-general-use.onnx", "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a", 1024,
              "isnet"),
    "birefnet_lite": SEGMENTERS["birefnet_lite"],
}
DEFAULT_PUBSEG = "isnet"
_SHA_CACHE: dict = {}


def _sha256_file(p: str) -> str:
    st = os.stat(p)
    key = (p, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def model_path(name: str) -> str:
    fn, sha, _, _ = SEGMENTERS[name]
    p = os.path.join(MODELS_DIR, fn)
    got = _sha256_file(p)
    if got != sha:
        raise ValueError(f"segmenter {name}: SHA-256 {got} != pinned {sha}")
    return p


def policy_record(name: str, app: str, a_max: int = 2048, psi: str | None = None, thr_ppm: int = 500000) -> dict:
    _, sha, size, _ = SEGMENTERS[name]
    return pi_cap_record(app=app, seg_sha256_hex=sha, input_px=size, thr_ppm=thr_ppm, a_max_px_per_MP=a_max, psi=psi)


def pubseg_model_path(name: str) -> str:
    fn, sha, _, _ = PUBSEGS[name]
    p = os.path.join(MODELS_DIR, fn)
    got = _sha256_file(p)
    if got != sha:
        raise ValueError(f"PubSeg {name}: SHA-256 {got} != pinned {sha}")
    return p
