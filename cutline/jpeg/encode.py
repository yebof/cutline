"""cutline.jpeg.encode - the E0 encode wrapper and the encoder-zoo helper.

- ``encode_e0(X, q, sampling, progressive=False, optimize=False, restart_interval=0) -> bytes``: libjpeg-turbo 2.1.0
  (jpeglib ``turbo210``) ``islow`` encode of the compiled canvas X (uint8 [H, W, 3]) with simple chroma downsampling
  (turbo's h2v2, bias 1, 2 = Phi's h2v2), at IJG quality ``q`` (int) or the explicit tables
  ``q = (luma64, chroma64)`` (natural order bytes; Cb and Cr share table 1). ``sampling`` 0 = 4:4:4, 1 = 4:2:0.
  jpeglib cannot write a restart interval, so ``restart_interval`` must be 0 (restart-marker files are produced with
  ``encode_with("opencv:rst<n>", …)``).
- ``encode_with(encoder_id, X, q, sampling) -> bytes``: the zoo encoders, ids
  ``pillow`` / ``pillow:prog`` (progressive + optimised) / ``opencv`` / ``opencv:rst<n>`` /
  ``jpeglib:<ver>:<dct>[:<flag>,<flag>…]`` (e.g. ``jpeglib:9f:islow:-DO_FANCY_DOWNSAMPLING``,
  ``jpeglib:mozjpeg403:islow:-TRELLIS_QUANT,-TRELLIS_QUANT_DC,-OVERSHOOT_DERINGING``) / ``sharp:fast`` /
  ``sharp:mozjpeg`` (sharp 0.35 / libvips via node; ``NODEDIR`` holds ``node_modules`` with sharp) / ``go`` (Go
  ``image/jpeg``, 4:2:0 only, the ``GOENC`` binary). ``q`` is an IJG quality for every encoder but jpeglib (which
  also takes tables).
- ``NODEDIR`` = ``$CUTLINE_NODE_DIR`` or ``<repo root>/encoders/node``; ``GOENC`` = ``$CUTLINE_GOENC`` or
  ``<repo root>/encoders/bin/goenc``.
- ``zoo_encoder_id(name) -> encoder_id``: encoder-zoo configuration names (``6b/islow``, ``7/islow/nofancy``,
  ``mozjpeg403/islow/fast``, ``sharp/fast``, ``pillow``, …) -> ``encode_with`` ids.
- ``E0_ZOO_CONFIGS``: the 42 E0 configurations (the islow family of the encoder zoo).
Temporary files live in ``$TMPDIR`` (``cutline.jpeg._io``).
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from cutline.jpeg._io import ROOT, temp_path

NODEDIR = os.environ.get("CUTLINE_NODE_DIR") or os.path.join(ROOT, "encoders", "node")
GOENC = os.environ.get("CUTLINE_GOENC") or os.path.join(ROOT, "encoders", "bin", "goenc")
MOZ_FAST = ("-TRELLIS_QUANT", "-TRELLIS_QUANT_DC", "-OVERSHOOT_DERINGING")

E0_ZOO_CONFIGS = (
    "6b/islow", "7/islow", "7/islow/nofancy", "8/islow", "8/islow/nofancy", "8a/islow", "8a/islow/nofancy", "8b/islow",
    "8b/islow/nofancy", "8c/islow", "8c/islow/nofancy", "8d/islow", "8d/islow/nofancy", "9/islow", "9/islow/nofancy",
    "9a/islow", "9a/islow/nofancy", "9b/islow", "9b/islow/nofancy", "9c/islow", "9c/islow/nofancy", "9d/islow",
    "9d/islow/nofancy", "9e/islow", "9e/islow/nofancy", "9f/islow", "9f/islow/nofancy", "mozjpeg101/islow/default",
    "mozjpeg101/islow/fast", "mozjpeg201/islow/default", "mozjpeg201/islow/fast", "mozjpeg300/islow/fast",
    "mozjpeg403/islow/fast", "opencv", "pillow", "sharp/fast", "turbo120/islow", "turbo130/islow", "turbo140/islow",
    "turbo150/islow", "turbo200/islow", "turbo210/islow")
IJG7PLUS = ("7", "8", "8a", "8b", "8c", "8d", "9", "9a", "9b", "9c", "9d", "9e", "9f")


def _check_x(X):
    X = np.ascontiguousarray(np.asarray(X))
    if X.dtype != np.uint8 or X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("X must be uint8 [H, W, 3]")
    return X


def _jpeglib_encode(X, q, sampling, ver, dct, flags) -> bytes:
    import jpeglib
    dm = {"islow": jpeglib.JDCT_ISLOW, "ifast": jpeglib.JDCT_IFAST, "float": jpeglib.JDCT_FLOAT}[dct]
    kw = {}
    if isinstance(q, (int, np.integer)):
        qt = int(q)
    else:
        luma, chroma = q
        qt = np.stack([np.frombuffer(bytes(luma), np.uint8).reshape(8, 8),
                       np.frombuffer(bytes(chroma), np.uint8).reshape(8, 8)]).astype(np.uint16)
        kw["quant_tbl_no"] = np.array([0, 1, 1])
    with temp_path() as p:
        with jpeglib.version(ver):
            im = jpeglib.from_spatial(X)
            im.samp_factor = ((2, 2), (1, 1), (1, 1)) if sampling else ((1, 1), (1, 1), (1, 1))
            im.write_spatial(p, qt=qt, dct_method=dm, flags=list(flags), **kw)
        b = open(p, "rb").read()
    return _renumber_components(b) if "quant_tbl_no" in kw else b


def _renumber_components(b: bytes) -> bytes:
    """jpeglib numbers the components 0, 1, 2 when it writes explicit tables; JFIF (and grammar G6) wants 1, 2, 3.
    A lossless byte edit of the SOF and SOS component ids (the coefficients and the entropy-coded data are
    unchanged)."""
    out = bytearray(b)
    i = 2
    while i + 4 <= len(out) and out[i] == 0xFF:
        m = out[i + 1]
        if m == 0xD9:
            break
        L = (out[i + 2] << 8) | out[i + 3]
        s = i + 4
        if m in (0xC0, 0xC2) and [out[s + 6 + 3 * c] for c in range(3)] == [0, 1, 2]:
            for c in range(3):
                out[s + 6 + 3 * c] += 1
        if m == 0xDA:
            ns = out[s]
            for j in range(ns):
                if out[s + 1 + 2 * j] <= 2:
                    out[s + 1 + 2 * j] += 1
            k = i + 2 + L
            while not (out[k] == 0xFF and out[k + 1] not in (0x00,) and not 0xD0 <= out[k + 1] <= 0xD7):
                k += 1
            i = k
            continue
        i += 2 + L
    return bytes(out)


def encode_e0(X, q, sampling: int, progressive: bool = False, optimize: bool = False,
              restart_interval: int = 0) -> bytes:
    X = _check_x(X)
    if restart_interval:
        raise NotImplementedError("jpeglib cannot write a restart interval; use encode_with('opencv:rst<n>', ...)")
    flags = []
    if optimize:
        flags.append("+OPTIMIZE_CODING")
    if progressive:
        flags.append("+PROGRESSIVE_MODE")
    return _jpeglib_encode(X, q, sampling, "turbo210", "islow", flags)


def _pillow(X, q, sampling, prog=False) -> bytes:
    from PIL import Image, ImageFile
    kw = {"progressive": True, "optimize": True} if prog else {}
    old = ImageFile.MAXBLOCK
    ImageFile.MAXBLOCK = max(old, 4 * X.size + 65536)   # the optimised/progressive writer cannot suspend mid-stream
    try:
        with temp_path() as p:
            Image.fromarray(X).save(p, "JPEG", quality=int(q), subsampling=2 if sampling else 0, **kw)
            return open(p, "rb").read()
    finally:
        ImageFile.MAXBLOCK = old


def _opencv(X, q, sampling, rst=0) -> bytes:
    import cv2
    params = [cv2.IMWRITE_JPEG_QUALITY, int(q), cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
              cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420 if sampling else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    if rst:
        params += [cv2.IMWRITE_JPEG_RST_INTERVAL, int(rst)]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(X, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


SHARP_JS = r"""
const sharp = require('sharp');
const j = JSON.parse(require('fs').readFileSync(process.argv[2]));
sharp(j.src).jpeg({quality: j.q, chromaSubsampling: j.sub, mozjpeg: j.moz}).toFile(j.dst)
  .catch(e => { console.error(e); process.exit(1); });
"""


def _sharp(X, q, sampling, moz) -> bytes:
    from PIL import Image
    with temp_path(".png") as png, temp_path(".jpg") as dst, temp_path(".json") as jf, temp_path(".js") as js:
        Image.fromarray(X).save(png)
        with open(jf, "w") as f:
            json.dump({"src": png, "dst": dst, "q": int(q), "sub": "4:2:0" if sampling else "4:4:4", "moz": bool(moz)},
                      f)
        with open(js, "w") as f:
            f.write(SHARP_JS)
        env = dict(os.environ, NODE_PATH=os.path.join(NODEDIR, "node_modules"))
        subprocess.run(["node", js, jf], check=True, cwd=NODEDIR, env=env, capture_output=True)
        return open(dst, "rb").read()


def _go(X, q, sampling) -> bytes:
    from PIL import Image
    if not sampling:
        raise ValueError("Go image/jpeg writes 4:2:0 only")
    with temp_path(".png") as png, temp_path(".jpg") as dst:
        Image.fromarray(X).save(png)
        subprocess.run([GOENC, "enc", png, dst, str(int(q))], check=True, capture_output=True)
        return open(dst, "rb").read()


def encode_with(encoder_id: str, X, q, sampling: int) -> bytes:
    X = _check_x(X)
    parts = encoder_id.split(":")
    kind = parts[0]
    if kind == "pillow":
        return _pillow(X, q, sampling, prog=len(parts) > 1 and parts[1] == "prog")
    if kind == "opencv":
        rst = int(parts[1][3:]) if len(parts) > 1 and parts[1].startswith("rst") else 0
        return _opencv(X, q, sampling, rst)
    if kind == "jpeglib":
        ver, dct = parts[1], parts[2]
        flags = parts[3].split(",") if len(parts) > 3 and parts[3] else []
        return _jpeglib_encode(X, q, sampling, ver, dct, flags)
    if kind == "sharp":
        return _sharp(X, q, sampling, moz=len(parts) > 1 and parts[1] == "mozjpeg")
    if kind == "go":
        return _go(X, q, sampling)
    raise ValueError(f"unknown encoder id {encoder_id!r}")


def zoo_encoder_id(name: str) -> str:
    """Encoder-zoo configuration name -> encode_with id."""
    if name in ("pillow", "opencv", "go"):
        return name
    if name.startswith("sharp/"):
        return "sharp:" + name.split("/")[1]
    parts = name.split("/")
    ver, dct = parts[0], parts[1]
    flags = []
    if len(parts) > 2:
        if parts[2] == "fast":
            flags = list(MOZ_FAST)
        elif parts[2] == "nofancy":
            flags = ["-DO_FANCY_DOWNSAMPLING"]
        elif parts[2] != "default":
            raise ValueError(name)
    return f"jpeglib:{ver}:{dct}" + (":" + ",".join(flags) if flags else "")
