"""cutline.jpeg.libjpeg - the pinned libjpeg-turbo 2.1.0 build and the zero-warning counter.

- ``turbo210_so_path()``: path of jpeglib 1.0.2's ``cjpeglib_turbo210.abi3.so``, checked against the SHA-256 pinned in
  ``ENV_PINS.json`` (``codecs.jpeglib.builds.turbo210.so_sha256``). The pin file is ``<repo root>/env/ENV_PINS.json``
  unless the environment variable ``CUTLINE_ENV_PINS`` names another path.
- ``ensure_helper() -> ctypes.CDLL``: the C helper ``_warncount/warncount.c``, compiled on first use with ``gcc``
  against the bundled libjpeg headers (``-DLIBVER=70``) into ``_warncount/build/warncount70.so`` (rebuilt when a
  source is newer); it dlopens the pinned build, binds ``jpeg_mem_src_tj``, installs its own ``error_exit``
  (longjmp) and silences ``output_message``, and counts warnings with libjpeg's own ``emit_message``.
  Asserts ``libver() == 70``.
- ``warnings(data) -> (err1, nwarn1, err2, nwarn2, msg)``: run 1 = ``jpeg_read_header`` + ``jpeg_read_coefficients``;
  run 2 = full decompress with ``JCS_RGB``, ``JDCT_ISLOW``, fancy upsampling; ``err`` is 1 on an error exit; ``msg`` is
  the first message (informational only).
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import os
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_warncount")
BUILD = os.path.join(SRC, "build")
HELPER = os.path.join(BUILD, "warncount70.so")
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))           # repository root
ENV_PINS = os.environ.get("CUTLINE_ENV_PINS") or os.path.join(ROOT, "env", "ENV_PINS.json")
LIBVER = 70

_LOCK = threading.Lock()
_STATE: dict = {}


def _pinned_sha() -> str:
    pins = json.load(open(ENV_PINS))
    return pins["codecs"]["jpeglib"]["builds"]["turbo210"]["so_sha256"]


def turbo210_so_path() -> str:
    """The pinned reference decoder's shared object (SHA-256 checked once per process)."""
    if "so" in _STATE:
        return _STATE["so"]
    import jpeglib
    cands = glob.glob(os.path.join(os.path.dirname(jpeglib.__file__), "cjpeglib", "cjpeglib_turbo210*.so"))
    if len(cands) != 1:
        raise RuntimeError(f"expected one cjpeglib_turbo210 build, found {cands}")
    so = cands[0]
    got = hashlib.sha256(open(so, "rb").read()).hexdigest()
    want = _pinned_sha()
    if got != want:
        raise RuntimeError(f"cjpeglib_turbo210 SHA-256 {got} != pinned {want} ({ENV_PINS})")
    _STATE["so"] = so
    return so


def _build() -> None:
    os.makedirs(BUILD, exist_ok=True)
    tmp = f"{HELPER}.{os.getpid()}.tmp"
    cmd = ["gcc", "-O2", "-shared", "-fPIC", f"-I{SRC}", f"-DLIBVER={LIBVER}", "-o", tmp,
           os.path.join(SRC, "warncount.c"), "-ldl"]
    subprocess.run(cmd, check=True, capture_output=True)
    os.replace(tmp, HELPER)


def ensure_helper() -> ctypes.CDLL:
    """Load (building if needed) the warning-counter helper bound to the pinned turbo 2.1.0 build."""
    with _LOCK:
        if "lib" in _STATE:
            return _STATE["lib"]
        src_mtime = max(os.path.getmtime(os.path.join(SRC, f)) for f in os.listdir(SRC) if f.endswith((".c", ".h")))
        if not os.path.exists(HELPER) or os.path.getmtime(HELPER) < src_mtime:
            _build()
        lib = ctypes.CDLL(HELPER)
        lib.libver.restype = ctypes.c_int
        lib.structsize.restype = ctypes.c_int
        if lib.libver() != LIBVER:
            raise RuntimeError(f"warncount helper built for JPEG_LIB_VERSION {lib.libver()}, need {LIBVER}")
        lib.load.argtypes = [ctypes.c_char_p]
        lib.load.restype = ctypes.c_int
        rc = lib.load(turbo210_so_path().encode())
        if rc != 0:
            raise RuntimeError(f"warncount load() failed with {rc}")
        lib.run.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.c_int, ctypes.POINTER(ctypes.c_long),
                            ctypes.c_char_p, ctypes.c_int]
        lib.run.restype = ctypes.c_int
        _STATE["lib"] = lib
        return lib


def warnings(data: bytes) -> tuple:
    """(err1, nwarn1, err2, nwarn2, msg) of the two pinned decodes."""
    lib = ensure_helper()
    data = bytes(data)
    res = []
    msg0 = ""
    with _LOCK:                       # the helper keeps its function pointers in statics; runs are cheap
        for mode in (0, 1):
            nw = ctypes.c_long(0)
            buf = ctypes.create_string_buffer(200)
            rc = lib.run(data, len(data), mode, ctypes.byref(nw), buf, 200)
            res += [int(rc), int(nw.value)]
            if not msg0:
                msg0 = buf.value.decode("latin-1")
    return (res[0], res[1], res[2], res[3], msg0)
