"""cutline.verify.ep - EP (pinned-encoder) registry entries and the class-7 block test.

An EP entry pins one encoder build and every compression parameter: ``buildSha256`` = SHA-256 of the executed shared
library, ``paramsSha256`` = H(canonical JSON of the parameters, keyed by the library's API names). The class-7 test
re-encodes the reference canvas with the entry and compares every coefficient.

Supported entries: Python-loadable builds of jpeglib 1.0.2's ``cjpeglib_mozjpeg403.abi3.so`` (mozjpeg 4.0.3; its
SHA-256 is pinned in ``ENV_PINS.json`` under ``codecs.jpeglib.builds.mozjpeg403.so_sha256``) with the library
defaults (``jpeg_set_defaults`` under mozjpeg's JCP_MAX_COMPRESSION profile: trellis quantisation, optimised
progressive scans, quant table set 3) at IJG quality 85. Every parameter is concrete, so the sampling is part of the
entry: entry 1 = 4:2:0, entry 2 = 4:4:4. The pin file is ``<repo root>/env/ENV_PINS.json`` unless the environment
variable ``CUTLINE_ENV_PINS`` names another path.

- ``EPSpec(entry, version, sampling, quality)``: ``.params_json() -> bytes`` (canonical JSON), ``.params_sha256()``
  (= H(params_json); H = SHA-256), ``.build_sha256()`` (SHA-256 of the loaded ``.so``, checked against the pin),
  ``.registry_entry() -> registries.EPEntry``.
- ``EP_ENTRIES`` {1: 4:2:0, 2: 4:4:4}; ``resolve(ep) -> EPSpec`` (an id or an EPSpec; ``E.HOP.CLASS`` for an unknown
  id); ``build_sha256(entry)``, ``params_json(entry)``.
- ``ep_encode(entry, X, sampling=None) -> bytes``: the re-encode of X (uint8 [H, W, 3]); ``sampling``, when given,
  must equal the entry's (``E.HOP.CLASS`` otherwise). Temporary files in ``$TMPDIR``.
- ``ep_test_file(ep, X, coefs, g, cert_mcus=None) -> list[TE]``: rows ``(mcu, comp, blk, k, c, c_ep)`` of every
  coefficient that differs from the re-encode, restricted to ``cert_mcus`` when given; a re-encode whose size,
  sampling or tables differ from the file's cannot match any coefficient and raises ``E.HOP.CODEC``.
- ``check_entry(spec, ledger_entry) -> bool``: the ledger's ``EPEntry`` names exactly this build and parameter set.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from cutline.crypto import H
from cutline.enc import jcs
from cutline.errors import HopError
from cutline.registries import EPEntry

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))       # the repository root
PINS_PATH = os.environ.get("CUTLINE_ENV_PINS") or os.path.join(ROOT, "env", "ENV_PINS.json")
_SO_SHA: dict = {}

MOZ_DEFAULTS = {
    "api": "jpeglib 1.0.2: jpeglib.from_spatial(X); samp_factor; write_spatial(path, qt=quality)",
    "color_space": "JCS_RGB in, JCS_YCbCr out (library default)",
    "dct_method": "JDCT_ISLOW (library default)",
    "base_quant_tbl_idx": "library default (mozjpeg quant_tbl_master_idx 3)",
    "trellis_quant": "library default (on)",
    "trellis_quant_dc": "library default (on)",
    "trellis_q_opt": "library default (off)",
    "lambda_log_scale1": "14.75",
    "lambda_log_scale2": "16.5",
    "optimize_coding": "library default (on)",
    "optimize_scans": "library default (on)",
    "progressive": "library default (on, jpeg_simple_progression)",
    "scan_script": "library default (mozjpeg optimize_scans search)",
    "smoothing_factor": "0",
    "overshoot_deringing": "library default (on)",
    "flags": [],
}


@dataclass(frozen=True)
class EPSpec:
    entry: int
    version: str                     # jpeglib build name
    sampling: int                    # 0 = 4:4:4, 1 = 4:2:0
    quality: int
    extra: dict = field(default_factory=lambda: dict(MOZ_DEFAULTS))

    def params(self) -> dict:
        return {**self.extra, "library": f"jpeglib {self.version}", "quality": str(self.quality),
                "sampling": "2x2,1x1,1x1" if self.sampling else "1x1,1x1,1x1"}

    def params_json(self) -> bytes:
        return jcs(self.params())

    def params_sha256(self) -> bytes:
        return H(self.params_json())

    def so_path(self) -> str:
        import jpeglib
        c = glob.glob(os.path.join(os.path.dirname(jpeglib.__file__), "cjpeglib", f"cjpeglib_{self.version}.*so"))
        if len(c) != 1:
            raise RuntimeError(f"expected one cjpeglib_{self.version} build, found {c}")
        return c[0]

    def build_sha256(self) -> bytes:
        if self.version not in _SO_SHA:
            got = hashlib.sha256(open(self.so_path(), "rb").read()).hexdigest()
            pins = json.load(open(PINS_PATH))
            want = pins["codecs"]["jpeglib"]["builds"][self.version]["so_sha256"]
            if got != want:
                raise RuntimeError(f"{self.version} SHA-256 {got} != pinned {want}")
            _SO_SHA[self.version] = bytes.fromhex(got)
        return _SO_SHA[self.version]

    def registry_entry(self) -> EPEntry:
        return EPEntry(self.build_sha256(), self.params_sha256())


EP_ENTRIES = {1: EPSpec(1, "mozjpeg403", 1, 85), 2: EPSpec(2, "mozjpeg403", 0, 85)}


def resolve(ep) -> EPSpec:
    if isinstance(ep, EPSpec):
        return ep
    s = EP_ENTRIES.get(int(ep)) if ep is not None else None
    if s is None:
        raise HopError("E.HOP.CLASS", f"EP entry {ep} is not defined")
    return s


def build_sha256(entry) -> bytes:
    return resolve(entry).build_sha256()


def params_json(entry) -> bytes:
    return resolve(entry).params_json()


def check_entry(spec, ledger_entry) -> bool:
    s = resolve(spec)
    return ledger_entry is not None and (bytes(ledger_entry.build_sha256), bytes(ledger_entry.params_sha256)) == \
        (s.build_sha256(), s.params_sha256())


def ep_encode(entry, X, sampling: int | None = None) -> bytes:
    import jpeglib
    from cutline.jpeg._io import temp_path
    s = resolve(entry)
    if sampling is not None and int(sampling) != s.sampling:
        raise HopError("E.HOP.CLASS", f"EP entry {s.entry} encodes sampling {s.sampling}, not {sampling}")
    X = np.ascontiguousarray(np.asarray(X))
    if X.dtype != np.uint8 or X.ndim != 3 or X.shape[2] != 3:
        raise ValueError("X must be uint8 [H, W, 3]")
    s.build_sha256()
    with temp_path() as p:
        with jpeglib.version(s.version):
            im = jpeglib.from_spatial(X)
            im.samp_factor = ((2, 2), (1, 1), (1, 1)) if s.sampling else ((1, 1), (1, 1), (1, 1))
            im.write_spatial(p, qt=s.quality)
        return open(p, "rb").read()


def ep_iter_rows(ep, X, coefs, g, cert_mcus=None, band_px=None):
    """``ep_test_file``'s rows, generated band by band (MCU rows) in ascending (mcu, comp, blk, k) order:
    the re-encode is of the whole file (an encoder need not be MCU-row local), the comparison is banded."""
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    from cutline.verify.blocktest import BAND_PX, block_mcu_map
    b = ep_encode(ep, X, g.sampling)
    ge = container_ok_v2(b, False)                     # rendition mode: a hop child need not be 16-aligned
    if (ge.W, ge.H, ge.sampling, ge.q3()) != (g.W, g.H, g.sampling, g.q3()):
        raise HopError("E.HOP.CODEC", "EP re-encode: size, sampling or tables differ from the file")
    ce = read_coefs(b, ge)
    del b
    cert = None if cert_mcus is None else np.unique(np.fromiter((int(v) for v in cert_mcus), np.int64))
    p = 16 if g.sampling else 8
    nmy = -(-int(g.H) // p)
    per = max(1, (BAND_PX if band_px is None else int(band_px)) // (p * max(1, int(g.W))))
    maps = [block_mcu_map(g, comp, *np.asarray(coefs[comp]).shape[:2]) for comp in range(3)]
    for R0 in range(0, nmy, per):
        R1 = min(nmy, R0 + per)
        rows = []
        for comp in range(3):
            f = 2 if (g.sampling and comp == 0) else 1
            sl = slice(f * R0, f * R1)
            a, e = np.asarray(coefs[comp][sl]).astype(np.int64), np.asarray(ce[comp][sl]).astype(np.int64)
            mcu_m, blk_m = maps[comp][0][sl], maps[comp][1][sl]
            for r, c, v, u in np.argwhere(a != e):
                m = int(mcu_m[r, c])
                if cert is not None and not _member(cert, m):
                    continue
                rows.append((m, comp, int(blk_m[r, c]), int(8 * v + u), int(a[r, c, v, u]), int(e[r, c, v, u])))
        rows.sort(key=lambda x: x[:4])
        yield from rows


def _member(sorted_arr, v: int) -> bool:
    i = int(np.searchsorted(sorted_arr, v))
    return i < len(sorted_arr) and int(sorted_arr[i]) == v


def ep_test_file(ep, X, coefs, g, cert_mcus=None) -> list:
    return list(ep_iter_rows(ep, X, coefs, g, cert_mcus))
