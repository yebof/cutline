"""cutline.verify.fast - the native E0 fast path and its cross-check.

Off-chain fast paths are allowed but never decide alone: a caller uses one only after ``crosscheck_e0`` has shown on
its corpus that it equals the integer path, and the integer path stays the reference.

- ``e0_native(X, listing_bytes, g, cert_mcus=None, coefs=None) -> bool | None``: re-encode the canvas X with
  ``jpeg.encode.encode_e0`` (libjpeg-turbo 2.1.0 islow, the file's own luma table and chroma table, the file's
  sampling; libjpeg's ``rgb_ycc`` and h2v2 downsampling are exactly Phi and its h2v2) and compare the stored
  coefficients of every real block of the MCUs in ``cert_mcus`` (None = all). True = every coefficient equal (E0
  PASS on that set). None when the fast path does not apply: Cb and Cr use different tables (``encode_e0`` shares
  one chroma table), the re-encode does not reproduce the file's tables, or X is not the file's size. Progressive
  mode and restart intervals do not matter (only coefficients are compared).
- ``native_fail_mcus(X, listing_bytes, g, coefs=None) -> set[int] | None``: the MCUs where the re-encode differs.
- ``crosscheck_e0(X, listing_bytes, g, cert_mcus=None) -> dict``: runs both paths and reports
  ``{"agree": bool, "native": bool | None, "integer_pass": bool, "native_fail_mcus": [...],
  "integer_fail_mcus": [...]}``; ``agree`` requires the same verdict and the same set of failing certified MCUs.
"""
from __future__ import annotations

import numpy as np


def _reencode_coefs(X, g):
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.encode import encode_e0
    from cutline.jpeg.grammar import container_ok_v2
    qy, qcb, qcr = g.q3()
    if qcb != qcr:
        return None
    X = np.asarray(X)
    if X.shape[:2] != (g.H, g.W):
        return None
    b = encode_e0(X, (qy, qcb), g.sampling)
    ge = container_ok_v2(b, False)                     # rendition mode: a hop child need not be 16-aligned
    if ge.q3() != g.q3() or (ge.W, ge.H, ge.sampling) != (g.W, g.H, g.sampling):
        return None
    return read_coefs(b, ge)


def native_fail_mcus(X, listing_bytes, g, coefs=None):
    from cutline.jpeg.coefs import read_coefs
    from cutline.verify.blocktest import block_mcu_map
    ce = _reencode_coefs(X, g)
    if ce is None:
        return None
    co = coefs if coefs is not None else read_coefs(listing_bytes, g)
    bad = set()
    for comp in range(3):
        d = (np.asarray(co[comp]) != np.asarray(ce[comp])).any(axis=(2, 3))
        hb, wb = d.shape
        mcu_m, _ = block_mcu_map(g, comp, hb, wb)
        bad.update(int(v) for v in np.unique(mcu_m[d]))
    return bad


def e0_native(X, listing_bytes, g, cert_mcus=None, coefs=None):
    bad = native_fail_mcus(X, listing_bytes, g, coefs)
    if bad is None:
        return None
    if cert_mcus is not None:
        bad &= set(int(v) for v in cert_mcus)
    return not bad


def crosscheck_e0(X, listing_bytes, g, cert_mcus=None, coefs=None) -> dict:
    from cutline.jpeg.coefs import read_coefs
    from cutline.verify.blocktest import e0_file_fails
    co = coefs if coefs is not None else read_coefs(listing_bytes, g)
    rows = e0_file_fails(X, co, g, cert_mcus)
    ifail = sorted({r[0] for r in rows})
    bad = native_fail_mcus(X, listing_bytes, g, co)
    if bad is None:
        return {"agree": True, "native": None, "integer_pass": not rows, "native_fail_mcus": None,
                "integer_fail_mcus": ifail}
    if cert_mcus is not None:
        bad &= set(int(v) for v in cert_mcus)
    nfail = sorted(bad)
    return {"agree": nfail == ifail, "native": not nfail, "integer_pass": not rows, "native_fail_mcus": nfail,
            "integer_fail_mcus": ifail}
