"""cutline.jpeg.grammar - ContainerOK v2.1.

Marker-level whitelist grammar parser with an exact error precedence, plus the libjpeg zero-warning condition
(counter helper, ``cutline.jpeg.libjpeg``); upload vs rendition mode.

- ``parse_grammar(data, upload) -> Grammar``: one pass over the bytes in file order (check steps 1-5); raises
  ``ContainerError("E.CONTAINER.<NAME>")`` at the first failing check.
- ``container_ok_v2(data, upload) -> Grammar``: the grammar, then both pinned libjpeg runs (step 6:
  ``LIBJPEG_ERROR`` / ``LIBJPEG_WARNING``, run 1 before run 2).
- ``marker_sequence(data) -> list[str]``: the segment names of a file that passes the grammar (``APP0 DQT SOF0 …``);
  informational.
- ``PROBE_REASON_TO_ERROR`` and ``probe_reason_to_error(reason)``: human-readable rejection reasons -> ``E.*`` names.

Length errors, the zero-entry check inside the DQT row, NO_EOI / TRUNCATED at the end of the data, and the
coefficient-major progressive check follow the precedence of the check steps.

G6 includes the size bound (``SOF_TOO_LARGE``: X, Y <= ``SOF_MAX_SIDE`` = 8192 and X * Y <= ``SOF_MAX_PIXELS`` = 2^25),
checked right after ``SOF_PARAMS`` and before any libjpeg call.
"""
from __future__ import annotations

import re

from cutline.consts import ZZ
from cutline.errors import ContainerError
from cutline.types import Grammar

E = "E.CONTAINER."

# G6 size bound: checked in the grammar, before libjpeg allocates the whole-image coefficient arrays. Admits every
# honest file (12 MP masters, i.e. 4032 x 3024 at l* = 0) with 2.75x headroom in pixels.
SOF_MAX_SIDE = 8192
SOF_MAX_PIXELS = 1 << 25                     # 33,554,432 = 8192 x 4096

MNAME = {0xE0: "APP0", 0xDB: "DQT", 0xC4: "DHT", 0xC0: "SOF0", 0xC2: "SOF2", 0xDA: "SOS", 0xDD: "DRI", 0xFE: "COM",
         0xD9: "EOI"}


def _err(name: str, detail: str = ""):
    return ContainerError(E + name, detail)


def _u16(b, i):
    return (b[i] << 8) | b[i + 1]


def _ecs_end(b: bytes, start: int) -> int | None:
    """Index of the first ``FF xx`` (xx not in {00, D0..D7}) at or after ``start``; None if the data ends first."""
    n = len(b)
    k = start
    while True:
        k = b.find(b"\xff", k)
        if k < 0 or k + 1 >= n:
            return None
        x = b[k + 1]
        if x != 0x00 and not 0xD0 <= x <= 0xD7:
            return k
        k += 2                              # FF 00 (stuffing) and FF D0..D7 (restart markers) stay in the ECS


def _segments(b: bytes):
    """Marker scanner (the checks of step 2). Yields (M, payload_start, payload_end, ecs_end_or_None);
    for EOI yields (0xD9, pos + 2, pos + 2, None) and stops. Raises at the first scanner failure."""
    n = len(b)
    if n < 2 or b[0] != 0xFF or b[1] != 0xD8:
        raise _err("NO_SOI")
    i = 2
    while True:
        if i >= n:
            raise _err("NO_EOI", "data ends at a segment boundary")
        if b[i] != 0xFF:
            raise _err("GARBAGE", f"at {i}")
        if i + 1 >= n:
            raise _err("TRUNCATED", "FF is the last byte")
        m = b[i + 1]
        if m == 0xFF:
            raise _err("FILL_BYTES", f"at {i}")
        if m == 0xD9:
            yield (m, i + 2, i + 2, None)
            return
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            raise _err("STRAY_MARKER", f"{m:02x} at {i}")
        if i + 4 > n:
            raise _err("TRUNCATED", "length field")
        L = _u16(b, i + 2)
        if L < 2 or i + 2 + L > n:
            raise _err("TRUNCATED", f"segment {m:02x} L={L}")
        s, e = i + 4, i + 2 + L
        if m == 0xDA:
            k = _ecs_end(b, e)
            yield (m, s, e, k if k is not None else n)
            if k is None:
                raise _err("NO_EOI", "data ends inside an entropy-coded segment")
            i = k
        else:
            yield (m, s, e, None)
            i = e


def parse_grammar(data: bytes, upload: bool) -> Grammar:
    """ContainerOK v2.1 grammar; raises ContainerError at the first failing check, in precedence order."""
    b = bytes(data)
    tables: dict = {}
    sof = None                      # dict(W, H, sampling, tq, progressive)
    dri_count = 0
    restart_interval = 0
    seen_sos = False
    scans = 0
    first = True
    coded: dict = {}                # (c, k) -> current Al (progressive)
    base_seen: set = set()
    eoi_end = None
    for m, s, e, x in _segments(b):
        if m == 0xD9:
            eoi_end = e
            break
        L = e - s + 2
        if m == 0xE0:                                                       # G1
            if not first:
                raise _err("APP0_NOT_FIRST")
            p = b[s:e]
            if L != 16 or p[:5] != b"JFIF\x00":
                raise _err("APP0_NOT_JFIF")
            if p[5] != 1 or p[6] not in (1, 2) or p[7] > 2:
                raise _err("JFIF_VERSION_UNITS")
            xd, yd = _u16(p, 8), _u16(p, 10)
            if xd != yd or xd == 0 or p[12] or p[13]:
                raise _err("JFIF_DENSITY_THUMBNAIL")
        elif m == 0xFE:                                                     # G2
            if sof is not None:
                raise _err("COM_AFTER_SOF")
        elif m == 0xDB:                                                     # G3
            if sof is not None:
                raise _err("DQT_AFTER_SOF")
            p = s
            while p < e:
                pq, tq = b[p] >> 4, b[p] & 15
                if pq != 0:
                    raise _err("DQT_16BIT")
                if tq > 3 or tq in tables:
                    raise _err("DQT_REDEFINED", f"table {tq}")
                if p + 65 > e:
                    raise _err("DQT_LENGTH")
                zz = b[p + 1:p + 65]
                if min(zz) == 0:
                    raise _err("DQT_ZERO_ENTRY", f"table {tq}")
                nat = bytearray(64)
                for zi, ni in enumerate(ZZ):
                    nat[ni] = zz[zi]
                tables[tq] = bytes(nat)
                p += 65
        elif m == 0xC4:                                                     # G4
            pass
        elif m == 0xDD:                                                     # G5
            if seen_sos or dri_count:
                raise _err("DRI_POSITION")
            if L != 4:
                raise _err("DRI_LENGTH")
            dri_count += 1
            restart_interval = _u16(b, s)
        elif m in (0xC0, 0xC2):                                             # G6
            if sof is not None:
                raise _err("SECOND_SOF")
            if L != 17:
                raise _err("SOF_LENGTH")
            P, Y, X, nf = b[s], _u16(b, s + 1), _u16(b, s + 3), b[s + 5]
            if P != 8 or nf != 3 or X == 0 or Y == 0:
                raise _err("SOF_PARAMS")
            if X > SOF_MAX_SIDE or Y > SOF_MAX_SIDE or X * Y > SOF_MAX_PIXELS:      # size bound
                raise _err("SOF_TOO_LARGE", f"{X} x {Y}")
            comps = []
            for c in range(3):
                cid, hv, tq = b[s + 6 + 3 * c], b[s + 7 + 3 * c], b[s + 8 + 3 * c]
                if cid != c + 1 or tq not in tables:
                    raise _err("SOF_COMPONENTS")
                comps.append((hv >> 4, hv & 15, tq))
            samp = [(h, v) for h, v, _ in comps]
            if samp == [(1, 1)] * 3:
                sampling = 0
            elif samp == [(2, 2), (1, 1), (1, 1)]:
                sampling = 1
            else:
                raise _err("SAMPLING")
            if upload and (X % 16 or Y % 16):
                raise _err("UPLOAD_NOT_MULT16")
            sof = {"W": X, "H": Y, "sampling": sampling, "tq": tuple(c[2] for c in comps), "progressive": m == 0xC2}
        elif m == 0xDA:                                                     # G7-G10
            if sof is None:
                raise _err("SOS_BEFORE_SOF")
            ns = b[s] if L >= 3 else 0
            if not 1 <= ns <= 3 or L != 6 + 2 * ns:
                raise _err("SOS_LENGTH")
            seen_sos = True
            scans += 1
            cidx = [b[s + 1 + 2 * j] - 1 for j in range(ns)]
            Ss, Se, ahal = b[s + 1 + 2 * ns], b[s + 2 + 2 * ns], b[s + 3 + 2 * ns]
            Ah, Al = ahal >> 4, ahal & 15
            if any(c not in (0, 1, 2) for c in cidx) or len(set(cidx)) != ns:
                raise _err("SCAN_COMPONENTS")
            if not sof["progressive"]:                                      # G8
                if (Ss, Se, Ah, Al) != (0, 63, 0, 0) or base_seen & set(cidx):
                    raise _err("BASELINE_SCRIPT")
                base_seen |= set(cidx)
            else:                                                           # G9
                if Ss == 0:
                    if Se != 0:
                        raise _err("PROG_DC_WITH_AC")
                    ks = (0,)
                else:
                    if ns != 1 or Se < Ss or Se > 63:
                        raise _err("PROG_AC_SHAPE")
                    if (cidx[0], 0) not in coded:
                        raise _err("AC_BEFORE_DC")
                    ks = range(Ss, Se + 1)
                for c in cidx:                                              # coefficient-major
                    for k in ks:
                        cur = coded.get((c, k))
                        if cur is None:
                            if Ah != 0:
                                raise _err("REFINE_BEFORE_FIRST", f"(c={c}, k={k})")
                        elif Ah != cur or Al != Ah - 1:
                            raise _err("BAD_REFINEMENT", f"(c={c}, k={k})")
                        coded[(c, k)] = Al
            ecs = b[e:x]                                                    # G10
            rst = []
            j = ecs.find(b"\xff")
            while 0 <= j < len(ecs) - 1:
                y = ecs[j + 1]
                if 0xD0 <= y <= 0xD7:
                    rst.append(y)
                    j = ecs.find(b"\xff", j + 2)
                elif y == 0xFF:
                    j = j + 1
                else:
                    j = ecs.find(b"\xff", j + 2)
            if any(r != 0xD0 + (i % 8) for i, r in enumerate(rst)):
                raise _err("RST_SEQUENCE")
            if rst and not dri_count:
                raise _err("RST_WITHOUT_DRI")
        else:                                                               # G11
            raise _err("MARKER_NOT_IN_GRAMMAR", f"marker {m:02x}")
        first = False
    # step 5 (a missing EOI raised inside the scanner)
    if eoi_end != len(b):
        raise _err("DATA_AFTER_EOI")
    if scans == 0:
        raise _err("NO_SCAN")
    if sof["progressive"]:
        if any(coded.get((c, k)) != 0 for c in range(3) for k in range(64)):
            raise _err("PROG_INCOMPLETE")
    elif base_seen != {0, 1, 2}:
        raise _err("BASELINE_INCOMPLETE")
    return Grammar(W=sof["W"], H=sof["H"], sampling=sof["sampling"], tables=dict(sorted(tables.items())),
                   tq=sof["tq"], progressive=sof["progressive"], restart_interval=restart_interval, scans=scans)


def libjpeg_condition(data: bytes) -> None:
    """Step 6: both pinned decodes, zero error exits and zero warnings."""
    from cutline.jpeg.libjpeg import warnings
    e1, w1, e2, w2, msg = warnings(data)
    if e1:
        raise _err("LIBJPEG_ERROR", f"run 1: {msg}")
    if w1:
        raise _err("LIBJPEG_WARNING", f"run 1: {w1} warning(s): {msg}")
    if e2:
        raise _err("LIBJPEG_ERROR", f"run 2: {msg}")
    if w2:
        raise _err("LIBJPEG_WARNING", f"run 2: {w2} warning(s): {msg}")


def container_ok_v2(data: bytes, upload: bool) -> Grammar:
    """ContainerOK v2.1: the grammar, then the libjpeg zero-warning condition."""
    g = parse_grammar(data, upload)
    libjpeg_condition(data)
    return g


def marker_sequence(data: bytes) -> list:
    """Segment names in file order (the file must pass the scanner); unknown markers as ``FFxx``."""
    out = []
    for m, _s, _e, _x in _segments(bytes(data)):
        out.append(MNAME.get(m, f"FF{m:02X}"))
    return out


# ------------------------------------------------------------------------------------------------ reason strings
PROBE_REASON_TO_ERROR = {
    "no SOI": E + "NO_SOI",
    "no EOI": E + "NO_EOI",
    "garbage": E + "GARBAGE",
    "fill bytes": E + "FILL_BYTES",
    "stray marker": E + "STRAY_MARKER",
    "truncated segment": E + "TRUNCATED",
    "APP0 not first": E + "APP0_NOT_FIRST",
    "APP0 not a plain JFIF segment": E + "APP0_NOT_JFIF",
    "JFIF version/units": E + "JFIF_VERSION_UNITS",
    "JFIF density/thumbnail": E + "JFIF_DENSITY_THUMBNAIL",
    "COM after SOF": E + "COM_AFTER_SOF",
    "DQT after SOF": E + "DQT_AFTER_SOF",
    "16-bit DQT": E + "DQT_16BIT",
    "DQT id redefined": E + "DQT_REDEFINED",
    "DQT length": E + "DQT_LENGTH",
    "DQT entry 0": E + "DQT_ZERO_ENTRY",
    "DRI after first SOS or repeated": E + "DRI_POSITION",
    "second SOF": E + "SECOND_SOF",
    "SOF precision/components/dims": E + "SOF_PARAMS",
    "SOF too large": E + "SOF_TOO_LARGE",
    "SOF component ids/tables": E + "SOF_COMPONENTS",
    "sampling": E + "SAMPLING",
    "upload canvas not a multiple of 16": E + "UPLOAD_NOT_MULT16",
    "SOS before SOF": E + "SOS_BEFORE_SOF",
    "scan components": E + "SCAN_COMPONENTS",
    "baseline scan script": E + "BASELINE_SCRIPT",
    "progressive DC scan with AC": E + "PROG_DC_WITH_AC",
    "progressive AC scan shape": E + "PROG_AC_SHAPE",
    "AC before DC": E + "AC_BEFORE_DC",
    "refinement before first scan": E + "REFINE_BEFORE_FIRST",
    "coefficient first-coded twice or bad refinement": E + "BAD_REFINEMENT",
    "RST sequence": E + "RST_SEQUENCE",
    "RST without DRI": E + "RST_WITHOUT_DRI",
    "marker not in the grammar": E + "MARKER_NOT_IN_GRAMMAR",
    "data after EOI": E + "DATA_AFTER_EOI",
    "no scan": E + "NO_SCAN",
    "incomplete progressive scan script": E + "PROG_INCOMPLETE",
    "baseline components missing": E + "BASELINE_INCOMPLETE",
    "libjpeg warning": E + "LIBJPEG_WARNING",
    "libjpeg error": E + "LIBJPEG_ERROR",
}


def probe_reason_to_error(reason: str) -> str | None:
    """Map a human-readable rejection reason ("ok" -> None) to its ``E.CONTAINER.*`` error name."""
    if reason == "ok":
        return None
    if reason.startswith("libjpeg warning: exception"):
        return E + "LIBJPEG_ERROR"            # jpeglib exceptions (error exits) reported inside a warning text
    if reason.startswith("libjpeg warning"):
        return E + "LIBJPEG_WARNING"
    if reason.startswith("garbage at"):
        return E + "GARBAGE"
    if reason.startswith("stray marker"):
        return E + "STRAY_MARKER"
    if re.fullmatch(r"marker [0-9a-f]{2} not in the grammar", reason):
        return E + "MARKER_NOT_IN_GRAMMAR"
    if re.fullmatch(r"DQT entry 0 \(table \d+\)", reason):
        return E + "DQT_ZERO_ENTRY"
    return PROBE_REASON_TO_ERROR.get(reason)
