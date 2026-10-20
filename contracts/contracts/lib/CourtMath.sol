// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// @notice CUTLINE v1 court arithmetic: exact resize weights and taps, Phi (the one libjpeg rgb_ycc_convert
/// constant set), jpeg_fdct_islow, the islow quantiser, the C40 fixed-point DCT and the CELL threshold.
/// Python reference: cutline.phi.phi, cutline.dct.fdct_islow / quant_islow / d80 / cell_fail,
/// cutline.compiler.kernels.taps.
library CourtMath {
    /// Per-output tap buffer of the court: rule 3 (every per-axis scale < 2) bounds an output
    /// by 3 nonzero area taps and 4 nonzero triangle taps, so 4 is the smallest conforming buffer.
    uint256 internal constant KMAX = 4;

    /// C40[8u + x] = round(2^40 a(u) cos((2x + 1) u pi / 16)), int64 big-endian, u-major; SHA-256 3b9edd6b….
    bytes internal constant C40 =
        hex"0000005a827999fd0000005a827999fd0000005a827999fd0000005a827999fd0000005a827999fd0000005a827999fd0000005a827999fd0000005a827999fd"
        hex"0000007d8a5f3fdd0000006a6d98a43b000000471cece6ba00000018f8b83c6affffffe70747c396ffffffb8e3131946ffffff9592675bc5ffffff8275a0c023"
        hex"0000007641af3cca00000030fbc54d5dffffffcf043ab2a3ffffff89be50c336ffffff89be50c336ffffffcf043ab2a300000030fbc54d5d0000007641af3cca"
        hex"0000006a6d98a43bffffffe70747c396ffffff8275a0c023ffffffb8e3131946000000471cece6ba0000007d8a5f3fdd00000018f8b83c6affffff9592675bc5"
        hex"0000005a827999fdffffffa57d866603ffffffa57d8666030000005a827999fd0000005a827999fdffffffa57d866603ffffffa57d8666030000005a827999fd"
        hex"000000471cece6baffffff8275a0c02300000018f8b83c6a0000006a6d98a43bffffff9592675bc5ffffffe70747c3960000007d8a5f3fddffffffb8e3131946"
        hex"00000030fbc54d5dffffff89be50c3360000007641af3ccaffffffcf043ab2a3ffffffcf043ab2a30000007641af3ccaffffff89be50c33600000030fbc54d5d"
        hex"00000018f8b83c6affffffb8e31319460000006a6d98a43bffffff8275a0c0230000007d8a5f3fddffffff9592675bc5000000471cece6baffffffe70747c396";

    // =====================================================================================================
    // Exact resize weights and taps
    // =====================================================================================================

    /// w[j][i]: area max(0, min((i+1) n_out, (j+1) n_in) - max(i n_out, j n_in)); triangle
    /// max(0, 2 max(n_in, n_out) - |(2i+1) n_out - (2j+1) n_in|).
    function weight(uint256 kernel, uint256 i, uint256 j, uint256 nIn, uint256 nOut) internal pure returns (uint256) {
        unchecked {
            if (kernel == 0) {
                uint256 hi1 = (i + 1) * nOut;
                uint256 hi2 = (j + 1) * nIn;
                uint256 lo1 = i * nOut;
                uint256 lo2 = j * nIn;
                uint256 hi_ = hi1 < hi2 ? hi1 : hi2;
                uint256 lo_ = lo1 > lo2 ? lo1 : lo2;
                return hi_ > lo_ ? hi_ - lo_ : 0;
            }
            uint256 s2 = 2 * (nIn > nOut ? nIn : nOut);
            uint256 a = (2 * i + 1) * nOut;
            uint256 c2 = (2 * j + 1) * nIn;
            uint256 dd = a > c2 ? a - c2 : c2 - a;
            return s2 > dd ? s2 - dd : 0;
        }
    }

    /// The nonzero taps of output j are exactly i in [first, last] (both kernels are positive on one interval):
    /// area: [floor(j n_in / n_out), ceil((j+1) n_in / n_out) − 1]; triangle: |(2i+1) n_out − (2j+1) n_in| < s2 with
    /// s2 = 2 max(n_in, n_out), clipped to [0, n_in). Requires n_in, n_out >= 1 and j < n_out.
    function tapRange(uint256 kernel, uint256 j, uint256 nIn, uint256 nOut) internal pure returns (uint256 first, uint256 last) {
        unchecked {
            if (kernel == 0) {
                first = (j * nIn) / nOut;
                last = ((j + 1) * nIn + nOut - 1) / nOut - 1;
            } else {
                uint256 s2 = 2 * (nIn > nOut ? nIn : nOut);
                uint256 c2 = (2 * j + 1) * nIn;
                uint256 n2 = 2 * nOut;
                // lower: (2i+1) n_out > c2 − s2
                first = c2 < s2 + nOut ? 0 : (c2 - s2 - nOut) / n2 + 1;
                // upper: (2i+1) n_out < c2 + s2, i.e. i <= (c2 + s2 − n_out − 1) div 2 n_out
                last = (c2 + s2 - nOut - 1) / n2;
                if (last > nIn - 1) last = nIn - 1;
            }
        }
    }

    /// kernels.taps(n_in, n_out, kind, j): (i0, w[j][i0 .. i0 + |w| − 1]), the nonzero taps.
    function taps(uint256 nIn, uint256 nOut, uint256 kernel, uint256 j) internal pure returns (uint256 first, uint256[] memory w) {
        uint256 last;
        (first, last) = tapRange(kernel, j, nIn, nOut);
        w = new uint256[](last - first + 1);
        for (uint256 i = first; i <= last; ++i) w[i - first] = weight(kernel, i, j, nIn, nOut);
    }

    // =====================================================================================================
    // Phi: libjpeg jccolor.c rgb_ycc_convert, SCALEBITS 16, one constant set
    // =====================================================================================================

    function ycc(uint256 r, uint256 g, uint256 b, uint256 comp) internal pure returns (int256) {
        unchecked {
            if (comp == 0) return int256((19595 * r + 38470 * g + 7471 * b + 32768) >> 16);
            if (comp == 1) return int256((32768 * b + 8421375 - 11059 * r - 21709 * g) >> 16);
            return int256((32768 * r + 8421375 - 27439 * g - 5329 * b) >> 16);
        }
    }

    // =====================================================================================================
    // jpeg_fdct_islow (jfdctint.c; CONST_BITS 13, PASS1_BITS 2, DESCALE = arithmetic shift)
    // =====================================================================================================

    /// One 1-D pass over a[base + stride·i], i = 0..7, written to o[base + stride·k] (k = output index).
    function fdctPass(int256[64] memory a, uint256 base, uint256 stride, bool first, int256[64] memory o) internal pure {
        unchecked {
            int256 d0 = a[base];
            int256 d1 = a[base + stride];
            int256 d2 = a[base + 2 * stride];
            int256 d3 = a[base + 3 * stride];
            int256 d4 = a[base + 4 * stride];
            int256 d5 = a[base + 5 * stride];
            int256 d6 = a[base + 6 * stride];
            int256 d7 = a[base + 7 * stride];
            int256 t10 = d0 + d7 + d3 + d4;
            int256 t13 = d0 + d7 - d3 - d4;
            int256 t11 = d1 + d6 + d2 + d5;
            int256 t12 = d1 + d6 - d2 - d5;
            (int256 sh, int256 rnd) = first ? (int256(11), int256(1024)) : (int256(15), int256(16384));
            if (first) {
                o[base] = (t10 + t11) << 2;
                o[base + 4 * stride] = (t10 - t11) << 2;
            } else {
                o[base] = (t10 + t11 + 2) >> 2;
                o[base + 4 * stride] = (t10 - t11 + 2) >> 2;
            }
            int256 z1 = (t12 + t13) * 4433;
            o[base + 2 * stride] = (z1 + t13 * 6270 + rnd) >> uint256(sh);
            o[base + 6 * stride] = (z1 - t12 * 15137 + rnd) >> uint256(sh);
            int256 t4 = d3 - d4;
            int256 t5 = d2 - d5;
            int256 t6 = d1 - d6;
            int256 t7 = d0 - d7;
            int256 z5 = (t4 + t6 + t5 + t7) * 9633;
            int256 y1 = (t4 + t7) * -7373;
            int256 y2 = (t5 + t6) * -20995;
            int256 y3 = (t4 + t6) * -16069 + z5;
            int256 y4 = (t5 + t7) * -3196 + z5;
            o[base + 7 * stride] = (t4 * 2446 + y1 + y3 + rnd) >> uint256(sh);
            o[base + 5 * stride] = (t5 * 16819 + y2 + y4 + rnd) >> uint256(sh);
            o[base + 3 * stride] = (t6 * 25172 + y2 + y3 + rnd) >> uint256(sh);
            o[base + stride] = (t7 * 12299 + y1 + y4 + rnd) >> uint256(sh);
        }
    }

    /// Output k of one 1-D pass over a[base + stride·i] (the court needs one coefficient only).
    function fdctAt(int256[64] memory a, uint256 base, uint256 stride, bool first, uint256 k) internal pure returns (int256) {
        unchecked {
            int256 d0 = a[base];
            int256 d1 = a[base + stride];
            int256 d2 = a[base + 2 * stride];
            int256 d3 = a[base + 3 * stride];
            int256 d4 = a[base + 4 * stride];
            int256 d5 = a[base + 5 * stride];
            int256 d6 = a[base + 6 * stride];
            int256 d7 = a[base + 7 * stride];
            uint256 sh = first ? 11 : 15;
            int256 rnd = first ? int256(1024) : int256(16384);
            if (k & 1 == 0) {
                int256 t10 = d0 + d7 + d3 + d4;
                int256 t13 = d0 + d7 - d3 - d4;
                int256 t11 = d1 + d6 + d2 + d5;
                int256 t12 = d1 + d6 - d2 - d5;
                if (k == 0) return first ? (t10 + t11) << 2 : (t10 + t11 + 2) >> 2;
                if (k == 4) return first ? (t10 - t11) << 2 : (t10 - t11 + 2) >> 2;
                int256 z1 = (t12 + t13) * 4433;
                if (k == 2) return (z1 + t13 * 6270 + rnd) >> sh;
                return (z1 - t12 * 15137 + rnd) >> sh;
            }
            int256 t4 = d3 - d4;
            int256 t5 = d2 - d5;
            int256 t6 = d1 - d6;
            int256 t7 = d0 - d7;
            int256 z5 = (t4 + t6 + t5 + t7) * 9633;
            if (k == 1) return (t7 * 12299 + (t4 + t7) * -7373 + (t5 + t7) * -3196 + z5 + rnd) >> sh;
            if (k == 3) return (t6 * 25172 + (t5 + t6) * -20995 + (t4 + t6) * -16069 + z5 + rnd) >> sh;
            if (k == 5) return (t5 * 16819 + (t5 + t6) * -20995 + (t5 + t7) * -3196 + z5 + rnd) >> sh;
            return (t4 * 2446 + (t4 + t7) * -7373 + (t4 + t6) * -16069 + z5 + rnd) >> sh;
        }
    }

    /// jpeg_fdct_islow on a level-shifted block P[8y + x] -> t[8v + u] (scaled by 8); dct.fdct_islow.
    function fdctIslow(int256[64] memory P) internal pure returns (int256[64] memory t) {
        int256[64] memory ws;
        for (uint256 y; y < 8; ++y) fdctPass(P, 8 * y, 1, true, ws);   // rows: ws[8y + u]
        for (uint256 u; u < 8; ++u) fdctPass(ws, u, 8, false, t);       // columns: t[8v + u]
    }

    /// t[8v + u] of jpeg_fdct_islow(P) for one coefficient k = 8v + u.
    function fdctIslowAt(int256[64] memory P, uint256 k) internal pure returns (int256) {
        int256[64] memory col;
        uint256 u = k & 7;
        for (uint256 y; y < 8; ++y) col[y] = fdctAt(P, 8 * y, 1, true, u);
        return fdctAt(col, 0, 1, false, k >> 3);
    }

    /// The libjpeg islow quantiser sign(t)·floor((|t| + 4q) / (8q)) (dct.quant_islow); q >= 1.
    function quantIslow(int256 t, uint256 q) internal pure returns (int256) {
        unchecked {
            int256 qv = int256(q) * 8;
            int256 a = t < 0 ? -t : t;
            int256 e = (a + qv / 2) / qv;
            return t < 0 ? -e : e;
        }
    }

    /// EXACT: fails iff c != quant_islow(t, q).
    function exactFails(int256 c, int256 t, uint256 q) internal pure returns (bool) {
        return c != quantIslow(t, q);
    }

    // =====================================================================================================
    // CELL: C40 fixed-point DCT and the threshold (dct.d80, dct.cell_fail)
    // =====================================================================================================

    function c40(bytes memory t, uint256 u, uint256 x) internal pure returns (int256 v) {
        assembly ("memory-safe") { v := signextend(7, shr(192, mload(add(add(t, 32), shl(3, add(shl(3, u), x)))))) }
    }

    /// D80_k = Σ_y C40[v][y] Σ_x C40[u][x] P[y][x] for one k = 8v + u.
    function d80At(int256[64] memory P, uint256 k) internal pure returns (int256 val) {
        bytes memory t = C40;
        uint256 u = k & 7;
        uint256 v = k >> 3;
        unchecked {
            for (uint256 y; y < 8; ++y) {
                int256 rowSum;
                for (uint256 x; x < 8; ++x) rowSum += c40(t, u, x) * P[8 * y + x];
                val += c40(t, v, y) * rowSum;
            }
        }
    }

    /// Every D80_k of one block (k = 8v + u).
    function d80(int256[64] memory P) internal pure returns (int256[64] memory D) {
        bytes memory t = C40;
        int256[64] memory row; // row[8y + u] = Σ_x C40[u][x] P[y][x]
        unchecked {
            for (uint256 y; y < 8; ++y) {
                for (uint256 u; u < 8; ++u) {
                    int256 s;
                    for (uint256 x; x < 8; ++x) s += c40(t, u, x) * P[8 * y + x];
                    row[8 * y + u] = s;
                }
            }
            for (uint256 v; v < 8; ++v) {
                for (uint256 u; u < 8; ++u) {
                    int256 s;
                    for (uint256 y; y < 8; ++y) s += c40(t, v, y) * row[8 * y + u];
                    D[8 * v + u] = s;
                }
            }
        }
    }

    /// CELL: fails iff |c·q·2^80 - D80| > (q·2^19 + eta20 + q)·2^60.
    function cellFails(int256 c, int256 D, uint256 q, uint256 eta20) internal pure returns (bool) {
        int256 diff = c * int256(q) * (int256(1) << 80) - D;
        if (diff < 0) diff = -diff;
        return diff > (int256(q) * (int256(1) << 19) + int256(eta20) + int256(q)) * (int256(1) << 60);
    }

    // =====================================================================================================
    // jpeg_idct_islow (jidctint.c) with libjpeg's range-limit table (dct.idct_islow); not used by the court;
    // kept for decoder parity with the Python reference
    // =====================================================================================================

    function _idctPass(int256[64] memory a, uint256 base, uint256 stride, uint256 sh, int256[64] memory o, uint256 obase,
        uint256 ostride) private pure
    {
        unchecked {
            int256 rnd = int256(1) << (sh - 1);
            int256 x2 = a[base + 2 * stride];
            int256 x6 = a[base + 6 * stride];
            int256 z1 = (x2 + x6) * 4433;
            int256 e2 = z1 + x6 * -15137;
            int256 e3 = z1 + x2 * 6270;
            int256 e0 = (a[base] + a[base + 4 * stride]) << 13;
            int256 e1 = (a[base] - a[base + 4 * stride]) << 13;
            int256 t10 = e0 + e3;
            int256 t13 = e0 - e3;
            int256 t11 = e1 + e2;
            int256 t12 = e1 - e2;
            int256 o0 = a[base + 7 * stride];
            int256 o1 = a[base + 5 * stride];
            int256 o2 = a[base + 3 * stride];
            int256 o3 = a[base + stride];
            int256 z5 = (o0 + o2 + o1 + o3) * 9633;
            int256 y1 = (o0 + o3) * -7373;
            int256 y2 = (o1 + o2) * -20995;
            int256 y3 = (o0 + o2) * -16069 + z5;
            int256 y4 = (o1 + o3) * -3196 + z5;
            o0 = o0 * 2446 + y1 + y3;
            o1 = o1 * 16819 + y2 + y4;
            o2 = o2 * 25172 + y2 + y3;
            o3 = o3 * 12299 + y1 + y4;
            o[obase] = (t10 + o3 + rnd) >> sh;
            o[obase + ostride] = (t11 + o2 + rnd) >> sh;
            o[obase + 2 * ostride] = (t12 + o1 + rnd) >> sh;
            o[obase + 3 * ostride] = (t13 + o0 + rnd) >> sh;
            o[obase + 4 * ostride] = (t13 - o0 + rnd) >> sh;
            o[obase + 5 * ostride] = (t12 - o1 + rnd) >> sh;
            o[obase + 6 * ostride] = (t11 - o2 + rnd) >> sh;
            o[obase + 7 * ostride] = (t10 - o3 + rnd) >> sh;
        }
    }

    /// coef[8v + u] (quantised), q[8v + u] -> samples[8y + x], exactly jpeg_idct_islow incl. range_limit wrap-around.
    function idctIslow(int256[64] memory coef, uint256[64] memory q) internal pure returns (uint256[64] memory s) {
        int256[64] memory d;
        int256[64] memory ws;   // ws[8u + y]
        int256[64] memory px;   // px[8y + x]
        unchecked {
            for (uint256 i; i < 64; ++i) d[i] = coef[i] * int256(q[i]);
            for (uint256 u; u < 8; ++u) _idctPass(d, u, 8, 11, ws, 8 * u, 1);      // columns
            for (uint256 y; y < 8; ++y) _idctPass(ws, y, 8, 18, px, 8 * y, 1);     // rows
            for (uint256 i; i < 64; ++i) {
                uint256 v = uint256(px[i]) & 1023;
                s[i] = v < 128 ? v + 128 : (v < 512 ? 255 : (v < 896 ? 0 : v - 896));
            }
        }
    }
}
