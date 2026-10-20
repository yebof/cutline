// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {ProofIn, Cell, ClaimCore} from "./Types.sol";
import {CutlineLib} from "./CutlineLib.sol";
import {CourtMath} from "./CourtMath.sol";
import {IClaimRegistry} from "../interfaces/IClaimRegistry.sol";
import {IAdjudicator} from "../interfaces/IAdjudicator.sol";
import "./Errors.sol";

/// @notice The block court's check: steps 1-19 run in order and the error of the first failing check is the revert
/// reason. Steps: 1-3 claim (any status), tier E, class kind EXACT or CELL; 4-7 rec, cutPre, occupancy and program
/// commitments; 8 the ordered static preconditions with exact lengths; 9-11 header, coefficient index and q, D
/// membership (binary search); 12 the MCU leaf and R_Y; 13 level-l* occupancy; 14 taps and footprint (KMAX = 4, the
/// rule-3 bound); 15 the exact list of occupied cells; 16-17 alpha checks (the fill and its path only when
/// min alpha < 255); 18 the compiled block X; 19 Phi, one coefficient and the class predicate.
/// Python reference: cutline.ledger.proofin.court_reference.
abstract contract CourtCore {
    using CutlineLib for bytes;

    IClaimRegistry public immutable reg;
    IAdjudicator public immutable adj;

    uint256 internal constant KMAX = CourtMath.KMAX;
    uint256 internal constant PROG_FIXED = 794;
    uint256 internal constant HEADER_LEN = 197;
    uint256 internal constant COEFS_LEN = 384;
    uint256 internal constant CELL_LEN = 1024;
    uint256 internal constant FILL_LEN = 192;

    /// Claim, program and block geometry of one proof (memory).
    struct Ctx {
        bytes32 claimId;
        uint256 kind;       // 1 EXACT, 2 CELL
        uint256 eta20;
        uint256 l;          // l*
        uint256 y0;
        uint256 x0;
        uint256 ch;
        uint256 cw;
        uint256 oh;
        uint256 ow;
        uint256 oy;
        uint256 ox;
        uint256 hc;
        uint256 wc;
        uint256 kernel;
        uint256 nMcu;
        uint256 bx;
        uint256 by;
        uint256 ny;         // level-l* cell grid
        uint256 nx;
        bytes32 levelRoot;  // R_l*
        uint256 q;
        int256 c;
        // footprint: crop-relative source ranges and the cell rectangle
        uint256 rmin;
        uint256 rmax;
        uint256 cmin;
        uint256 cmax;
        uint256 cyA;
        uint256 cxA;
        uint256 gw;
        uint256 gh;
    }

    /// Taps of the block's 8 outputs on one axis, as one flat uint256[56]: [d] first source index (crop-relative),
    /// [8 + d] tap count (0 = outside the placed rectangle), [16 + d] tap sum, [24 + 4d + t] weight t (KMAX = 4; unused
    /// weights are 0, so the unrolled 4-tap sums below read at most 3 words past a row with weight 0).
    uint256 internal constant T_N = 8;
    uint256 internal constant T_TOT = 16;
    uint256 internal constant T_W = 24;

    /// Intermediate buffers of steps 14-18.
    struct Work {
        uint256[56] tx;
        uint256[56] ty;
        uint256[] off;      // cell data calldata offsets + 1 (0 = unoccupied), gw·gh
        uint256[] src;      // footprint pixels, fw·fh (+ 3 guard words), four 64-bit lanes [R|G|B|A]
        uint256[] hor;      // horizontal pass, (fh + 3)·8 (3 guard rows)
        uint256[128] v;     // vertical pass: [i] the lanes of pixel i, [64 + i] its divisor D (0 = outside)
        uint256 fw;
        uint256 fh;
        uint256 flags;      // bit 0: some a > 0 (step 16); bit 1: some a < 255 (step 17)
    }

    constructor(IClaimRegistry reg_, IAdjudicator adj_) {
        reg = reg_;
        adj = adj_;
    }

    // =====================================================================================================
    // The whole check (steps 1-19)
    // =====================================================================================================

    /// Steps 1-19; returns the verdict instead of requiring it (BlockCourt requires `fails`).
    /// w is the islow output t (EXACT, scaled by 8) or D80_k (CELL); c the file's quantised coefficient.
    function _check(ProofIn calldata p)
        internal view returns (bytes32 claimId, uint256 q, int256 c, uint8[192] memory X, int256 w, bool fails)
    {
        Ctx memory g = _prelude(p);
        X = _block(p, g);
        (w, fails) = _test(X, g.kind, g.eta20, g.q, g.c, p.comp, p.k);
        return (g.claimId, g.q, g.c, X, w, fails);
    }

    /// Steps 1-13.
    function _prelude(ProofIn calldata p) internal view returns (Ctx memory g) {
        _claim(p, g);          // 1-3
        _commitments(p);       // 4-7
        _static(p, g);         // 8
        _header(p, g);         // 9-11
        _listing(p, g);        // 12
        _occ(p, g);            // 13
    }

    /// Steps 14-18: the footprint, the cells, the compiled block X (alpha checks of steps 16-17 included).
    function _block(ProofIn calldata p, Ctx memory g) internal view returns (uint8[192] memory X) {
        Work memory w;
        _footprint(g, w.tx, w.ty);           // 14
        _cells(p, g, w);                     // 15
        _gather(g, w);
        _horizontal(g, w);
        _vertical(g, w);
        _finish(p, w, X);                    // 18 (X), alpha flags for 16-17
        _alphaChecks(p, g, w.flags);         // 16, 17
    }

    // =====================================================================================================
    // Steps 1-13
    // =====================================================================================================

    /// 1 claim (any status), 2 tier E, 3 class kind EXACT or CELL.
    function _claim(ProofIn calldata p, Ctx memory g) internal view {
        ClaimCore memory core = p.core;
        g.claimId = CutlineLib.claimId(p.nf, p.claimant, core);
        (bool exists, bytes32 nf_,,,,,,,,) = reg.claimInfo(g.claimId);
        if (!exists || nf_ != p.nf) revert E_COURT_CLAIM();
        if (core.tier != 1) revert E_COURT_TIER();
        (, uint8 kind, uint32 eta20) = adj.cls(core.encClass);
        if (kind != 1 && kind != 2) revert E_COURT_CLASS();
        g.kind = kind;
        g.eta20 = eta20;
    }

    /// 4 rec, 5 cutPre vs R_cut, 6 occHashes vs R_occ, 7 prog vs tHash.
    function _commitments(ProofIn calldata p) internal pure {
        if (p.rec.length != CutlineLib.REC_LEN || sha256(p.rec) != p.core.recHash) revert E_COURT_REC();
        bytes calldata cp = p.cutPre;
        if (cp.length < 5) revert E_COURT_ROOTS();
        uint256 lmax = cp.u8(4);
        if (cp.length != 5 + 32 * (lmax + 1) || CutlineLib.rCut(cp) != p.rec.word(1)) revert E_COURT_ROOTS();
        if (p.occHashes.length != lmax + 1 || CutlineLib.rOcc(p.occHashes) != p.rec.word(2)) revert E_COURT_OCC();
        if (CutlineLib.tHash(p.prog) != p.core.tHash) revert E_COURT_PROGRAM();
    }

    /// 8 the static preconditions, in court order; any failure E_COURT_STATIC.
    function _static(ProofIn calldata p, Ctx memory g) internal pure {
        bytes calldata t = p.prog;
        // rule 5(9): exact length
        if (t.length < PROG_FIXED || t.length != PROG_FIXED + 4 * t.u16(792)) revert E_COURT_STATIC();
        // rule 7: flags bit 0 = 0 and bits 2-7 = 0
        if (t.u8(23) & 0xfd != 0) revert E_COURT_STATIC();
        // rules 5(0), 5(1): kernel <= 1, sampling <= 1
        g.kernel = t.u8(21);
        if (g.kernel > 1 || t.u8(22) > 1) revert E_COURT_STATIC();
        g.l = t.u8(0);
        g.y0 = t.u16(1);
        g.x0 = t.u16(3);
        g.ch = t.u16(5);
        g.cw = t.u16(7);
        g.oh = t.u16(9);
        g.ow = t.u16(11);
        g.oy = t.u16(13);
        g.ox = t.u16(15);
        g.hc = t.u16(17);
        g.wc = t.u16(19);
        // rule 3: 1 <= oh <= ch < 2 oh and 1 <= ow <= cw < 2 ow
        if (g.oh == 0 || g.oh > g.ch || g.ch >= 2 * g.oh || g.ow == 0 || g.ow > g.cw || g.cw >= 2 * g.ow) {
            revert E_COURT_STATIC();
        }
        // rules 5(3), 5(5), 5(7), 5(6)
        if (g.hc % 16 != 0 || g.wc % 16 != 0) revert E_COURT_STATIC();
        if (g.oy + g.oh > g.hc || g.ox + g.ow > g.wc) revert E_COURT_STATIC();
        bytes calldata cp = p.cutPre;
        if (g.l > cp.u8(4)) revert E_COURT_STATIC();
        uint256 hl = (cp.u16(0) + (uint256(1) << g.l) - 1) >> g.l;   // h* = ceil(H0 / 2^l*)
        uint256 wl = (cp.u16(2) + (uint256(1) << g.l) - 1) >> g.l;   // w*
        if (g.y0 + g.ch > hl || g.x0 + g.cw > wl) revert E_COURT_STATIC();
        g.ny = (hl + 15) >> 4;
        g.nx = (wl + 15) >> 4;
    }

    /// 9 header, 10 index and q, 11 D membership.
    function _header(ProofIn calldata p, Ctx memory g) internal pure {
        bytes calldata h = p.header;
        if (h.length != HEADER_LEN) revert E_COURT_HEADER();
        if (h.u8(4) != p.prog.u8(22) || h.u16(0) != g.wc || h.u16(2) != g.hc) revert E_COURT_STATIC();
        if (h.u8(4) != 0) revert E_COURT_SAMPLING();
        uint256 bpr = g.wc >> 3;
        g.nMcu = bpr * (g.hc >> 3);
        if (p.mcu >= g.nMcu || p.comp >= 3 || p.k >= 64) revert E_COURT_INDEX();
        g.q = h.u8(5 + 64 * uint256(p.comp) + uint256(p.k));
        if (g.q == 0) revert E_COURT_ZERO_Q();
        g.bx = p.mcu % bpr;
        g.by = p.mcu / bpr;
        bytes calldata t = p.prog;
        uint256 lo;
        uint256 hi = t.u16(792);
        uint256 mcu = p.mcu;
        while (lo < hi) {
            uint256 mid = (lo + hi) >> 1;
            uint256 v = t.u32(PROG_FIXED + 4 * mid);
            if (v == mcu) revert E_COURT_DECLARED();
            if (v < mcu) lo = mid + 1;
            else hi = mid;
        }
    }

    /// 12 coefs, the MCU leaf's path to R_mcu, R_Y; c.
    function _listing(ProofIn calldata p, Ctx memory g) internal view {
        if (p.coefs.length != COEFS_LEN) revert E_COURT_COEFS();
        (bool ok, bytes32 rMcu) = CutlineLib.tryRootFromPath(CutlineLib.mcuLeaf(p.mcu, p.coefs), p.mcu, g.nMcu, p.mcuPath);
        if (!ok) revert E_COURT_MCU_PATH();
        if (CutlineLib.listingRoot(p.header, rMcu) != p.core.rY) revert E_COURT_RY();
        g.c = p.coefs.i16(128 * uint256(p.comp) + 2 * uint256(p.k));
    }

    /// 13 R_l* and the level-l* occupancy bits.
    function _occ(ProofIn calldata p, Ctx memory g) internal pure {
        g.levelRoot = bytes32(p.cutPre[5 + 32 * g.l:37 + 32 * g.l]);
        if (!CutlineLib.checkPackbitsPadding(p.occBits, g.ny * g.nx) || sha256(p.occBits) != p.occHashes[g.l]) {
            revert E_COURT_OCC();
        }
    }


    // =====================================================================================================
    // Steps 14-15: taps, footprint, cells
    // =====================================================================================================

    function _axisTaps(Ctx memory g, uint256[56] memory t, bool xAxis) internal pure {
        (uint256 nIn, uint256 nOut, uint256 o, uint256 b) = xAxis ? (g.cw, g.ow, g.ox, g.bx) : (g.ch, g.oh, g.oy, g.by);
        uint256 kernel = g.kernel;
        unchecked {
            for (uint256 d; d < 8; ++d) {
                uint256 cc = 8 * b + d;
                if (cc < o || cc >= o + nOut) continue;
                uint256 j = cc - o;
                (uint256 first, uint256 last) = CourtMath.tapRange(kernel, j, nIn, nOut);
                // unreachable after step 8 (rule 3 bounds the taps by 3 / 4): a buffer guard, not a protocol check
                if (last - first >= KMAX) revert E_COURT_STATIC();
                uint256 tot;
                for (uint256 i = first; i <= last; ++i) {
                    uint256 wv = CourtMath.weight(kernel, i, j, nIn, nOut);
                    t[T_W + 4 * d + (i - first)] = wv;
                    tot += wv;
                }
                t[d] = first;
                t[T_N + d] = last - first + 1;
                t[T_TOT + d] = tot;
            }
        }
    }

    /// 14 taps of both axes and the footprint; E_COURT_NOT_CERTIFIED if the block is outside the placement.
    function _footprint(Ctx memory g, uint256[56] memory tx_, uint256[56] memory ty_) internal pure {
        _axisTaps(g, tx_, true);
        _axisTaps(g, ty_, false);
        uint256 cmin = type(uint256).max;
        uint256 rmin = type(uint256).max;
        uint256 cmax;
        uint256 rmax;
        unchecked {
            for (uint256 d; d < 8; ++d) {
                uint256 n = tx_[T_N + d];
                if (n != 0) {
                    uint256 i0 = tx_[d];
                    if (i0 < cmin) cmin = i0;
                    if (i0 + n - 1 > cmax) cmax = i0 + n - 1;
                }
                n = ty_[T_N + d];
                if (n != 0) {
                    uint256 i0 = ty_[d];
                    if (i0 < rmin) rmin = i0;
                    if (i0 + n - 1 > rmax) rmax = i0 + n - 1;
                }
            }
        }
        if (cmin == type(uint256).max || rmin == type(uint256).max) revert E_COURT_NOT_CERTIFIED();
        g.cmin = cmin;
        g.cmax = cmax;
        g.rmin = rmin;
        g.rmax = rmax;
        g.cxA = (g.x0 + cmin) >> 4;
        g.cyA = (g.y0 + rmin) >> 4;
        g.gw = ((g.x0 + cmax) >> 4) - g.cxA + 1;
        g.gh = ((g.y0 + rmax) >> 4) - g.cyA + 1;
    }

    /// H(0x00 ‖ "CUTLINE-cell-v1" ‖ u8 l ‖ u16 cy ‖ u16 cx ‖ salt16 ‖ data[1024]), hashed in scratch memory
    /// above the free pointer (not advanced); |data| = 1024 is checked by the caller.
    function _cellLeaf(uint256 l, uint256 cy, uint256 cx, bytes16 salt, bytes calldata data) private view returns (bytes32 h) {
        assembly ("memory-safe") {
            let q := mload(0x40)
            mstore(q, 0x004355544c494e452d63656c6c2d763100000000000000000000000000000000) // 0x00 "CUTLINE-cell-v1"
            mstore8(add(q, 16), l)
            mstore(add(q, 17), shl(240, cy))
            mstore(add(q, 19), shl(240, cx))
            mstore(add(q, 21), salt)
            calldatacopy(add(q, 37), data.offset, 1024)
            if iszero(staticcall(gas(), 0x02, q, 1061, 0x00, 0x20)) { revert(0, 0) }
            h := mload(0x00)
        }
    }

    /// 15 exactly the occupied cells of the rectangle, row-major; records their data calldata offsets + 1 (0 =
    /// unoccupied, all-zero RGBA).
    function _cells(ProofIn calldata p, Ctx memory g, Work memory w) internal view {
        uint256 gw = g.gw;
        uint256[] memory off = new uint256[](gw * g.gh);
        w.off = off;
        uint256 j;
        uint256 nx = g.nx;
        uint256 n = g.ny * nx;
        uint256 l = g.l;
        bytes32 root = g.levelRoot;
        bytes calldata occ = p.occBits;
        Cell[] calldata cs = p.cells;
        for (uint256 gi; gi < off.length; ++gi) {
            uint256 cy = g.cyA + gi / gw;
            uint256 cx = g.cxA + gi % gw;
            uint256 idx = cy * nx + cx;
            if ((occ.u8(idx >> 3) >> (7 - (idx & 7))) & 1 == 0) continue;
            if (j >= cs.length) revert E_COURT_CELLS();
            Cell calldata c = cs[j++];
            bytes calldata d = c.data;
            if (c.cy != cy || c.cx != cx || d.length != CELL_LEN) revert E_COURT_CELLS();
            (bool ok, bytes32 r) = CutlineLib.tryRootFromPath(_cellLeaf(l, cy, cx, c.salt, d), idx, n, c.path);
            if (!ok || r != root) revert E_COURT_CELL_PATH();
            uint256 o;
            assembly ("memory-safe") { o := d.offset }
            off[gi] = o + 1;
        }
        if (j != cs.length) revert E_COURT_CELLS();
    }

    // =====================================================================================================
    // Steps 16-18: the compiled block, in Yul: gather, horizontal pass, and the composite split into a vertical
    // pass and a finishing pass to keep the loops on the stack
    // =====================================================================================================

    /// RGBA byte quadruple -> four 64-bit lanes [R|G|B|A]: the copies px·2^0, ·2^56, ·2^112, ·2^168 do not overlap.
    uint256 private constant SPREAD = 0x1000000000000010000000000000100000000000001; // 2^168 + 2^112 + 2^56 + 1
    uint256 private constant LANES = 0xff00000000000000ff00000000000000ff00000000000000ff;

    /// src[r·fw + c] = RGBA of level pixel (y0 + rmin + r, x0 + cmin + c) in four lanes; 3 zero guard words at the end.
    function _gather(Ctx memory g, Work memory w) internal pure {
        uint256 fw = g.cmax - g.cmin + 1;
        uint256 fh = g.rmax - g.rmin + 1;
        w.fw = fw;
        w.fh = fh;
        uint256[] memory off = w.off;
        uint256[] memory src = new uint256[](fw * fh + 3);
        w.src = src;
        uint256 yb = g.y0 + g.rmin;
        uint256 xb = g.x0 + g.cmin;
        uint256 cyA = g.cyA;
        uint256 cxA = g.cxA;
        uint256 gw = g.gw;
        assembly ("memory-safe") {
            let sp := add(src, 0x20)
            let ye := add(yb, fh)
            let xe := add(xb, fw)
            for { let Y := yb } lt(Y, ye) { Y := add(Y, 1) } {
                let orow := add(add(off, 0x20), shl(5, mul(sub(shr(4, Y), cyA), gw)))
                let base := sub(shl(6, and(Y, 15)), 1) // in-cell row offset, minus the +1 of off
                for { let Xl := xb } lt(Xl, xe) { Xl := add(Xl, 1) } {
                    let o := mload(add(orow, shl(5, sub(shr(4, Xl), cxA))))
                    if o {
                        mstore(sp, and(mul(shr(224, calldataload(add(o, add(base, shl(2, and(Xl, 15)))))), SPREAD), LANES))
                    }
                    sp := add(sp, 0x20)
                }
            }
        }
    }

    /// hor[r·8 + dx] = Σ_{t<4} wx[dx][t]·src[r·fw + i0x[dx] − cmin + t]; 3 zero guard rows at the end.
    function _horizontal(Ctx memory g, Work memory w) internal pure {
        uint256 fw = w.fw;
        uint256 fh = w.fh;
        uint256[] memory src = w.src;
        uint256[] memory hor = new uint256[]((fh + 3) * 8);
        w.hor = hor;
        uint256[56] memory txp = w.tx;
        uint256 cmin = g.cmin;
        assembly ("memory-safe") {
            let stride := shl(5, fw)
            for { let dx := 0 } lt(dx, 8) { dx := add(dx, 1) } {
                if mload(add(txp, shl(5, add(T_N, dx)))) {
                    let wp := add(txp, shl(5, add(T_W, shl(2, dx))))
                    let sp := add(add(src, 0x20), shl(5, sub(mload(add(txp, shl(5, dx))), cmin)))
                    let hp := add(add(hor, 0x20), shl(5, dx))
                    let he := add(hp, shl(8, fh))
                    for { } lt(hp, he) { hp := add(hp, 256) } {
                        mstore(hp, add(add(mul(mload(wp), mload(sp)), mul(mload(add(wp, 0x20)), mload(add(sp, 0x20)))),
                            add(mul(mload(add(wp, 0x40)), mload(add(sp, 0x40))), mul(mload(add(wp, 0x60)), mload(add(sp, 0x60))))))
                        sp := add(sp, stride)
                    }
                }
            }
        }
    }

    /// v[i] = Σ_{t<4} wy[dy][t]·hor[(i0y[dy] − rmin + t)·8 + dx] and v[64 + i] = toty[dy]·totx[dx] for the pixels
    /// inside the placed rectangle (v[64 + i] = 0 outside).
    function _vertical(Ctx memory g, Work memory w) internal pure {
        uint256[56] memory txp = w.tx;
        uint256[56] memory typ = w.ty;
        uint256[] memory hor = w.hor;
        uint256[128] memory v = w.v;
        uint256 rmin = g.rmin;
        assembly ("memory-safe") {
            for { let i := 0 } lt(i, 64) { i := add(i, 1) } {
                let dy := shr(3, i)
                let dx := and(i, 7)
                if mul(mload(add(typ, shl(5, add(T_N, dy)))), mload(add(txp, shl(5, add(T_N, dx))))) {
                    let wp := add(typ, shl(5, add(T_W, shl(2, dy))))
                    let hp := add(add(add(hor, 0x20), shl(8, sub(mload(add(typ, shl(5, dy))), rmin))), shl(5, dx))
                    mstore(add(v, shl(5, i)), add(add(mul(mload(wp), mload(hp)), mul(mload(add(wp, 0x20)), mload(add(hp, 256)))),
                        add(mul(mload(add(wp, 0x40)), mload(add(hp, 512))), mul(mload(add(wp, 0x60)), mload(add(hp, 768))))))
                    mstore(add(v, shl(5, add(64, i))),
                        mul(mload(add(typ, shl(5, add(T_TOT, dy)))), mload(add(txp, shl(5, add(T_TOT, dx))))))
                }
            }
        }
    }

    /// 18 X (court semantics v2): rnd(u) = min(255, (2u + D) div 2D); a = rnd(acc_A); comp_ch = min(255,
    /// rnd(acc_ch) + ((255 − a)·Bg_ch + 127) div 255); outside the placed rectangle comp = Bg (a = 0); X_ch =
    /// LUT[ch][comp_ch]. Bg is read from p.fill at its calldata offset whatever its length: when step 17 accepts an
    /// absent fill every a = 255, so Bg has weight 0 and contributes (0 + 127) div 255 = 0. Sets w.flags (steps 16-17).
    function _finish(ProofIn calldata p, Work memory w, uint8[192] memory X) internal pure {
        bytes calldata lut = p.prog[24:792];
        bytes calldata fill = p.fill;
        uint256[128] memory v = w.v;
        uint256 flags;
        assembly ("memory-safe") {
            let lo := lut.offset
            let fo := fill.offset
            for { let i := 0 } lt(i, 64) { i := add(i, 1) } {
                let f := shr(232, calldataload(add(fo, mul(i, 3))))
                let xp := add(X, mul(i, 96))
                let d2 := mload(add(v, shl(5, add(64, i))))
                switch d2
                case 0 {
                    flags := or(flags, 2)
                    mstore(xp, shr(248, calldataload(add(lo, shr(16, f)))))
                    mstore(add(xp, 32), shr(248, calldataload(add(lo, add(256, and(shr(8, f), 0xff))))))
                    mstore(add(xp, 64), shr(248, calldataload(add(lo, add(512, and(f, 0xff))))))
                }
                default {
                    let acc := mload(add(v, shl(5, i)))
                    let a := div(add(shl(1, and(acc, 0xffffffffffffffff)), d2), shl(1, d2))
                    if gt(a, 255) { a := 255 }
                    if a { flags := or(flags, 1) }
                    if lt(a, 255) { flags := or(flags, 2) }
                    let rest := sub(255, a)
                    let c := div(add(shl(1, shr(192, acc)), d2), shl(1, d2))
                    if gt(c, 255) { c := 255 }
                    c := add(c, div(add(mul(rest, shr(16, f)), 127), 255))
                    if gt(c, 255) { c := 255 }
                    mstore(xp, shr(248, calldataload(add(lo, c))))
                    c := div(add(shl(1, and(shr(128, acc), 0xffffffffffffffff)), d2), shl(1, d2))
                    if gt(c, 255) { c := 255 }
                    c := add(c, div(add(mul(rest, and(shr(8, f), 0xff)), 127), 255))
                    if gt(c, 255) { c := 255 }
                    mstore(add(xp, 32), shr(248, calldataload(add(lo, add(256, c)))))
                    c := div(add(shl(1, and(shr(64, acc), 0xffffffffffffffff)), d2), shl(1, d2))
                    if gt(c, 255) { c := 255 }
                    c := add(c, div(add(mul(rest, and(f, 0xff)), 127), 255))
                    if gt(c, 255) { c := 255 }
                    mstore(add(xp, 64), shr(248, calldataload(add(lo, add(512, c)))))
                }
            }
        }
        w.flags = flags;
    }

    /// 16 max a > 0; 17 the fill and its path iff min a < 255, else "" and [].
    function _alphaChecks(ProofIn calldata p, Ctx memory g, uint256 flags) internal view {
        if (flags & 1 == 0) revert E_COURT_NOT_CERTIFIED();
        if (flags & 2 != 0) {
            if (p.fill.length != FILL_LEN) revert E_COURT_FILL();
            (bool ok, bytes32 r) = CutlineLib.tryRootFromPath(CutlineLib.fillLeaf(p.mcu, p.fill), p.mcu, g.nMcu, p.fillPath);
            if (!ok || r != p.core.rFill) revert E_COURT_FILL();
        } else if (p.fill.length != 0 || p.fillPath.length != 0) {
            revert E_COURT_FILL();
        }
    }

    // =====================================================================================================
    // Step 19: Phi, one coefficient, the class predicate
    // =====================================================================================================

    /// P[i] = Phi_comp(X[i]) − 128 (libjpeg rgb_ycc_convert; 4:4:4, no downsampling). The weighted sums are
    /// non-negative for every comp, so the shift is exact.
    function _phiBlock(uint8[192] memory X, uint256 comp) internal pure returns (int256[64] memory P) {
        // two's-complement coefficients: the Yul sums wrap to the exact non-negative value
        (uint256 kr, uint256 kg, uint256 kb, uint256 k0) = comp == 0
            ? (uint256(19595), uint256(38470), uint256(7471), uint256(32768))
            : comp == 1
                ? (uint256(int256(-11059)), uint256(int256(-21709)), uint256(32768), uint256(8421375))
                : (uint256(32768), uint256(int256(-27439)), uint256(int256(-5329)), uint256(8421375));
        assembly ("memory-safe") {
            for { let i := 0 } lt(i, 64) { i := add(i, 1) } {
                let xp := add(X, mul(i, 96))
                let s := add(add(add(mul(mload(xp), kr), mul(mload(add(xp, 32)), kg)), mul(mload(add(xp, 64)), kb)), k0)
                mstore(add(P, shl(5, i)), sub(shr(16, s), 128))
            }
        }
    }

    function _test(uint8[192] memory X, uint256 kind, uint256 eta20, uint256 q, int256 c, uint256 comp, uint256 k)
        internal pure returns (int256 w, bool fails)
    {
        int256[64] memory P = _phiBlock(X, comp);
        if (kind == 1) {
            w = CourtMath.fdctIslowAt(P, k);
            fails = CourtMath.exactFails(c, w, q);
        } else {
            w = CourtMath.d80At(P, k);
            fails = CourtMath.cellFails(c, w, q, eta20);
        }
    }
}
