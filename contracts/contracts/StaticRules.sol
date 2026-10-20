// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {IClaimRegistry} from "./interfaces/IClaimRegistry.sol";
import {IAdjudicator} from "./interfaces/IAdjudicator.sol";
import {StaticIn} from "./lib/Types.sol";
import {Consts} from "./lib/Consts.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import "./lib/Errors.sol";

/// @notice Path (a') static proofs: `proveStaticViolation(StaticIn)` (selector 0x8d02fe1f), checks 1-11 below in
/// order. Called directly, not through the Adjudicator. The Python reference model is
/// `cutline.compiler.static_rules.prove_static_violation`.
/// - Q_pol is the 192-byte genesis constant below (IJG q85 luma ‖ chroma ‖ chroma, natural order; SHA-256
///   0984e213…; `qPol()` exposes it for re-hashing).
/// - Rule 8 reads `tierTProfile` / `kfam` from the Adjudicator; the claim lookup and the strike go to
///   ClaimRegistry (a static proof strikes with byCourt = true, final like the court).
/// - Rule 9, the size bounds: 9(0) the capture master is at most 2^28 pixels,
///   H0·W0 <= 2^28, read from the verified R_cut preimage (cutPre = u16 H0 ‖ u16 W0 ‖ …; no prog); it keeps every block
///   failure provable within the per-transaction gas cap (court liveness), and its own proof costs O(|rec| + |cutPre|).
///   9(1) the level-l* crop is at most 2^25 pixels, ch·cw <= 2^25, read from enc(π); it bounds the
///   verifier's Compile.
contract StaticRules {
    IGovernance public immutable gov;
    IClaimRegistry public immutable reg;
    IAdjudicator public immutable adj;

    /// Q_pol = IJG q85 (luma ‖ chroma ‖ chroma), natural order.
    bytes internal constant Q_POL =
        hex"05030305070c0f120404040608111211040405070c111511040507090f1a181305070b1114211f17070b1113181f221c0f13171a1f24241e161c1d1d221e1f1e"
        hex"0505070e1e1e1e1e050608141e1e1e1e0708111e1e1e1e1e0e141e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e"
        hex"0505070e1e1e1e1e050608141e1e1e1e0708111e1e1e1e1e0e141e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e1e";

    // row flags: the preimages a row reads
    uint256 private constant R_PROG = 1;
    uint256 private constant R_HDR = 2;
    uint256 private constant R_REC = 4;

    // enc(π) offsets
    uint256 private constant P_LSTAR = 0;
    uint256 private constant P_Y0 = 1;
    uint256 private constant P_X0 = 3;
    uint256 private constant P_CH = 5;
    uint256 private constant P_CW = 7;
    uint256 private constant P_OH = 9;
    uint256 private constant P_OW = 11;
    uint256 private constant P_OY = 13;
    uint256 private constant P_OX = 15;
    uint256 private constant P_HC = 17;
    uint256 private constant P_WC = 19;
    uint256 private constant P_KERNEL = 21;
    uint256 private constant P_SAMPLING = 22;
    uint256 private constant P_FLAGS = 23;
    uint256 private constant P_LUT = 24;
    uint256 private constant P_ND = 792;
    uint256 private constant P_D = 794;

    uint256 private constant HEADER_LEN = 197;   // listing header length
    uint256 private constant ASPECT_A = 1_000_000;
    uint256 private constant ASPECT_B = 1_020_202;
    uint256 private constant MASTER_MAX_PIXELS = 1 << 28;   // static rule 9(0): H0·W0 <= 2^28
    uint256 private constant CROP_MAX_PIXELS = 1 << 25;     // static rule 9(1): ch·cw <= 2^25

    event StaticViolation(bytes32 indexed claimId, uint8 rule, uint8 i, uint16 j, address challenger);

    constructor(IGovernance g, IClaimRegistry r, IAdjudicator a) {
        gov = g;
        reg = r;
        adj = a;
    }

    /// The 192 bytes of Q_pol (so its SHA-256 can be checked).
    function qPol() external pure returns (bytes memory) {
        return Q_POL;
    }

    /// Checks 1-11 in order (the first failing check is the revert).
    function proveStaticViolation(StaticIn calldata s) external returns (bytes32 claimId) {
        // 1 caller WATCHER or validator
        if (!gov.hasRole(Consts.ROLE_WATCHER, msg.sender) && !gov.isValidator(msg.sender)) revert E_ROLE();
        // 2 (rule, i) is a row; j in the row's constant range
        (bool isRow, uint256 fl, uint256 jmax, bool jDyn) = _row(s.rule, s.i);
        if (!isRow || (!jDyn && s.j > jmax)) revert E_STATIC_RULE();
        // 3 the claim exists (any status)
        claimId = CutlineLib.claimId(s.nf, s.claimant, s.core);
        (bool exists,,,,,,,,,) = reg.claimInfo(claimId);
        if (!exists) revert E_STATIC_CLAIM();
        // 4 every preimage the row does not read is absent
        if ((fl & R_PROG == 0 && s.prog.length != 0)
            || (fl & R_HDR == 0 && (s.header.length != 0 || s.rMcu != bytes32(0)))
            || (fl & R_REC == 0 && (s.rec.length != 0 || s.cutPre.length != 0))) revert E_STATIC_ARGS();
        bool is59 = s.rule == 5 && s.i == 9;
        if (fl & R_PROG != 0) {
            // 5 tHash binds the program
            if (CutlineLib.tHash(s.prog) != s.core.tHash) revert E_STATIC_PROGRAM();
            // 6 exact length (except the row 5(9) itself)
            if (!is59 && !_progLenOk(s.prog)) revert E_STATIC_PROGRAM_LEN();
        }
        // 7 the header binds to core.rY
        if (fl & R_HDR != 0) {
            if (s.header.length != HEADER_LEN || CutlineLib.listingRoot(s.header, s.rMcu) != s.core.rY) revert E_STATIC_RY();
        }
        // 8 rec and the R_cut preimage
        if (fl & R_REC != 0) {
            bytes calldata cp = s.cutPre;
            if (s.rec.length != CutlineLib.REC_LEN || sha256(s.rec) != s.core.recHash || cp.length < 5
                || cp.length != 5 + 32 * (CutlineLib.u8(cp, 4) + 1) || CutlineLib.rCut(cp) != CutlineLib.word(s.rec, 1))
            {
                revert E_STATIC_REC();
            }
        }
        // 9 rows 4 and 5(8): j against nD of the verified program
        if (jDyn) {
            uint256 nD = CutlineLib.u16(s.prog, P_ND);
            if (s.rule == 4 ? s.j + 1 >= nD : s.j >= nD) revert E_STATIC_RULE();
        }
        // 10 the row's predicate
        if (!_violated(s)) revert E_STATIC_NO_VIOLATION();
        // 11 strike (self-verifying, final) and emit
        reg.strike(claimId, true);
        emit StaticViolation(claimId, s.rule, s.i, s.j, msg.sender);
    }

    // ------------------------------------------------------------------ rows and predicates

    /// (isRow, preimage flags, constant j range [0, jmax], j range depends on nD (rows 4 and 5(8))).
    function _row(uint256 rule, uint256 i) private pure returns (bool, uint256, uint256, bool) {
        if (rule == 1) return (i < 3, R_PROG, 254, false);
        if (rule == 2) return (i < 3, R_HDR, 63, false);
        if (rule == 3) return (i < 2, R_PROG, 0, false);
        if (rule == 4) return (i == 0, R_PROG, 0, true);
        if (rule == 5) {
            uint256 fl = R_PROG;
            if (i == 2 || i == 4) fl |= R_HDR;
            if (i == 6 || i == 7) fl |= R_REC;
            return (i <= 9, fl, 0, i == 8);
        }
        if (rule == 6) return (i < 2, R_PROG, 0, false);
        if (rule == 7) return (i < 8 && i != 1, R_PROG, 0, false);   // no row for bit 1 (dGen)
        if (rule == 8) {
            if (i == 0) return (true, R_PROG, 0, false);
            if (i == 1) return (true, R_HDR, 0, false);
            if (i == 2) return (true, 0, 0, false);
        }
        if (rule == 9) return (i < 2, i == 0 ? R_REC : R_PROG, 0, false);   // 9(0) master (rec + cutPre), 9(1) crop (prog)
        return (false, 0, 0, false);
    }

    /// |prog| >= 794 and |prog| = 794 + 4·nD.
    function _progLenOk(bytes calldata p) private pure returns (bool) {
        return p.length >= P_D && p.length == P_D + 4 * CutlineLib.u16(p, P_ND);
    }

    function _violated(StaticIn calldata s) private view returns (bool) {
        bytes calldata p = s.prog;
        uint256 rule = s.rule;
        uint256 i = s.i;
        uint256 j = s.j;
        if (rule == 1) {
            uint256 o = P_LUT + 256 * i + j;
            return CutlineLib.u8(p, o) > CutlineLib.u8(p, o + 1);
        }
        if (rule == 2) {
            uint256 q = CutlineLib.u8(s.header, 5 + 64 * i + j);
            return q == 0 || q > uint8(Q_POL[64 * i + j]);
        }
        if (rule == 3) {
            (uint256 inN, uint256 outN) = i == 0
                ? (CutlineLib.u16(p, P_CH), CutlineLib.u16(p, P_OH))
                : (CutlineLib.u16(p, P_CW), CutlineLib.u16(p, P_OW));
            return !(1 <= outN && outN <= inN && inN < 2 * outN);
        }
        if (rule == 4) return _d(p, j) >= _d(p, j + 1);
        if (rule == 5) return _rule5(s, p, i, j);
        if (rule == 6) {
            uint256 cw = CutlineLib.u16(p, P_CW);
            uint256 ch = CutlineLib.u16(p, P_CH);
            uint256 ow = CutlineLib.u16(p, P_OW);
            uint256 oh = CutlineLib.u16(p, P_OH);
            return i == 0 ? ASPECT_A * cw * oh > ASPECT_B * ch * ow : ASPECT_A * ch * ow > ASPECT_B * cw * oh;
        }
        if (rule == 7) return (CutlineLib.u8(p, P_FLAGS) >> i) & 1 == 1;
        // rule 9: 9(0) H0·W0 > 2^28; 9(1) ch·cw > 2^25 (u16 · u16 < 2^32, no overflow)
        if (rule == 9) {
            if (i == 0) return CutlineLib.u16(s.cutPre, 0) * CutlineLib.u16(s.cutPre, 2) > MASTER_MAX_PIXELS;
            return CutlineLib.u16(p, P_CH) * CutlineLib.u16(p, P_CW) > CROP_MAX_PIXELS;
        }
        // rule 8 (8c): vacuous for tier E; t = tierT[core.encClass]
        if (s.core.tier != 2) return false;
        (bool defined, uint8 fam, uint8 sampling, uint8 codec) = adj.tierTProfile(s.core.encClass);
        if (!defined) return false;   // cannot happen for a registered tier-T claim (rule 8b)
        if (i == 0) {
            (bool kDefined, uint8 exact) = adj.kfam(fam);
            return !kDefined || exact != CutlineLib.u8(p, P_KERNEL);
        }
        if (i == 1) return sampling != CutlineLib.u8(s.header, 4);
        return codec != 0;
    }

    function _rule5(StaticIn calldata s, bytes calldata p, uint256 i, uint256 j) private pure returns (bool) {
        if (i == 9) return !_progLenOk(p);
        if (i == 0) return CutlineLib.u8(p, P_KERNEL) > 1;
        uint256 sampling = CutlineLib.u8(p, P_SAMPLING);
        if (i == 1) return sampling > 1;
        if (i == 2) return sampling != CutlineLib.u8(s.header, 4);
        uint256 hc = CutlineLib.u16(p, P_HC);
        uint256 wc = CutlineLib.u16(p, P_WC);
        if (i == 3) return hc % 16 != 0 || wc % 16 != 0;
        if (i == 4) return CutlineLib.u16(s.header, 0) != wc || CutlineLib.u16(s.header, 2) != hc;
        if (i == 5) {
            return CutlineLib.u16(p, P_OY) + CutlineLib.u16(p, P_OH) > hc
                || CutlineLib.u16(p, P_OX) + CutlineLib.u16(p, P_OW) > wc;
        }
        uint256 l = CutlineLib.u8(p, P_LSTAR);
        if (i == 7) return l > CutlineLib.u8(s.cutPre, 4);
        if (i == 6) {
            uint256 hs = (CutlineLib.u16(s.cutPre, 0) + (uint256(1) << l) - 1) >> l;   // ceil(H0 / 2^l*)
            uint256 ws = (CutlineLib.u16(s.cutPre, 2) + (uint256(1) << l) - 1) >> l;
            return CutlineLib.u16(p, P_Y0) + CutlineLib.u16(p, P_CH) > hs
                || CutlineLib.u16(p, P_X0) + CutlineLib.u16(p, P_CW) > ws;
        }
        // i == 8
        uint256 pp = sampling == 0 ? 8 : 16;
        return _d(p, j) >= (hc / pp) * (wc / pp);
    }

    function _d(bytes calldata p, uint256 j) private pure returns (uint256) {
        return CutlineLib.u32(p, P_D + 4 * j);
    }
}
