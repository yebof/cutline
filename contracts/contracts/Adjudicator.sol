// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {IClaimRegistry} from "./interfaces/IClaimRegistry.sol";
import {IAdjudicator} from "./interfaces/IAdjudicator.sol";
import {ClaimCore, Codes} from "./lib/Types.sol";
import {Consts} from "./lib/Consts.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 Adjudicator: path (b) (challenges, votes, the tally, the f+2 appeal) and the write-once
/// registries (encoder classes, tier-T tables with successor pointers, declared kernel families, EP entries). The
/// court (BlockCourt) and the static proofs (StaticRules) are called directly and read the registries through
/// IAdjudicator.
/// - Governance acts (setClass, setTierTable, setKernelFamily, setEPEntry) are called by each validator with the same
///   arguments; the Adjudicator forwards msg.sender to `Governance.vote(subject, validator)` and executes when the
///   2f+1 quorum is reached. The registries are write-once, so a subject is simply the act's tag
///   and arguments. The precondition checks run on every vote, so a late vote after execution reverts
///   E_REG_DEFINED.
/// - The genesis state is written by the constructor: classes {1: (EXACT, 0), 3: (CELL, 52,429), 5: (0, 262,144),
///   7: (0, 0)} and kernel families 0-4 with exact kernels (0, 1, 1, 1, 1).
/// - Path (b) windows: one open CHALLENGE per target (`latestChallenge`), one open APPEAL per target (`latestAppeal`).
///   A window is open iff now < openedAt + W_vote; `now` is block.number.
contract Adjudicator is IAdjudicator {
    IGovernance public immutable gov;
    IClaimRegistry public immutable reg;

    struct Cls { bool defined; uint8 kind; uint32 eta20; }
    struct Kfam { bool defined; uint8 exact; }
    struct TierT {
        bool defined;
        uint8 kernelFamily;
        uint8 sampling;
        uint8 codec;
        uint64 successorHeight;   // 0 = current
        uint64 setHeight;
        bytes record;             // the 55-byte record
    }
    struct EP { bytes32 buildSha256; bytes32 paramsSha256; }
    /// Storage form of a path-(b) window: the `Challenge` struct (Types.sol) with kind, openedAt and finalised packed
    /// into one slot (3 slots instead of 4); `challenges(id)` returns the `Challenge` fields in their order.
    struct Win { bytes32 target; bytes32 pk; uint64 openedAt; uint8 kind; bool finalised; }

    // ---- registries
    mapping(uint16 => Cls) internal _cls;
    mapping(uint16 => TierT) internal _tierT;
    mapping(uint24 => uint16) internal _currentTierT;   // profile key (family << 16 | sampling << 8 | codec) -> id
    mapping(uint8 => Kfam) internal _kfam;
    mapping(uint32 => EP) public ep;

    // ---- path (b)
    uint64 public challengeCount;
    mapping(uint64 => Win) internal _wins;
    mapping(bytes32 => uint64) public latestChallenge;           // target -> id of its last CHALLENGE window
    mapping(bytes32 => uint64) public latestAppeal;              // target -> id of its last APPEAL window
    mapping(bytes32 => bool) public passed;                      // target passed at >= 2f+1 PASS
    mapping(bytes32 => uint64) public lastUndecided;             // target -> openedAt of its last undecided challenge
    mapping(uint64 => uint32) public passCount;
    mapping(uint64 => uint32) public maxFail;
    mapping(uint64 => mapping(bytes32 => uint32)) public failCount;
    mapping(uint64 => mapping(address => bool)) public voted;

    event ChallengeOpened(uint64 indexed id, bytes32 indexed target, string bundleLocator);
    event Verdict(uint64 indexed id, address indexed validator, uint8 status, bytes32 transcript);
    event ChallengeFinalized(uint64 indexed id, uint8 outcome);
    event AppealOpened(uint64 indexed id, bytes32 indexed target);
    event AppealFinalized(uint64 indexed id, bool cleared);
    event ClassSet(uint16 indexed id, uint8 kind, uint32 eta20);
    event TierTableSet(uint16 indexed id, uint16 predecessor, bytes32 tableDigest);
    event KernelFamilySet(uint8 indexed f, uint8 exact, string name);
    event EPEntrySet(uint32 indexed id, bytes32 buildSha256, bytes32 paramsSha256);

    uint256 private constant TIER_T_RECORD_LEN = 55;
    uint256 private constant HEADER_LEN = 197;

    constructor(IGovernance g, IClaimRegistry r) {
        gov = g;
        reg = r;
        _cls[1] = Cls(true, Codes.CLASS_EXACT, 0);
        _cls[3] = Cls(true, Codes.CLASS_CELL, 52_429);
        _cls[5] = Cls(true, Codes.CLASS_NONE, 262_144);
        _cls[7] = Cls(true, Codes.CLASS_NONE, 0);
        _kfam[0] = Kfam(true, 0);
        for (uint8 f = 1; f <= 4; ++f) _kfam[f] = Kfam(true, 1);
    }

    // =====================================================================================================
    // Registry views (IAdjudicator and the registries)
    // =====================================================================================================

    /// challenges[id] = (kind, target, pk, openedAt, finalised); an unknown id gives zeros.
    function challenges(uint64 id)
        external view returns (uint8 kind, bytes32 target, bytes32 pk, uint64 openedAt, bool finalised)
    {
        Win storage w = _wins[id];
        return (w.kind, w.target, w.pk, w.openedAt, w.finalised);
    }

    function cls(uint16 id) external view returns (bool, uint8, uint32) {
        Cls storage c = _cls[id];
        return (c.defined, c.kind, c.eta20);
    }

    function kfam(uint8 f) external view returns (bool, uint8) {
        Kfam storage k = _kfam[f];
        return (k.defined, k.exact);
    }

    /// tierT[id] = (record, successorHeight); an undefined id gives ("", 0).
    function tierT(uint16 id) external view returns (bytes memory record, uint64 successorHeight) {
        TierT storage t = _tierT[id];
        return (t.record, t.successorHeight);
    }

    /// Height at which setTierTable stored table id (0 = undefined).
    function tierTSetHeight(uint16 id) external view returns (uint64) {
        return _tierT[id].setHeight;
    }

    function tierTCurrent(uint16 id) external view returns (bool defined, bool current) {
        TierT storage t = _tierT[id];
        return (t.defined, t.defined && t.successorHeight == 0);
    }

    function tierTProfile(uint16 id) external view returns (bool, uint8, uint8, uint8) {
        TierT storage t = _tierT[id];
        return (t.defined, t.kernelFamily, t.sampling, t.codec);
    }

    /// currentTierT[profile] (0 = the profile has no table yet).
    function currentTierT(uint8 family, uint8 sampling, uint8 codec) external view returns (uint16 id) {
        return _currentTierT[_profileKey(family, sampling, codec)];
    }

    // =====================================================================================================
    // Registry acts (governance quorum; write-once)
    // =====================================================================================================

    /// cls[id] = (kind, eta20), write-once (E_REG_DEFINED); kind in {0, 1, 2} (E_REG_PROFILE).
    function setClass(uint16 id, uint8 kind, uint32 eta20) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (_cls[id].defined) revert E_REG_DEFINED();
        if (kind > Codes.CLASS_CELL) revert E_REG_PROFILE();
        if (gov.vote(keccak256(abi.encode("adj.setClass", id, kind, eta20)), msg.sender)) {
            _cls[id] = Cls(true, kind, eta20);
            emit ClassSet(id, kind, eta20);
        }
    }

    /// record = u16 id ‖ u16 predecessor ‖ u8 kernelFamily ‖ u8 sampling ‖ u8 codec ‖ 9 × u32 T ‖ u32 G4 ‖
    /// u32 G8 ‖ u32 noiseMilli. Check order: TABLE_LEN, DEFINED, PROFILE (incl. id in [1024, 65535]),
    /// PREDECESSOR (= currentTierT[profile], re-checked at execution).
    function setTierTable(bytes calldata record) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (record.length != TIER_T_RECORD_LEN) revert E_REG_TABLE_LEN();
        uint16 id = uint16(CutlineLib.u16(record, 0));
        uint16 pred = uint16(CutlineLib.u16(record, 2));
        uint8 fam = uint8(CutlineLib.u8(record, 4));
        uint8 sampling = uint8(CutlineLib.u8(record, 5));
        uint8 codec = uint8(CutlineLib.u8(record, 6));
        if (_tierT[id].defined) revert E_REG_DEFINED();
        if (id < Consts.TIER_T_MIN_ID || !_kfam[fam].defined || sampling > 1 || codec != 0) revert E_REG_PROFILE();
        uint24 pk = _profileKey(fam, sampling, codec);
        if (pred != _currentTierT[pk]) revert E_REG_PREDECESSOR();
        if (gov.vote(keccak256(abi.encode("adj.setTierTable", record)), msg.sender)) {
            _tierT[id] = TierT(true, fam, sampling, codec, 0, uint64(block.number), record);
            if (pred != 0) _tierT[pred].successorHeight = uint64(block.number);
            _currentTierT[pk] = id;
            emit TierTableSet(id, pred, CutlineLib.tierTDigest(record));
        }
    }

    /// A declared kernel family f >= 5 (0-4 are genesis), write-once; exact in {0, 1}.
    function setKernelFamily(uint8 f, uint8 exact, string calldata name) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (_kfam[f].defined) revert E_REG_DEFINED();
        if (exact > 1) revert E_REG_PROFILE();
        if (gov.vote(keccak256(abi.encode("adj.setKernelFamily", f, exact, name)), msg.sender)) {
            _kfam[f] = Kfam(true, exact);
            emit KernelFamilySet(f, exact, name);
        }
    }

    /// ep[id] = (buildSha256, paramsSha256), id >= 1, write-once. Id 0 ("no entry") counts as defined.
    function setEPEntry(uint32 id, bytes32 buildSha256, bytes32 paramsSha256) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (id == 0 || ep[id].buildSha256 != bytes32(0) || ep[id].paramsSha256 != bytes32(0)) revert E_REG_DEFINED();
        if (buildSha256 == bytes32(0) && paramsSha256 == bytes32(0)) revert E_REG_PROFILE();
        if (gov.vote(keccak256(abi.encode("adj.setEPEntry", id, buildSha256, paramsSha256)), msg.sender)) {
            ep[id] = EP(buildSha256, paramsSha256);
            emit EPEntrySet(id, buildSha256, paramsSha256);
        }
    }

    // =====================================================================================================
    // Path (b): challenges
    // =====================================================================================================

    function openChallenge(bytes32 nf, address claimant, ClaimCore calldata core, string calldata bundleLocator,
                           bytes calldata header, bytes32 rMcu) external returns (uint64 id) {
        _requireWatcherOrValidator();
        bytes32 target = CutlineLib.claimId(nf, claimant, core);
        (bool exists,,,,,, bytes32 pk, bool courtFailed, bool quorumFailed, bool cleared) = reg.claimInfo(target);
        if (!exists) revert E_CHAL_UNKNOWN();
        if (core.tier != Codes.TIER_T) {
            // 4:2:0 eligibility: the header binds to core.rY and names 4:2:0 (header[4] = 1)
            if (header.length != HEADER_LEN || CutlineLib.u8(header, 4) != 1
                || CutlineLib.listingRoot(header, rMcu) != core.rY) revert E_CHAL_NOT_PATH_B();
        }
        uint64 c0 = latestChallenge[target];
        if (c0 != 0) {
            Win storage prev = _wins[c0];
            if (block.number < uint256(prev.openedAt) + Consts.W_VOTE) revert E_CHAL_OPEN();
            if (!prev.finalised) {
                _finalize(c0, prev);   // implicit finalisation; may strike, so re-read the flags
                (,,,,,,, courtFailed, quorumFailed, cleared) = reg.claimInfo(target);
            }
        }
        if (courtFailed || quorumFailed || cleared) revert E_CHAL_DECIDED();
        if (passed[target]) revert E_CHAL_PASSED();
        uint64 lu = lastUndecided[target];
        if (lu != 0 && block.number - lu < Consts.W_REG) revert E_CHAL_UNDECIDED_RATE();
        id = ++challengeCount;
        _wins[id] = Win(target, pk, uint64(block.number), Codes.KIND_CHALLENGE, false);
        latestChallenge[target] = id;
        reg.extendBlock(pk, uint64(block.number) + Consts.W_VOTE);
        emit ChallengeOpened(id, target, bundleLocator);
    }

    /// A validator's vote on an open window. An id that names no window reverts E_CHAL_UNKNOWN.
    function submitVerdict(uint64 id, uint8 status, bytes32 transcript) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();
        Win storage c = _get(id);
        if (block.number >= uint256(c.openedAt) + Consts.W_VOTE) revert E_VOTE_WINDOW();
        if (voted[id][msg.sender]) revert E_VOTE_DUP();
        if (status == Codes.VERDICT_PASS) {
            if (transcript != CutlineLib.passTranscript(c.target)) revert E_VOTE_PASS_TRANSCRIPT();
            ++passCount[id];
        } else if (status == Codes.VERDICT_FAIL) {
            if (c.kind != Codes.KIND_CHALLENGE || transcript == bytes32(0) || transcript == Consts.EMPTY_ROWS) {
                revert E_VOTE_FAIL_TRANSCRIPT();
            }
            uint32 n = ++failCount[id][transcript];
            if (n > maxFail[id]) maxFail[id] = n;
        } else {
            revert E_VOTE_STATUS();
        }
        voted[id][msg.sender] = true;
        emit Verdict(id, msg.sender, status, transcript);
    }

    /// Finalises a window after it closes (anyone). A finalised window, or one still open, reverts E_CHAL_WINDOW;
    /// an APPEAL id E_CHAL_UNKNOWN.
    function finalizeChallenge(uint64 id) external {
        Win storage c = _get(id);
        if (c.kind != Codes.KIND_CHALLENGE) revert E_CHAL_UNKNOWN();
        if (c.finalised || block.number < uint256(c.openedAt) + Consts.W_VOTE) revert E_CHAL_WINDOW();
        _finalize(id, c);
    }

    // =====================================================================================================
    // Path (b): the f+2 appeal
    // =====================================================================================================

    function openAppeal(bytes32 target) external returns (uint64 id) {
        _requireWatcherOrValidator();
        (,,,,,,, bool courtFailed, bool quorumFailed,) = reg.claimInfo(target);
        if (!quorumFailed || courtFailed) revert E_APPEAL_NOT_QUORUM_FAILED();
        uint64 a0 = latestAppeal[target];
        if (a0 != 0 && block.number < uint256(_wins[a0].openedAt) + Consts.W_VOTE) revert E_APPEAL_OPEN();
        id = ++challengeCount;
        _wins[id] = Win(target, bytes32(0), uint64(block.number), Codes.KIND_APPEAL, false);
        latestAppeal[target] = id;
        emit AppealOpened(id, target);
    }

    /// Anyone, after the window: >= f+2 PASS (f of the set at finalisation) clears the quorum failure. The clear is
    /// skipped (cleared = false) if the target is no longer quorum-failed (e.g. an earlier appeal already cleared it)
    /// or has been court-failed since, so every appeal window can be finalised.
    function finalizeAppeal(uint64 id) external {
        Win storage c = _get(id);
        if (c.kind != Codes.KIND_APPEAL) revert E_CHAL_UNKNOWN();
        if (c.finalised || block.number < uint256(c.openedAt) + Consts.W_VOTE) revert E_CHAL_WINDOW();
        c.finalised = true;
        bool cleared;
        if (passCount[id] >= gov.f() + 2) {
            (,,,,,,, bool courtFailed, bool quorumFailed,) = reg.claimInfo(c.target);
            if (quorumFailed && !courtFailed) {
                reg.clearQuorumFailed(c.target);
                cleared = true;
            }
        }
        emit AppealFinalized(id, cleared);
    }

    // =====================================================================================================
    // Internals
    // =====================================================================================================

    /// Finalize(c), with n and f of the validator set at finalisation.
    function _finalize(uint64 id, Win storage c) internal {
        uint256 f = (gov.validatorCount() - 1) / 3;
        uint256 pc = passCount[id];
        uint8 outcome;
        c.finalised = true;
        if (maxFail[id] >= f + 1 && pc <= f) {
            reg.strike(c.target, false);
            outcome = Codes.OUTCOME_FAILED;
        } else if (pc >= 2 * f + 1) {
            passed[c.target] = true;
            outcome = Codes.OUTCOME_PASSED;
        } else {
            lastUndecided[c.target] = c.openedAt;
            outcome = Codes.OUTCOME_UNDECIDED;
        }
        emit ChallengeFinalized(id, outcome);
    }

    function _get(uint64 id) internal view returns (Win storage c) {
        if (id == 0 || id > challengeCount) revert E_CHAL_UNKNOWN();
        c = _wins[id];
    }

    function _requireWatcherOrValidator() internal view {
        if (!gov.hasRole(Consts.ROLE_WATCHER, msg.sender) && !gov.isValidator(msg.sender)) revert E_ROLE();
    }

    function _profileKey(uint8 family, uint8 sampling, uint8 codec) internal pure returns (uint24) {
        return (uint24(family) << 16) | (uint24(sampling) << 8) | uint24(codec);
    }
}
