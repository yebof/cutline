// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {IDeviceRegistry} from "./interfaces/IDeviceRegistry.sol";
import {ICaptureAnchor} from "./interfaces/ICaptureAnchor.sol";
import {Use, Codes} from "./lib/Types.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import {Consts} from "./lib/Consts.sol";
import {RelayVerifier} from "./lib/Relay.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 attested device keys.
/// - `enroll` (ENROLLER) writes a PENDING key; `confirm` (validators) activates it at the governance quorum 2f+1.
/// - Revocation is forward-only: t_c is set once, to the height at which the first revocation evidence is
///   finalised (list tally, double use, retirement); `state` never changes after activation.
/// Deployed second (after Governance) and wired by a one-time init.
contract DeviceRegistry is IDeviceRegistry, RelayVerifier {
    /// 3 slots: pkHash; (holder, enrolledAt, rateN, state); tc.
    struct Device {
        bytes32 pkHash;
        address holder;
        uint64 enrolledAt;
        uint16 rateN;
        uint8 state;
        uint64 tc;
    }

    /// One revocation-list window per key. `round` numbers the windows; counts are keyed by round, so opening
    /// a window resets them in O(1).
    struct RevWindow {
        uint64 openedAt;
        uint32 round;
        uint16 keep;
        uint16 maxRevoke;
        bool finalised;
    }

    bytes4 internal constant SEL_RETIRE = bytes4(keccak256("retireKey(bytes32)"));

    address internal immutable _deployer;
    IGovernance public gov;
    ICaptureAnchor public anchor;

    mapping(bytes32 => Device) public devices;
    mapping(bytes32 => bytes32) public enrolledPk;                  // pkHash -> keyId (unique)
    mapping(bytes32 => RevWindow) public revocationWindow;          // keyId -> current window
    mapping(bytes32 => mapping(uint32 => mapping(address => bool))) public revocationVoted;      // keyId, round, validator
    mapping(bytes32 => mapping(uint32 => mapping(bytes32 => uint16))) public revokeCount;       // keyId, round, entryHash

    event Enrolled(bytes32 indexed keyId, address indexed holder, bytes32 pkHash, bytes32 attHash, uint16 rateN, bytes16 nonceE);
    event Activated(bytes32 indexed keyId, uint64 height);
    event Revoked(bytes32 indexed keyId, uint64 tc, uint8 source);
    event RevocationVoted(bytes32 indexed keyId, address indexed validator, uint32 round, uint8 status, bytes32 entryHash);
    event RevocationFinalized(bytes32 indexed keyId, uint32 round, bool revoke);

    constructor() {
        _deployer = msg.sender;
    }

    /// One-time wiring by the deployer (see scripts/lib/deploy.js for the deploy order).
    function init(IGovernance g, ICaptureAnchor a) external {
        if (msg.sender != _deployer || address(gov) != address(0)) revert E_INIT();
        gov = g;
        anchor = a;
    }

    // ------------------------------------------------------------------ enrolment

    function enroll(bytes32 keyId, bytes32 pkHash, address holder, bytes32 attHash, uint16 rateN, bytes16 nonceE) external {
        if (!gov.hasRole(Consts.ROLE_ENROLLER, msg.sender)) revert E_ROLE();
        // keyId 0 is the "not enrolled" sentinel of enrolledPk, so it can never be a device id
        if (keyId == bytes32(0) || devices[keyId].state != Codes.DEV_UNKNOWN) revert E_DEV_EXISTS();
        if (enrolledPk[pkHash] != bytes32(0)) revert E_DEV_PK_ENROLLED();
        if (rateN < Consts.RATE_N_MIN || rateN > Consts.RATE_N_MAX) revert E_DEV_RATE();
        if (holder == address(0)) revert E_DEV_HOLDER_ZERO();
        devices[keyId] = Device(pkHash, holder, uint64(block.number), rateN, Codes.DEV_PENDING, 0);
        enrolledPk[pkHash] = keyId;
        emit Enrolled(keyId, holder, pkHash, attHash, rateN, nonceE);
    }

    /// Validator vote under keccak256(abi.encode("enrol", keyId, pkHash)); ACTIVE at 2f+1 (Governance.vote reverts
    /// E_ROLE for a non-validator and E_VOTE_DUP for a repeated vote).
    function confirm(bytes32 keyId) external {
        Device storage d = devices[keyId];
        if (d.state != Codes.DEV_PENDING) revert E_DEV_STATE();
        if (gov.vote(CutlineLib.enrolSubject(keyId, d.pkHash), msg.sender)) {
            d.state = Codes.DEV_ACTIVE;
            emit Activated(keyId, uint64(block.number));
        }
    }

    // ------------------------------------------------------------------ validity

    function validAt(bytes32 keyId, uint64 h) external view returns (bool) {
        Device storage d = devices[keyId];
        return d.state == Codes.DEV_ACTIVE && (d.tc == 0 || h < d.tc);
    }

    // ------------------------------------------------------------------ revocation sources

    /// Source (e): signed relay by the holder (canonical retireKey(bytes32)); E_DEV_NOT_HOLDER.
    function retireKey(bytes32 keyId, address signer, bytes calldata sig) external {
        _relay(SEL_RETIRE, keccak256(abi.encode(keyId)), signer, sig);
        if (signer != devices[keyId].holder) revert E_DEV_NOT_HOLDER();
        _setTc(keyId, Codes.REVOKE_RETIRE);
    }

    /// Source (c): both uses pass capture checks 1-3 and 5 under (pkx, pky) with recomputed v2 leaves; then equal
    /// (keyId, epoch, slot) (E_DEV_NOT_SAME_SLOT) and different records (E_DEV_SAME_RECORD). t_c = now, once.
    function proveDoubleUse(bytes32 pkx, bytes32 pky, Use calldata a, Use calldata b) external {
        bytes32 ph = CutlineLib.pkHash(pkx, pky);
        (bytes32 ka, uint256 sa, bytes32 ha) = _checkUse(ph, pkx, pky, a);
        (bytes32 kb, uint256 sb, bytes32 hb) = _checkUse(ph, pkx, pky, b);
        if (ka != kb || sa != sb) revert E_DEV_NOT_SAME_SLOT();
        if (ha == hb) revert E_DEV_SAME_RECORD();
        _setTc(ka, Codes.REVOKE_DOUBLE_USE);
    }

    /// Capture checks 1 (rec), 2 (pk), 3 (low-s, then P256VERIFY) and 5 (inclusion of the recomputed leaf v2).
    /// Returns (keyId, epoch << 16 | slot, recHash).
    function _checkUse(bytes32 ph, bytes32 pkx, bytes32 pky, Use calldata u)
        internal view returns (bytes32 keyId, uint256 slotKey, bytes32 recHash)
    {
        CutlineLib.checkRec(u.rec);
        keyId = CutlineLib.recKeyId(u.rec);
        if (devices[keyId].pkHash != ph) revert E_CAP_PK();
        if (!CutlineLib.lowS(u.s)) revert E_CAP_HIGH_S();
        recHash = sha256(u.rec);
        if (!CutlineLib.p256Verify(recHash, u.r, u.s, pkx, pky)) revert E_CAP_DEVICE_SIG();
        uint32 epoch = CutlineLib.recEpoch(u.rec);
        uint16 slot = CutlineLib.recSlot(u.rec);
        bytes32 leaf = CutlineLib.leafV2(recHash, u.r, u.s, CutlineLib.slotTag(keyId, epoch, slot));
        anchor.checkInclusion(u.batchId, leaf, u.leafIndex, u.path);
        slotKey = (uint256(epoch) << 16) | slot;
    }

    /// Source (b): the revocation-list tally. Validator only (E_ROLE); status REVOKE (1) or KEEP (2), a KEEP
    /// vote carrying H("keep" ‖ keyId) (E_VOTE_STATUS); a known key (E_DEV_STATE); one vote per validator and window
    /// (E_VOTE_DUP). A vote at or after the end of the current window finalises it (if not yet finalised) and opens a
    /// new one at now.
    function voteRevocation(bytes32 keyId, uint8 status, bytes32 entryHash) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();
        if (status == Codes.RVOTE_KEEP) {
            if (entryHash != CutlineLib.keepEntry(keyId)) revert E_VOTE_STATUS();
        } else if (status != Codes.RVOTE_REVOKE) {
            revert E_VOTE_STATUS();
        }
        if (devices[keyId].state == Codes.DEV_UNKNOWN) revert E_DEV_STATE();
        RevWindow storage w = revocationWindow[keyId];
        if (w.openedAt == 0 || block.number >= uint256(w.openedAt) + Consts.W_VOTE) {
            if (w.openedAt != 0 && !w.finalised) _finalizeRevocation(keyId, w);
            w.openedAt = uint64(block.number);
            w.round += 1;
            w.keep = 0;
            w.maxRevoke = 0;
            w.finalised = false;
        }
        uint32 round = w.round;
        if (revocationVoted[keyId][round][msg.sender]) revert E_VOTE_DUP();
        revocationVoted[keyId][round][msg.sender] = true;
        if (status == Codes.RVOTE_KEEP) {
            w.keep += 1;
        } else {
            uint16 c = revokeCount[keyId][round][entryHash] + 1;
            revokeCount[keyId][round][entryHash] = c;
            if (c > w.maxRevoke) w.maxRevoke = c;
        }
        emit RevocationVoted(keyId, msg.sender, round, status, entryHash);
    }

    /// Anyone, after the window (E_VOTE_WINDOW if no window, still open, or already finalised).
    function finalizeRevocation(bytes32 keyId) external {
        RevWindow storage w = revocationWindow[keyId];
        if (w.openedAt == 0 || w.finalised || block.number < uint256(w.openedAt) + Consts.W_VOTE) revert E_VOTE_WINDOW();
        _finalizeRevocation(keyId, w);
    }

    /// REVOKE iff max identical REVOKE count >= f + 1 and KEEP count <= f (f of the set at finalisation).
    function _finalizeRevocation(bytes32 keyId, RevWindow storage w) internal {
        w.finalised = true;
        uint256 f_ = gov.f();
        bool revoke = w.maxRevoke >= f_ + 1 && w.keep <= f_;
        emit RevocationFinalized(keyId, w.round, revoke);
        if (revoke) _setTc(keyId, Codes.REVOKE_LIST);
    }

    /// tc := now only if tc = 0; a later source succeeds and changes nothing.
    function _setTc(bytes32 keyId, uint8 source) internal {
        Device storage d = devices[keyId];
        if (d.tc == 0) {
            d.tc = uint64(block.number);
            emit Revoked(keyId, uint64(block.number), source);
        }
    }
}
