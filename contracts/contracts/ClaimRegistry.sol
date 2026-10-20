// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {IDeviceRegistry} from "./interfaces/IDeviceRegistry.sol";
import {ICaptureAnchor} from "./interfaces/ICaptureAnchor.sol";
import {IClaimRegistry} from "./interfaces/IClaimRegistry.sol";
import {IAdjudicator} from "./interfaces/IAdjudicator.sol";
import {ClaimCore, FirstIn, Codes} from "./lib/Types.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import {Consts} from "./lib/Consts.sol";
import {RelayVerifier} from "./lib/Relay.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 captures, claims, principals, strikes, delegations, exclusive grants, transfers.
/// - Check order: CapValid, then the claim part, then the principal, grant, transfer and withdraw rules.
/// - Owner, recipient and claimant transactions are signed relays: canonical arguments,
///   then (address signer, bytes sig); the relay digest names the canonical selector.
/// - `blockedUntil[pk]` lives here, written by the Adjudicator's openChallenge via extendBlock.
/// Deployed first and wired by a one-time init (scripts/lib/deploy.js).
contract ClaimRegistry is IClaimRegistry, RelayVerifier {
    /// captures[nf]; heights are u32. 3 slots:
    /// (owner, registeredAt, batchId, timeUnverified); keyId; (epoch, lastTransfer, licCount).
    struct Capture {
        address owner;
        uint32 registeredAt;
        uint32 batchId;
        bool timeUnverified;
        bytes32 keyId;
        uint32 epoch;
        uint32 lastTransfer;
        uint32 licCount;
    }

    /// claims[claimId]. 3 slots: (claimant, height, status, tier, marketplace, kind); nf;
    /// (principal id, courtFailed, quorumFailed, cleared). principalKey = H(nf ‖ kind ‖ id) is recomputed, not stored.
    struct Claim {
        address claimant;
        uint64 height;
        uint8 status;
        uint8 tier;
        uint8 marketplace;
        uint8 kind;
        bytes32 nf;
        uint32 id;
        bool courtFailed;
        bool quorumFailed;
        bool cleared;
    }

    /// strikes[principalKey] (u32 quorumStrikes). 1 slot.
    struct Strike {
        bool courtStruck;
        uint32 quorumStrikes;
    }

    /// delegations[nf][d]. 1 slot.
    struct Delegation {
        uint32 epoch;
        uint64 scope;
        uint64 expiry;
        bool revoked;
    }

    /// exclusiveOf[nf][m]. 1 slot.
    struct Exclusive {
        address licensee;
        uint32 licId;
        uint64 expiry;
    }

    bytes4 internal constant SEL_DELEGATE = bytes4(keccak256("delegate(bytes32,address,uint64,bool,uint64,bytes32[])"));
    bytes4 internal constant SEL_REVOKE = bytes4(keccak256("revokeDelegation(bytes32,address)"));
    bytes4 internal constant SEL_OFFER = bytes4(keccak256("offerTransfer(bytes32,address)"));
    bytes4 internal constant SEL_ACCEPT = bytes4(keccak256("acceptTransfer(bytes32)"));
    bytes4 internal constant SEL_WITHDRAW = bytes4(keccak256("withdraw(bytes32)"));

    uint8 internal constant P_OK = 0;
    uint8 internal constant P_EXCLUSIVE = 1;
    uint8 internal constant P_NOT_AUTHORISED = 2;

    address internal immutable _deployer;
    IGovernance public gov;
    IDeviceRegistry public dev;
    ICaptureAnchor public anchor;
    address public adjudicator;
    address public staticRules;
    address public blockCourt;

    mapping(bytes32 => Capture) public captures;
    mapping(bytes32 => Claim) public claims;
    mapping(bytes32 => Strike) public strikes;
    mapping(bytes32 => mapping(address => Delegation)) public delegations;
    mapping(bytes32 => mapping(uint8 => Exclusive)) public exclusiveOf;
    mapping(bytes32 => mapping(uint8 => uint64)) public lastExclusiveGrant;
    mapping(bytes32 => mapping(uint8 => uint32)) public activeOwn;
    mapping(bytes32 => uint64) public blockedUntil;
    mapping(bytes32 => address) public pendingTransfer;

    event CaptureRegistered(bytes32 indexed nf, address indexed owner, bytes32 indexed keyId, uint32 batchId);
    event ClaimRegistered(bytes32 indexed claimId, bytes32 indexed nf, address indexed claimant, ClaimCore core, bytes32 principalKey);
    event Struck(bytes32 indexed nf, bytes32 indexed claimId, bytes32 principalKey, bool byCourt);
    event Cleared(bytes32 indexed claimId);
    event Delegated(bytes32 indexed nf, address indexed to, uint64 scope, uint64 expiry, bool exclusive, uint32 licId);
    event DelegationRevoked(bytes32 indexed nf, address indexed d);
    event TransferOffered(bytes32 indexed nf, address indexed to);
    event Transferred(bytes32 indexed nf, address oldOwner, address newOwner, uint32 epoch);
    event StatusChanged(bytes32 indexed claimId, uint8 status);
    event BlockExtended(bytes32 indexed pk, uint64 until);

    constructor() {
        _deployer = msg.sender;
    }

    /// One-time wiring by the deployer.
    function init(IGovernance g, IDeviceRegistry d, ICaptureAnchor a, address adj, address sr, address bc) external {
        if (msg.sender != _deployer || address(gov) != address(0)) revert E_INIT();
        gov = g;
        dev = d;
        anchor = a;
        adjudicator = adj;
        staticRules = sr;
        blockCourt = bc;
    }

    // =================================================================================================================
    // Registration
    // =================================================================================================================

    /// Catalogue shoots: a relayer of some platforms[m] or AGGREGATOR (E_ROLE); idempotent.
    function registerCapture(FirstIn calldata f, bytes calldata sigHold) external returns (bytes32 nf) {
        if (!gov.isAnyRelayer(msg.sender) && !gov.hasRole(Consts.ROLE_AGGREGATOR, msg.sender)) revert E_ROLE();
        (nf,) = _registerCapture(f, sigHold);
    }

    function registerFirst(FirstIn calldata f, bytes calldata sigHold, ClaimCore calldata core, bytes calldata sigClaim)
        external returns (bytes32)
    {
        (bytes32 nf, bytes32 rh) = _registerCapture(f, sigHold);
        if (core.recHash != rh) revert E_CLAIM_REC();
        return _claimPart(nf, core, sigClaim);
    }

    function registerMore(ClaimCore calldata core, bytes calldata sigClaim) external returns (bytes32) {
        bytes32 nf = CutlineLib.nf(core.recHash);
        if (captures[nf].owner == address(0)) revert E_CLAIM_UNKNOWN_CAPTURE();
        return _claimPart(nf, core, sigClaim);
    }

    /// RegisterCapture: length, idempotence, then CapValid checks 1-8.
    function _registerCapture(FirstIn calldata f, bytes calldata sigHold) internal returns (bytes32 nf, bytes32 rh) {
        if (f.rec.length != CutlineLib.REC_LEN) revert E_REC_LEN();
        rh = sha256(f.rec);
        nf = CutlineLib.nf(rh);
        if (captures[nf].owner != address(0)) return (nf, rh);
        // 1. rec words
        CutlineLib.checkRec(f.rec);
        bytes32 keyId = CutlineLib.recKeyId(f.rec);
        // 2. the device key
        (bytes32 pkHash_, address holder,, uint16 rateN, uint8 state, uint64 tc) = dev.devices(keyId);
        if (pkHash_ != CutlineLib.pkHash(f.pkx, f.pky)) revert E_CAP_PK();
        // 3. low-s, then the device signature
        if (!CutlineLib.lowS(f.s)) revert E_CAP_HIGH_S();
        if (!CutlineLib.p256Verify(rh, f.r, f.s, f.pkx, f.pky)) revert E_CAP_DEVICE_SIG();
        // 4. rate slot and epoch
        uint64 hBeta = CutlineLib.recHBeta(f.rec);
        uint32 epoch = CutlineLib.recEpoch(f.rec);
        uint16 slot = CutlineLib.recSlot(f.rec);
        if (slot >= rateN || epoch != hBeta / Consts.RATE_EPOCH) revert E_CAP_SLOT();
        // 5. inclusion of the recomputed leaf v2
        uint64 ha = anchor.checkInclusion(
            f.batchId, CutlineLib.leafV2(rh, f.r, f.s, CutlineLib.slotTag(keyId, epoch, slot)), f.leafIndex, f.path);
        // 6. the beacon bracket
        (, bytes32 beacon, uint64 beaconHeight,,) = anchor.batches(f.batchId);
        if (beacon != CutlineLib.recBeta(f.rec) || beaconHeight != hBeta) revert E_CAP_BRACKET();
        // 7. validAt(keyId, h_a), from the device record read in step 2
        if (!(state == Codes.DEV_ACTIVE && (tc == 0 || ha < tc))) revert E_CAP_KEY_INVALID();
        // 8. the vouch σ_hold
        address h = CutlineLib.recover65(CutlineLib.vouchDigest(block.chainid, address(this), nf), sigHold);
        if (h == address(0) || h != holder) revert E_CAP_HOLDER();
        // 9. time-unverified after W_reg
        address owner = CutlineLib.recOwner(f.rec);
        Capture storage c = captures[nf];
        c.owner = owner;
        c.registeredAt = uint32(block.number);
        c.batchId = f.batchId;
        c.timeUnverified = block.number > uint256(ha) + Consts.W_REG;
        c.keyId = keyId;
        emit CaptureRegistered(nf, owner, keyId, f.batchId);
    }

    /// ClaimPart: the claim checks after the capture is registered.
    function _claimPart(bytes32 nf, ClaimCore calldata core, bytes calldata sigClaim) internal returns (bytes32 claimId) {
        uint8 m = core.marketplace;
        if (m >= Consts.MARKETPLACES) revert E_CLAIM_MARKETPLACE();
        if (!gov.isRelayer(m, msg.sender)) revert E_ROLE();
        _rule8(core.tier, core.encClass);
        ClaimCore memory cm = core;
        address claimant = CutlineLib.recover65(CutlineLib.claimDigest(block.chainid, address(this), cm), sigClaim);
        if (claimant == address(0)) revert E_CLAIM_SIG();
        (uint8 err, uint8 kind, uint32 id) = _principal(nf, claimant, m);
        if (err == P_EXCLUSIVE) revert E_PRINCIPAL_EXCLUSIVE();
        if (err == P_NOT_AUTHORISED) revert E_PRINCIPAL_NOT_AUTHORISED();
        bytes32 pk = CutlineLib.principalKey(nf, kind, id);
        if (_struck(pk)) revert E_PRINCIPAL_STRUCK();
        claimId = CutlineLib.claimId(nf, claimant, cm);
        Claim storage cl = claims[claimId];
        if (cl.claimant != address(0)) revert E_CLAIM_DUP();
        cl.claimant = claimant;
        cl.height = uint64(block.number);
        cl.status = Codes.STATUS_ACTIVE;
        cl.tier = core.tier;
        cl.marketplace = m;
        cl.kind = kind;
        cl.nf = nf;
        cl.id = id;
        if (kind == Codes.PRINCIPAL_OWN) ++activeOwn[pk][m];
        emit ClaimRegistered(claimId, nf, claimant, cm, pk);
    }

    /// Rule 8 v2 at registration: (tier E, encClass 1) or (tier T, encClass >= 1024, the table is
    /// defined and current).
    function _rule8(uint8 tier, uint16 encClass) internal view {
        if (tier == Codes.TIER_E) {
            if (encClass == 1) return;
        } else if (tier == Codes.TIER_T && encClass >= Consts.TIER_T_MIN_ID) {
            (bool defined, bool current) = IAdjudicator(adjudicator).tierTCurrent(encClass);
            if (defined && current) return;
        }
        revert E_CLAIM_RULE8();
    }

    function claimDigest(ClaimCore calldata core) external view returns (bytes32) {
        return CutlineLib.claimDigest(block.chainid, address(this), core);
    }

    // =================================================================================================================
    // Principals
    // =================================================================================================================

    /// Non-reverting principal(nf, who, m) at the current state: (P_OK | P_EXCLUSIVE | P_NOT_AUTHORISED, kind, id).
    function _principal(bytes32 nf, address who, uint8 m) internal view returns (uint8 err, uint8 kind, uint32 id) {
        Exclusive storage x = exclusiveOf[nf][m];
        if (x.licensee != address(0) && x.expiry >= block.number) {
            if (who == x.licensee) return (P_OK, Codes.PRINCIPAL_LIC, x.licId);
            return (P_EXCLUSIVE, 0, 0);
        }
        Capture storage c = captures[nf];
        uint32 e = c.epoch;
        if (who == c.owner) return (P_OK, Codes.PRINCIPAL_OWN, e);
        Delegation storage d = delegations[nf][who];
        if (d.epoch == e && d.expiry != 0 && !d.revoked && d.expiry >= block.number && (uint256(d.scope) >> m) & 1 == 1) {
            return (P_OK, Codes.PRINCIPAL_OWN, e);
        }
        return (P_NOT_AUTHORISED, 0, 0);
    }

    /// principal(nf, who, m); reverts E_PRINCIPAL_EXCLUSIVE / E_PRINCIPAL_NOT_AUTHORISED.
    function principalOf(bytes32 nf, address who, uint8 m) external view returns (uint8 kind, uint32 id) {
        uint8 err;
        (err, kind, id) = _principal(nf, who, m);
        if (err == P_EXCLUSIVE) revert E_PRINCIPAL_EXCLUSIVE();
        if (err == P_NOT_AUTHORISED) revert E_PRINCIPAL_NOT_AUTHORISED();
    }

    function _struck(bytes32 pk) internal view returns (bool) {
        Strike storage s = strikes[pk];
        return s.courtStruck || s.quorumStrikes > 0;
    }

    function _claimPk(Claim storage cl) internal view returns (bytes32) {
        return CutlineLib.principalKey(cl.nf, cl.kind, cl.id);
    }

    // =================================================================================================================
    // Strikes and the challenge block
    // =================================================================================================================

    function strike(bytes32 claimId, bool byCourt) external {
        if (msg.sender != adjudicator && msg.sender != blockCourt && msg.sender != staticRules) revert E_ROLE();
        // The strike kind is bound to the caller; the Adjudicator (path b) strikes quorum-only,
        // BlockCourt / StaticRules (paths a, a') strike final. The flag stays in the selector and must agree.
        if (byCourt != (msg.sender != adjudicator)) revert E_ROLE();
        Claim storage cl = claims[claimId];
        if (cl.claimant == address(0)) revert E_COURT_CLAIM();
        bytes32 pk = _claimPk(cl);
        if (byCourt) {
            cl.courtFailed = true;
            strikes[pk].courtStruck = true;
        } else if (!cl.cleared && !cl.quorumFailed) {
            cl.quorumFailed = true;
            strikes[pk].quorumStrikes += 1;
        }
        emit Struck(cl.nf, claimId, pk, byCourt);
    }

    function clearQuorumFailed(bytes32 claimId) external {
        if (msg.sender != adjudicator) revert E_ROLE();
        Claim storage cl = claims[claimId];
        if (!cl.quorumFailed) revert E_STRIKE_NOT_QUORUM_FAILED();
        cl.quorumFailed = false;
        cl.cleared = true;
        strikes[_claimPk(cl)].quorumStrikes -= 1;
        emit Cleared(claimId);
    }

    function extendBlock(bytes32 pk, uint64 until) external {
        if (msg.sender != adjudicator) revert E_ROLE();
        if (until > blockedUntil[pk]) blockedUntil[pk] = until;
        emit BlockExtended(pk, until);
    }

    // =================================================================================================================
    // Delegations and exclusive grants; signed relays by the owner
    // =================================================================================================================

    function delegate(bytes32 nf, address to, uint64 scope, bool exclusive, uint64 expiry, bytes32[] calldata withdraw_,
        address signer, bytes calldata sig) external
    {
        _relay(SEL_DELEGATE, keccak256(abi.encode(nf, to, scope, exclusive, expiry, withdraw_)), signer, sig);
        Capture storage c = captures[nf];
        if (signer != c.owner) revert E_OWNER();
        uint32 e = c.epoch;
        bytes32 pk = CutlineLib.principalKey(nf, Codes.PRINCIPAL_OWN, e);
        if (_struck(pk)) revert E_PRINCIPAL_STRUCK();
        if (scope == 0) revert E_DELEG_SCOPE();
        uint32 licId;
        if (!exclusive) {
            if (withdraw_.length != 0 || expiry < block.number) revert E_DELEG_ARGS();
            delegations[nf][to] = Delegation(e, scope, expiry, false);
        } else {
            if (to == address(0)) revert E_DELEG_ARGS();        // an exclusive licence needs a licensee
            licId = _grant(nf, c, pk, e, to, scope, expiry, withdraw_);
        }
        emit Delegated(nf, to, scope, expiry, exclusive, licId);
    }

    /// The exclusive grant (withdraws the owner principal's ACTIVE claims on the scope's marketplaces).
    function _grant(bytes32 nf, Capture storage c, bytes32 pk, uint32 e, address to, uint64 scope, uint64 expiry,
        bytes32[] calldata withdraw_) internal returns (uint32 licId)
    {
        if (block.number < blockedUntil[pk]) revert E_GRANT_CHALLENGED();
        if (expiry < block.number || expiry > block.number + Consts.LIC_CAP) revert E_GRANT_EXPIRY();
        for (uint256 s = scope; s != 0; s &= s - 1) {
            uint8 m = _lowBit(s);
            Exclusive storage x = exclusiveOf[nf][m];
            if (x.licensee != address(0) && x.expiry >= block.number) revert E_GRANT_TAKEN();
            uint64 last = lastExclusiveGrant[nf][m];
            if (last != 0 && block.number - last < Consts.W_REG) revert E_GRANT_RATE();
        }
        for (uint256 i; i < withdraw_.length; ++i) {
            bytes32 cid = withdraw_[i];
            Claim storage cl = claims[cid];
            if (cl.nf != nf || cl.kind != Codes.PRINCIPAL_OWN || cl.id != e || cl.status != Codes.STATUS_ACTIVE
                || (uint256(scope) >> cl.marketplace) & 1 == 0) revert E_GRANT_WITHDRAW_LIST();
            cl.status = Codes.STATUS_WITHDRAWN;
            --activeOwn[pk][cl.marketplace];
            emit StatusChanged(cid, Codes.STATUS_WITHDRAWN);
        }
        for (uint256 s = scope; s != 0; s &= s - 1) {
            if (activeOwn[pk][_lowBit(s)] != 0) revert E_GRANT_OWNER_CLAIMS();
        }
        licId = c.licCount + 1;
        c.licCount = licId;
        for (uint256 s = scope; s != 0; s &= s - 1) {
            uint8 m = _lowBit(s);
            exclusiveOf[nf][m] = Exclusive(to, licId, expiry);
            lastExclusiveGrant[nf][m] = uint64(block.number);
        }
    }

    /// Index of the lowest set bit of s != 0 (s < 2^64).
    function _lowBit(uint256 s) internal pure returns (uint8 m) {
        uint256 b = s & (~s + 1);
        while (b > 1) {
            b >>= 1;
            ++m;
        }
    }

    function revokeDelegation(bytes32 nf, address d, address signer, bytes calldata sig) external {
        _relay(SEL_REVOKE, keccak256(abi.encode(nf, d)), signer, sig);
        if (signer != captures[nf].owner) revert E_OWNER();
        delegations[nf][d].revoked = true;
        emit DelegationRevoked(nf, d);
    }

    // =================================================================================================================
    // Two-step transfer
    // =================================================================================================================

    function offerTransfer(bytes32 nf, address to, address signer, bytes calldata sig) external {
        _relay(SEL_OFFER, keccak256(abi.encode(nf, to)), signer, sig);
        if (signer != captures[nf].owner) revert E_OWNER();
        pendingTransfer[nf] = to;
        emit TransferOffered(nf, to);
    }

    function acceptTransfer(bytes32 nf, address signer, bytes calldata sig) external {
        _relay(SEL_ACCEPT, keccak256(abi.encode(nf)), signer, sig);
        if (pendingTransfer[nf] != signer) revert E_TRANSFER_NO_OFFER();
        Capture storage c = captures[nf];
        bytes32 pk = CutlineLib.principalKey(nf, Codes.PRINCIPAL_OWN, c.epoch);
        if (block.number < blockedUntil[pk]) revert E_TRANSFER_CHALLENGED();
        if (_struck(pk)) revert E_TRANSFER_STRUCK();
        if (c.lastTransfer != 0 && block.number - c.lastTransfer < Consts.W_REG) revert E_TRANSFER_RATE();
        address old = c.owner;
        uint32 e = c.epoch + 1;
        c.owner = signer;
        c.epoch = e;
        c.lastTransfer = uint32(block.number);
        pendingTransfer[nf] = address(0);
        emit Transferred(nf, old, signer, e);
    }

    // =================================================================================================================
    // withdraw and levelCap
    // =================================================================================================================

    function withdraw(bytes32 claimId, address signer, bytes calldata sig) external {
        _relay(SEL_WITHDRAW, keccak256(abi.encode(claimId)), signer, sig);
        Claim storage cl = claims[claimId];
        if (signer != cl.claimant) revert E_CLAIM_NOT_CLAIMANT();
        if (cl.status != Codes.STATUS_ACTIVE) revert E_CLAIM_NOT_ACTIVE();
        cl.status = Codes.STATUS_WITHDRAWN;
        if (cl.kind == Codes.PRINCIPAL_OWN) --activeOwn[_claimPk(cl)][cl.marketplace];
        emit StatusChanged(claimId, Codes.STATUS_WITHDRAWN);
    }

    function levelCap(bytes32 claimId) external view returns (uint8) {
        Claim storage cl = claims[claimId];
        if (cl.claimant == address(0)) return 0;
        Capture storage cp = captures[cl.nf];
        (,,, uint64 ha,) = anchor.batches(cp.batchId);
        if (!dev.validAt(cp.keyId, ha)) return 0;
        if (_struck(_claimPk(cl)) || cl.courtFailed || cl.quorumFailed || cl.status == Codes.STATUS_WITHDRAWN) return 1;
        (uint8 err, uint8 kind, uint32 id) = _principal(cl.nf, cl.claimant, cl.marketplace);
        if (err != P_OK || kind != cl.kind || id != cl.id) return 1;
        if (cp.timeUnverified) return 2;
        return 4;
    }

    /// (exists, nf, claimant, status, tier, marketplace, principalKey, courtFailed, quorumFailed, cleared).
    function claimInfo(bytes32 claimId)
        external view returns (bool, bytes32, address, uint8, uint8, uint8, bytes32, bool, bool, bool)
    {
        Claim storage cl = claims[claimId];
        if (cl.claimant == address(0)) return (false, 0, address(0), 0, 0, 0, 0, false, false, false);
        return (true, cl.nf, cl.claimant, cl.status, cl.tier, cl.marketplace, _claimPk(cl), cl.courtFailed,
            cl.quorumFailed, cl.cleared);
    }
}
