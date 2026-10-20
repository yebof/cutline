// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE consortium membership.
/// - Validators, each with its regulatory bloc; at most one seat per bloc; 4 <= n <= 255.
/// - Roles WATCHER / ENROLLER / AGGREGATOR (Consts.ROLE_*), and relayers: `relayers[a] = {platformId, markets}` where
///   bit m of `markets` lists a in platforms[m] (claims for marketplace m) and platformId (>= 1) is
///   platformOf[a] (hop batches). Each relayer account belongs to exactly one platform.
/// - The tally `vote(subject, validator)` is namespaced by (msg.sender, subject, setVersion): a registry passes its own
///   msg.sender as the validator, so a forged direct call only fills the forger's own namespace and can neither count
///   nor block a registry's vote. The quorum is 2 floor((n-1)/3) + 1 of the current set. Any change of the
///   validator set increments setVersion, which starts every open tally afresh (bitmaps are indexed by the set).
/// - Governance acts (setRole, setRelayer, addValidator, removeValidator) are called by each validator with the same
///   arguments and execute when the quorum is reached. Their subject carries a per-target change counter, so the same
///   act can be voted again after it was reversed. The genesis state is written by the constructor, with no vote.
contract Governance is IGovernance {
    struct RoleGrant { bytes32 role; address account; }
    struct RelayerGrant { address relayer; uint32 platformId; uint64 markets; }
    struct Relayer { uint32 platformId; uint64 markets; }
    struct Tally { uint256 bitmap; bool done; }

    address[] public validators;
    mapping(address => uint256) public validatorIdx1;   // 1-based index into validators, 0 = not a validator
    mapping(address => bytes32) public blocOf;          // the validator's regulatory bloc (e.g. bytes32("EU"))
    mapping(bytes32 => address) public seatOf;          // bloc -> its validator (one seat per bloc)
    mapping(bytes32 => mapping(address => bool)) public hasRole;
    mapping(address => Relayer) public relayers;
    uint64 public setVersion;                           // +1 per validator-set change
    mapping(bytes32 => uint64) public changeNonce;      // per governance target: executed changes
    mapping(bytes32 => Tally) internal _tallies;        // key keccak256(abi.encode(caller, subject, setVersion))

    event Voted(address indexed caller, bytes32 indexed subject, address indexed validator, uint256 count);
    event RoleSet(bytes32 indexed role, address indexed account, bool on);
    event RelayerSet(address indexed relayer, uint32 indexed platformId, uint64 markets);
    event ValidatorAdded(address indexed validator, bytes32 bloc);
    event ValidatorRemoved(address indexed validator, bytes32 bloc);

    constructor(address[] memory vals, bytes32[] memory blocs, RoleGrant[] memory roles, RelayerGrant[] memory rels) {
        if (vals.length != blocs.length || vals.length < 4 || vals.length > 255) revert E_GOV_MIN();
        for (uint256 i; i < vals.length; ++i) _add(vals[i], blocs[i]);
        for (uint256 i; i < roles.length; ++i) _setRole(roles[i].role, roles[i].account, true);
        for (uint256 i; i < rels.length; ++i) _setRelayer(rels[i].relayer, rels[i].platformId, rels[i].markets);
    }

    // ------------------------------------------------------------------ views

    function isValidator(address a) public view returns (bool) { return validatorIdx1[a] != 0; }
    function validatorCount() public view returns (uint256) { return validators.length; }
    function f() public view returns (uint256) { return (validators.length - 1) / 3; }
    function quorum() public view returns (uint256) { return 2 * ((validators.length - 1) / 3) + 1; }
    function isRelayer(uint8 m, address a) external view returns (bool) { return m < 64 && (relayers[a].markets >> m) & 1 == 1; }
    function isAnyRelayer(address a) external view returns (bool) { return relayers[a].markets != 0; }
    function platformOf(address a) external view returns (uint32) { return relayers[a].platformId; }
    function getValidators() external view returns (address[] memory) { return validators; }

    /// (count, done) of the tally of `caller` on `subject` in the current set version.
    function tallyOf(address caller, bytes32 subject) external view returns (uint256 count, bool done) {
        Tally storage t = _tallies[keccak256(abi.encode(caller, subject, setVersion))];
        return (_popcount(t.bitmap), t.done);
    }

    // ------------------------------------------------------------------ the tally

    function vote(bytes32 subject, address validator) external returns (bool reached) {
        return _vote(msg.sender, subject, validator);
    }

    function _vote(address caller, bytes32 subject, address validator) internal returns (bool reached) {
        uint256 i1 = validatorIdx1[validator];
        if (i1 == 0) revert E_ROLE();
        Tally storage t = _tallies[keccak256(abi.encode(caller, subject, setVersion))];
        uint256 bit = uint256(1) << (i1 - 1);
        if (t.bitmap & bit != 0) revert E_VOTE_DUP();
        t.bitmap |= bit;
        uint256 c = _popcount(t.bitmap);
        emit Voted(caller, subject, validator, c);
        if (!t.done && c >= quorum()) {
            t.done = true;
            return true;
        }
        return false;
    }

    // ------------------------------------------------------------------ governance acts (quorum)

    function setRole(bytes32 role, address a, bool on) external {
        bytes32 k = keccak256(abi.encode("role", role, a));
        if (_act(keccak256(abi.encode("gov.setRole", role, a, on, changeNonce[k])))) {
            ++changeNonce[k];
            _setRole(role, a, on);
        }
    }

    function setRelayer(address relayer, uint32 platformId, uint64 markets) external {
        bytes32 k = keccak256(abi.encode("relayer", relayer));
        if (_act(keccak256(abi.encode("gov.setRelayer", relayer, platformId, markets, changeNonce[k])))) {
            ++changeNonce[k];
            _setRelayer(relayer, platformId, markets);
        }
    }

    function addValidator(address a, bytes32 bloc) external {
        bytes32 k = keccak256(abi.encode("validator", a));
        if (_act(keccak256(abi.encode("gov.addValidator", a, bloc, changeNonce[k])))) {
            ++changeNonce[k];
            if (validators.length >= 255) revert E_GOV_MIN();
            _add(a, bloc);
            ++setVersion;
        }
    }

    function removeValidator(address a) external {
        bytes32 k = keccak256(abi.encode("validator", a));
        if (_act(keccak256(abi.encode("gov.removeValidator", a, changeNonce[k])))) {
            ++changeNonce[k];
            _remove(a);
            ++setVersion;
        }
    }

    // ------------------------------------------------------------------ internals

    function _act(bytes32 subject) internal returns (bool) {
        if (validatorIdx1[msg.sender] == 0) revert E_ROLE();
        return _vote(address(this), subject, msg.sender);
    }

    function _add(address a, bytes32 bloc) internal {
        if (a == address(0) || validatorIdx1[a] != 0) revert E_GOV_MEMBER();
        if (bloc == bytes32(0) || seatOf[bloc] != address(0)) revert E_GOV_BLOC();
        validators.push(a);
        validatorIdx1[a] = validators.length;
        blocOf[a] = bloc;
        seatOf[bloc] = a;
        emit ValidatorAdded(a, bloc);
    }

    function _remove(address a) internal {
        uint256 i1 = validatorIdx1[a];
        if (i1 == 0) revert E_GOV_MEMBER();
        if (validators.length <= 4) revert E_GOV_MIN();
        address last = validators[validators.length - 1];
        validators[i1 - 1] = last;
        validatorIdx1[last] = i1;
        validators.pop();
        delete validatorIdx1[a];
        bytes32 bloc = blocOf[a];
        delete seatOf[bloc];
        delete blocOf[a];
        emit ValidatorRemoved(a, bloc);
    }

    function _setRole(bytes32 role, address a, bool on) internal {
        hasRole[role][a] = on;
        emit RoleSet(role, a, on);
    }

    function _setRelayer(address relayer, uint32 platformId, uint64 markets) internal {
        if (relayer == address(0) || (platformId == 0 && markets != 0)) revert E_REG_PLATFORM();
        relayers[relayer] = Relayer(platformId, markets);
        emit RelayerSet(relayer, platformId, markets);
    }

    function _popcount(uint256 b) internal pure returns (uint256 c) {
        for (; b != 0; b &= b - 1) ++c;
    }
}
