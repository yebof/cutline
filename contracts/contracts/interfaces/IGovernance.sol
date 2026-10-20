// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// Governance interface: validators, roles, platform relayers and quorum votes.
/// Roles: Consts.ROLE_WATCHER / ROLE_ENROLLER / ROLE_AGGREGATOR.
interface IGovernance {
    function isValidator(address a) external view returns (bool);
    function hasRole(bytes32 role, address a) external view returns (bool);
    /// a is a relayer of marketplace m (Governance platforms[m]); false for m >= 64.
    function isRelayer(uint8 m, address a) external view returns (bool);
    /// a is a relayer listed in some platforms[m].
    function isAnyRelayer(address a) external view returns (bool);
    /// the platform id (>= 1) of relayer a, 0 = not a platform.
    function platformOf(address a) external view returns (uint32);
    function validatorCount() external view returns (uint256);
    /// f = floor((n - 1) / 3) and the quorum 2f + 1 for the current set.
    function f() external view returns (uint256);
    function quorum() external view returns (uint256);
    /// Counts `validator`'s vote on `subject` in the tally namespaced by (msg.sender, subject, validator-set version);
    /// returns true exactly once, when the quorum is reached. Reverts E_ROLE (not a validator), E_VOTE_DUP.
    function vote(bytes32 subject, address validator) external returns (bool reached);
}
