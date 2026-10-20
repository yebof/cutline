// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// ClaimRegistry interface used by the court contracts.
interface IClaimRegistry {
    /// (exists, nf, claimant, status, tier, marketplace, principalKey, courtFailed, quorumFailed, cleared).
    function claimInfo(bytes32 claimId) external view returns (bool, bytes32, address, uint8, uint8, uint8, bytes32, bool, bool, bool);
    /// Adjudicator | BlockCourt | StaticRules only (E_ROLE).
    function strike(bytes32 claimId, bool byCourt) external;
    /// Adjudicator only (E_STRIKE_NOT_QUORUM_FAILED).
    function clearQuorumFailed(bytes32 claimId) external;
    /// Adjudicator only (openChallenge): blockedUntil[pk] = max(blockedUntil[pk], until).
    function extendBlock(bytes32 pk, uint64 until) external;
}
