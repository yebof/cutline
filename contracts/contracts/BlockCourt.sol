// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {ProofIn} from "./lib/Types.sol";
import {Consts} from "./lib/Consts.sol";
import {CourtCore} from "./lib/CourtCore.sol";
import {IGovernance} from "./interfaces/IGovernance.sol";
import {IClaimRegistry} from "./interfaces/IClaimRegistry.sol";
import {IAdjudicator} from "./interfaces/IAdjudicator.sol";
import "./lib/Errors.sol";

/// @notice Path (a): the block court. `proveBlockFailure` re-executes one coefficient (mcu, comp, k) of a
/// tier-E, 4:4:4 claim of class kind EXACT or CELL on chain and strikes the claim's principal when it fails
/// (ClaimRegistry.strike(claimId, byCourt = true)). Called directly, not through the Adjudicator;
/// selector 0x00aa6b4d. Deployed as BlockCourt(gov, claimRegistry, adjudicator) (scripts/lib/deploy.js).
contract BlockCourt is CourtCore {
    IGovernance public immutable gov;

    event BlockFailure(bytes32 indexed claimId, uint32 mcu, uint8 comp, uint8 k, int16 c, int256 w, address challenger);

    constructor(IGovernance gov_, IClaimRegistry reg_, IAdjudicator adj_) CourtCore(reg_, adj_) {
        gov = gov_;
    }

    /// Caller WATCHER or validator (E_ROLE), steps 1-19 (E_COURT_*; E_COURT_CONSISTENT when the block passes),
    /// then step 20: strike and BlockFailure.
    function proveBlockFailure(ProofIn calldata p) external returns (bytes32 claimId) {
        if (!gov.hasRole(Consts.ROLE_WATCHER, msg.sender) && !gov.isValidator(msg.sender)) revert E_ROLE();
        (bytes32 id, , int256 c, , int256 w, bool fails) = _check(p);
        if (!fails) revert E_COURT_CONSISTENT();
        claimId = id;
        reg.strike(claimId, true);
        emit BlockFailure(claimId, p.mcu, p.comp, p.k, int16(c), w, msg.sender);
    }
}
