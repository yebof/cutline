// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// Adjudicator view interface: encoder classes, tier-T tables and kernel families.
interface IAdjudicator {
    /// cls[id] = (defined, kind, eta20); kind 1 EXACT, 2 CELL, 0 not court-checkable.
    function cls(uint16 id) external view returns (bool, uint8, uint32);
    /// tier-T table id is defined, and current (successorHeight = 0).
    function tierTCurrent(uint16 id) external view returns (bool defined, bool current);
    /// (defined, kernelFamily, sampling, codec) of tier-T table id.
    function tierTProfile(uint16 id) external view returns (bool, uint8, uint8, uint8);
    /// kfam[f] = (defined, exact).
    function kfam(uint8 f) external view returns (bool, uint8);
}
