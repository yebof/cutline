// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// CaptureAnchor view interface: anchored capture batches.
interface ICaptureAnchor {
    /// MerkleVerify against batch batchId; returns its anchorHeight h_a. Reverts E_ANCHOR_UNKNOWN_BATCH,
    /// E_ANCHOR_INCLUSION.
    function checkInclusion(uint32 batchId, bytes32 leaf, uint32 leafIndex, bytes32[] calldata path) external view returns (uint64);
    /// batches[id] = (root, beacon, beaconHeight, anchorHeight, count).
    function batches(uint32 id) external view returns (bytes32, bytes32, uint64, uint64, uint32);
}
