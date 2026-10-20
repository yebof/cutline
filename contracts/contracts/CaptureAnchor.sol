// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {ICaptureAnchor} from "./interfaces/ICaptureAnchor.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import {Consts} from "./lib/Consts.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 capture anchors.
/// Batches of anchor leaves v2, each bracketed by a boundary beacon h_β ≡ 0 (mod 15) and landing within
/// Delta = 240 blocks of it. Batch ids are 0-based (the number of batches before this one).
contract CaptureAnchor is ICaptureAnchor {
    /// 3 slots: root; beacon; (beaconHeight, anchorHeight, count).
    struct Batch {
        bytes32 root;
        bytes32 beacon;
        uint64 beaconHeight;
        uint64 anchorHeight;
        uint32 count;
    }

    IGovernance public immutable gov;
    Batch[] internal _batches;

    event Anchored(uint32 indexed id, bytes32 root, uint32 count, uint64 hBeta);
    event Pulse(address indexed validator, bytes32 r);

    constructor(IGovernance g) {
        gov = g;
    }

    /// AGGREGATOR (E_ROLE); count >= 1 (COUNT); h_β ≡ 0 mod 15 (BEACON_BOUNDARY); h_β < now <= h_β + Delta
    /// (STALE); blockhash(h_β) != 0 (BEACON).
    function anchorBatch(bytes32 root, uint32 count, uint64 hBeta) external returns (uint32 id) {
        if (!gov.hasRole(Consts.ROLE_AGGREGATOR, msg.sender)) revert E_ROLE();
        if (count == 0) revert E_ANCHOR_COUNT();
        if (hBeta % Consts.BEACON_EPOCH != 0) revert E_ANCHOR_BEACON_BOUNDARY();
        if (!(hBeta < block.number && block.number <= uint256(hBeta) + Consts.DELTA)) revert E_ANCHOR_STALE();
        bytes32 b = blockhash(hBeta);
        if (b == bytes32(0)) revert E_ANCHOR_BEACON();
        id = uint32(_batches.length);
        _batches.push(Batch(root, b, hBeta, uint64(block.number), count));
        emit Anchored(id, root, count, hBeta);
    }

    /// E_ANCHOR_UNKNOWN_BATCH, then MerkleVerify under the batch's count and root (any Merkle failure is
    /// E_ANCHOR_INCLUSION). Returns the anchor height h_a.
    function checkInclusion(uint32 batchId, bytes32 leaf, uint32 leafIndex, bytes32[] calldata path)
        external view returns (uint64)
    {
        if (batchId >= _batches.length) revert E_ANCHOR_UNKNOWN_BATCH();
        Batch storage b = _batches[batchId];
        (bool ok, bytes32 root) = CutlineLib.tryRootFromPath(leaf, leafIndex, b.count, path);
        if (!ok || root != b.root) revert E_ANCHOR_INCLUSION();
        return b.anchorHeight;
    }

    /// batches[id] = (root, beacon, beaconHeight, anchorHeight, count); E_ANCHOR_UNKNOWN_BATCH for id >= batchCount.
    function batches(uint32 id) external view returns (bytes32, bytes32, uint64, uint64, uint32) {
        if (id >= _batches.length) revert E_ANCHOR_UNKNOWN_BATCH();
        Batch storage b = _batches[id];
        return (b.root, b.beacon, b.beaconHeight, b.anchorHeight, b.count);
    }

    function batchCount() external view returns (uint32) {
        return uint32(_batches.length);
    }

    /// Pulse: validator sidecars only (E_ROLE); emits an event and changes no state.
    function pulse(bytes32 r) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();
        emit Pulse(msg.sender, r);
    }
}
