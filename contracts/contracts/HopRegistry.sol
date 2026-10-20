// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import {CutlineLib} from "./lib/CutlineLib.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 hop registry. Platforms anchor batches of hop leaves
/// H(0x00 ‖ "CUTLINE-hop-v1" ‖ R_Y(parent) ‖ R_Y(child) ‖ opHash); a batch records {root, platformId, height, count}
/// with a 0-based u32 id. There is no failed flag in v1. Per-platform, versioned hop ceilings
/// Q_hop(p, v): version 0 is IJG q75 at height 0 for every platform; `setHopCeiling` (governance quorum) appends
/// version v = count(p) + 1 with q = elementwise max(Q_hop(p, v - 1), T) and height = now (write-once).
/// The ceiling in force for a batch is the last version whose height is <= the batch's height.
contract HopRegistry {
    IGovernance public immutable gov;

    struct HopBatch { bytes32 root; uint32 platformId; uint64 height; uint32 count; }
    struct Ceiling { bytes q; uint64 height; }

    HopBatch[] internal _batches;
    mapping(uint32 => Ceiling[]) internal _ceilings;    // platform -> versions 1..count (index v - 1)

    /// IJG q75 (luma ‖ chroma ‖ chroma), natural order; SHA-256 85eeea02….
    bytes internal constant Q75 =
        hex"080605080c141a1f0606070a0d1d1e1c0707080c141d231c07090b0f1a2c281f090b131c223734270c121c202934392e1920272c343d3c33242e303138323432"
        hex"09090c1832323232090b0d21323232320c0d1c323232323218213232323232323232323232323232323232323232323232323232323232323232323232323232"
        hex"09090c1832323232090b0d21323232320c0d1c323232323218213232323232323232323232323232323232323232323232323232323232323232323232323232";

    uint256 private constant Q_LEN = 192;

    event HopBatchAnchored(uint32 indexed id, uint32 indexed platformId, bytes32 root, uint32 count);
    event HopCeilingSet(uint32 indexed platformId, uint32 v, uint64 height);

    constructor(IGovernance g) {
        gov = g;
    }

    // ------------------------------------------------------------------ batches

    /// platformOf(msg.sender) != 0 (E_REG_PLATFORM); count >= 1 (E_ANCHOR_COUNT; there is no hop-specific count error).
    function anchorHopBatch(bytes32 root, uint32 count) external returns (uint32 id) {
        uint32 p = gov.platformOf(msg.sender);
        if (p == 0) revert E_REG_PLATFORM();
        if (count == 0) revert E_ANCHOR_COUNT();
        id = uint32(_batches.length);
        _batches.push(HopBatch(root, p, uint64(block.number), count));
        emit HopBatchAnchored(id, p, root, count);
    }

    function batchCount() external view returns (uint256) {
        return _batches.length;
    }

    /// batches[id] = (root, platformId, height, count); E_HOP_UNKNOWN_BATCH for id >= the batch count.
    function batches(uint32 id) external view returns (bytes32 root, uint32 platformId, uint64 height, uint32 count) {
        HopBatch storage b = _batch(id);
        return (b.root, b.platformId, b.height, b.count);
    }

    /// MerkleVerify(leaf, leafIndex, count, path, root) against batch batchId; a malformed path is false.
    function hopIncluded(uint32 batchId, bytes32 leaf, uint32 leafIndex, bytes32[] calldata path) external view returns (bool) {
        HopBatch storage b = _batch(batchId);
        (bool ok, bytes32 r) = CutlineLib.tryRootFromPath(leaf, leafIndex, b.count, path);
        return ok && r == b.root;
    }

    // ------------------------------------------------------------------ hop ceilings

    /// Governance quorum. q192 = T[3][64] (natural order, luma ‖ Cb ‖ Cr), every entry in [1, 255]. Check order:
    /// E_REG_PLATFORM (platformId 0), E_REG_TABLE_LEN (|q192| != 192), E_REG_TABLE (an entry 0). The subject carries
    /// the next version number, so the same T can be registered again later as a new (no-op) version.
    function setHopCeiling(uint32 platformId, bytes calldata q192) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (platformId == 0) revert E_REG_PLATFORM();
        if (q192.length != Q_LEN) revert E_REG_TABLE_LEN();
        for (uint256 k; k < Q_LEN; ++k) if (q192[k] == 0) revert E_REG_TABLE();
        Ceiling[] storage cs = _ceilings[platformId];
        uint32 v = uint32(cs.length) + 1;
        if (gov.vote(keccak256(abi.encode("hop.setHopCeiling", platformId, q192, v)), msg.sender)) {
            bytes memory prev = v == 1 ? Q75 : cs[v - 2].q;
            bytes memory q = new bytes(Q_LEN);
            for (uint256 k; k < Q_LEN; ++k) q[k] = q192[k] > prev[k] ? q192[k] : prev[k];
            cs.push(Ceiling(q, uint64(block.number)));
            emit HopCeilingSet(platformId, v, uint64(block.number));
        }
    }

    /// Number of registered versions of platform p (version 0 is implicit).
    function hopCeilingCount(uint32 platformId) external view returns (uint32) {
        return uint32(_ceilings[platformId].length);
    }

    /// Q_hop(p, v) and its height; v = 0 is IJG q75 at height 0. An unregistered v reverts E_REG_TABLE.
    function hopCeiling(uint32 platformId, uint32 v) public view returns (bytes memory q, uint64 height) {
        if (v == 0) return (Q75, 0);
        Ceiling[] storage cs = _ceilings[platformId];
        if (v > cs.length) revert E_REG_TABLE();
        Ceiling storage c = cs[v - 1];
        return (c.q, c.height);
    }

    /// The ceiling in force for a hop anchored in batch batchId: p = its platform,
    /// v* = max{v : hopCeiling[p][v].height <= batches[batchId].height}.
    function ceilingInForce(uint32 batchId) external view returns (uint32 platformId, uint32 v, bytes memory q) {
        HopBatch storage b = _batch(batchId);
        platformId = b.platformId;
        Ceiling[] storage cs = _ceilings[platformId];
        // heights are non-decreasing in v: binary search for the last version with height <= b.height
        uint256 lo;
        uint256 hi = cs.length;
        while (lo < hi) {
            uint256 mid = (lo + hi) >> 1;
            if (cs[mid].height <= b.height) lo = mid + 1;
            else hi = mid;
        }
        v = uint32(lo);
        (q,) = hopCeiling(platformId, v);
    }

    function _batch(uint32 id) internal view returns (HopBatch storage) {
        if (id >= _batches.length) revert E_HOP_UNKNOWN_BATCH();
        return _batches[id];
    }
}
