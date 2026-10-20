// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {ClaimCore} from "./Types.sol";
import {Consts} from "./Consts.sol";
import "./Errors.sol";

/// @notice CUTLINE v1 digests, Merkle paths, calldata readers and signature checks: capture record, nullifier,
/// slot tag and anchor leaf; enrolment, vouch and claim digests; principal keys and relay digests; court transcripts;
/// listing MCU leaves, R_Y, program and fill digests; tier-T table digests. H = SHA-256 (precompile 0x02).
library CutlineLib {
    // =====================================================================================================
    // Merkle
    // =====================================================================================================

    /// H(0x01 ‖ a ‖ b), 65-byte preimage hashed in scratch memory above the free pointer (not advanced).
    function node(bytes32 a, bytes32 b) internal view returns (bytes32 r) {
        assembly ("memory-safe") {
            let p := mload(0x40)
            mstore8(p, 1)
            mstore(add(p, 1), a)
            mstore(add(p, 33), b)
            if iszero(staticcall(gas(), 0x02, p, 65, 0x00, 0x20)) { revert(0, 0) }
            r := mload(0x00)
        }
    }

    /// d(n) = ceil(log2 n) for n >= 1 (d(1) = 0).
    function depthOf(uint256 n) internal pure returns (uint256 d) {
        while ((uint256(1) << d) < n) ++d;
    }

    /// MerkleRootFromPath(leaf, i, n, path): the root implied by `path` (the loop with both requires, before
    /// the final root comparison). Reverts E_MERKLE_PATH (i >= n or |path| != d(n)) and E_MERKLE_ODD (a self-paired
    /// node whose sibling is not itself). Callers remap both to their own error.
    function merkleRootFromPath(bytes32 leaf, uint256 i, uint256 n, bytes32[] calldata path)
        internal view returns (bytes32 h)
    {
        if (i >= n || path.length != depthOf(n)) revert E_MERKLE_PATH();
        h = leaf;
        uint256 m = n;
        for (uint256 k; k < path.length; ++k) {
            bytes32 s = path[k];
            if (i & 1 == 0) {
                if (i == m - 1 && s != h) revert E_MERKLE_ODD();
                h = node(h, s);
            } else {
                h = node(s, h);
            }
            i >>= 1;
            m = (m + 1) >> 1;
        }
    }

    /// Non-reverting variant: (ok, root). ok = false exactly where merkleRootFromPath reverts.
    function tryRootFromPath(bytes32 leaf, uint256 i, uint256 n, bytes32[] calldata path)
        internal view returns (bool ok, bytes32 h)
    {
        if (i >= n || path.length != depthOf(n)) return (false, 0);
        h = leaf;
        uint256 m = n;
        for (uint256 k; k < path.length; ++k) {
            bytes32 s = path[k];
            if (i & 1 == 0) {
                if (i == m - 1 && s != h) return (false, 0);
                h = node(h, s);
            } else {
                h = node(s, h);
            }
            i >>= 1;
            m = (m + 1) >> 1;
        }
        ok = true;
    }

    /// MerkleVerify (reverts E_MERKLE_PATH / E_MERKLE_ODD / E_MERKLE_ROOT).
    function merkleVerify(bytes32 leaf, uint256 i, uint256 n, bytes32[] calldata path, bytes32 root) internal view {
        if (merkleRootFromPath(leaf, i, n, path) != root) revert E_MERKLE_ROOT();
    }

    // =====================================================================================================
    // Calldata readers (big-endian); callers check lengths first
    // =====================================================================================================

    function u8(bytes calldata b, uint256 i) internal pure returns (uint256 v) {
        assembly ("memory-safe") { v := shr(248, calldataload(add(b.offset, i))) }
    }

    function u16(bytes calldata b, uint256 i) internal pure returns (uint256 v) {
        assembly ("memory-safe") { v := shr(240, calldataload(add(b.offset, i))) }
    }

    function u32(bytes calldata b, uint256 i) internal pure returns (uint256 v) {
        assembly ("memory-safe") { v := shr(224, calldataload(add(b.offset, i))) }
    }

    function u64(bytes calldata b, uint256 i) internal pure returns (uint256 v) {
        assembly ("memory-safe") { v := shr(192, calldataload(add(b.offset, i))) }
    }

    function i16(bytes calldata b, uint256 i) internal pure returns (int256 v) {
        assembly ("memory-safe") { v := signextend(1, shr(240, calldataload(add(b.offset, i)))) }
    }

    /// word(b, w) = b[32w : 32w + 32].
    function word(bytes calldata b, uint256 w) internal pure returns (bytes32 v) {
        assembly ("memory-safe") { v := calldataload(add(b.offset, mul(w, 32))) }
    }

    // =====================================================================================================
    // The capture record rec v3
    // =====================================================================================================

    uint256 internal constant REC_LEN = 416;

    /// Decoder checks in precedence order: E_REC_LEN, E_REC_DOMAIN, E_REC_NONCANONICAL (words 4, 9, 10, 11,
    /// 12 high bytes), E_REC_OWNER_ZERO.
    function checkRec(bytes calldata rec) internal pure {
        if (rec.length != REC_LEN) revert E_REC_LEN();
        if (word(rec, 0) != Consts.REC_DOMAIN) revert E_REC_DOMAIN();
        if (uint256(word(rec, 4)) >> 64 != 0 || uint256(word(rec, 9)) >> 160 != 0 || uint256(word(rec, 10)) >> 32 != 0
            || uint256(word(rec, 11)) >> 16 != 0 || uint256(word(rec, 12)) >> 16 != 0) revert E_REC_NONCANONICAL();
        if (uint256(word(rec, 9)) == 0) revert E_REC_OWNER_ZERO();
    }

    function recRCut(bytes calldata rec) internal pure returns (bytes32) { return word(rec, 1); }
    function recROcc(bytes calldata rec) internal pure returns (bytes32) { return word(rec, 2); }
    function recHBeta(bytes calldata rec) internal pure returns (uint64) { return uint64(uint256(word(rec, 4))); }
    function recBeta(bytes calldata rec) internal pure returns (bytes32) { return word(rec, 5); }
    function recKeyId(bytes calldata rec) internal pure returns (bytes32) { return word(rec, 8); }
    function recOwner(bytes calldata rec) internal pure returns (address) { return address(uint160(uint256(word(rec, 9)))); }
    function recEpoch(bytes calldata rec) internal pure returns (uint32) { return uint32(uint256(word(rec, 10))); }
    function recSlot(bytes calldata rec) internal pure returns (uint16) { return uint16(uint256(word(rec, 11))); }

    // =====================================================================================================
    // Untagged and tagged SHA-256 digests
    // =====================================================================================================

    /// nf = H("nf" ‖ recHash).
    function nf(bytes32 recHash) internal pure returns (bytes32) {
        return sha256(abi.encodePacked("nf", recHash));
    }

    /// slotTag = H("slot" ‖ keyId ‖ u32 epoch ‖ u16 slot).
    function slotTag(bytes32 keyId, uint32 epoch, uint16 slot) internal pure returns (bytes32) {
        return sha256(abi.encodePacked("slot", keyId, epoch, slot));
    }

    /// leaf_v2 = H(0x00 ‖ "CUTLINE-anchor-v2" ‖ recHash ‖ H(r ‖ s) ‖ slotTag).
    function leafV2(bytes32 recHash, bytes32 r, bytes32 s, bytes32 slotTag_) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x00), Consts.TAG_ANCHOR, recHash, sha256(abi.encodePacked(r, s)), slotTag_));
    }

    /// pkHash = H(pk_x ‖ pk_y).
    function pkHash(bytes32 x, bytes32 y) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(x, y));
    }

    /// keyId = H(pk_x ‖ pk_y ‖ enrolSalt) (computed off-chain; exposed as a helper).
    function keyIdOf(bytes32 x, bytes32 y, bytes32 enrolSalt) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(x, y, enrolSalt));
    }

    /// Enrolment challenge H("CUTLINE-ENROL-v1" ‖ holder ‖ u16 rateN ‖ u256 chainId ‖ nonce_E).
    function enrolChallenge(address holder, uint16 rateN, uint256 chainId, bytes16 nonceE) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(Consts.TAG_ENROL, holder, rateN, chainId, nonceE));
    }

    /// claimId = H(abi.encode(nf, claimant, core)) (352-byte preimage).
    function claimId(bytes32 nf_, address claimant, ClaimCore memory core) internal pure returns (bytes32) {
        return sha256(abi.encode(nf_, claimant, core));
    }

    /// principalKey = H(nf ‖ u8 kind ‖ u32 id).
    function principalKey(bytes32 nf_, uint8 kind, uint32 id) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(nf_, kind, id));
    }

    /// PASS transcript H("pass" ‖ target).
    function passTranscript(bytes32 target) internal pure returns (bytes32) {
        return sha256(abi.encodePacked("pass", target));
    }

    /// KEEP entry of a revocation vote H("keep" ‖ keyId).
    function keepEntry(bytes32 keyId) internal pure returns (bytes32) {
        return sha256(abi.encodePacked("keep", keyId));
    }

    /// Listing MCU leaf v2 H(0x00 ‖ "CUTLINE-mcu-v2" ‖ u32 mcu ‖ coefs).
    function mcuLeaf(uint32 mcu, bytes calldata coefs) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x00), Consts.TAG_MCU, mcu, coefs));
    }

    /// R_Y v2 = H(0x04 ‖ "CUTLINE-listing-v2" ‖ header ‖ R_mcu); callers check |header| = 197 first.
    function listingRoot(bytes calldata header, bytes32 rMcu) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x04), Consts.TAG_LISTING, header, rMcu));
    }

    /// Fill leaf H(0x00 ‖ "CUTLINE-fill-v1" ‖ u32 mcu ‖ RGB).
    function fillLeaf(uint32 mcu, bytes calldata rgb) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x00), Consts.TAG_FILL, mcu, rgb));
    }

    /// Cell leaf H(0x00 ‖ "CUTLINE-cell-v1" ‖ u8 l ‖ u16 cy ‖ u16 cx ‖ salt16 ‖ RGBA[1024]).
    function cellLeaf(uint8 l, uint16 cy, uint16 cx, bytes16 salt, bytes calldata data) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x00), Consts.TAG_CELL, l, cy, cx, salt, data));
    }

    /// R_cut = H(0x02 ‖ "CUTLINE-roots" ‖ cutPre), cutPre = u16 H0 ‖ u16 W0 ‖ u8 Lmax ‖ R_0 ‖ ….
    function rCut(bytes calldata cutPre) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x02), Consts.TAG_ROOTS, cutPre));
    }

    /// R_occ = H(0x03 ‖ H(packbits(O_0)) ‖ … ‖ H(packbits(O_Lmax))).
    function rOcc(bytes32[] calldata occHashes) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x03), occHashes));
    }

    /// tHash = H(0x05 ‖ "CUTLINE-prog-v1" ‖ enc(π)).
    function tHash(bytes calldata prog) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(bytes1(0x05), Consts.TAG_PROG, prog));
    }

    /// Tier-T table digest H("CUTLINE-tierT-v1" ‖ record[55]).
    function tierTDigest(bytes memory record) internal pure returns (bytes32) {
        return sha256(abi.encodePacked(Consts.TAG_TIER_T, record));
    }

    /// packbits canonical form: |bits| = ceil(nbits / 8) and the padding bits of the last byte are zero.
    function checkPackbitsPadding(bytes calldata bits, uint256 nbits) internal pure returns (bool) {
        if (bits.length != (nbits + 7) >> 3) return false;
        uint256 rem = nbits & 7;
        if (rem == 0) return true;
        return u8(bits, bits.length - 1) & ((uint256(1) << (8 - rem)) - 1) == 0;
    }

    // =====================================================================================================
    // keccak256 digests (secp256k1-signed or contract-internal)
    // =====================================================================================================

    /// keccak256(abi.encode("CUTLINE-CLAIM-v2", chainId, registry, core)) (448-byte encoding).
    function claimDigest(uint256 chainId, address registry, ClaimCore memory core) internal pure returns (bytes32) {
        return keccak256(abi.encode(Consts.DOM_CLAIM, chainId, registry, core));
    }

    /// keccak256(abi.encode("CUTLINE-VOUCH-v1", chainId, registry, nf)) (192-byte encoding).
    function vouchDigest(uint256 chainId, address registry, bytes32 nf_) internal pure returns (bytes32) {
        return keccak256(abi.encode(Consts.DOM_VOUCH, chainId, registry, nf_));
    }

    /// keccak256(abi.encode("CUTLINE-RELAY-v1", chainId, target, signer, nonce, selector, argsHash)).
    function relayDigest(uint256 chainId, address target, address signer, uint256 nonce, bytes4 selector, bytes32 argsHash)
        internal pure returns (bytes32)
    {
        return keccak256(abi.encode(Consts.DOM_RELAY, chainId, target, signer, nonce, selector, argsHash));
    }

    /// Governance tally subject of `confirm`: keccak256(abi.encode("enrol", keyId, pkHash)).
    function enrolSubject(bytes32 keyId, bytes32 pkHash_) internal pure returns (bytes32) {
        return keccak256(abi.encode("enrol", keyId, pkHash_));
    }

    // =====================================================================================================
    // Signatures
    // =====================================================================================================

    /// P256VERIFY (EIP-7951, address 0x100) on digest ‖ r ‖ s ‖ x ‖ y: true iff the call succeeds and returns exactly
    /// 32 bytes equal to 1. It does NOT check low-s; see lowS.
    function p256Verify(bytes32 digest, bytes32 r, bytes32 s, bytes32 x, bytes32 y) internal view returns (bool ok) {
        assembly ("memory-safe") {
            let p := mload(0x40)
            mstore(p, digest)
            mstore(add(p, 0x20), r)
            mstore(add(p, 0x40), s)
            mstore(add(p, 0x60), x)
            mstore(add(p, 0x80), y)
            mstore(0x00, 0)
            let success := staticcall(gas(), 0x100, p, 0xa0, 0x00, 0x20)
            ok := and(success, and(eq(returndatasize(), 0x20), eq(mload(0x00), 1)))
        }
    }

    /// s <= n/2 on P-256 (callers revert E_CAP_HIGH_S otherwise); rejects the malleable twin (r, n - s).
    function lowS(bytes32 s) internal pure returns (bool) {
        return uint256(s) <= Consts.P256_HALF_N;
    }

    /// ecrecover of a 65-byte r ‖ s ‖ v signature with v in {27, 28}; returns address(0) for any other length or v
    /// (no low-s check for secp256k1).
    function recover65(bytes32 digest, bytes calldata sig) internal pure returns (address) {
        if (sig.length != 65) return address(0);
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly ("memory-safe") {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        if (v != 27 && v != 28) return address(0);
        return ecrecover(digest, v, r, s);
    }
}
