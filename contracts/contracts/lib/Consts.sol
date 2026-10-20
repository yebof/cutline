// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// CUTLINE v1 genesis constants and domain strings. Heights and windows are in blocks.
library Consts {
    // ---- heights and windows
    uint64 internal constant DELTA = 240;                 // anchor window after h_beta
    uint64 internal constant BEACON_EPOCH = 15;           // beacon boundaries h ≡ 0 (mod 15)
    uint64 internal constant RATE_EPOCH = 1800;           // rec.epoch = h_beta div 1800
    uint64 internal constant W_REG = 1_296_000;           // registration window, transfer and grant rate limits
    uint64 internal constant W_VOTE = 150;                // challenge / appeal / revocation window
    uint64 internal constant LIC_CAP = 31_536_000;        // exclusive licence expiry cap (2 years)
    uint16 internal constant RATE_N_MIN = 30;
    uint16 internal constant RATE_N_MAX = 600;
    uint16 internal constant RATE_N_DEFAULT = 120;
    uint8 internal constant MARKETPLACES = 64;            // m < 64
    uint16 internal constant TIER_T_MIN_ID = 1024;        // tier-T table ids 1,024-65,535

    // ---- P-256
    uint256 internal constant P256_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551;
    uint256 internal constant P256_HALF_N = 0x7FFFFFFF800000007FFFFFFFFFFFFFFFDE737D56D38BCF4279DCE5617E3192A8; // n div 2
    address internal constant P256VERIFY = address(0x100);   // EIP-7951

    // ---- fixed hashes
    /// H(abi.encode(T[] = [])) = H(u256 0x20 ‖ u256 0): rejected as a FAIL transcript.
    bytes32 internal constant EMPTY_ROWS = 0x1cca23f135d6a051c5bceab95d8bbc3983a92021783ffe3540c209f682c5c93b;
    /// bytes32("CUTLINE-REC-v3"): word 0 of a capture record.
    bytes32 internal constant REC_DOMAIN = "CUTLINE-REC-v3";

    // ---- governance roles
    bytes32 internal constant ROLE_WATCHER = keccak256("WATCHER");
    bytes32 internal constant ROLE_ENROLLER = keccak256("ENROLLER");
    bytes32 internal constant ROLE_AGGREGATOR = keccak256("AGGREGATOR");

    // ---- domain strings; tagged ones are hashed as tag byte ‖ string ‖ fields
    string internal constant TAG_CELL = "CUTLINE-cell-v1";
    string internal constant TAG_ANCHOR = "CUTLINE-anchor-v2";
    string internal constant TAG_FILL = "CUTLINE-fill-v1";
    string internal constant TAG_MCU = "CUTLINE-mcu-v2";
    string internal constant TAG_HOP = "CUTLINE-hop-v1";
    string internal constant TAG_ROOTS = "CUTLINE-roots";
    string internal constant TAG_LISTING = "CUTLINE-listing-v2";
    string internal constant TAG_PROG = "CUTLINE-prog-v1";
    string internal constant TAG_TIER_T = "CUTLINE-tierT-v1";
    string internal constant TAG_ENROL = "CUTLINE-ENROL-v1";
    string internal constant DOM_CLAIM = "CUTLINE-CLAIM-v2";
    string internal constant DOM_VOUCH = "CUTLINE-VOUCH-v1";
    string internal constant DOM_RELAY = "CUTLINE-RELAY-v1";
}
