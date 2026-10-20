// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

// CUTLINE v1 shared types. Field order and types are part of the ABI and of the hashed encodings; do not reorder.

/// The hashed core of a claim: 9 static words, 288 bytes.
struct ClaimCore {
    bytes32 recHash;
    bytes32 rY;
    bytes32 tHash;
    bytes32 rFill;
    uint16 encClass;
    uint8 tier;         // 1 = tier E, 2 = tier T
    uint8 marketplace;  // m < 64
    uint8 category;     // 0 general, 1 cosmetics, 2 apparel; 3-255 reserved (general)
    bytes32 listingRef;
}

/// The capture part of registerFirst / registerCapture.
struct FirstIn {
    bytes rec;          // 416-byte capture record
    bytes32 pkx;
    bytes32 pky;
    bytes32 r;
    bytes32 s;          // low-s
    uint32 batchId;
    uint32 leafIndex;
    bytes32[] path;
}

/// One use of a device key, as submitted to proveDoubleUse.
struct Use {
    bytes rec;
    bytes32 r;
    bytes32 s;
    uint32 batchId;
    uint32 leafIndex;
    bytes32[] path;
}

/// One opened codec cell with its Merkle path.
struct Cell {
    uint16 cy;
    uint16 cx;
    bytes16 salt;
    bytes data;         // 1,024 bytes RGBA[16][16][4]
    bytes32[] path;
}

/// Input of proveBlockFailure(ProofIn) (selector 0x00aa6b4d).
struct ProofIn {
    bytes32 nf;
    address claimant;
    ClaimCore core;
    bytes rec;
    bytes cutPre;
    bytes32[] occHashes;
    bytes occBits;
    bytes prog;
    bytes header;
    uint32 mcu;
    bytes coefs;
    bytes32[] mcuPath;
    bytes fill;
    bytes32[] fillPath;
    Cell[] cells;
    uint8 comp;
    uint8 k;
}

/// Input of proveStaticViolation(StaticIn) (selector 0x8d02fe1f).
struct StaticIn {
    bytes32 nf;
    address claimant;
    ClaimCore core;
    uint8 rule;
    uint8 i;
    uint16 j;
    bytes prog;
    bytes header;
    bytes32 rMcu;
    bytes rec;
    bytes cutPre;
}

/// A path-(b) window (kind CHALLENGE or APPEAL). The bundle locator is emitted, not stored.
struct Challenge {
    uint8 kind;         // Codes.KIND_CHALLENGE | Codes.KIND_APPEAL
    bytes32 target;     // claimId
    bytes32 pk;         // principalKey of the target's principal (CHALLENGE only)
    uint64 openedAt;
    bool finalised;
}

/// Fixed numeric codes (uint8 on the wire and in storage).
library Codes {
    // claim status (2 is unused)
    uint8 internal constant STATUS_ACTIVE = 1;
    uint8 internal constant STATUS_WITHDRAWN = 3;
    // tiers
    uint8 internal constant TIER_E = 1;
    uint8 internal constant TIER_T = 2;
    // principal kinds
    uint8 internal constant PRINCIPAL_OWN = 0;
    uint8 internal constant PRINCIPAL_LIC = 1;
    // device states
    uint8 internal constant DEV_UNKNOWN = 0;
    uint8 internal constant DEV_PENDING = 1;
    uint8 internal constant DEV_ACTIVE = 2;
    // revocation sources in Revoked(keyId, tc, source)
    uint8 internal constant REVOKE_LIST = 2;
    uint8 internal constant REVOKE_DOUBLE_USE = 3;
    uint8 internal constant REVOKE_RETIRE = 5;
    // revocation votes
    uint8 internal constant RVOTE_REVOKE = 1;
    uint8 internal constant RVOTE_KEEP = 2;
    // path-(b) windows
    uint8 internal constant KIND_CHALLENGE = 1;
    uint8 internal constant KIND_APPEAL = 2;
    // verdicts
    uint8 internal constant VERDICT_PASS = 1;
    uint8 internal constant VERDICT_FAIL = 2;
    // ChallengeFinalized outcomes
    uint8 internal constant OUTCOME_FAILED = 1;
    uint8 internal constant OUTCOME_PASSED = 2;
    uint8 internal constant OUTCOME_UNDECIDED = 3;
    // encoder-class kinds
    uint8 internal constant CLASS_NONE = 0;
    uint8 internal constant CLASS_EXACT = 1;
    uint8 internal constant CLASS_CELL = 2;
}
