// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

// CUTLINE v1 custom errors. Each error maps one-to-one to a dotted error name:
// `E_<AREA>_<NAME>()` <-> `E.<AREA>.<NAME>` (the first underscore after `E_` separates AREA from NAME; no AREA contains
// an underscore). `E_ROLE()` and `E_OWNER()` have no NAME part.

// ---- roles and ownership
error E_ROLE();
error E_OWNER();
// governance mechanics and one-time wiring
error E_GOV_BLOC();          // the validator's regulatory bloc already holds a seat (one seat per bloc)
error E_GOV_MEMBER();        // addValidator of a validator / removeValidator of a non-validator
error E_GOV_MIN();           // the validator set would fall below 4 (or exceed 255)
error E_INIT();              // init(...) called twice or by an account other than the deployer
error E_LABEL_VOCAB();       // a label name outside the pinned vocabulary
error E_LABEL_MONOTONE();    // codes not monotone in the level
error E_LABEL_NO_PROVENANCE(); // G0/G1 code differs from the regime's registered no-provenance code

// ---- Merkle; internal: every caller remaps them to its own error
error E_MERKLE_PATH();
error E_MERKLE_ODD();
error E_MERKLE_ROOT();

// ---- capture record and CapValid
error E_REC_LEN();
error E_REC_DOMAIN();
error E_REC_NONCANONICAL();
error E_REC_OWNER_ZERO();
error E_CAP_PK();
error E_CAP_DEVICE_SIG();
error E_CAP_HIGH_S();
error E_CAP_SLOT();
error E_CAP_BRACKET();
error E_CAP_KEY_INVALID();
error E_CAP_HOLDER();
error E_ANCHOR_COUNT();
error E_ANCHOR_BEACON_BOUNDARY();
error E_ANCHOR_STALE();
error E_ANCHOR_BEACON();
error E_ANCHOR_UNKNOWN_BATCH();
error E_ANCHOR_INCLUSION();

// ---- devices
error E_DEV_EXISTS();
error E_DEV_PK_ENROLLED();
error E_DEV_RATE();
error E_DEV_HOLDER_ZERO();
error E_DEV_STATE();
error E_DEV_NOT_HOLDER();
error E_DEV_NOT_SAME_SLOT();
error E_DEV_SAME_RECORD();

// ---- claims, principals, strikes, delegations, grants, transfers, relays
error E_CLAIM_REC();
error E_CLAIM_UNKNOWN_CAPTURE();
error E_CLAIM_MARKETPLACE();
error E_CLAIM_RULE8();
error E_CLAIM_SIG();
error E_CLAIM_DUP();
error E_CLAIM_NOT_ACTIVE();
error E_CLAIM_NOT_CLAIMANT();
error E_PRINCIPAL_EXCLUSIVE();
error E_PRINCIPAL_NOT_AUTHORISED();
error E_PRINCIPAL_STRUCK();
error E_STRIKE_NOT_QUORUM_FAILED();
error E_DELEG_SCOPE();
error E_DELEG_ARGS();
error E_GRANT_CHALLENGED();
error E_GRANT_EXPIRY();
error E_GRANT_TAKEN();
error E_GRANT_RATE();
error E_GRANT_WITHDRAW_LIST();
error E_GRANT_OWNER_CLAIMS();
error E_TRANSFER_NO_OFFER();
error E_TRANSFER_CHALLENGED();
error E_TRANSFER_STRUCK();
error E_TRANSFER_RATE();
error E_RELAY_SIG();
error E_RELAY_NONCE();       // declared for the one-to-one map; unreachable in the v1 relay shape (the nonce is in the digest)

// ---- static rules
error E_STATIC_RULE();
error E_STATIC_CLAIM();
error E_STATIC_ARGS();
error E_STATIC_PROGRAM();
error E_STATIC_PROGRAM_LEN();
error E_STATIC_RY();
error E_STATIC_REC();
error E_STATIC_NO_VIOLATION();

// ---- registries and hops
error E_REG_DEFINED();
error E_REG_PROFILE();
error E_REG_PREDECESSOR();
error E_REG_TABLE();
error E_REG_TABLE_LEN();
error E_REG_PLATFORM();
error E_HOP_UNKNOWN_BATCH();

// ---- path (b): challenges, votes, appeals
error E_CHAL_UNKNOWN();
error E_CHAL_NOT_PATH_B();
error E_CHAL_DECIDED();
error E_CHAL_OPEN();
error E_CHAL_PASSED();
error E_CHAL_UNDECIDED_RATE();
error E_CHAL_WINDOW();
error E_VOTE_WINDOW();
error E_VOTE_DUP();
error E_VOTE_STATUS();
error E_VOTE_PASS_TRANSCRIPT();
error E_VOTE_FAIL_TRANSCRIPT();
error E_APPEAL_NOT_QUORUM_FAILED();
error E_APPEAL_OPEN();

// ---- block court
error E_COURT_CLAIM();
error E_COURT_TIER();
error E_COURT_CLASS();
error E_COURT_REC();
error E_COURT_ROOTS();
error E_COURT_OCC();
error E_COURT_PROGRAM();
error E_COURT_STATIC();
error E_COURT_HEADER();
error E_COURT_SAMPLING();
error E_COURT_INDEX();
error E_COURT_ZERO_Q();
error E_COURT_DECLARED();
error E_COURT_COEFS();
error E_COURT_MCU_PATH();
error E_COURT_RY();
error E_COURT_NOT_CERTIFIED();
error E_COURT_CELLS();
error E_COURT_CELL_PATH();
error E_COURT_FILL();
error E_COURT_CONSISTENT();
