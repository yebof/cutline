// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {IGovernance} from "./interfaces/IGovernance.sol";
import "./lib/Errors.sol";

/// @notice CUTLINE v1 label tables.
/// - `tables[regime][version] = {uint8[5] code, bytes32[5] name, uint64 height}` for the levels G0..G4; versions are
///   1, 2, … per regime, write-once (a new table appends a version; nothing is modified or deleted). Relying parties
///   stamp the version they used (the label records the table version).
/// - `setTable` (governance quorum) reverts, in this order: E_LABEL_NO_PROVENANCE if the regime has no registered
///   "no provenance information" code; E_LABEL_MONOTONE if code[l] > code[l + 1] for some l (codes are ordinal
///   favourability, non-decreasing in the level); E_LABEL_NO_PROVENANCE if code[0] or code[1] differs from the
///   regime's registered no-provenance code; E_LABEL_VOCAB if a name is outside the pinned vocabulary at its level.
/// - The pinned vocabulary (below, `minLevel`) contains no AI-positive and no capture-positive term below G3, and no
///   term asserting "no AI", "AI-generated" or `digitalCapture` at any level.
/// - Regimes: genesis EU, CN, IPTC, GMC with no-provenance code 0; `setRegime` (quorum, write-once) adds one.
contract LabelPolicy {
    IGovernance public immutable gov;

    struct Regime { bool defined; uint8 noProvenanceCode; }
    struct Table { uint8[5] code; bytes32[5] name; uint64 height; }

    mapping(bytes4 => Regime) public regimes;
    mapping(bytes4 => uint64) public tableVersion;                  // number of versions of the regime's table
    mapping(bytes4 => mapping(uint64 => Table)) internal _tables;

    // ---- the pinned vocabulary (bytes32 ASCII names); minimum level in parentheses
    // any level (0): negative or neutral statements
    bytes32 internal constant N_NO_PROVENANCE = "NO_PROVENANCE_INFO";
    bytes32 internal constant N_NOT_EVALUATED = "NOT_EVALUATED";
    bytes32 internal constant N_REGIME_DEFAULT = "REGIME_DEFAULT";
    bytes32 internal constant N_PLATFORM_DEFAULT = "PLATFORM_DEFAULT";
    bytes32 internal constant N_PLATFORM_NOTICE = "PLATFORM_OWN_NOTICE";          // CN Art. 6 three-tier notice
    bytes32 internal constant N_SELLER_TAGS_KEPT = "SELLER_TAGS_KEPT";
    bytes32 internal constant N_NONE = "NONE";                                    // IPTC: none asserted
    // G2 and above
    bytes32 internal constant N_PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED";
    // G3 and above: capture-positive and AI-disclosure terms
    bytes32 internal constant N_CAPTURE_VERIFIED = "CAPTURE_REGION_VERIFIED";
    bytes32 internal constant N_CAPTURE_VERIFIED_TOL = "CAPTURE_VERIFIED_ENC_TOLERANCE";
    bytes32 internal constant N_CAPTURE_VERIFIED_PENDING = "CAPTURE_VERIFIED_PENDING";
    bytes32 internal constant N_PROVENANCE_NOTE = "PROVENANCE_NOTE_REGION_VERIFIED";
    bytes32 internal constant N_AI_BACKGROUND = "AI_BACKGROUND_DECLARED";
    bytes32 internal constant N_RETOUCHED = "RETOUCHED_AREAS_MARKED";
    bytes32 internal constant N_GEN_RETOUCH = "GENERATIVE_RETOUCH_DECLARED";
    bytes32 internal constant N_COLOUR_ADJUSTED = "COLOUR_ADJUSTED";
    bytes32 internal constant N_AI_ELEMENT_NOTICE = "AI_ELEMENT_NOTICE";
    bytes32 internal constant N_IPTC_COMPOSITE_SYNTHETIC = "IPTC_COMPOSITE_SYNTHETIC";
    bytes32 internal constant N_IPTC_COMPOSITE_TRAINED = "IPTC_COMPOSITE_TRAINED_ALGO";
    bytes32 internal constant N_GMC_COMPOSITE_SYNTHETIC = "GMC_COMPOSITE_SYNTHETIC";

    event RegimeSet(bytes4 indexed regime, uint8 noProvenanceCode);
    event TableSet(bytes4 indexed regime, uint64 version, uint8[5] code, bytes32[5] name);

    constructor(IGovernance g) {
        gov = g;
        _setRegime("EU", 0);
        _setRegime("CN", 0);
        _setRegime("IPTC", 0);
        _setRegime("GMC", 0);
    }

    // ------------------------------------------------------------------ vocabulary

    /// (in the vocabulary, minimum level at which the name may appear).
    function minLevel(bytes32 n) public pure returns (bool known, uint8 level) {
        if (n == N_NO_PROVENANCE || n == N_NOT_EVALUATED || n == N_REGIME_DEFAULT || n == N_PLATFORM_DEFAULT
            || n == N_PLATFORM_NOTICE || n == N_SELLER_TAGS_KEPT || n == N_NONE) return (true, 0);
        if (n == N_PARTIALLY_VERIFIED) return (true, 2);
        if (n == N_CAPTURE_VERIFIED || n == N_CAPTURE_VERIFIED_TOL || n == N_CAPTURE_VERIFIED_PENDING
            || n == N_PROVENANCE_NOTE || n == N_AI_BACKGROUND || n == N_RETOUCHED || n == N_GEN_RETOUCH
            || n == N_COLOUR_ADJUSTED || n == N_AI_ELEMENT_NOTICE || n == N_IPTC_COMPOSITE_SYNTHETIC
            || n == N_IPTC_COMPOSITE_TRAINED || n == N_GMC_COMPOSITE_SYNTHETIC) return (true, 3);
        return (false, 0);
    }

    // ------------------------------------------------------------------ governance acts (quorum)

    /// Registers a regime and its "no provenance information" code (write-once, E_REG_DEFINED).
    function setRegime(bytes4 regime, uint8 noProvenanceCode) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        if (regimes[regime].defined) revert E_REG_DEFINED();
        if (gov.vote(keccak256(abi.encode("label.setRegime", regime, noProvenanceCode)), msg.sender)) {
            _setRegime(regime, noProvenanceCode);
        }
    }

    /// Appends version tableVersion[regime] + 1 (see the contract header for the checks and their order).
    function setTable(bytes4 regime, uint8[5] calldata code, bytes32[5] calldata name) external {
        if (!gov.isValidator(msg.sender)) revert E_ROLE();   // governance act: validators only (checked first)
        Regime storage r = regimes[regime];
        if (!r.defined) revert E_LABEL_NO_PROVENANCE();
        for (uint256 l; l < 4; ++l) if (code[l] > code[l + 1]) revert E_LABEL_MONOTONE();
        if (code[0] != r.noProvenanceCode || code[1] != r.noProvenanceCode) revert E_LABEL_NO_PROVENANCE();
        for (uint256 l; l < 5; ++l) {
            (bool known, uint8 ml) = minLevel(name[l]);
            if (!known || l < ml) revert E_LABEL_VOCAB();
        }
        uint64 v = tableVersion[regime] + 1;
        if (gov.vote(keccak256(abi.encode("label.setTable", regime, code, name, v)), msg.sender)) {
            tableVersion[regime] = v;
            _tables[regime][v] = Table(code, name, uint64(block.number));
            emit TableSet(regime, v, code, name);
        }
    }

    // ------------------------------------------------------------------ views

    /// The current table's (code, name) at `level` and its version; (0, 0, 0) if the regime has no table.
    /// level > 4 reverts E_REG_TABLE.
    function labelOf(bytes4 regime, uint8 level) external view returns (uint8 code, bytes32 name, uint64 version) {
        version = tableVersion[regime];
        (code, name) = labelAt(regime, version, level);
    }

    /// (code, name) of version `version` at `level`; version 0 or an unwritten version gives (0, 0).
    function labelAt(bytes4 regime, uint64 version, uint8 level) public view returns (uint8 code, bytes32 name) {
        if (level > 4) revert E_REG_TABLE();
        Table storage t = _tables[regime][version];
        return (t.code[level], t.name[level]);
    }

    /// The whole table of a version and the height at which it was written.
    function table(bytes4 regime, uint64 version) external view returns (uint8[5] memory code, bytes32[5] memory name, uint64 height) {
        Table storage t = _tables[regime][version];
        return (t.code, t.name, t.height);
    }

    /// The version in force after block h (the last version written at a height <= h; 0 = none).
    function versionAt(bytes4 regime, uint64 h) external view returns (uint64 v) {
        v = tableVersion[regime];
        while (v != 0 && _tables[regime][v].height > h) --v;
    }

    function _setRegime(bytes4 regime, uint8 np) internal {
        regimes[regime] = Regime(true, np);
        emit RegimeSet(regime, np);
    }
}
