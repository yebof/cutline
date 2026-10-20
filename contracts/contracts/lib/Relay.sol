// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

import {CutlineLib} from "./CutlineLib.sol";
import "./Errors.sol";

/// @notice Signed relays with nonces. A relayed entry point takes its canonical
/// arguments followed by `(address signer, bytes sig)`; the relay digest names the canonical selector (without the
/// relay parameters) and argsHash = keccak256(abi.encode(canonical args in declaration order)). The nonce is
/// relayNonce[signer] of the target contract (this contract), incremented on success. Check order: SIG, then NONCE;
/// with the nonce inside the digest a stale or replayed signature fails the SIG check, so E_RELAY_NONCE is
/// never reached in this shape.
abstract contract RelayVerifier {
    mapping(address => uint256) public relayNonce;

    /// Verifies σ over the relay digest for `signer` and consumes its nonce. Returns the nonce used.
    function _relay(bytes4 canonicalSel, bytes32 argsHash, address signer, bytes calldata sig) internal returns (uint256 nonce) {
        nonce = relayNonce[signer];
        bytes32 d = CutlineLib.relayDigest(block.chainid, address(this), signer, nonce, canonicalSel, argsHash);
        address a = CutlineLib.recover65(d, sig);
        if (a == address(0) || a != signer) revert E_RELAY_SIG();
        relayNonce[signer] = nonce + 1;
    }
}
