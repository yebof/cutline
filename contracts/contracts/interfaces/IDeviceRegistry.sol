// SPDX-License-Identifier: MIT
pragma solidity 0.8.34;

/// DeviceRegistry view interface: enrolled device keys.
interface IDeviceRegistry {
    /// devices[keyId] = (pkHash, holder, enrolledAt, rateN, state, tc); state 0 unknown, 1 PENDING, 2 ACTIVE.
    function devices(bytes32 keyId) external view returns (bytes32, address, uint64, uint16, uint8, uint64);
    /// state = ACTIVE and (tc = 0 or h < tc); revocation is forward-only.
    function validAt(bytes32 keyId, uint64 h) external view returns (bool);
}
