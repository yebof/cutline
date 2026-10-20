// Hardhat 3 configuration for the CUTLINE v1 contracts: solc 0.8.34 with viaIR, EVM Osaka, EDR simulated networks.
// The toolchain is pinned to Node v22.22.2.
import hardhatEthers from "@nomicfoundation/hardhat-ethers";
import { defineConfig } from "hardhat/config";
import path from "node:path";

if (process.version !== "v22.22.2") {
  throw new Error(`CUTLINE: Node ${process.version} is not the pinned v22.22.2`);
}

// Build output directory (artifacts and cache); override with CUTLINE_BUILD_DIR.
const buildDir = process.env.CUTLINE_BUILD_DIR || ".build";
// Accounts for the `besu` network: comma-separated hex private keys (deploySystem uses the first 16).
const besuKeys = (process.env.BESU_PRIVATE_KEYS || "").split(",").map((k) => k.trim()).filter(Boolean);

export default defineConfig({
  plugins: [hardhatEthers],
  solidity: {
    profiles: {
      default: {
        version: "0.8.34",
        settings: {
          optimizer: { enabled: true, runs: 200 },
          viaIR: true,
          evmVersion: "osaka",
          outputSelection: { "*": { "*": ["abi", "evm.bytecode", "evm.deployedBytecode", "evm.methodIdentifiers", "metadata", "storageLayout"] } },
        },
      },
    },
  },
  paths: {
    artifacts: path.join(buildDir, "artifacts"),
    cache: path.join(buildDir, "cache"),
  },
  networks: {
    default: {
      type: "edr-simulated",
      chainType: "l1",
      hardfork: "osaka",
      chainId: 20260926,
      blockGasLimit: 30_000_000n,
      allowUnlimitedContractSize: false,
      accounts: { count: 64 },
    },
    cutline: {
      type: "edr-simulated",
      chainType: "l1",
      hardfork: "osaka",
      chainId: 20260926,
      blockGasLimit: 30_000_000n,
      allowUnlimitedContractSize: false,
      accounts: { count: 64 },
    },
    besu: {
      type: "http",
      chainType: "l1",
      url: process.env.BESU_URL || "http://127.0.0.1:8545",
      accounts: besuKeys,
    },
  },
});
