// Full-system deployment of the CUTLINE v1 contracts in a fixed order. With a fresh deployer account (nonce 0) the
// addresses are deterministic: ClaimRegistry at nonce 0, DeviceRegistry at nonce 1, and so on. Contracts deployed
// before Governance (ClaimRegistry, DeviceRegistry) are wired by a one-time, deployer-only init(...); every later
// contract takes its dependencies as constructor arguments (immutables, cheaper reads):
//
//   0 ClaimRegistry()                   init(gov, dev, anchor, adjudicator, staticRules, blockCourt)
//   1 DeviceRegistry()                  init(gov, anchor)
//   2 Governance(vals, blocs, roles, relayers)
//   3 CaptureAnchor(gov)
//   4 HopRegistry(gov)
//   5 Adjudicator(gov, claimRegistry)
//   6 StaticRules(gov, claimRegistry, adjudicator)
//   7 BlockCourt(gov, claimRegistry, adjudicator)
//   8 LabelPolicy(gov)
//
// Account roles: 0 deployer; 1..n validators (n <= 10); 11 ENROLLER; 12 AGGREGATOR; 13 WATCHER; 14 relayer of
// platform 1; 15 relayer of platform 2. The network must provide at least 16 accounts.
import { ethers as E } from "ethers";

export const ROLE = { WATCHER: E.id("WATCHER"), ENROLLER: E.id("ENROLLER"), AGGREGATOR: E.id("AGGREGATOR") };
/** Regulatory blocs of the genesis validators (one seat per bloc). */
export const BLOCS = ["CN", "EU", "US", "UK", "SG", "JP", "BR", "IN", "AU", "CA"].map((b) => E.encodeBytes32String(b));
/** Default relayer markets: platform 1 relays marketplaces m = 0..31, platform 2 m = 32..63. */
export const MARKETS1 = (1n << 32n) - 1n;
export const MARKETS2 = ((1n << 64n) - 1n) ^ MARKETS1;

export const ORDER = ["ClaimRegistry", "DeviceRegistry", "Governance", "CaptureAnchor", "HopRegistry", "Adjudicator",
  "StaticRules", "BlockCourt", "LabelPolicy"];

export async function deployContract(ethers, name, args = [], from = undefined) {
  const f = await ethers.getContractFactory(name, from);
  const c = await f.deploy(...args);
  await c.waitForDeployment();
  return c;
}

/** Signers split by role (see the header). */
export async function accounts(ethers, n = 7) {
  const s = await ethers.getSigners();
  if (s.length < 16) throw new Error(`deploySystem needs at least 16 accounts (got ${s.length})`);
  return {
    all: s, deployer: s[0], validators: s.slice(1, 1 + n), enroller: s[11], aggregator: s[12], watcher: s[13],
    relayer1: s[14], relayer2: s[15],
  };
}

/** Constructor arguments of Governance for n validators (distinct blocs), the three roles and the two relayers. */
export function governanceArgs(acc, n = 7) {
  const vals = acc.validators.slice(0, n).map((v) => v.address);
  const roles = [[ROLE.ENROLLER, acc.enroller.address], [ROLE.AGGREGATOR, acc.aggregator.address], [ROLE.WATCHER, acc.watcher.address]];
  const rels = [[acc.relayer1.address, 1, MARKETS1], [acc.relayer2.address, 2, MARKETS2]];
  return [vals, BLOCS.slice(0, n), roles, rels];
}

/**
 * Deploys and wires all nine contracts in ORDER. Must run on a fresh network so that the deployer's nonce is 0.
 * Returns {acc, c: {name: contract}, addr: {name: address}}.
 */
export async function deploySystem(ethers, { n = 7 } = {}) {
  const acc = await accounts(ethers, n);
  const nonce = await ethers.provider.getTransactionCount(acc.deployer.address);
  if (nonce !== 0) throw new Error(`deploySystem needs a fresh network (deployer nonce ${nonce})`);
  const c = {};
  c.ClaimRegistry = await deployContract(ethers, "ClaimRegistry");
  c.DeviceRegistry = await deployContract(ethers, "DeviceRegistry");
  c.Governance = await deployContract(ethers, "Governance", governanceArgs(acc, n));
  const gov = await c.Governance.getAddress();
  c.CaptureAnchor = await deployContract(ethers, "CaptureAnchor", [gov]);
  c.HopRegistry = await deployContract(ethers, "HopRegistry", [gov]);
  const reg = await c.ClaimRegistry.getAddress();
  c.Adjudicator = await deployContract(ethers, "Adjudicator", [gov, reg]);
  const adj = await c.Adjudicator.getAddress();
  c.StaticRules = await deployContract(ethers, "StaticRules", [gov, reg, adj]);
  c.BlockCourt = await deployContract(ethers, "BlockCourt", [gov, reg, adj]);
  c.LabelPolicy = await deployContract(ethers, "LabelPolicy", [gov]);
  const addr = {};
  for (const k of ORDER) addr[k] = await c[k].getAddress();
  await (await c.ClaimRegistry.init(gov, addr.DeviceRegistry, addr.CaptureAnchor, adj, addr.StaticRules, addr.BlockCourt)).wait();
  await (await c.DeviceRegistry.init(gov, addr.CaptureAnchor)).wait();
  return { acc, c, addr };
}
