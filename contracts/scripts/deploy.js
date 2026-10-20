// Deploys the full CUTLINE v1 system (see lib/deploy.js for the order and the account roles), then checks the wiring
// (every immutable and init pointer), the runtime sizes (< 24,576 bytes) and the genesis registries written by the
// constructors (encoder classes, kernel families, label regimes, validators, roles, the two relayer platforms).
// Prints a JSON summary with the addresses; set DEPLOY_OUT=<file> to also write it to a file.
//
//   npx hardhat run scripts/deploy.js --network cutline                  (in-process EDR, chainId 20260926)
//   BESU_URL=... BESU_PRIVATE_KEYS=k0,k1,...,k15 DEPLOY_N=4 \
//     npx hardhat run scripts/deploy.js --network besu                   (needs a fresh chain)
//
// DEPLOY_N is the number of genesis validators (default 7, at most 10).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { network } from "hardhat";
import { ORDER, deploySystem } from "./lib/deploy.js";

const conn = await network.getOrCreate();
const { ethers } = conn;
const NET = conn.networkName;
const n = Number(process.env.DEPLOY_N || 7);
const t0 = Date.now();
const { acc, c, addr } = await deploySystem(ethers, { n });
const P = ethers.provider;
const lc = (a) => a.toLowerCase();

// wiring
const wiring = {
  "ClaimRegistry.gov": [await c.ClaimRegistry.gov(), addr.Governance],
  "ClaimRegistry.dev": [await c.ClaimRegistry.dev(), addr.DeviceRegistry],
  "ClaimRegistry.anchor": [await c.ClaimRegistry.anchor(), addr.CaptureAnchor],
  "ClaimRegistry.adjudicator": [await c.ClaimRegistry.adjudicator(), addr.Adjudicator],
  "ClaimRegistry.staticRules": [await c.ClaimRegistry.staticRules(), addr.StaticRules],
  "ClaimRegistry.blockCourt": [await c.ClaimRegistry.blockCourt(), addr.BlockCourt],
  "DeviceRegistry.gov": [await c.DeviceRegistry.gov(), addr.Governance],
  "DeviceRegistry.anchor": [await c.DeviceRegistry.anchor(), addr.CaptureAnchor],
  "CaptureAnchor.gov": [await c.CaptureAnchor.gov(), addr.Governance],
  "HopRegistry.gov": [await c.HopRegistry.gov(), addr.Governance],
  "Adjudicator.gov": [await c.Adjudicator.gov(), addr.Governance],
  "Adjudicator.reg": [await c.Adjudicator.reg(), addr.ClaimRegistry],
  "StaticRules.gov": [await c.StaticRules.gov(), addr.Governance],
  "StaticRules.reg": [await c.StaticRules.reg(), addr.ClaimRegistry],
  "StaticRules.adj": [await c.StaticRules.adj(), addr.Adjudicator],
  "LabelPolicy.gov": [await c.LabelPolicy.gov(), addr.Governance],
};
for (const [k, [got, want]] of Object.entries(wiring)) assert.equal(lc(got), lc(want), k);

// deployment nonces: contract i is the deployer's i-th creation
const nonceAddr = ORDER.map((name, i) => [name, ethers.getCreateAddress({ from: acc.deployer.address, nonce: i })]);
for (const [name, a] of nonceAddr) assert.equal(lc(addr[name]), lc(a), `${name} not at nonce ${ORDER.indexOf(name)}`);

// runtime sizes
const sizes = {};
for (const name of ORDER) {
  const code = await P.getCode(addr[name]);
  sizes[name] = (code.length - 2) / 2;
  assert.ok(sizes[name] > 0 && sizes[name] < 24576, `${name} runtime ${sizes[name]} B`);
}

// genesis state
const cls = {};
for (const id of [1, 3, 5, 7]) { const r = await c.Adjudicator.cls(id); cls[id] = [r[0], Number(r[1]), Number(r[2])]; }
const kfam = {};
for (let f = 0; f <= 4; ++f) { const r = await c.Adjudicator.kfam(f); kfam[f] = [r[0], Number(r[1])]; }
const regimes = {};
for (const r of ["EU", "CN", "IPTC", "GMC"]) {
  const b4 = ethers.hexlify(ethers.toUtf8Bytes(r.padEnd(4, "\0")));
  const x = await c.LabelPolicy.regimes(b4);
  regimes[r] = [x[0], Number(x[1])];
}
const gov = {
  validators: Number(await c.Governance.validatorCount()), quorum: Number(await c.Governance.quorum()), f: Number(await c.Governance.f()),
  relayer1_platform: Number(await c.Governance.platformOf(acc.relayer1.address)),
  relayer2_platform: Number(await c.Governance.platformOf(acc.relayer2.address)),
};
const out = {
  network: NET, chainId: Number((await P.getNetwork()).chainId), deployer: acc.deployer.address, order: ORDER,
  addresses: addr, wiring: Object.fromEntries(Object.keys(wiring).map((k) => [k, "ok"])), runtimeSizes: sizes,
  genesis: { classes: cls, kernelFamilies: kfam, labelRegimes: regimes, governance: gov },
  seconds: (Date.now() - t0) / 1000,
};
// On the simulated network the deployer is the default Hardhat account 0, so the first two addresses are fixed.
if (out.chainId === 20260926) {
  assert.equal(lc(addr.ClaimRegistry), "0x5fbdb2315678afecb367f032d93f642f64180aa3");
  assert.equal(lc(addr.DeviceRegistry), "0xe7f1725e7734ce288f8367e1bb143e90bb3f0512");
}
const json = JSON.stringify(out, null, 1);
if (process.env.DEPLOY_OUT) {
  const file = path.resolve(process.env.DEPLOY_OUT);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, json);
}
console.log(json);
