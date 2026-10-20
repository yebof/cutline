"""sidecar.config - one validator sidecar's configuration.

``SidecarConfig`` (a dataclass; ``from_json(path_or_dict)`` / ``to_json()``):

| Field | Meaning |
|---|---|
| ``index`` | the validator index i (1..n; the Governance validator of bloc i) |
| ``key`` | the secp256k1 private key (0x hex); the sender of every transaction |
| ``rpc`` | the own node's HTTP JSON-RPC URL; ``peer_rpcs`` the other nodes' (cross-node alarm) |
| ``platform_templates`` / ``listing_templates`` | bundle sources with a ``{claim_id}`` placeholder (0x hex) |
| ``peer_caches`` | other validators' / watchers' cache servers (``<base>/bundle/<claim_id>``) |
| ``cache_dir`` | the local bundle cache (verifying copies are stored here and served by ``sources.CacheServer``) |
| ``fetch_limit`` / ``fetch_timeout_s`` | per-copy byte limit (at least the bundle size bound) and HTTP timeout |
| ``pe_url`` | P_E's attestation-chain publication (``<pe_url>/chain/<keyId>.json``) |
| ``statuslist_url`` | the revocation status list (JSON ``{"entries": {serial: {...}}}``); ``revocation_poll_s`` |
| ``pinned_root_sha256`` | the pinned attestation root's DER SHA-256 (None: the emulation root) |
| ``pulse`` | send the beacon pulse (``--no-pulse`` clears it) |
| ``byzantine`` | fault injection (tests): ``None`` / ``"fail-all"`` / ``"pass-all"`` / ``"silent"``; ``silent_until`` ends
  ``silent`` |
| ``watcher`` | court duty (static proofs, block-failure proofs); ``watch_claims`` also scans ``ClaimRegistered`` |
| ``isolation`` | ``"process"`` (the vote runs in a child with ``vote_rss_mb`` / ``vote_timeout_s``) or ``"inline"`` |
| ``deadline_margin`` | a vote is not sent later than ``openedAt + W_vote − margin`` (5 blocks) |
| ``block_period_s`` | the chain's period (turns block deadlines into wall budgets) |
| ``max_queue`` | vote-queue bound (overload: the latest-deadline jobs abstain ``overload``) |
| ``finalize`` / ``finalize_stagger`` | send finalize calls, validator i waiting ``(i − 1) · stagger`` blocks |
| ``log_range`` | ``eth_getLogs`` range per request (≤ 500, the Bonsai history limit) |
| ``start_block`` | the first block scanned on a fresh state (None: head − 2·W_vote) |
| ``alarm_lag`` | the cross-node alarm compares the hash at head − lag (peers may trail by a block) |
| ``poll_s`` | the head poll period (250 ms) |
| ``state_path`` / ``metrics_path`` | the durable state JSON and the metrics JSONL |
| ``chain_id`` / ``registry`` | override ``ledger().chain_id`` / ``.registry`` (normally None) |
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

BYZANTINE_MODES = (None, "fail-all", "pass-all", "silent")


@dataclass
class SidecarConfig:
    index: int = 1
    key: str = ""
    rpc: str = ""
    peer_rpcs: list = field(default_factory=list)
    platform_templates: list = field(default_factory=list)
    listing_templates: list = field(default_factory=list)
    peer_caches: list = field(default_factory=list)
    cache_dir: str | None = None
    fetch_limit: int = 256 << 20
    fetch_timeout_s: float = 20.0
    pe_url: str | None = None
    statuslist_url: str | None = None
    revocation_poll_s: float = 60.0
    pinned_root_sha256: str | None = None
    pulse: bool = True
    byzantine: str | None = None
    silent_until: int | None = None
    watcher: bool = False
    watch_claims: bool = False
    isolation: str = "process"
    vote_rss_mb: int = 1800
    vote_timeout_s: float = 240.0
    deadline_margin: int = 5
    block_period_s: float = 2.0
    max_queue: int = 64
    finalize: bool = True
    finalize_stagger: int = 1
    log_range: int = 500
    start_block: int | None = None
    alarm_lag: int = 2
    poll_s: float = 0.25
    state_path: str | None = None
    metrics_path: str | None = None
    chain_id: int | None = None
    registry: str | None = None

    def __post_init__(self):
        if self.byzantine not in BYZANTINE_MODES:
            raise ValueError(f"byzantine must be one of {BYZANTINE_MODES}")
        if self.isolation not in ("process", "inline"):
            raise ValueError("isolation must be 'process' or 'inline'")
        if not 1 <= self.log_range <= 500:
            raise ValueError("log_range must be in 1..500 (Bonsai history)")

    @classmethod
    def from_json(cls, src) -> "SidecarConfig":
        d = src if isinstance(src, dict) else json.load(open(src))
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown sidecar config keys {sorted(unknown)}")
        return cls(**d)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def private_key(self) -> int:
        k = self.key[2:] if self.key.startswith("0x") else self.key
        return int(k, 16)

    def address(self) -> bytes:
        from cutline.crypto import address_of
        return address_of(self.private_key())

    def with_(self, **kw) -> "SidecarConfig":
        return dataclasses.replace(self, **kw)

    def ensure_dirs(self) -> None:
        for p in (self.state_path, self.metrics_path):
            if p:
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)


__all__ = ["SidecarConfig", "BYZANTINE_MODES"]
