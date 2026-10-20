"""``python -m sidecar``: the validator sidecar command line.

Commands:

- ``run --out DIR [--config SUP.json] [--accounts KEYS.json] [--validators 1..n] [--rpc-base 8545] [--no-pulse]
  [--byzantine MODE@i,j ...] [--silent-until H] [--watcher] [--isolation process|inline] [--print-configs]``: the
  supervisor. With ``--accounts`` (a JSON list of 0x private keys indexed by account number), validator i uses
  account i (its node key = its Governance validator key) and RPC ``http://127.0.0.1:<rpc-base + i - 1>``;
  ``--watcher`` adds a watcher sidecar with account 13 on node 1. ``SUP.json`` = ``{"common": {config fields},
  "validators": [{...}], "watcher": {...} | null}`` overrides these defaults; one of ``--accounts`` or a
  ``validators`` list is required. Every sidecar's peers are the other nodes' RPCs (cross-node alarm); peer caches
  are not configured by default (``peer_caches`` in ``common`` if wanted). Writes one config per sidecar under
  ``--out`` (plus ``state/``, ``metrics/``, ``cache/`` and ``logs/``) and runs them as children
  (``service.Supervisor``).
- ``one --config CFG.json``: one sidecar over ``rpcport.RpcPort`` (the supervisor's child).
- ``serve-cache --dir D [--port P]``: a peer cache server (``sources.CacheServer``) in the foreground.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

WATCHER_ACCOUNT = 13


def _range(s: str) -> list:
    if ".." in s:
        a, b = s.split("..")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _defaults(a) -> dict:
    if not a.accounts:
        if a.watcher:
            raise SystemExit("--watcher needs --accounts")
        return {"common": {}, "validators": [], "watcher": None}
    keys = json.load(open(a.accounts))
    vals = _range(a.validators)
    rpcs = {i: f"http://127.0.0.1:{a.rpc_base + i - 1}" for i in vals}
    sup = {"common": {}, "validators": [{"index": i, "key": keys[i], "rpc": rpcs[i],
                                         "peer_rpcs": [rpcs[j] for j in vals if j != i]} for i in vals],
           "watcher": None}
    if a.watcher:
        sup["watcher"] = {"index": 100 + WATCHER_ACCOUNT, "key": keys[WATCHER_ACCOUNT], "rpc": rpcs[vals[0]],
                          "watcher": True, "watch_claims": True, "pulse": False, "finalize": False}
    return sup


def cmd_run(a) -> int:
    from sidecar.service import Supervisor, write_configs
    sup = _defaults(a)
    if a.config:
        user = json.load(open(a.config))
        sup["common"].update(user.get("common", {}))
        if "validators" in user:
            sup["validators"] = user["validators"]
        if "watcher" in user:
            sup["watcher"] = user["watcher"]
    if not sup["validators"]:
        raise SystemExit("no validators: pass --accounts or --config with a 'validators' list")
    common = dict(sup["common"])
    if a.no_pulse:
        common["pulse"] = False
    if a.isolation:
        common["isolation"] = a.isolation
    over = {}
    for spec in a.byzantine or []:
        mode, _, who = spec.partition("@")
        for i in _range(who):
            over.setdefault(i, {})["byzantine"] = mode
            if mode == "silent" and a.silent_until is not None:
                over[i]["silent_until"] = a.silent_until
    vals = list(sup["validators"]) + ([sup["watcher"]] if sup.get("watcher") else [])
    paths = write_configs(common, vals, a.out, over)
    if a.print_configs:
        print(json.dumps(paths))
        return 0
    Supervisor(paths, log_dir=os.path.join(a.out, "logs")).run()
    return 0


def cmd_one(a) -> int:
    import signal
    from sidecar.config import SidecarConfig
    from sidecar.rpcport import RpcPort
    from sidecar.service import Sidecar
    cfg = SidecarConfig.from_json(a.config)
    port = RpcPort(cfg.rpc, cfg.private_key(), chain_id=cfg.chain_id)
    peers = [RpcPort(u) for u in cfg.peer_rpcs]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    Sidecar(cfg, port, peers).run(stop)
    return 0


def _serve(server) -> int:
    print(server.start(), flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.stop()
    return 0


def cmd_serve_cache(a) -> int:
    from sidecar.sources import BundleCache, CacheServer
    return _serve(CacheServer(BundleCache(a.dir), port=a.port))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sidecar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config")
    r.add_argument("--accounts", help="JSON list of 0x private keys, indexed by account number")
    r.add_argument("--validators", default="1..4")
    r.add_argument("--rpc-base", type=int, default=8545)
    r.add_argument("--no-pulse", action="store_true")
    r.add_argument("--byzantine", action="append", help="MODE@i,j or MODE@i..j (fail-all, pass-all, silent)")
    r.add_argument("--silent-until", type=int)
    r.add_argument("--watcher", action="store_true")
    r.add_argument("--isolation", choices=("process", "inline"))
    r.add_argument("--out", required=True, help="directory for the configs, state, metrics, caches and logs")
    r.add_argument("--print-configs", action="store_true")
    r.set_defaults(fn=cmd_run)
    o = sub.add_parser("one")
    o.add_argument("--config", required=True)
    o.set_defaults(fn=cmd_one)
    c = sub.add_parser("serve-cache")
    c.add_argument("--dir", required=True)
    c.add_argument("--port", type=int, default=0)
    c.set_defaults(fn=cmd_serve_cache)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
