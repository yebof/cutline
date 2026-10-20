"""sidecar.sources - bundle sources, the local cache and the peer cache server.

A sidecar takes the bundle **from any source**: every verifying copy yields the same tested set and transcript, so
it abstains only when no source yields a verifying bundle within W_vote.

- ``fetch(url, limit, timeout) -> (bytes | None, reason)``: ``http(s)://`` (urllib; at most ``limit`` bytes, else
  ``"too_large"``), ``file://`` or a plain path (a ustar file, or a bundle directory re-serialised with
  ``bundle.ustar_bytes``). Never raises: a missing, unreachable or oversized copy is ``(None, reason)``.
- ``BundleCache(dir)``: ``get(claim_id) -> bytes | None``, ``put(claim_id, data)`` (atomic), ``path(claim_id)``;
  file name ``<claim_id hex>.tar``. Only copies that produced a vote (or a claim-wide abstention) are stored.
- ``CacheServer(cache, host="127.0.0.1", port=0)``: stdlib HTTP server for ``GET /bundle/0x<claim_id>`` (the peer
  cache); ``start()`` returns the base URL; ``stop()``.
- ``source_list(cfg, target, locators, cache) -> [descriptor]``: the order tried: the locators of every
  ``ChallengeOpened`` on the target (the triggering one first), the platform templates, the listing-page templates,
  the peer caches, the local cache (``"cache:"``). Duplicates are dropped.
- ``Copies(descriptors, limit, timeout, cache, target)``: an iterator of fetched copies for
  ``verify.vote.vote_any`` (lazy: the next source is fetched only when the previous copy did not verify);
  ``.log`` = [(descriptor, reason)], ``.last`` = the descriptor of the last copy yielded, ``.last_bytes``.
"""
from __future__ import annotations

import http.server
import os
import tempfile
import threading
import urllib.error
import urllib.request


def _hex(b: bytes) -> str:
    return "0x" + bytes(b).hex()


def fetch(url: str, limit: int, timeout: float = 20.0) -> tuple:
    try:
        if url.startswith(("http://", "https://")):
            req = urllib.request.Request(url, headers={"User-Agent": "cutline-sidecar/1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read(limit + 1)
            if len(data) > limit:
                return None, "too_large"
            return data, "ok"
        path = url[len("file://"):] if url.startswith("file://") else url
        if os.path.isdir(path):
            from cutline import bundle as B
            m, _ = B.load_members(path)
            data = B.ustar_bytes(m)
        else:
            if os.path.getsize(path) > limit:
                return None, "too_large"
            with open(path, "rb") as f:
                data = f.read()
        return data, "ok"
    except urllib.error.HTTPError as e:
        return None, f"http_{e.code}"
    except FileNotFoundError:
        return None, "missing"
    except Exception as e:                      # unreachable host, timeout, a broken directory, ...
        return None, "fetch:" + type(e).__name__


class BundleCache:
    def __init__(self, directory: str | None):
        self.dir = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    def path(self, claim_id: bytes) -> str | None:
        return None if not self.dir else os.path.join(self.dir, bytes(claim_id).hex() + ".tar")

    def get(self, claim_id: bytes) -> bytes | None:
        p = self.path(claim_id)
        if p is None or not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def put(self, claim_id: bytes, data: bytes) -> None:
        p = self.path(claim_id)
        if p is None:
            return
        fd, tmp = tempfile.mkstemp(prefix=".bundle-", dir=self.dir)
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)


class _Handler(http.server.BaseHTTPRequestHandler):
    cache: BundleCache = None

    def do_GET(self):  # noqa: N802
        parts = self.path.strip("/").split("/")
        data = None
        if len(parts) == 2 and parts[0] == "bundle":
            h = parts[1][2:] if parts[1].startswith("0x") else parts[1]
            try:
                cid = bytes.fromhex(h)
            except ValueError:
                cid = b""
            if len(cid) == 32:
                data = self.cache.get(cid)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-tar")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class CacheServer:
    def __init__(self, cache: BundleCache, host: str = "127.0.0.1", port: int = 0):
        handler = type("H", (_Handler,), {"cache": cache})
        self.httpd = http.server.ThreadingHTTPServer((host, port), handler)
        self.thread = None

    @property
    def url(self) -> str:
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}"

    def start(self) -> str:
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def source_list(cfg, target: bytes, locators, cache: BundleCache | None = None) -> list:
    cid = _hex(target)
    out = []
    for loc in locators or []:
        out.append(str(loc))
    for t in list(cfg.platform_templates) + list(cfg.listing_templates):
        out.append(t.format(claim_id=cid))
    for base in cfg.peer_caches:
        out.append(base.rstrip("/") + "/bundle/" + cid)
    if cache is not None and cache.dir:
        out.append("cache:")
    seen, res = set(), []
    for d in out:
        if d and d not in seen:
            seen.add(d)
            res.append(d)
    return res


class Copies:
    """Lazy iterator of fetched copies (bytes) for ``vote_any``; see the module docstring."""

    def __init__(self, descriptors, limit: int, timeout: float, cache: BundleCache | None, target: bytes):
        self.descriptors = list(descriptors)
        self.limit, self.timeout, self.cache, self.target = int(limit), float(timeout), cache, bytes(target)
        self.log = []
        self.last = None
        self.last_bytes = None

    def __iter__(self):
        for d in self.descriptors:
            if d == "cache:":
                data = self.cache.get(self.target) if self.cache is not None else None
                reason = "ok" if data is not None else "missing"
            else:
                data, reason = fetch(d, self.limit, self.timeout)
            self.log.append((d, reason))
            if data is None:
                continue
            self.last, self.last_bytes = d, data
            yield data


__all__ = ["fetch", "BundleCache", "CacheServer", "source_list", "Copies"]
