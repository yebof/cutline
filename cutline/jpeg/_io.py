"""cutline.jpeg._io - temporary files for the path-based jpeglib calls.

``tmpdir()`` = ``$TMPDIR`` if set, else ``<system temp dir>/cutline``; ``temp_jpeg(data)`` is a context manager
yielding the path of a temporary file holding ``data`` (removed on exit).
"""
from __future__ import annotations

import contextlib
import os
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # repository root


def tmpdir() -> str:
    d = os.environ.get("TMPDIR") or os.path.join(tempfile.gettempdir(), "cutline")
    os.makedirs(d, exist_ok=True)
    return d


@contextlib.contextmanager
def temp_path(suffix: str = ".jpg"):
    fd, p = tempfile.mkstemp(suffix=suffix, dir=tmpdir())
    os.close(fd)
    try:
        yield p
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(p)


@contextlib.contextmanager
def temp_jpeg(data: bytes, suffix: str = ".jpg"):
    with temp_path(suffix) as p:
        with open(p, "wb") as f:
            f.write(data)
        yield p
