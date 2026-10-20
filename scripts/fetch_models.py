#!/usr/bin/env python3
"""Download the pinned ONNX segmenters into ./models (or $CUTLINE_MODELS_DIR) and check their SHA-256."""
import hashlib
import os
import sys
import urllib.request

BASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
MODELS = {
    "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx": "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
    "u2netp.onnx": "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
    "isnet-general-use.onnx": "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.environ.get("CUTLINE_MODELS_DIR") or os.path.join(root, "models")
    os.makedirs(out, exist_ok=True)
    for name, digest in MODELS.items():
        path = os.path.join(out, name)
        if not (os.path.exists(path) and sha256(path) == digest):
            print(f"downloading {name}")
            urllib.request.urlretrieve(BASE + name, path)
        if sha256(path) != digest:
            sys.exit(f"{name}: SHA-256 mismatch")
        print(f"ok  {name}")


if __name__ == "__main__":
    main()
