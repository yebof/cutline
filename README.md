# CUTLINE

CUTLINE makes the product pixels of an e-commerce image publicly verifiable when everything around the product,
such as an AI-generated background, may be synthetic. Any platform, buyer or regulator can check it, and the check
still holds after platforms re-encode the image.

How it works:

1. **Capture.** An attested capture app fixes the product region when the photo is taken. It commits to the region
   cell by cell in a salted Merkle tree and signs a capture record with a hardware-backed key.
2. **Listing.** At listing time the seller opens exactly the committed cells that the product occupies. An exact
   integer compiler rebuilds the listing from these cells and a public background.
3. **Verification.** A verifier requires every certified JPEG block of the listing to *equal* libjpeg's integer
   encoding of that reference. There is no tolerance, so the test survives compression by construction.
4. **Re-encodes.** Each platform re-encode ("hop") is checked the same way against its public parent.
5. **Ledger.** The outcome is a certified outline and an evidence level from G0 (no evidence) to G4. A consortium
   ledger holds capture anchors, revocation, claim status and an on-chain court. Anyone can convict a false claim
   there with a single failing block.

## Repository layout

| Path | Contents |
|---|---|
| `cutline/` | Reference implementation in Python. See the table below. |
| `capture_emu/` | Software emulation of the attested capture app, its attestation chain and the seller's back office. |
| `sidecar/` | Validator sidecar for the consortium ledger: beacon pulses, revocation, watch and alarm, court submission and finalisation. |
| `contracts/` | Solidity contracts (Hardhat 3, solc 0.8.34, EVM Osaka) and a deployment script. |
| `abi/` | Contract ABI export and storage layout, read by `cutline.ledger`. |
| `env/ENV_PINS.json` | SHA-256 pins of the libjpeg builds loaded at run time. |
| `scripts/fetch_models.py` | Downloads the pinned ONNX segmenters. |

What each part of `cutline/` does:

| Module | Purpose |
|---|---|
| `jpeg/` | Whitelist container grammar and coefficient reading. |
| `commit/` | Pyramid, cells, Merkle trees and the capture record. |
| `compiler/` | Listing program, exact kernels, compositing and static rules. |
| `region/` | Capture-time region and the printed uncovered-product statistic. |
| `verify/` | Codec tests, hops, the evidence record and levels. |
| `ledger/` | Encodings, receipts and the client. |
| `bundle.py` | Claim bundle. |
| `cli.py` | Command-line tool. |

## Requirements

- **Python 3.12** on x86-64 Linux.
- **gcc and glibc**, to build a small C helper on first use. The helper is built automatically into
  `cutline/jpeg/_warncount/build/`. It dlopens the libjpeg-turbo 2.1.0 build that ships with the `jpeglib` package.
- **Node.js 22.22.2**, for the contracts only.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .
python scripts/fetch_models.py        # writes ./models (or $CUTLINE_MODELS_DIR)
```

Environment overrides:

| Variable | Default | What it points to |
|---|---|---|
| `CUTLINE_MODELS_DIR` | `./models` | the ONNX segmenters |
| `CUTLINE_ENV_PINS` | `./env/ENV_PINS.json` | the codec pins |
| `CUTLINE_ABI_DIR` | `./abi` | the ABI and storage layout |
| `CUTLINE_NODE_DIR` | `./encoders/node` | optional `sharp` encoder |
| `CUTLINE_GOENC` | `./encoders/bin/goenc` | optional Go encoder |

The two optional encoders are only needed to test additional encoder classes.

## Quick start

The capture side runs against the emulated app. Each command prints one JSON line.

```bash
cutline capture product.jpg --out cap                      # region fixed at capture, cells committed, record signed
cutline compile cap --out comp --auto --bg gen:1           # exact listing on a public background
cutline encode comp/X.png --out listing.jpg                # libjpeg islow encode (E0)
cutline bundle cap --compiled comp --listing listing.jpg --out bundle
cutline selfcheck bundle                                   # the seller's self-check (exit 0 iff the claim passes)
cutline verify bundle                                      # verdict and evidence fields
```

`cutline evaluate` returns the evidence level and the label. Its ledger is emulated unless you pass
`--chain-id`/`--registry`.

```bash
SALT=$(python -c "import hashlib;print(hashlib.sha256(b'cutline-cli salt_ref B0CLI00001').hexdigest())")
cutline evaluate bundle --listing-id B0CLI00001 --seller seller-cli --salt-ref $SALT
```

More subcommands:

- `cutline hop PARENT CHILD --crop Y,X,H,W` checks one platform re-encode.
- `cutline prove-block` produces the calldata of a single-block court proof against a failing listing.
- `cutline prove-static` produces the calldata of a static-rule violation proof.

## Contracts

```bash
cd contracts
npm ci
npx hardhat build
npx hardhat run scripts/deploy.js --network cutline        # in-process EDR chain
```

To deploy to a Besu QBFT network, set `BESU_URL` and `BESU_PRIVATE_KEYS` (comma-separated), then run
`npx hardhat run scripts/deploy.js --network besu`. `DEPLOY_N` sets the number of validators (default 7).

## Validator sidecar

```bash
python -m sidecar --help
python -m sidecar run --accounts KEYS.json --out sup --print-configs
```

## Author

Yebo Feng, Nanyang Technological University (NTU), fengyebo@gmail.com

## License

MIT, see [LICENSE](LICENSE). The libjpeg headers in `cutline/jpeg/_warncount/` keep their original notices.
