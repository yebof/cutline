"""cutline - the normative Python reference of CUTLINE.

Every bit of the capture commitment, the claim compiler, the listing format, the verifier and the ledger encodings is
defined here; the seller tool, the platform verifier, the validator sidecars and the Solidity contracts must agree with
it bit for bit. No floating point is normative: only the heuristic fields U and tone use pinned float or neural
implementations.

Subpackages and modules:
  consts, types, errors, crypto, enc, abi, bundle, registries, labels, phi, dct, edge, cli,
  jpeg/, commit/, compiler/, verify/, region/, ledger/
"""

__version__ = "0.0.0"
