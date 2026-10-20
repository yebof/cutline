"""Emulated attested capture app, seller back office and aggregator.

A software stand-in for the capture side: the master from a decoded phone frame, the certified region with bounded
hole filling, commitments and rec v3 signed by a software P-256 key (``cryptography``), provisioned owner addresses,
and an aggregator that anchors v2 leaves. The attestation chain is an emulated test-root X.509 chain whose leaf
carries the Android KeyDescription extension (DER via ``asn1crypto``, OID 1.3.6.1.4.1.11129.2.1.17) with
attestationChallenge = H("CUTLINE-ENROL-v1" || holder || rateN || chainId || nonce_E). The back office implements the
normative holder policy (check 0 on sigma_dev, owner address, slot, provisioning order, compromise signal, idempotent
retries).

Modules: ``attestation``, ``app``, ``backoffice``, ``aggregator``, ``chain_stub``, ``session``.
"""
