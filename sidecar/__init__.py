"""Validator sidecar.

Per validator: pulse transactions every beacon epoch (15 blocks); path-(b) PASS/FAIL votes with integer transcripts
within W_vote, evaluated at the ChallengeOpened block (tier T under the claim's own table); abstention only without
a verifying bundle **from any source** (every ChallengeOpened locator on the target, the relaying platform, the
listing page, peer caches, the local cache; ``verify.vote.vote_any``); appeal votes (f+2, PASS only); finalisation
calls; revocation-list polling with REVOKE / KEEP votes; enrolment checks incl. the holder challenge against the
pinned attestation root; the watcher's court duty (static proofs and 4:4:4 block-failure proofs, checked against the
reference models first); the cross-node hash alarm; restartable state; backoff and a bounded vote queue.

Modules: ``chain`` (the ``ChainPort`` boundary, function / event tables), ``rpcport`` (JSON-RPC port on
``cutline.ledger``), ``config``, ``state``, ``metrics``, ``watch``, ``sources``, ``runner``, ``verdict``
(challenge and appeal votes), ``pulse``, ``finalize``, ``enrol``, ``revocation``, ``court``, ``service``
(``Sidecar``, ``Supervisor``), ``__main__`` (CLI).
"""
