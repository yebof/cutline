"""cutline.verify.excess - the integer excess E_inf and E_2 of a CELL block.

- ``e_inf_block(D80, q, eta20) -> (einf_2p30, E2, A, sigma)``: for k = 0..63,
  ``b_k = (2·⌊D80_k/(q_k·2^80)⌋ + 1)·q_k·2^79`` (signed floor), tie set ``A = {k : |D80_k − b_k| ≤ (eta20 + q_k)·2^60}``,
  ``σ_k = +1 if b_k > D80_k else −1``, ``E_inf·2^30 = max_p max(Σ_{k∈A} max(0, σ_k q_k B30s[k][p]),
  Σ_{k∈A} max(0, −σ_k q_k B30s[k][p]))``, ``E_2 = isqrt(Σ_{k∈A} q_k^2)``. ``sigma`` is a list aligned with ``A``.
- ``nearest_rank(values, P)``: the value at 1-based rank ⌈P/100·N⌉ of the ascending values; None if empty.
- ``excess_summary(per_block)``: {"p50", "p99", "max"} of the per-block values (zeros included).
A table entry q_k < 1 (or a wrong length) raises ``E.CONTAINER.DQT_ZERO_ENTRY``.
EXACT and EP blocks have E_inf = E_2 = 0 (no call needed). Python integers only.
"""
from __future__ import annotations

import math

from cutline import consts
from cutline.errors import ContainerError


def e_inf_block(D80, q, eta20: int) -> tuple:
    D = [int(v) for v in D80]
    qq = [int(v) for v in (q.reshape(-1) if hasattr(q, "reshape") else q)]
    if len(qq) != 64 or len(D) != 64 or min(qq) < 1:
        # unreachable after ContainerOK (which rejects zero DQT entries); named instead of ZeroDivisionError
        raise ContainerError("E.CONTAINER.DQT_ZERO_ENTRY", "q_k must be in [1, 255] for k = 0..63")
    A, sig = [], []
    for k in range(64):
        qk = qq[k]
        bk = (2 * (D[k] // (qk << 80)) + 1) * qk * (1 << 79)
        if abs(D[k] - bk) <= (eta20 + qk) << 60:
            A.append(k)
            sig.append(1 if bk > D[k] else -1)
    B = consts.B30S
    best = 0
    for p in range(64):
        pos = neg = 0
        for k, s in zip(A, sig):
            v = s * qq[k] * B[k][p]
            if v > 0:
                pos += v
            else:
                neg -= v
        best = max(best, pos, neg)
    e2 = math.isqrt(sum(qq[k] ** 2 for k in A))
    return best, e2, A, sig


def nearest_rank(values, P):
    v = sorted(int(x) for x in values)
    if not v:
        return None
    n = len(v)
    r = -(-P * n // 100) if isinstance(P, int) else math.ceil(P / 100 * n)
    return v[max(1, r) - 1]


def excess_summary(per_block) -> dict:
    return {"p50": nearest_rank(per_block, 50), "p99": nearest_rank(per_block, 99),
            "max": max((int(x) for x in per_block), default=None)}
