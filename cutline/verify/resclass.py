"""cutline.verify.resclass - the integer resolution class and Lambda_chain.

- ``lam_2p51(Q, eta20=ETA20_CLS)`` = max_p Σ_k (Q_k·2^20 + 2·eta20)·B30[k][p]; ``ms_2p51(Q)`` = (Q_0·2^20 + 2·eta20)·B30[0][0].
- ``res_class(q3) -> (cls, lam_y, lam_c, ms_y, ms_c)`` for the three per-component tables (Y, Cb, Cr): "fine",
  "standard" or "coarse" with exact integer thresholds (S = 2^51).
- ``worst(a, b)``: the worse of two classes (fine < standard < coarse).
- ``class_hop(q3, ceiling192)``: "coarse" when some entry exceeds the platform ceiling Q_hop(p, v*), else
  ``res_class(q3)[0]``.
- ``lambda_chain(q3_list) -> int``: Σ lam_y over the files of a chain (upload first); ``chain_class(classes)``.
Tables are 64 natural-order entries (bytes, lists or arrays). Python integers only.
"""
from __future__ import annotations

from cutline import consts
from cutline.consts import ETA20_CLS

S51 = 1 << 51
ORDER = ("fine", "standard", "coarse")


def _q(Q) -> list:
    q = [int(v) for v in (Q if not hasattr(Q, "reshape") else Q.reshape(-1))]
    if len(q) != 64:
        raise ValueError("a table has 64 entries")
    return q


def lam_2p51(Q, eta20: int = ETA20_CLS) -> int:
    q = _q(Q)
    B30 = consts.B30
    w = [(q[k] << 20) + 2 * eta20 for k in range(64)]
    return max(sum(w[k] * B30[k][p] for k in range(64)) for p in range(64))


def ms_2p51(Q, eta20: int = ETA20_CLS) -> int:
    q = _q(Q)
    return ((q[0] << 20) + 2 * eta20) * consts.B30[0][0]


def res_class(q3) -> tuple:
    qy, qcb, qcr = q3
    ly, lc = lam_2p51(qy), max(lam_2p51(qcb), lam_2p51(qcr))
    my, mc = ms_2p51(qy), max(ms_2p51(qcb), ms_2p51(qcr))
    if ly <= 50 * S51 and lc <= 70 * S51:
        c = "fine"
    elif ly <= 115 * S51 and lc <= 160 * S51 and my <= S51 and mc <= S51:
        c = "standard"
    else:
        c = "coarse"
    return c, ly, lc, my, mc


def worst(a: str, b: str) -> str:
    return ORDER[max(ORDER.index(a), ORDER.index(b))]


def class_hop(q3, ceiling192: bytes) -> str:
    flat = [v for t in q3 for v in _q(t)]
    if any(v > int(c) for v, c in zip(flat, bytes(ceiling192))):
        return "coarse"
    return res_class(q3)[0]


def lambda_chain(q3_list) -> int:
    return sum(lam_2p51(q3[0]) for q3 in q3_list)


def chain_class(classes) -> str:
    out = "fine"
    for c in classes:
        out = worst(out, c)
    return out
