"""cutline.commit.pyramid - octave premultiplied RGBA pyramid.

For a master M (H0 x W0 x 3, uint8) and the certified region A (H0 x W0, bool), level l has
h_l x w_l = ceil(H0/2^l) x ceil(W0/2^l) pixels; Lmax is the smallest l with max(h_l, w_l) <= 256. For level pixel
(i, j) with master box B (2^l x 2^l, clipped at the master edge, n = |B|):

    S_c = sum_{p in B} M_c(p) A(p),  N = sum_{p in B} A(p)
    C_l[i, j, c] = floor((2 S_c + n) / (2 n)),   alpha_l[i, j] = floor((2 * 255 * N + n) / (2 n))

Every level is a function of the exact integer box sums of M (never of a rounded coarser/finer level). The sums are
built hierarchically (the 2 x 2 sum of the exact level-l sums is the exact level-(l+1) sum), one channel at a time,
so a 12-MP master needs well under 2 GB of memory.

- ``lmax(H0, W0) -> int``.
- ``level_dims(H0, W0, l) -> (h_l, w_l)``.
- ``level_planes(M, A, l) -> (C uint8[h,w,3], alpha uint8[h,w])``: one level.
- ``pyramid(M, A, Lmax=None) -> list[(C, alpha)]``: levels 0..Lmax (default ``lmax``).
"""
from __future__ import annotations

import numpy as np

LEVEL_LIMIT = 256            # Lmax: smallest l with max(h_l, w_l) <= 256


def lmax(H0: int, W0: int) -> int:
    if H0 < 1 or W0 < 1:
        raise ValueError("master must be at least 1 x 1")
    l = 0
    while max(level_dims(H0, W0, l)) > LEVEL_LIMIT:
        l += 1
    return l


def level_dims(H0: int, W0: int, l: int) -> tuple:
    f = 1 << l
    return -(-H0 // f), -(-W0 // f)


def _check(M, A):
    M = np.asarray(M)
    A = np.asarray(A)
    if M.ndim != 3 or M.shape[2] != 3 or M.dtype != np.uint8:
        raise ValueError("M must be uint8 (H0, W0, 3)")
    if A.shape != M.shape[:2]:
        raise ValueError("A must be (H0, W0)")
    H0, W0 = A.shape
    if not (1 <= H0 < 65536 and 1 <= W0 < 65536):
        raise ValueError("H0, W0 must be in [1, 65535]")
    return M, A.astype(bool)


def _box_counts(n0: int, l: int) -> np.ndarray:
    """Master pixels per level-l row (or column): min(2^l, n0 - i 2^l)."""
    f = 1 << l
    n = -(-n0 // f)
    return np.minimum(f, n0 - np.arange(n, dtype=np.int64) * f)


def _down2(S: np.ndarray) -> np.ndarray:
    """Exact 2 x 2 sums (zero padding at an odd edge)."""
    h, w = S.shape
    if h % 2 or w % 2:
        S = np.pad(S, ((0, h % 2), (0, w % 2)))
    return S.reshape(S.shape[0] // 2, 2, S.shape[1] // 2, 2).sum(axis=(1, 3))


def _levels(M, A, L: int, want=None):
    """Yield (l, C_l, alpha_l) for l in 0..L (only the levels in ``want`` are materialised)."""
    H0, W0 = A.shape
    want = set(range(L + 1)) if want is None else set(want)
    out = {l: [None, None] for l in want}
    N = A.astype(np.int64)                     # alpha: region-pixel counts
    for l in range(L + 1):
        if l:
            N = _down2(N)
        if l in want:
            n = _box_counts(H0, l)[:, None] * _box_counts(W0, l)[None, :]
            out[l][1] = ((2 * 255 * N + n) // (2 * n)).astype(np.uint8)
    del N
    for l in want:
        out[l][0] = np.empty(level_dims(H0, W0, l) + (3,), np.uint8)
    for c in range(3):
        S = M[..., c].astype(np.int64)
        S[~A] = 0
        for l in range(L + 1):
            if l:
                S = _down2(S)
            if l in want:
                n = _box_counts(H0, l)[:, None] * _box_counts(W0, l)[None, :]
                out[l][0][..., c] = (2 * S + n) // (2 * n)
        del S
    return [(l, out[l][0], out[l][1]) for l in sorted(want)]


def level_planes(M, A, l: int) -> tuple:
    M, A = _check(M, A)
    if l < 0:
        raise ValueError("level must be >= 0")
    (_, C, al), = _levels(M, A, l, want=[l])
    return C, al


def pyramid(M, A, Lmax: int | None = None) -> list:
    M, A = _check(M, A)
    L = lmax(*A.shape) if Lmax is None else Lmax
    return [(C, al) for _, C, al in _levels(M, A, L)]
