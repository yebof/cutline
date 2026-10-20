"""cutline.consts - domain constants.

Every tag and domain byte string (the exact bytes, including the tag byte), the retired tags (for rejection), the
genesis constants, the zig-zag map, the IJG tables (``STD_L``, ``STD_C``, ``ijg_table``), Q_pol and IJG q75 (192 bytes
each, with SHA-256 self-checks at import), T_A and Q_hop(Amazon, 1) of the worked example, the encoder-class registry
{1, 3, 5, 7} (2, 4 and 6 reserved), the genesis kernel families, the jccolor constants of Phi and Phi_go, the frozen
selectors, and the two integer tables:

- ``C40``: 64 ints, u-major (``C40[8*u + x]`` = round(2^40 a(u) cos((2x+1) u pi / 16))), SHA-256 of the 512 bytes
  int64 big-endian ``3b9edd6b...``.
- ``B30S``: 64 x 64 signed ints, k-major (``B30S[k][p]``, k = 8v + u, p = 8y + x), SHA-256 of the 4,096 int64
  big-endian values ``8a4a7b60...``; ``B30`` = the absolute values. ``b30s_np()`` / ``c40_np()`` give int64 arrays.

``C40``, ``B30S`` and ``B30`` are computed on first access with mpmath at 60 digits (at least 50 are required) and
cached; their SHA-256 is asserted when they are built (module ``__getattr__``, so ``import cutline.consts`` stays
cheap). The same constants appear in the Solidity library ``CutlineLib.sol``.
"""
from __future__ import annotations

import hashlib
import struct
from functools import lru_cache

# ------------------------------------------------------------------------------------------------ tagged SHA-256
TAG_CELL = b"\x00CUTLINE-cell-v1"            # || u8 l || u16 cy || u16 cx || salt16 || RGBA[1024]      (1,061 B)
TAG_ANCHOR_V2 = b"\x00CUTLINE-anchor-v2"     # || recHash || H(r || s_low) || slotTag                   (114 B)
TAG_FILL = b"\x00CUTLINE-fill-v1"            # || u32 mcu || RGB[p*p*3]                               (212 / 788 B)
TAG_FILLOV = b"\x00CUTLINE-fillov-v1"        # reserved for v1.1; MUST NOT appear in a v1 claim
TAG_MCU_V2 = b"\x00CUTLINE-mcu-v2"           # || u32 mcu || i16[64] per real block                    (19 + 128 b)
TAG_HOP = b"\x00CUTLINE-hop-v1"              # || R_Y(parent) || R_Y(child) || opHash                  (111 B)
TAG_NODE = b"\x01"                           # || left || right                                        (65 B)
TAG_ROOTS = b"\x02CUTLINE-roots"             # || u16 H0 || u16 W0 || u8 Lmax || R_0 .. R_Lmax         (19 + 32 k)
TAG_OCC = b"\x03"                            # || H(packbits(O_0)) || ...                              (1 + 32 k)
TAG_LISTING_V2 = b"\x04CUTLINE-listing-v2"   # || header[197] || R_mcu                                 (248 B)
TAG_PROG = b"\x05CUTLINE-prog-v1"            # || enc(pi)                                              (810 + 4 nD)

# Retired in v1: MUST be rejected wherever v1 expects the current version.
RETIRED_TAGS = (b"\x00CUTLINE-anchor-v1", b"\x00CUTLINE-mcu-v1", b"\x04CUTLINE-listing-v1")
RETIRED_STRINGS = ("CUTLINE-anchor-v1", "CUTLINE-mcu-v1", "CUTLINE-listing-v1", "CUTLINE-REC-v1", "CUTLINE-REC-v2",
                   "CUTLINE-CLAIM-v1")

# ------------------------------------------------------------------------------------------------ untagged prefixes
PFX_NF = b"nf"                               # || recHash                                              (34 B)
PFX_SLOT = b"slot"                           # || keyId || u32 epoch || u16 slot                       (42 B)
PFX_ENROL = b"CUTLINE-ENROL-v1"              # || holder[20] || u16 rateN || u256 chainId || nonce_E[16] (86 B)
PFX_LREF = b"CUTLINE-lref"                   # || u8 m || H(l) || H(s) || salt_ref[32]                 (109 B)
PFX_PASS = b"pass"                           # || target (claimId)                                     (36 B)
PFX_KEEP = b"keep"                           # || keyId                                                (36 B)
PFX_TIERT = b"CUTLINE-tierT-v1"              # || table record[55]                                     (71 B)
PFX_PUBSEG = b"CUTLINE-pubseg-v1"            # || H(model) || H(JCS manifest)                          (81 B)
PFX_UREC = b"CUTLINE-urec-v1"                # first bytes of the U record

# ------------------------------------------------------------------------------------------------ HMAC messages
HM_SALT = b"salt"                            # k_cap: || u8 l || u16 cy || u16 cx -> first 16 bytes
HM_MASTER = b"master"                        # k_cap: || master RGB bytes -> R_M
HM_STROKES = b"strokes"                      # k_cap: || packbits(m) -> H_m
HM_SLOT = b"slot"                            # k_slot: || u32 epoch || u32 counter -> PRP stream

# ------------------------------------------------------------------------------------------------ keccak (ABI strings)
DOM_CLAIM = "CUTLINE-CLAIM-v2"
DOM_VOUCH = "CUTLINE-VOUCH-v1"
DOM_PRESENT = "CUTLINE-PRESENT-v1"
DOM_RELAY = "CUTLINE-RELAY-v1"
DOM_ENROL_VOTE = "enrol"                     # keccak256(abi.encode("enrol", keyId, pkHash)), unsigned

# ------------------------------------------------------------------------------------------------ domain words
REC_DOMAIN_STR = b"CUTLINE-REC-v3"
REC_DOMAIN = REC_DOMAIN_STR + b"\x00" * (32 - len(REC_DOMAIN_STR))          # rec word 0 (bytes32)
BUNDLE_VERSION_STR = b"CUTLINE-bundle-v1"
BUNDLE_VERSION = BUNDLE_VERSION_STR + b"\x00" * (32 - len(BUNDLE_VERSION_STR))   # word 1 of claim.bin
PICAP_V = "CUTLINE-picap-v1"
PUBSEG_MANIFEST_V = "CUTLINE-pubseg-manifest-v1"
HKDF_OWNER_INFO = b"cutline-owner"           # info = || u32 capIndex; salt empty; L = 32

# evaluation-only domains (never in a protocol digest)
E5_DOMAINS = ("CUTLINE-E5-draw-v1", "CUTLINE-E5-bg-v1", "CUTLINE-E5-pc-v1", "CUTLINE-E5-prompt-v1",
              "CUTLINE-E5-seed-v1")

# Empty-row FAIL transcript H(abi.encode(T[] = [])) = H(u256 0x20 || u256 0): a constant REJECTED by submitVerdict
EMPTY_ROW_TRANSCRIPT = hashlib.sha256((32).to_bytes(32, "big") + bytes(32)).digest()

# ------------------------------------------------------------------------------------------------ genesis constants
DELTA = 240                                  # blocks
BEACON_EPOCH = 15                            # h_beta = 0 mod 15 (E.ANCHOR.BEACON_BOUNDARY)
RATE_EPOCH = 1800                            # epoch = h_beta div 1800
RATEN_MIN, RATEN_MAX, RATEN_DEFAULT = 30, 600, 120
W_REG = 1_296_000
W_VOTE = 150
EXCL_LICENCE_CAP = 31_536_000
K_RECEIPT = 30                               # receipt freshness, blocks
T_WALL = 300                                 # seconds
MARKETPLACE_LIMIT = 64                       # 0 <= m < 64
D_HOP = 3                                    # G3/G4 depth cap
RESAMPLING_HOPS_MAX = 1                      # resampling hops for G3/G4
CHAIN_LEN_CAP = 8
ASPECT_A = 1_000_000                         # rule 6: 10^6 * cw * oh <= 1020202 * ch * ow (and symmetric)
ASPECT_B = 1_020_202
MASTER_MAX_PIXELS = 1 << 28                  # static rule 9(0): H0 * W0 <= 2^28 = 268,435,456 px
CROP_MAX_PIXELS = 1 << 25                    # static rule 9(1): ch * cw <= 2^25 = 33,554,432 px (the SOF cap)
TAU_U = 32                                   # block equivalents per listing MP
ADJ_CHEBYSHEV = 8                            # 17 x 17 square
A_MAX_PPM = 2048                             # a_max, px per master MP
PHI_D_PPM = 20_000                           # phi_D = 2 %
ETA20_CLASS3 = 52_429                        # eta 0.05 * 2^20
ETA20_CLS = 262_144                          # eta 0.25 * 2^20 (class 5, resolution classes, Lambda_chain)
NU20 = 1                                     # nu = 2^-20 steps
TIERT_ID_MIN, TIERT_ID_MAX = 1024, 65535
ATT_VERSION = 300                            # emulated attestation profile v1
KEYMINT_VERSION = 300
CATEGORY_GENERAL, CATEGORY_COSMETICS, CATEGORY_APPAREL = 0, 1, 2
TIER_E, TIER_T = 1, 2
SPEC_VERSION = "cutline-spec-v1"             # evidence record versions.spec
U_IS_LEVEL_CRITERION = False                 # release constant: U is reported, not a level criterion

# ------------------------------------------------------------------------------------------------ registries
# encClass -> (kind, eta20); kind 1 = EXACT, 2 = CELL, 0 = not court-checkable
CLASS_REGISTRY = {1: (1, 0), 3: (2, 52_429), 5: (0, 262_144), 7: (0, 0)}
RESERVED_CLASSES = frozenset({2, 4, 6})
KIND_NONE, KIND_EXACT, KIND_CELL = 0, 1, 2
# declared kernel family f -> exact kernel (0 = area, 1 = triangle), genesis
KFAM_GENESIS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
KERNEL_AREA, KERNEL_TRIANGLE = 0, 1

# ------------------------------------------------------------------------------------------------ frozen selectors
SEL_PROVE_BLOCK = bytes.fromhex("00aa6b4d")          # proveBlockFailure(ProofIn)
SEL_PROVE_STATIC = bytes.fromhex("8d02fe1f")         # proveStaticViolation(StaticIn)
SEL_DELEGATE = bytes.fromhex("90e75851")             # delegate(bytes32,address,uint64,bool,uint64,bytes32[])
CLAIM_CORE_T = "(bytes32,bytes32,bytes32,bytes32,uint16,uint8,uint8,uint8,bytes32)"
SIG_PROVE_STATIC = ("proveStaticViolation((bytes32,address," + CLAIM_CORE_T +
                    ",uint8,uint8,uint16,bytes,bytes,bytes32,bytes,bytes))")
SIG_DELEGATE = "delegate(bytes32,address,uint64,bool,uint64,bytes32[])"
SIG_PROVE_BLOCK = ("proveBlockFailure((bytes32,address," + CLAIM_CORE_T + ",bytes,bytes,bytes32[],bytes,bytes,bytes,"
                   "uint32,bytes,bytes32[],bytes,bytes32[],(uint16,uint16,bytes16,bytes,bytes32[])[],uint8,uint8))")

# ------------------------------------------------------------------------------------------------ JPEG tables
# zig-zag index -> natural index
ZZ = (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
      28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
      54, 47, 55, 62, 63)
STD_L = (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
         51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121,
         120, 101, 72, 92, 95, 98, 112, 100, 103, 99)
STD_C = (17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99,
         99, 99, 99, 99) + (99,) * 32


def ijg_table(std, q: int) -> bytes:
    """IJG quality scaling (natural order, 64 bytes): s = 5000/q (q < 50) else 200 - 2q; clip((std*s + 50)/100)."""
    if not 1 <= q <= 100:
        raise ValueError("quality in 1..100")
    s = 5000 // q if q < 50 else 200 - 2 * q
    return bytes(min(255, max(1, (v * s + 50) // 100)) for v in std)


def q3_bytes(luma: bytes, chroma: bytes, chroma2: bytes | None = None) -> bytes:
    """The 192-byte ``q[0] || q[1] || q[2]`` (Y, Cb, Cr) of a header."""
    return bytes(luma) + bytes(chroma) + bytes(chroma if chroma2 is None else chroma2)


Q_POL_L = ijg_table(STD_L, 85)
Q_POL_C = ijg_table(STD_C, 85)
Q_POL = q3_bytes(Q_POL_L, Q_POL_C)                    # genesis constant (192 B)
IJG_Q75_L = ijg_table(STD_L, 75)
IJG_Q75_C = ijg_table(STD_C, 75)
IJG_Q75 = q3_bytes(IJG_Q75_L, IJG_Q75_C)              # Q_hop(p, 0) for every platform
TA_L = bytes((11, 8, 8, 11, 16, 19, 22, 21, 8, 8, 8, 11, 14, 18, 17, 21, 8, 8, 9, 11, 18, 19, 25, 33, 11, 11, 11, 19, 18,
              26, 33, 48, 16, 14, 18, 18, 27, 36, 47, 54, 19, 18, 19, 26, 36, 47, 54, 54, 22, 17, 25, 33, 47, 54, 54, 54,
              21, 21, 33, 48, 54, 54, 54, 54))
TA_C = bytes((12, 11, 12, 15, 19, 20, 26, 34, 11, 13, 13, 14, 14, 18, 24, 30, 12, 13, 14, 15, 18, 21, 32, 37, 15, 14, 15,
              20, 21, 27, 37, 47, 19, 14, 18, 21, 27, 34, 47, 54, 20, 18, 21, 27, 34, 44, 54, 54, 26, 24, 32, 37, 47, 54,
              54, 54, 34, 30, 37, 47, 54, 54, 54, 54))
T_A = q3_bytes(TA_L, TA_C)                           # Amazon's served tables (worked example)
Q_HOP_AMAZON_1 = bytes(max(a, b) for a, b in zip(IJG_Q75, T_A))

SHA_Q_POL = "0984e21391e025c2a441ff74f3cbf66a18aadfaa082994c7e2114b0ad8b0c317"
SHA_IJG_Q75 = "85eeea021be85a084773c9ad479cf1bf57eec2c9544535588b418765d43a8191"
SHA_T_A = "bd9752b8ddf689bcf3b0935154e84281f7b8683ab932ff25b4dc170af0db1dd6"
SHA_Q_HOP_AMAZON_1 = "2d0eeea6f807868f565482980b70af645ac585b6b208a4f2626a1291ca271066"
SHA_C40 = "3b9edd6b43249fd314ef32ba61d470cbdcb02ef5249083101939ad0d4459d379"
SHA_B30S = "8a4a7b60b54d7006302d066e7d4cc195b4f5121f9945dc957e80a883ee82d74d"

for _b, _h in ((Q_POL, SHA_Q_POL), (IJG_Q75, SHA_IJG_Q75), (T_A, SHA_T_A), (Q_HOP_AMAZON_1, SHA_Q_HOP_AMAZON_1)):
    assert hashlib.sha256(_b).hexdigest() == _h, "table self-check failed"
assert Q_POL_L[:8] == bytes((5, 3, 3, 5, 7, 12, 15, 18))

# ------------------------------------------------------------------------------------------------ jccolor
# Phi (libjpeg rgb_ycc_convert, SCALEBITS 16)
PHI_Y = (19595, 38470, 7471, 32768)                  # Y  = (a R + b G + c B + 32768) >> 16
PHI_CB = (-11059, -21709, 32768, 8421375)            # Cb = (32768 B - 11059 R - 21709 G + 8421375) >> 16
PHI_CR = (32768, -27439, -5329, 8421375)             # Cr = (32768 R - 27439 G - 5329 B + 8421375) >> 16
# Phi_go (Go image/color.RGBToYCbCr): chroma offset 257 << 15, clamp via x & 0xff000000
PHI_GO_CB = (-11056, -21712, 32768, 257 << 15)
PHI_GO_CR = (32768, -27440, -5328, 257 << 15)
H2V2_BIAS = (1, 2)                                   # libjpeg h2v2_downsample: bias 1 (even j), 2 (odd j)
H2V2_GO_BIAS = 2                                     # Go: (sum + 2) >> 2


# ------------------------------------------------------------------------------------------------ C40 and B30 (lazy)
@lru_cache(maxsize=None)
def _c40() -> tuple:
    import mpmath
    with mpmath.workdps(60):
        a = [mpmath.sqrt(mpmath.mpf(1) / 8)] + [mpmath.mpf(1) / 2] * 7
        vals = tuple(int(mpmath.nint(a[u] * mpmath.cos((2 * x + 1) * u * mpmath.pi / 16) * mpmath.mpf(2) ** 40))
                     for u in range(8) for x in range(8))
    assert hashlib.sha256(b"".join(struct.pack(">q", v) for v in vals)).hexdigest() == SHA_C40, "C40 self-check"
    return vals


@lru_cache(maxsize=None)
def _b30s() -> tuple:
    import mpmath
    with mpmath.workdps(60):
        a = [mpmath.sqrt(mpmath.mpf(1) / 8)] + [mpmath.mpf(1) / 2] * 7
        cosv = [[mpmath.cos((2 * x + 1) * u * mpmath.pi / 16) for x in range(8)] for u in range(8)]
        two30, half = mpmath.mpf(2) ** 30, mpmath.mpf(1) / 2
        rows = []
        for v in range(8):
            for u in range(8):
                row = []
                for y in range(8):
                    for x in range(8):
                        val = a[u] * a[v] * cosv[u][x] * cosv[v][y] * two30
                        r = int(mpmath.floor(abs(val) + half))     # round half away from zero
                        row.append(r if val >= 0 else -r)
                rows.append(tuple(row))
    out = tuple(rows)
    flat = b"".join(struct.pack(">q", x) for row in out for x in row)
    assert hashlib.sha256(flat).hexdigest() == SHA_B30S, "B30S self-check"
    return out


@lru_cache(maxsize=None)
def _b30() -> tuple:
    return tuple(tuple(abs(x) for x in row) for row in _b30s())


def c40_np():
    """C40 as an int64 array [8 (u), 8 (x)]."""
    import numpy as np
    return np.array(_c40(), np.int64).reshape(8, 8)


def b30s_np():
    """B30S as an int64 array [64 (k), 64 (p)]."""
    import numpy as np
    return np.array(_b30s(), np.int64)


_LAZY = {"C40": _c40, "B30S": _b30s, "B30": _b30}


def __getattr__(name):
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module 'cutline.consts' has no attribute {name!r}")
