"""cutline.bundle - the claim bundle: byte formats, writer, strict reader (stages A1-A4) and fill decoder.

A bundle is a directory, or an uncompressed POSIX ustar archive, holding exactly these members:
``listing.jpg``, ``claim.bin``, ``cellidx.bin``, ``cells.bin.zz``, ``proof.bin``, ``fill.bin.zz`` and, for
renditions only, ``hops.bin``. The three ABI files are Solidity's ``abi.encode(s)`` of ONE struct value (leading
``u256 0x20`` word, the top-level form); the field-list form is invalid.

- ``ClaimFile``, ``ProofFile``, ``FillFile`` (dataclasses) with ``.encode()`` (tuple form) and
  ``.encode_fields()`` (the invalid field-list form, for negative test cases); ``CLAIM_T``, ``PROOF_T``, ``FILL_T``,
  ``HOPS_T`` the ABI type strings; ``MEMBERS``, ``REQUIRED``.
- ``bundle_members(listing_jpg, claim, proof, fill, cellidx, cells, hops=None) -> dict[name, bytes]`` (zlib level 9);
  ``write_bundle(out, listing_jpg, claim, proof, fill, cellidx, cells, hops=None, archive=False) -> path``
  (directory form, or the ustar file ``out`` when ``archive``); ``write_members(out, members, archive=False)``;
  ``ustar_bytes(members) -> bytes`` (tarfile ``USTAR_FORMAT``, bare names, regular files, deterministic headers).
- ``load_members(src) -> (members, form)``: directory, ``.ustar`` path, raw archive bytes or a ``{name: bytes}``
  dict; strict member and archive rules (``E.BUNDLE.MEMBER``, ``E.BUNDLE.ARCHIVE``). ``parse_ustar_strict(b)``.
- ``zz_strict(data, limit) -> bytes``: exactly one zlib stream, ``eof``, no trailing byte, output <= limit
  (``E.BUNDLE.ZLIB``).
- ``read_bundle(src, chain_id, registry) -> RawBundle``: stage A1 (framing) and A2 (claim identity), in
  order: members/archive; ``claim.bin`` word 0 = 0x20 and word 1 = the version (``E.BUNDLE.VERSION``), canonical
  (``E.BUNDLE.ABI``), exact lengths; ``proof.bin`` canonical with cutPre / occHashes exact lengths
  (``E.BUNDLE.PROOF``); ``cellidx.bin`` records (multiple of 20, strict raster order; ``E.BUNDLE.CELLS``);
  ``cells.bin.zz`` one stream of exactly 1,024 N bytes; ``hops.bin`` canonical; then claimant = ecrecover
  (``E.BUNDLE.CLAIMANT``) and H(rec) = core.recHash (``E.BUNDLE.REC``). Returns nf and claimId.
- ``stage_a(src, chain_id, registry, listing=True) -> RawBundle``: A1-A2 plus A3 (ContainerOK v2.1 upload mode and
  R_Y v2 = core.rY, ``E.CONTAINER.*`` / ``E.BUNDLE.RY``) and A4 (tHash, R_cut, R_occ preimages; ``E.BUNDLE.PROGRAM``,
  ``E.BUNDLE.PROOF``). A5 (anchor data) is not an Evaluate input and is never checked here.
- ``read_fill(raw, n_mcu, sampling) -> FillFile``: the stage-C decode of ``fill.bin.zz`` (limit
  256 + (64 + 3 p^2) nMCU; ``E.BUNDLE.ZLIB``; word 0 / canonical ``E.BUNDLE.ABI``; ``|rgb| = 3 p^2 |mcus|``, strictly
  ascending, < nMCU: ``E.BUNDLE.FILL``; empty ``mcus`` with a nonempty ``fillProof``: ``E.MERKLE.MULTI``, the
  MultiVerify name). The other stage-C checks (occupancy, completeness set, multiproofs, tier-E fill coverage) and
  stage B belong to VerifyE; ``cutline.commit.cells.verify_cells`` and ``cutline.compiler.fill.verify_fill``
  implement the multiproof parts.
- ``assemble(...)``: builds the member dict of an honest claim from a capture ``Witness``, the program, the
  background and the signed core (used by the CLI).

Error names for cases the format leaves implicit: rec length != 416 inside a canonical claim.bin -> ``E.REC.LEN``
(the exact-length rule; ``E.BUNDLE.REC`` stays reserved for H(rec) != core.recHash, stage A2); sigHold or sigClaim
length != 65 -> ``E.BUNDLE.ABI``; a ``cells.bin`` shorter than 1,024 N -> ``E.BUNDLE.CELLS`` (longer:
``E.BUNDLE.ZLIB`` via the limit); an unknown bare member name (directory or archive) -> ``E.BUNDLE.MEMBER``, a name
with a directory part or ``./`` -> ``E.BUNDLE.ARCHIVE``.
"""
from __future__ import annotations

import io
import os
import re
import tarfile
import zlib
from dataclasses import dataclass, field

from . import abi
from .consts import BUNDLE_VERSION, CLAIM_CORE_T, TAG_OCC, TAG_PROG, TAG_ROOTS
from .crypto import H
from .enc import u256
from .errors import ABIError, BundleError, CutlineError, MerkleError, RecError
from .types import ClaimCore, claim_id

CLAIM_FIELDS = ["bytes32", "bytes", "bytes32", "bytes32", "bytes32", "bytes32", "bytes", "uint32", "uint32",
                "bytes32[]", "bytes", CLAIM_CORE_T, "bytes", "address"]
PROOF_FIELDS = ["bytes", "bytes32[]", "bytes", "bytes32[]"]
FILL_FIELDS = ["uint32[]", "bytes", "bytes32[]"]
HOP_ENTRY_T = "(bytes,bytes32,uint32,uint32,bytes32[],string)"
CLAIM_T = "(" + ",".join(CLAIM_FIELDS) + ")"
PROOF_T = "(" + ",".join(PROOF_FIELDS) + ")"
FILL_T = "(" + ",".join(FILL_FIELDS) + ")"
HOPS_T = HOP_ENTRY_T + "[]"

MEMBERS = ("listing.jpg", "claim.bin", "cellidx.bin", "cells.bin.zz", "proof.bin", "fill.bin.zz", "hops.bin")
REQUIRED = MEMBERS[:6]
CELL_BYTES = 1024
CELLIDX_REC = 20
WORD_20 = u256(0x20)


def _err(name: str, detail: str = "") -> BundleError:
    return BundleError("E.BUNDLE." + name, detail)


# ------------------------------------------------------------------------------------------------ file types
@dataclass
class ClaimFile:
    rec: bytes
    pk_x: bytes
    pk_y: bytes
    r: bytes
    s: bytes
    sig_hold: bytes
    batch_id: int
    leaf_index: int
    anchor_path: list
    prog: bytes
    core: ClaimCore
    sig_claim: bytes
    claimant: bytes
    version: bytes = BUNDLE_VERSION

    def as_tuple(self) -> tuple:
        return (self.version, self.rec, self.pk_x, self.pk_y, self.r, self.s, self.sig_hold, self.batch_id,
                self.leaf_index, list(self.anchor_path), self.prog, self.core.as_tuple(), self.sig_claim,
                self.claimant)

    @classmethod
    def from_tuple(cls, t) -> "ClaimFile":
        (version, rec, pk_x, pk_y, r, s, sig_hold, batch_id, leaf_index, path, prog, core, sig_claim, claimant) = t
        return cls(rec, pk_x, pk_y, r, s, sig_hold, batch_id, leaf_index, list(path), prog,
                   ClaimCore.from_tuple(core), sig_claim, claimant, version)

    def encode(self) -> bytes:
        return abi.encode_single(CLAIM_T, self.as_tuple())

    def encode_fields(self) -> bytes:
        return abi.encode(CLAIM_FIELDS, list(self.as_tuple()))


@dataclass
class ProofFile:
    cut_pre: bytes
    occ_hashes: list
    occ_bits: bytes
    cell_proof: list

    def as_tuple(self) -> tuple:
        return (self.cut_pre, list(self.occ_hashes), self.occ_bits, list(self.cell_proof))

    def encode(self) -> bytes:
        return abi.encode_single(PROOF_T, self.as_tuple())

    def encode_fields(self) -> bytes:
        return abi.encode(PROOF_FIELDS, list(self.as_tuple()))


@dataclass
class FillFile:
    mcus: list
    rgb: bytes
    fill_proof: list

    def as_tuple(self) -> tuple:
        return (list(self.mcus), self.rgb, list(self.fill_proof))

    def encode(self) -> bytes:
        return abi.encode_single(FILL_T, self.as_tuple())

    def encode_fields(self) -> bytes:
        return abi.encode(FILL_FIELDS, list(self.as_tuple()))


@dataclass(eq=False)
class RawBundle:
    members: dict                    # name -> raw member bytes (as stored)
    form: str                        # "dir", "ustar" or "dict"
    claim: ClaimFile
    proof: ProofFile
    cellidx: list                    # [(cy, cx, salt16)] in raster order
    cells_: bytes | None             # decompressed cells.bin (1,024 N bytes), or None until first use
    hops: list | None                # decoded HopEntry[] (renditions) or None
    nf: bytes = b""
    claim_id: bytes = b""
    notes: list = field(default_factory=list)

    @property
    def lmax(self) -> int:
        return self.proof.cut_pre[4]

    @property
    def cells(self) -> bytes:
        """cells.bin, inflated on first use (stage C2, after the opening-set check), never at A1: A1
        checks the zlib framing and the exact length by streaming, without holding the output."""
        if self.cells_ is None:
            n = len(self.cellidx)
            self.cells_ = zz_strict(self.members["cells.bin.zz"], CELL_BYTES * n)
        return self.cells_


# ------------------------------------------------------------------------------------------------ zlib
def zz_strict(data: bytes, limit: int) -> bytes:
    d = zlib.decompressobj()
    try:
        out = d.decompress(bytes(data), limit + 1)
    except zlib.error as e:
        raise _err("ZLIB", f"corrupt stream: {e}") from None
    if len(out) > limit:
        raise _err("ZLIB", f"output exceeds the limit {limit}")
    if not d.eof or d.unused_data or d.unconsumed_tail:
        raise _err("ZLIB", "not exactly one complete stream (trailing bytes or truncated)")
    return out


def zz_check(data: bytes, limit: int, chunk: int = 1 << 20) -> int:
    """``len(zz_strict(data, limit))`` with the same errors, streaming (at most ``chunk`` output bytes held)."""
    d = zlib.decompressobj()
    n = 0
    buf = bytes(data)
    try:
        while True:
            out = d.decompress(buf, chunk)
            n += len(out)
            if n > limit:
                raise _err("ZLIB", f"output exceeds the limit {limit}")
            buf = d.unconsumed_tail
            if d.eof or (not buf and not out):
                break
    except zlib.error as e:
        raise _err("ZLIB", f"corrupt stream: {e}") from None
    if not d.eof or d.unused_data or d.unconsumed_tail:
        raise _err("ZLIB", "not exactly one complete stream (trailing bytes or truncated)")
    return n


def zz_write(b: bytes) -> bytes:
    return zlib.compress(bytes(b), 9)


# ------------------------------------------------------------------------------------------------ ustar
_OCTAL = re.compile(rb"([0-7]+)[\0 ]+")


def _octal(field_b: bytes, what: str) -> int:
    # fullmatch (not match with "$", which also matches before a final "\n"); parse the digits only
    m = _OCTAL.fullmatch(field_b)
    if not m:
        raise _err("ARCHIVE", f"{what} field is not octal ASCII terminated by NUL or space")
    return int(m.group(1), 8)


def parse_ustar_strict(a: bytes) -> dict:
    """Archive form: POSIX.1-1988 ustar headers, regular files, bare member names, zero padding, end blocks."""
    a = bytes(a)
    pos, seen = 0, {}
    while True:
        hdr = a[pos:pos + 512]
        if len(hdr) < 512:
            raise _err("ARCHIVE", "archive ends without the two zero blocks")
        if hdr == bytes(512):
            if a[pos + 512:pos + 1024] != bytes(512):
                raise _err("ARCHIVE", "one zero block instead of two at the end")
            if any(a[pos + 1024:]):
                raise _err("ARCHIVE", "nonzero bytes after the end blocks")
            return seen
        chk = _octal(hdr[148:156], "chksum")
        if chk != sum(hdr[:148]) + 8 * 32 + sum(hdr[156:]):
            raise _err("ARCHIVE", "header checksum")
        if hdr[257:263] != b"ustar\0" or hdr[263:265] != b"00":
            raise _err("ARCHIVE", "not a POSIX ustar header (magic/version)")
        if hdr[156:157] not in (b"0", b"\0"):
            raise _err("ARCHIVE", f"typeflag {hdr[156:157]!r} is not a regular file")
        name_f = hdr[0:100]
        nul = name_f.find(b"\0")
        name = name_f if nul < 0 else name_f[:nul]
        if nul < 0 or any(name_f[nul:]):
            raise _err("ARCHIVE", "name field: bytes after the terminating NUL (or no NUL)")
        if any(hdr[157:257]):
            raise _err("ARCHIVE", "linkname is not all NUL")
        if any(hdr[345:500]):
            raise _err("ARCHIVE", "prefix is not empty")
        if hdr[124] & 0x80:
            raise _err("ARCHIVE", "base-256 size")
        size = _octal(hdr[124:136], "size")
        if not name or b"/" in name:
            raise _err("ARCHIVE", f"member name {name!r} has a directory part or ./")
        try:
            nm = name.decode("ascii")
        except UnicodeDecodeError:
            raise _err("MEMBER", f"member name {name!r}") from None
        if nm not in MEMBERS or nm in seen:
            raise _err("MEMBER", f"unexpected or repeated member {nm!r}")
        end = pos + 512 + size
        padded = pos + 512 + (size + 511) // 512 * 512
        if padded > len(a):
            raise _err("ARCHIVE", "member data truncated")
        if any(a[end:padded]):
            raise _err("ARCHIVE", "nonzero padding after member data")
        seen[nm] = a[pos + 512:end]
        pos = padded


def ustar_bytes(members: dict) -> bytes:
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for nm in sorted(members):
            ti = tarfile.TarInfo(nm)
            ti.size = len(members[nm])
            ti.mode, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0o644, 0, 0, 0, "", ""
            t.addfile(ti, io.BytesIO(bytes(members[nm])))
    return bio.getvalue()


# ------------------------------------------------------------------------------------------------ members
def _check_members(m: dict) -> None:
    extra = [n for n in m if n not in MEMBERS]
    if extra:
        raise _err("MEMBER", f"unexpected members {extra}")
    missing = [n for n in REQUIRED if n not in m]
    if missing:
        raise _err("MEMBER", f"missing members {missing}")


def load_members(src) -> tuple:
    """(members, form) from a directory, a ustar file path, raw archive bytes or a {name: bytes} dict."""
    if isinstance(src, dict):
        m, form = {k: bytes(v) for k, v in src.items()}, "dict"
    elif isinstance(src, (bytes, bytearray, memoryview)):
        m, form = parse_ustar_strict(bytes(src)), "ustar"
    elif os.path.isdir(src):
        m, form = {}, "dir"
        for nm in sorted(os.listdir(src)):
            p = os.path.join(src, nm)
            if os.path.islink(p) or not os.path.isfile(p):
                raise _err("MEMBER", f"{nm!r} is not a regular file")
            m[nm] = open(p, "rb").read()
    else:
        m, form = parse_ustar_strict(open(src, "rb").read()), "ustar"
    _check_members(m)
    return m, form


def bundle_members(listing_jpg: bytes, claim: ClaimFile, proof: ProofFile, fill: FillFile, cellidx: bytes,
                   cells: bytes, hops: bytes | None = None) -> dict:
    m = {"listing.jpg": bytes(listing_jpg), "claim.bin": claim.encode(), "cellidx.bin": bytes(cellidx),
         "cells.bin.zz": zz_write(cells), "proof.bin": proof.encode(), "fill.bin.zz": zz_write(fill.encode())}
    if hops is not None:
        m["hops.bin"] = bytes(hops)
    return m


def write_members(out: str, members: dict, archive: bool = False) -> str:
    if archive:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "wb") as f:
            f.write(ustar_bytes(members))
        return out
    os.makedirs(out, exist_ok=True)
    for nm in os.listdir(out):
        if nm not in members:
            os.remove(os.path.join(out, nm))
    for nm, b in members.items():
        with open(os.path.join(out, nm), "wb") as f:
            f.write(b)
    return out


def write_bundle(out: str, listing_jpg: bytes, claim: ClaimFile, proof: ProofFile, fill: FillFile, cellidx: bytes,
                 cells: bytes, hops: bytes | None = None, archive: bool = False) -> str:
    return write_members(out, bundle_members(listing_jpg, claim, proof, fill, cellidx, cells, hops), archive)


# ------------------------------------------------------------------------------------------------ reading
def _single(t: str, b: bytes, what: str):
    if len(b) < 32 or b[:32] != WORD_20:
        raise _err("ABI", f"{what}: word 0 is not 0x20 (field-list form?)")
    try:
        return abi.decode_single_canonical(t, b)
    except ABIError as e:
        raise _err("ABI", f"{what}: {e.detail}") from None


def decode_claim(b: bytes) -> ClaimFile:
    if len(b) < 64 or b[:32] != WORD_20 or b[32:64] != BUNDLE_VERSION:
        raise _err("VERSION", "claim.bin word 0 must be 0x20 and word 1 the version bytes32('CUTLINE-bundle-v1')")
    c = ClaimFile.from_tuple(_single(CLAIM_T, b, "claim.bin"))
    if len(c.rec) != 416:
        raise RecError("E.REC.LEN", f"rec has {len(c.rec)} bytes, not 416")     # the exact-length rule of rec
    if len(c.sig_hold) != 65 or len(c.sig_claim) != 65:
        raise _err("ABI", "sigHold and sigClaim must be 65 bytes")
    return c


def decode_proof(b: bytes) -> ProofFile:
    cut_pre, occ_hashes, occ_bits, cell_proof = _single(PROOF_T, b, "proof.bin")
    if len(cut_pre) < 5 or len(cut_pre) != 5 + 32 * (cut_pre[4] + 1):
        raise _err("PROOF", "cutPre length is not 5 + 32 (Lmax + 1)")
    if len(occ_hashes) != cut_pre[4] + 1:
        raise _err("PROOF", "occHashes does not have Lmax + 1 entries")
    return ProofFile(cut_pre, list(occ_hashes), occ_bits, list(cell_proof))


def decode_cellidx(b: bytes) -> list:
    if len(b) % CELLIDX_REC:
        raise _err("CELLS", "cellidx.bin length is not a multiple of 20")
    recs = [(int.from_bytes(b[o:o + 2], "big"), int.from_bytes(b[o + 2:o + 4], "big"), bytes(b[o + 4:o + 20]))
            for o in range(0, len(b), CELLIDX_REC)]
    keys = [(cy, cx) for cy, cx, _ in recs]
    if any(k2 <= k1 for k1, k2 in zip(keys, keys[1:])):
        raise _err("CELLS", "cellidx.bin records are not in strictly ascending raster order")
    return recs


def read_bundle(src, chain_id: int, registry: bytes) -> RawBundle:
    """Stage A1 (framing) and A2 (claim identity)."""
    from .ledger.abi import claim_digest, claimant_of          # local: ledger.abi imports types only
    from .commit.rec import nf as _nf
    m, form = load_members(src)
    claim = decode_claim(m["claim.bin"])
    proof = decode_proof(m["proof.bin"])
    n = len(m["cellidx.bin"]) // CELLIDX_REC
    cp = proof.cut_pre
    grid0 = -(-int.from_bytes(cp[0:2], "big") // 16) * -(-int.from_bytes(cp[2:4], "big") // 16)
    if n > grid0:        # normative bound: checked before any record is decoded or cells.bin inflated
        raise _err("CELLS", f"cellidx.bin lists {n} cells, more than the level-0 grid of cutPre ({grid0})")
    recs = decode_cellidx(m["cellidx.bin"])
    got = zz_check(m["cells.bin.zz"], CELL_BYTES * n)
    if got != CELL_BYTES * n:
        raise _err("CELLS", f"cells.bin has {got} bytes, not 1,024 x {n}")
    hops = None
    if "hops.bin" in m:
        hops = list(_single(HOPS_T, m["hops.bin"], "hops.bin"))
    # A2
    a = claimant_of(claim_digest(chain_id, registry, claim.core), claim.sig_claim)
    if a is None or a == bytes(20) or a != bytes(claim.claimant):
        raise _err("CLAIMANT", "ecrecover(claimDigest(core), sigClaim) differs from the claimant field")
    rh = H(claim.rec)
    if rh != claim.core.rec_hash:
        raise _err("REC", "H(rec) != core.recHash")
    nfv = _nf(rh)
    return RawBundle(members=m, form=form, claim=claim, proof=proof, cellidx=recs, cells_=None, hops=hops, nf=nfv,
                     claim_id=claim_id(nfv, claim.claimant, claim.core))


def check_listing(raw: RawBundle):
    """A3: ContainerOK v2.1 (upload mode) and R_Y v2 = core.rY. Returns the Grammar."""
    from .jpeg.coefs import read_coefs
    from .jpeg.grammar import container_ok_v2
    from .jpeg.ry import r_y_v2
    b = raw.members["listing.jpg"]
    g = container_ok_v2(b, True)
    if r_y_v2(g, read_coefs(b, g)) != raw.claim.core.r_y:
        raise _err("RY", "R_Y v2 of listing.jpg != core.rY")
    return g


def check_preimages(raw: RawBundle) -> None:
    """A4: tHash, R_cut and R_occ preimages (lengths were checked in A1)."""
    from .commit.rec import decode_rec_strict
    rec = decode_rec_strict(raw.claim.rec)
    if H(TAG_PROG + raw.claim.prog) != raw.claim.core.t_hash:
        raise _err("PROGRAM", "H(0x05 || 'CUTLINE-prog-v1' || prog) != core.tHash")
    if H(TAG_ROOTS + raw.proof.cut_pre) != rec.R_cut:
        raise _err("PROOF", "H(0x02 || 'CUTLINE-roots' || cutPre) != rec.R_cut")
    if H(TAG_OCC + b"".join(raw.proof.occ_hashes)) != rec.R_occ:
        raise _err("PROOF", "H(0x03 || occHashes) != rec.R_occ")


def stage_a(src, chain_id: int, registry: bytes, listing: bool = True) -> RawBundle:
    """Stage A (A1-A4; A5 is not an Evaluate input). Raises the first failing check's error."""
    raw = read_bundle(src, chain_id, registry)
    if listing:
        check_listing(raw)
    check_preimages(raw)
    return raw


def fill_limit(n_mcu: int, sampling: int) -> int:
    p = 8 if sampling == 0 else 16
    return 256 + (64 + 3 * p * p) * n_mcu


def read_fill(raw_or_members, n_mcu: int, sampling: int) -> FillFile:
    """Stage-C decode of fill.bin.zz (size limit, then framing). Needs nMCU and p from a well-formed program."""
    m = raw_or_members.members if isinstance(raw_or_members, RawBundle) else raw_or_members
    b = zz_strict(m["fill.bin.zz"], fill_limit(n_mcu, sampling))
    mcus, rgb, proof = _single(FILL_T, b, "fill.bin")
    p = 8 if sampling == 0 else 16
    if len(rgb) != 3 * p * p * len(mcus) or any(b2 <= b1 for b1, b2 in zip(mcus, mcus[1:])) \
            or any(x >= n_mcu for x in mcus):
        raise _err("FILL", "rgb length, order or MCU range")
    if not mcus and proof:
        # stage C3 / MultiVerify name this failure (same name as compiler.fill)
        raise MerkleError("E.MERKLE.MULTI", "an empty mcus needs fillProof = [] (the empty opening)")
    return FillFile(list(mcus), rgb, list(proof))


# ------------------------------------------------------------------------------------------------ assembling
def assemble(listing_jpg: bytes, witness, pk64: bytes, sig_hold: bytes, anchor, program, bg, core: ClaimCore,
             sig_claim: bytes, claimant: bytes, whole_fill: bool = False) -> dict:
    """Member dict of a claim: cells of the opened set at l*, proof.bin, and the fill openings.

    ``witness`` = capture Witness (k_cap, rec, sig_dev, commitment); ``anchor`` = types.AnchorProof;
    ``program`` = types.Program; ``bg`` = uint8[Hc, Wc, 3]; ``whole_fill`` opens every canvas MCU (constant class)."""
    from .commit.cells import occupancy, open_cells
    from .compiler.compose import compile_listing, opened_set
    from .compiler.fill import fill_open_set, open_fill
    cm = witness.commitment
    ls = program.lstar
    C, al, _ = cm.levels[ls]
    hs, ws = al.shape
    opened = opened_set(occupancy(al), program, hs, ws)
    idx, data, cproof = open_cells(cm, witness.k_cap, ls, opened)
    comp = compile_listing(program, {(cy, cx): data[k * CELL_BYTES:(k + 1) * CELL_BYTES]
                                     for k, (cy, cx) in enumerate(opened)}, hs, ws, bg)
    mcus = fill_open_set(comp.m_cert, comp.alpha, program.sampling, whole_fill)
    fm, rgb, fproof = open_fill(bg, program.sampling, mcus)
    r, s = witness.sig_dev
    claim = ClaimFile(rec=witness.rec.encode(), pk_x=bytes(pk64[:32]), pk_y=bytes(pk64[32:]), r=u256(r), s=u256(s),
                      sig_hold=bytes(sig_hold), batch_id=anchor.batch_id, leaf_index=anchor.leaf_index,
                      anchor_path=list(anchor.path), prog=program.encode(), core=core, sig_claim=bytes(sig_claim),
                      claimant=bytes(claimant))
    proof = ProofFile(cm.cut_pre, list(cm.occ_hashes), cm.occ_bits[ls], list(cproof))
    return bundle_members(listing_jpg, claim, proof, FillFile(fm, rgb, fproof), idx, data)


__all__ = ["ClaimFile", "ProofFile", "FillFile", "RawBundle", "CLAIM_T", "PROOF_T", "FILL_T", "HOPS_T", "MEMBERS",
           "REQUIRED", "zz_strict", "zz_write", "parse_ustar_strict", "ustar_bytes", "load_members",
           "bundle_members", "write_members", "write_bundle", "decode_claim", "decode_proof", "decode_cellidx",
           "read_bundle", "check_listing", "check_preimages", "stage_a", "fill_limit", "read_fill", "assemble",
           "CutlineError"]
