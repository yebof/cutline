"""capture_emu.session - one emulated capture-to-claim world, end to end (used by ``cutline.cli``).

EMULATION: a ChainStub, a BackOffice (holder + HKDF owner keys) and one EmulatedApp, all derived from fixed labels,
so every run gives the same bytes. The flow: enrol (emulated attestation chain) -> quorum confirm -> provision owner
addresses -> capture (soft map or segmenter -> certified region -> commitment -> rec v3) -> aggregate and anchor ->
ExportIntake in signing order -> CoSign (sigma_hold). Then a claim: ClaimCore v2 over the listing's R_Y v2, the
program's tHash and rFill, signed by the capture's owner key (claim digest v2), and the bundle members
(``cutline.bundle.assemble``).

Interface:

- ``Session(label="cutline-e2e", segmenter="u2netp", rateN=120, n_owner=16)``; ``.chain``, ``.bo``, ``.app``.
- ``.capture(M, soft_or_seg) -> recHash`` (queued; ``soft_or_seg`` as ``EmulatedApp.capture``).
- ``.anchor() -> None``: flush the aggregator to the chain, ExportIntake every new record in signing order, CoSign
  each (``Captured`` entries become complete).
- ``.captured[recHash] -> Captured(witness, rec_bytes, sig64, anchor (AnchorProof), sig_hold)``.
- ``.core(rh, listing_jpg, program, bg, marketplace=5, listing_id=..., seller=..., salt_ref=..., category=0, tier=1,
  enc_class=1) -> ClaimCore``; ``.sign_core(rh, core) -> (sig65, claimant)``.
- ``.claim_members(rh, listing_jpg, program, bg, whole_fill=False, core=None, **core_kw) -> (members, core)``.
- ``.chain_id``, ``.registry`` (what ``cutline.bundle.read_bundle`` needs for the claimant check).
- Program helpers (seller tooling, not normative): ``gain_lut(num) -> 768 B`` (monotone, LUT[c][v] = min(255,
  v num[c] // 100)); ``alpha_bbox(alpha) -> (y0, x0, ch, cw)``; ``occ_bbox(alpha) -> (y0, x0, ch, cw)`` (the occupied-cell bbox,
  the crop that keeps ``partial`` false); ``fit_program(crop, sampling, kernel, lut, flags=2,
  oy=7, ox=13, lstar=1) -> Program``: about 3/4 downscale (static rule 3), aspect kept (rule 6), non-aligned
  placement on a canvas whose sides are multiples of 16 (rule 5(3)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cutline.bundle import assemble
from cutline.commit.rec import rec_hash
from cutline.compiler.fill import r_fill
from cutline.compiler.static_rules import aspect_ok
from cutline.crypto import H, k1_key_from_label, k1_sign
from cutline.jpeg.ry import r_y_v2_bytes
from cutline.ledger.abi import att_hash, claim_digest, key_id, listing_ref, pk_hash
from cutline.types import AnchorProof, ClaimCore, Program

from .aggregator import Aggregator
from .app import EmulatedApp
from .attestation import ROOT_SHA256, verify_chain
from .backoffice import BackOffice
from .chain_stub import ChainStub


@dataclass
class Captured:
    witness: object
    rec_bytes: bytes
    sig64: bytes
    anchor: AnchorProof | None = None
    sig_hold: bytes | None = None


class Session:
    def __init__(self, label: str = "cutline-e2e", segmenter: str = "u2netp", rateN: int = 120, n_owner: int = 16):
        self.label = label
        self.chain = ChainStub()
        self.bo = BackOffice(k1_key_from_label(label + " holder"), self.chain, H((label + " seller seed").encode()))
        nonce = H((label + " nonce_E").encode())[:16]
        self.app = EmulatedApp(label + " device", self.bo.holder, rateN, self.chain.chain_id, nonce,
                               segmenter=segmenter)
        pk, ders, ch = self.app.enrol()
        if verify_chain(ders, ROOT_SHA256(), {"challenge": ch, "pk_hash": pk_hash(pk)}) is not None:
            raise RuntimeError("emulated attestation chain does not verify")
        kid = key_id(pk, H((label + " enrolSalt").encode()))
        self.chain.enroll(kid, pk_hash(pk), self.bo.holder, att_hash(ders), rateN, nonce)
        for v in range(self.chain.quorum()):
            self.chain.confirm(kid, v)
        self.app.set_key_id(kid)
        self.app.add_owner_addrs(self.bo.provision(kid, n_owner, pk))
        self.agg = Aggregator()
        self.captured: dict = {}
        self._pending: list = []

    @property
    def chain_id(self) -> int:
        return self.chain.chain_id

    @property
    def registry(self) -> bytes:
        return self.chain.registry

    def capture(self, M, soft_or_seg, strokes=None) -> bytes:
        rb, sig, leaf = self.app.capture(M, strokes, soft_or_seg, self.chain.beacon())
        rh = rec_hash(rb)
        self.agg.submit(leaf, rb, sig)
        self.captured[rh] = Captured(self.app.witnesses[rh], rb, sig)
        self._pending.append(rh)
        return rh

    def anchor(self) -> None:
        batches = self.agg.flush(self.chain)
        for rh in self._pending:
            c = self.captured[rh]
            st = self.bo.export_intake(c.rec_bytes, c.sig64)
            if st.status != "accepted":
                raise RuntimeError(f"ExportIntake: {st}")
        for rh in self._pending:
            c = self.captured[rh]
            for b in batches:
                try:
                    bid, i, path = b.proof(rh)
                    break
                except KeyError:
                    continue
            else:
                raise RuntimeError("record not anchored")
            sh = self.bo.cosign(c.rec_bytes, c.sig64, bid, i, path)
            if not isinstance(sh, bytes):
                raise RuntimeError(f"CoSign refused: {sh}")
            c.anchor, c.sig_hold = AnchorProof(bid, i, tuple(path)), sh
        self._pending = []

    def core(self, rh: bytes, listing_jpg: bytes, program, bg, marketplace: int = 5, listing_id: str = "B0E2E0000",
             seller: str = "seller-e2e", salt_ref: bytes | None = None, category: int = 0, tier: int = 1,
             enc_class: int = 1) -> ClaimCore:
        salt_ref = salt_ref if salt_ref is not None else H((self.label + " salt_ref " + listing_id).encode())
        return ClaimCore(rec_hash=rh, r_y=r_y_v2_bytes(listing_jpg, upload=True), t_hash=program.t_hash(),
                         r_fill=r_fill(bg, program.sampling), enc_class=enc_class, tier=tier,
                         marketplace=marketplace, category=category,
                         listing_ref=listing_ref(marketplace, listing_id, seller, salt_ref))

    def owner_of(self, rh: bytes) -> bytes:
        return self.captured[rh].witness.rec.owner

    def sign_core(self, rh: bytes, core: ClaimCore) -> tuple:
        owner = self.owner_of(rh)
        sk = self.bo.owner_sk(owner)
        return k1_sign(sk, claim_digest(self.chain_id, self.registry, core)), owner

    def claim_members(self, rh: bytes, listing_jpg: bytes, program, bg, whole_fill: bool = False, core=None,
                      **core_kw) -> tuple:
        c = self.captured[rh]
        if c.anchor is None:
            raise RuntimeError("anchor() first")
        core = core if core is not None else self.core(rh, listing_jpg, program, bg, **core_kw)
        sig_claim, claimant = self.sign_core(rh, core)
        m = assemble(listing_jpg, c.witness, self.app.pk64, c.sig_hold, c.anchor, program, bg, core, sig_claim,
                     claimant, whole_fill=whole_fill)
        return m, core


# ------------------------------------------------------------------------------------------------ program helpers
def gain_lut(num: tuple) -> bytes:
    """Monotone LUT[c][v] = min(255, v * num[c] // 100) (integer; static rule 1 holds)."""
    return bytes(min(255, v * n // 100) for n in num for v in range(256))


def _c16(x: int) -> int:
    return -(-x // 16) * 16


def fit_program(crop, sampling, kernel, lut, flags=0b10, oy=7, ox=13, lstar=1) -> Program:
    """Downscale the crop by about 3/4 (static rule 3: oh <= ch < 2 oh), keep the aspect (rule 6), place it at a
    non-aligned offset on a canvas whose sides are multiples of 16 (rule 5(3))."""
    y0, x0, ch, cw = crop
    oh = (3 * ch + 3) // 4
    ow = max(1, (cw * oh + ch // 2) // ch)
    assert oh <= ch < 2 * oh and ow <= cw < 2 * ow and aspect_ok(cw, ch, ow, oh)[0]
    Hc, Wc = _c16(oy + oh + 9), _c16(ox + ow + 11)
    return Program(lstar=lstar, crop=crop, out=(oh, ow), place=(oy, ox), canvas=(Hc, Wc), kernel=kernel,
                   sampling=sampling, flags=flags, lut=lut, D=())


def alpha_bbox(al) -> tuple:
    ys, xs = np.nonzero(al)
    return int(ys.min()), int(xs.min()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def occ_bbox(al) -> tuple:
    """bbox(O_l) as a crop (y0, x0, ch, cw): the union of the 16 x 16 pixel rectangles of the occupied cells of the
    level (``commit.cells.occupancy``), clipped to the level. A crop that contains it keeps ``partial`` false (a crop
    to the alpha bbox alone generally cuts occupied cells and makes the claim partial)."""
    from cutline.commit.cells import occupancy
    occ = occupancy(al)
    ys, xs = np.nonzero(occ)
    h, w = np.asarray(al).shape[:2]
    y0, x0 = int(ys.min()) * 16, int(xs.min()) * 16
    y1, x1 = min(h, (int(ys.max()) + 1) * 16), min(w, (int(xs.max()) + 1) * 16)
    return y0, x0, y1 - y0, x1 - x0
