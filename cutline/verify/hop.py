"""cutline.verify.hop - VerifyHop, chain assembly and the chain-level rules.

Every child MCU is tested against Dec* of the public parent, under the crop rule, no upscaling, the edge model,
platform binding, at most one resampling hop, summed excess, big-integer cumulative scale and a chain cap of 8.

VerifyHop is split into three layers so the chain rules can be tested without images:
``verify_hop`` = ``hop_checks`` -> ``hop_codec`` -> ``hop_update``.

- ``ChainState``: the chain part of the evidence carried from Y0 through every hop: ``cum = (Cw, Ch, Ow, Oh)``
  (Python ints, unbounded; the ``geometry`` of a rendition), ``dims = (H, W)`` and ``sampling`` of the current file,
  ``resamples`` (``resampling_hops``), ``hop_depth``, ``chain_class``, ``lambda_chain_2p51``, ``excess_sum`` /
  ``excess_max`` (·2^30), ``partial``, the bool masks ``sil``, ``c_adj`` (carried for information only: U is decided
  on Y0 and never re-tested on a rendition), ``cert_mask`` (None when not tracked: the pure chain-rule layer),
  ``ceilings`` = [(platformId, v*)], ``enc_classes`` (upload first, then every hop), ``hop_classes`` (Class_hop of
  every hop, informational).
- ``initial_state(y0_facts, program, *, sil=None, c_adj=None, cert_mask=None) -> ChainState``: ``y0_facts`` is a
  ``types.Evidence`` or a mapping / object with ``res_class``, ``lam_y_2p51`` (or ``q3``, the upload's tables, from
  which both are computed), ``e_inf_2p30`` (or Evidence ``excess["max"]``; 0 for class 1 and tier T), ``partial``,
  ``enc_classes`` (or ``enc_class``), optionally the masks; ``program`` = ``types.Program`` (or any object with
  ``crop = (y0, x0, ch, cw)``, ``out = (oh, ow)``, ``canvas = (Hc, Wc)``, ``sampling``): ``cum = (cw, ch, ow, oh)``
  of the upload at l* (the aspect is bounded from the capture crop, upload included).
- ``assemble_chain(r_disp, core_rY, hops, ledger, h) -> list[HopEntry]`` (Evaluate step 3): ``[]`` when
  ``r_disp == core_rY``; otherwise, in this order: more than ``CHAIN_LEN_CAP`` = 8 entries -> ``E.CHAIN.TOO_LONG``
  (before anything else); ``entries[0].rYParent != core.rY`` (or no entry) -> ``E.CHAIN.UNLINKED``; then per entry
  in order ``batchId >= hop_batch_count(h)`` -> ``E.HOP.UNKNOWN_BATCH`` and
  ``not hopIncluded(batchId, hopLeaf(rYParent, rYChild, H(op)), leafIndex, path)`` -> ``E.HOP.NOT_INCLUDED``
  (``hops`` = the ``hops.bin`` bytes or a decoded list; H(op) is over the raw op bytes, whatever their length).
  ``g0_reason_of(name)`` maps these names (and VerifyHop's) to the record's ``g0_reason``.
- ``hop_checks(e_par, op, J, ledger, entry, h=None) -> HopState``: every ``require`` of VerifyHop **except** the
  container and the codec test, in order (``E.HOP.OP``, ``PLATFORM``, ``DECODER``, ``KERNEL``, ``CLASS``,
  ``TABLES``, ``CLASS`` (class 3 above Q_pol), ``CROP``, ``OUT``, ``UPSCALE``, ``ASPECT``); ``J`` is the child's
  ``Grammar`` (or any object with ``H``, ``W``, ``sampling``, ``q3()``). ``HopState`` carries the decoded ``Op``,
  ``resamples``, ``cum``, ``v_star``, ``q_hop`` (the ceiling in force at the batch's height) and ``batch``.
- ``hop_codec(parent, op, child_bytes, J, *, coefs=None, ledger=None, e_inf=True) -> CodecResult(ok, rows,
  e_inf_2p30, tested_blocks)``: ``X1 = ResizeExact(Dec*(parent)[op.crop], op.out, op.kernelId)`` (the whole child
  canvas), then the class test on **every real block of every MCU** of the child through
  ``blocktest.iter_class_rows`` (``cert_mcus=None``): class 1 = E0 (rows ``(mcu, comp, blk, k, c, t)``); classes
  3 / 5 = CELL at the ledger's eta20 with the Phi / Phi_go front end (the float prefilter with exact re-checks,
  rows ``(mcu, comp, blk, k, c, D80)``); class 7 = re-encode X1 once with the EP entry (``verify.ep``; the local
  encoder must be the build and parameters ``ledger.ep(epEntry)`` names, else ``E.EVAL.EP_UNAVAILABLE``, a
  reference-internal error; so is an id that ``ledger.ep`` defines but this host has no encoder for: "cannot
  evaluate", never G0 "hop") and compare every coefficient (rows ``(mcu, comp, blk, k, c, c_ep)``). The test is
  streamed band by band and ``rows`` keeps at most ``blocktest.ROW_CAP`` (the verdict needs only the first).
  ``parent`` = the parent's bytes or its Dec* RGB array. ``e_inf_2p30`` = E_inf(child) (the max per-block
  E_inf·2^30 over every tested block, ``blocktest.iter_einf``; 0 for classes 1 and 7), computed only when the test
  passes.
- ``hop_update(e_par, hs, J, e_inf_2p30=0) -> ChainState``: VerifyHop's result: ``chain_class ← worst(…,
  Class_hop(q(Ych), Q_hop(p, v*)))``, ``lambda_chain_2p51 += lam_y``, depth, resamples, cum, dims, partial
  (``partial_hop``: bbox(cert_par) nonempty and not inside op.crop), excess, PushThrough of the masks, ceilings,
  enc_classes.
- ``push_through(mask, op, thr=128) -> bool array``: ``ResizeExact(255·M[op.crop], op.out, op.kernelId) >= thr``
  (Sil and C_adj use 128; CertMask uses "> 0", i.e. ``thr=1``). ``push_masks(state, op)`` applies the three.
- ``cert_bbox(cert_mask, sampling) -> (y0, x0, y1, x1) | None``: bbox(cert) (MCU rectangles, clipped).
- ``verify_hop(parent, e_par, child_bytes, entry, ledger, h=None, **codec_kw) -> ChainState``: ContainerOK v2.1 of
  the child in rendition mode (``E.HOP.CONTAINER``), then the three layers. Every failure raises ``HopError`` (⊥; the
  codec failure carries ``.rows``).
- ``fetch_parent(fetch, entry) -> bytes``: ``fetch(entry.parent_locator)``, accepted only if its R_Y v2 equals
  ``entry.rYParent`` (content addressing), else ``E.CHAIN.ANCESTOR_UNAVAILABLE``.
- ``verify_chain(entries, state, fetch, displayed, ledger, h, first_parent=None, **codec_kw) -> ChainOutcome``
  (Evaluate step 11): for each entry in order the parent and the child are fetched (the child of entry i is the
  parent of entry i + 1, or the displayed bytes), then ``verify_hop``. The outcome carries the final ``state`` or
  ``g0_reason`` ("ancestor" / "hop"), ``hop_index`` and the error name. Fetch order: the child of hop i is fetched
  before VerifyHop i runs, so an unavailable ancestor i + 1 is reported as "ancestor" with ``hop_index = i + 1``
  before hop i's own test.
- ``chain_cap(state) -> int``: the chain rules' contribution to Level: 2 if hop_depth > 3, resampling hops > 1,
  chain class coarse or partial; else 3 if some hop has an E1 class (3, 5); else 4. It is not Level (tone, D, U,
  tier, background and the ledger are ``levels.level``'s).
- ``aspect_holds(cum) -> bool`` and ``aspect_margin(cum) -> int`` (``max(10^6·Cw·Oh − 1,020,202·Ch·Ow, 10^6·Ch·Ow −
  1,020,202·Cw·Oh)``; > 0 fails); ``cum_after(cum, op) -> cum``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from cutline import consts
from cutline.consts import ASPECT_A, ASPECT_B, CHAIN_LEN_CAP, D_HOP, ETA20_CLS, Q_POL
from cutline.errors import ChainError, ContainerError, CutlineError, EvalError, HopError, RYError
from cutline.verify.hopcodec import HopEntry, Op, child_ry, decode_hops, decode_op, hop_leaf, op_hash
from cutline.verify.resclass import class_hop, lam_2p51, res_class, worst

V1_CLASSES = (1, 3, 5, 7)
E1_CLASSES = (3, 5)

G0_OF = {"E.CHAIN.TOO_LONG": "chain_too_long", "E.CHAIN.UNLINKED": "unlinked", "E.HOP.UNKNOWN_BATCH": "unlinked",
         "E.HOP.NOT_INCLUDED": "unlinked", "E.CHAIN.ANCESTOR_UNAVAILABLE": "ancestor"}


def g0_reason_of(name: str) -> str:
    """The record's g0_reason for a chain / hop error name: assembly -> chain_too_long / unlinked; fetch -> ancestor;
    any other E.HOP.* (a VerifyHop require) -> hop."""
    if name in G0_OF:
        return G0_OF[name]
    if name.startswith("E.HOP."):
        return "hop"
    raise ValueError(f"{name} is not a chain or hop error")


# ------------------------------------------------------------------------------------------------ state
@dataclass
class ChainState:
    cum: tuple                          # (Cw, Ch, Ow, Oh), Python ints
    dims: tuple                         # (H, W) of the current file
    sampling: int                       # of the current file
    chain_class: str
    lambda_chain_2p51: int
    resamples: int = 0
    hop_depth: int = 0
    excess_sum: int = 0
    excess_max: int = 0
    partial: bool = False
    sil: object = None                  # bool [H, W] or None
    c_adj: object = None
    cert_mask: object = None
    ceilings: list = field(default_factory=list)       # [(platformId, v*)]
    enc_classes: list = field(default_factory=list)    # upload first, then every hop
    hop_classes: list = field(default_factory=list)    # Class_hop of every hop (informational)

    def copy(self) -> "ChainState":
        return replace(self, ceilings=list(self.ceilings), enc_classes=list(self.enc_classes),
                       hop_classes=list(self.hop_classes))


def _get(o, k, default=None):
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


def initial_state(y0_facts, program, *, sil=None, c_adj=None, cert_mask=None) -> ChainState:
    q3 = _get(y0_facts, "q3")
    cls, lam = _get(y0_facts, "res_class"), _get(y0_facts, "lam_y_2p51")
    if q3 is not None and (cls is None or lam is None):
        r = res_class(q3)
        cls, lam = r[0], r[1]
    if cls is None or lam is None:
        raise ValueError("y0_facts needs res_class and lam_y_2p51 (or q3)")
    einf = _get(y0_facts, "e_inf_2p30")
    if einf is None:
        ex = _get(y0_facts, "excess")
        einf = (ex or {}).get("max") or 0
    ecs = _get(y0_facts, "enc_classes")
    if ecs is None:
        ecs = [_get(y0_facts, "enc_class", 1)]
    y0, x0, ch, cw = _get(program, "crop")
    oh, ow = _get(program, "out")
    Hc, Wc = _get(program, "canvas")
    pick = lambda a, k: a if a is not None else _get(y0_facts, k)  # noqa: E731
    return ChainState(cum=(int(cw), int(ch), int(ow), int(oh)), dims=(int(Hc), int(Wc)),
                      sampling=int(_get(program, "sampling")), chain_class=cls, lambda_chain_2p51=int(lam),
                      excess_sum=int(einf), excess_max=int(einf), partial=bool(_get(y0_facts, "partial", False)),
                      sil=_mask(pick(sil, "sil")), c_adj=_mask(pick(c_adj, "c_adj")),
                      cert_mask=_mask(pick(cert_mask, "cert_mask")), enc_classes=[int(c) for c in ecs])


def _mask(m):
    return None if m is None else np.asarray(m).astype(bool)


# ------------------------------------------------------------------------------------------------ aspect
def aspect_margin(cum) -> int:
    Cw, Ch, Ow, Oh = (int(v) for v in cum)
    return max(ASPECT_A * Cw * Oh - ASPECT_B * Ch * Ow, ASPECT_A * Ch * Ow - ASPECT_B * Cw * Oh)


def aspect_holds(cum) -> bool:
    return aspect_margin(cum) <= 0


def cum_after(cum, op: Op) -> tuple:
    Cw, Ch, Ow, Oh = cum
    return (Cw * op.crop[3], Ch * op.crop[2], Ow * op.out[1], Oh * op.out[0])


# ------------------------------------------------------------------------------------------------ chain assembly
def assemble_chain(r_disp: bytes, core_rY: bytes, hops, ledger, h: int) -> list:
    if bytes(r_disp) == bytes(core_rY):
        return []
    entries = decode_hops(hops) if isinstance(hops, (bytes, bytearray, memoryview)) else list(hops)
    if len(entries) > CHAIN_LEN_CAP:
        raise ChainError("E.CHAIN.TOO_LONG", f"{len(entries)} hops > {CHAIN_LEN_CAP}")
    if not entries or bytes(entries[0].r_y_parent) != bytes(core_rY):
        raise ChainError("E.CHAIN.UNLINKED", "the chain does not start at core.rY")
    n = ledger.hop_batch_count(h)
    for i, e in enumerate(entries):
        if not 0 <= e.batch_id < n:
            raise HopError("E.HOP.UNKNOWN_BATCH", f"hop {i}: batch {e.batch_id} >= {n}")
        leaf = hop_leaf(e.r_y_parent, child_ry(entries, i, r_disp), op_hash(bytes(e.op)))
        if not ledger.hop_included(e.batch_id, leaf, e.leaf_index, e.path, h):
            raise HopError("E.HOP.NOT_INCLUDED", f"hop {i}")
    return entries


# ------------------------------------------------------------------------------------------------ layer 1: requires
@dataclass(frozen=True)
class HopState:
    op: Op
    resamples: int
    cum: tuple
    v_star: int
    q_hop: bytes
    batch: object


def _q192(J) -> bytes:
    return b"".join(bytes(t) for t in J.q3())


def hop_checks(e_par: ChainState, op, J, ledger, entry, h: int | None = None) -> HopState:
    h = ledger.height if h is None else h
    if not isinstance(op, Op):
        op = decode_op(op)                                                           # E.HOP.OP
    Hp, Wp = e_par.dims
    Hch, Wch = J.H, J.W
    batch = ledger.hop_batch(entry.batch_id, h)                                      # E.HOP.UNKNOWN_BATCH (assembly)
    if op.platform_id != batch.platform_id:
        raise HopError("E.HOP.PLATFORM", f"op.platformId {op.platform_id} != batch {batch.platform_id}")
    if op.decoder_id != 0:
        raise HopError("E.HOP.DECODER", f"decoderId {op.decoder_id}")
    if op.kernel_id not in (0, 1):
        raise HopError("E.HOP.KERNEL", f"kernelId {op.kernel_id}")
    c = op.enc_class
    if c not in V1_CLASSES or ledger.cls(c) is None:
        raise HopError("E.HOP.CLASS", f"encClass {c}")
    if (c == 7 and ledger.ep(op.ep_entry) is None) or (c != 7 and op.ep_entry != 0):
        raise HopError("E.HOP.CLASS", f"encClass {c} with epEntry {op.ep_entry}")
    if bytes(op.q) != _q192(J):
        raise HopError("E.HOP.TABLES", "op.q != the child's header tables")
    if c == 3 and any(not 1 <= v <= p for v, p in zip(bytes(op.q), Q_POL)):
        raise HopError("E.HOP.CLASS", "class 3 with a table entry above Q_pol")
    cy, cx, chh, cww = op.crop
    if chh < 1 or cww < 1 or cy + chh > Hp or cx + cww > Wp:
        raise HopError("E.HOP.CROP", f"crop {op.crop} in parent {(Hp, Wp)}")
    if tuple(op.out) != (Hch, Wch):
        raise HopError("E.HOP.OUT", f"out {op.out} != child {(Hch, Wch)}")
    if op.out[0] > chh or op.out[1] > cww:
        raise HopError("E.HOP.UPSCALE", f"out {op.out} > crop {(chh, cww)}")
    resamples = e_par.resamples + (1 if op.resampling else 0)
    cum = cum_after(e_par.cum, op)
    if not aspect_holds(cum):
        raise HopError("E.HOP.ASPECT", f"cumulative aspect margin {aspect_margin(cum)} > 0")
    v_star, q_hop = ledger.hop_ceiling(op.platform_id, batch.height)
    return HopState(op, resamples, cum, v_star, bytes(q_hop), batch)


# ------------------------------------------------------------------------------------------------ layer 2: codec
@dataclass(frozen=True)
class CodecResult:
    ok: bool
    rows: list                  # failing coefficients (mcu, comp, blk, k, c, witness), ascending
    e_inf_2p30: int
    tested_blocks: int


def _decoded(parent):
    if isinstance(parent, np.ndarray):
        return parent
    from cutline.dct import dec_star
    return dec_star(bytes(parent))


def hop_canvas(parent, op: Op):
    """X1 = ResizeExact(Dec*(parent)[op.crop], op.out, op.kernelId) (uint8 RGB)."""
    from cutline.compiler.kernels import resize_exact
    P = _decoded(parent)
    y, x, h, w = op.crop
    if P.shape[0] < y + h or P.shape[1] < x + w:
        raise HopError("E.HOP.CROP", f"crop {op.crop} outside the decoded parent {P.shape[:2]}")
    return resize_exact(P[y:y + h, x:x + w], op.out[0], op.out[1], op.kernel_id).astype(np.uint8)


def _real_blocks(J, coefs):
    return sum(int(np.asarray(c).shape[0] * np.asarray(c).shape[1]) for c in coefs)


def _ep_spec(op: Op, ledger):
    """The local EP encoder of op.epEntry (``verify.ep``), required to be the build and parameters the ledger names;
    a verifier without that encoder cannot evaluate the hop (``E.EVAL.EP_UNAVAILABLE``, not a protocol outcome)."""
    from cutline.verify import ep as EP
    try:
        spec = EP.resolve(op.ep_entry)                                         # E.HOP.CLASS for an unknown id
    except HopError:
        # an id the ledger defines (setEPEntry) but this host has no encoder for cannot be evaluated here; it is not
        # a failing hop (the level must not depend on which EP builds a host carries)
        if ledger is not None and ledger.ep(op.ep_entry) is not None:
            raise EvalError("E.EVAL.EP_UNAVAILABLE", f"ep[{op.ep_entry}] is defined on the ledger but has no local "
                            "encoder: cannot evaluate") from None
        raise
    if ledger is not None and not EP.check_entry(spec, ledger.ep(op.ep_entry)):
        raise EvalError("E.EVAL.EP_UNAVAILABLE", f"ep[{op.ep_entry}] names another build or parameter set")
    return spec


def hop_codec(parent, op: Op, child_bytes: bytes, J, *, coefs=None, ledger=None, e_inf: bool = True) -> CodecResult:
    from cutline.jpeg.coefs import read_coefs
    from cutline.verify.blocktest import ROW_CAP, head_rows, iter_class_rows, iter_einf
    if coefs is None:
        coefs = read_coefs(bytes(child_bytes), J)
    X1 = hop_canvas(parent, op)
    if X1.shape[:2] != (J.H, J.W):
        raise HopError("E.HOP.OUT", f"canvas {X1.shape[:2]} != child {(J.H, J.W)}")
    c = op.enc_class
    reg = {c: ledger.cls(c)} if ledger is not None and ledger.cls(c) is not None else None
    ep = _ep_spec(op, ledger) if c == 7 else None
    # every MCU is tested; the verdict needs the first failing row, the report keeps at most ROW_CAP
    rows = [tuple(int(v) for v in r) for r in head_rows(iter_class_rows(c, X1, coefs, J, None, ep, reg), ROW_CAP)]
    einf = 0
    if not rows and e_inf and c in E1_CLASSES:
        eta20 = int((reg or consts.CLASS_REGISTRY)[c][1])
        einf = max((v for *_, v in iter_einf(X1, coefs, J, eta20, "go" if c == 5 else "phi")), default=0)
    return CodecResult(not rows, rows, int(einf), _real_blocks(J, coefs))


# ------------------------------------------------------------------------------------------------ layer 3: update
def push_through(mask, op: Op, thr: int = 128):
    """ResizeExact(255·M[op.crop], op.out, op.kernelId) >= thr (thr = 128 for Sil / C_adj, 1 for CertMask)."""
    from cutline.compiler.kernels import resize_exact
    y, x, h, w = op.crop
    M = np.asarray(mask).astype(bool)[y:y + h, x:x + w].astype(np.uint8) * 255
    return resize_exact(M, op.out[0], op.out[1], op.kernel_id) >= thr


def cert_bbox(cert_mask, sampling: int):
    """bbox(cert): the union of the MCU rectangles that contain a CertMask pixel, clipped to the file; as
    (y0, x0, y1, x1) half-open, or None when cert is empty."""
    M = np.asarray(cert_mask).astype(bool)
    ys, xs = np.nonzero(M)
    if not len(ys):
        return None
    p = 16 if sampling else 8
    H, W = M.shape
    return (int(ys.min()) // p * p, int(xs.min()) // p * p, min(H, (int(ys.max()) // p + 1) * p),
            min(W, (int(xs.max()) // p + 1) * p))


def partial_hop(e_par: ChainState, op: Op) -> bool:
    if e_par.cert_mask is None:
        return False
    bb = cert_bbox(e_par.cert_mask, e_par.sampling)
    if bb is None:
        return False
    y, x, h, w = op.crop
    return not (y <= bb[0] and x <= bb[1] and bb[2] <= y + h and bb[3] <= x + w)


def push_masks(e_par: ChainState, op: Op) -> tuple:
    sil = None if e_par.sil is None else push_through(e_par.sil, op, 128)
    cadj = None if e_par.c_adj is None else push_through(e_par.c_adj, op, 128)   # informational only
    cert = None if e_par.cert_mask is None else push_through(e_par.cert_mask, op, 1)
    return sil, cadj, cert


def hop_update(e_par: ChainState, hs: HopState, J, e_inf_2p30: int = 0) -> ChainState:
    op = hs.op
    q3 = J.q3()
    cls = class_hop(q3, hs.q_hop)
    ph = partial_hop(e_par, op)
    sil, cadj, cert = push_masks(e_par, op)
    einf = 0 if op.enc_class in (1, 7) else int(e_inf_2p30)
    return ChainState(cum=hs.cum, dims=(int(J.H), int(J.W)), sampling=int(J.sampling),
                      chain_class=worst(e_par.chain_class, cls),
                      lambda_chain_2p51=e_par.lambda_chain_2p51 + lam_2p51(q3[0], ETA20_CLS),
                      resamples=hs.resamples, hop_depth=e_par.hop_depth + 1,
                      excess_sum=e_par.excess_sum + einf, excess_max=max(e_par.excess_max, einf),
                      partial=bool(e_par.partial or ph), sil=sil, c_adj=cadj, cert_mask=cert,
                      ceilings=list(e_par.ceilings) + [(op.platform_id, hs.v_star)],
                      enc_classes=list(e_par.enc_classes) + [op.enc_class],
                      hop_classes=list(e_par.hop_classes) + [cls])


# ------------------------------------------------------------------------------------------------ VerifyHop
def child_grammar(child_bytes: bytes):
    """ContainerOK v2.1 in rendition mode (upload = false) and the coefficients; ``E.HOP.CONTAINER`` on failure."""
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    try:
        J = container_ok_v2(bytes(child_bytes), False)
        return J, read_coefs(bytes(child_bytes), J)
    except (ContainerError, RYError) as e:
        raise HopError("E.HOP.CONTAINER", e.name) from e


def verify_hop(parent, e_par: ChainState, child_bytes: bytes, entry: HopEntry, ledger, h: int | None = None,
               **codec_kw) -> ChainState:
    J, coefs = child_grammar(child_bytes)
    hs = hop_checks(e_par, entry.op, J, ledger, entry, h)
    if isinstance(parent, np.ndarray):
        P = parent
    else:
        P = _decoded(parent)
    if P.shape[:2] != tuple(e_par.dims):
        raise EvalError("E.EVAL.PARENT_DIMS", f"parent dims {P.shape[:2]} != chain state {e_par.dims}")
    res = hop_codec(P, hs.op, child_bytes, J, coefs=coefs, ledger=ledger, **codec_kw)
    if not res.ok:
        err = HopError("E.HOP.CODEC", f"failing coefficients (first {len(res.rows)} kept)")
        err.rows = res.rows
        raise err
    return hop_update(e_par, hs, J, res.e_inf_2p30)


def fetch_parent(fetch, entry: HopEntry) -> bytes:
    from cutline.jpeg.ry import r_y_v2_bytes
    try:
        b = fetch(entry.parent_locator)
    except Exception as e:                                                  # noqa: BLE001 (any fetch failure)
        raise ChainError("E.CHAIN.ANCESTOR_UNAVAILABLE", f"{entry.parent_locator}: {e}") from e
    if b is None:
        raise ChainError("E.CHAIN.ANCESTOR_UNAVAILABLE", entry.parent_locator)
    try:
        ok = r_y_v2_bytes(bytes(b), False) == bytes(entry.r_y_parent)
    except CutlineError:
        ok = False
    if not ok:
        raise ChainError("E.CHAIN.ANCESTOR_UNAVAILABLE", f"{entry.parent_locator}: R_Y != rYParent")
    return bytes(b)


@dataclass
class ChainOutcome:
    state: ChainState | None
    g0_reason: str | None = None
    hop_index: int | None = None
    error: str | None = None
    rows: list | None = None


def verify_chain(entries, state: ChainState, fetch, displayed: bytes, ledger, h: int, first_parent=None,
                 **codec_kw) -> ChainOutcome:
    st = state
    parent = first_parent
    for i, e in enumerate(entries):
        try:
            if parent is None:
                parent = fetch_parent(fetch, e)
            child = bytes(displayed) if i + 1 == len(entries) else None
            if child is None:
                try:
                    child = fetch_parent(fetch, entries[i + 1])
                except ChainError as ce:
                    return ChainOutcome(None, "ancestor", i + 1, ce.name)
        except ChainError as ce:
            return ChainOutcome(None, "ancestor", i, ce.name)
        try:
            st = verify_hop(parent, st, child, e, ledger, h, **codec_kw)
        except HopError as he:
            return ChainOutcome(None, "hop", i, he.name, getattr(he, "rows", None))
        parent = child
    return ChainOutcome(st)


# ------------------------------------------------------------------------------------------------ chain-level rules
def chain_cap(st: ChainState) -> int:
    if st.hop_depth > D_HOP or st.resamples > 1 or st.chain_class == "coarse" or st.partial:
        return 2
    if any(c in E1_CLASSES for c in st.enc_classes[1:]):
        return 3
    return 4


__all__ = ["ChainState", "initial_state", "assemble_chain", "g0_reason_of", "HopState", "hop_checks", "CodecResult",
           "hop_canvas", "hop_codec", "push_through", "push_masks", "cert_bbox", "partial_hop", "hop_update",
           "child_grammar", "verify_hop", "fetch_parent", "ChainOutcome", "verify_chain", "chain_cap",
           "aspect_margin", "aspect_holds", "cum_after", "V1_CLASSES", "E1_CLASSES"]
