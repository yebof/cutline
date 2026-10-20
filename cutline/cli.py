"""cutline.cli - command-line entry point.

The capture-side subcommands run the EMULATED capture world of ``capture_emu.session``:

- ``capture IMAGE --out DIR [--seg u2netp | --mask PNG | --full] [--side 512]``: enrol an emulated device, capture
  the master (box-downscaled to <= side, cropped to multiples of 16), anchor, ExportIntake, CoSign. Writes
  ``DIR/session.pkl`` (emulation state: keys and witnesses, local use only) and ``DIR/capture.json``.
- ``compile DIR --out ODIR [--program prog.json | --auto] [--sampling 0|1] [--kernel 0|1] [--bg gen:SEED |
  color:R,G,B | PNG]``: Compile from the capture's level-l* cells; writes ``X.png``, ``alpha.png``,
  ``bg.png``, ``program.bin``, ``compiled.json``. ``prog.json`` = {lstar, crop, out, place, canvas, kernel,
  sampling, flags, gain: [r, g, b] | lut: 768-byte hex, D}.
- ``encode X.png --out listing.jpg [--q 85] [--sampling 0|1] [--encoder ID]``: ``encode_e0`` (libjpeg-turbo 2.1
  islow) or ``encode_with(ID)``.
- ``bundle DIR --compiled ODIR --listing listing.jpg --out BUNDLE [--archive] [--whole-fill] [--listing-id ID]
  [--seller S] [--marketplace M]``: ClaimCore v2, sigma_claim, the bundle members; then bundle stage A as a
  self-check. Prints the claimId.
- ``prove-static BUNDLE --session DIR [--out calldata.hex]``: stage A, then the first static violation in search
  order; if there is one, the ``proveStaticViolation`` calldata with exactly the preimages its row reads,
  checked against the reference model ``prove_static_violation``.

Verifier-side subcommands:

- ``selfcheck BUNDLE [--chain-id N] [--registry 0x..] [--tier-t RECORD_HEX ...]``: the seller's self-check: bundle
  stage A, then VerifyE (tier E) or VerifyT (tier T, under the tier-T records given) with stages B and C; prints one
  JSON line (``verify_e.selfcheck``); exit 0 iff the claim passes, 1 otherwise.
- ``verify BUNDLE [same options]``: stage A, then VerifyE / VerifyT; prints the verdict, the rows' count, the first
  row, the evidence fields computed (canonical values: integers as decimal strings, bytes as 0x hex) and the
  ``versions`` parts; exit 0 when the verifier ran (any verdict), 1 when stage A failed, 2 when BUNDLE does not exist.
- ``prove-block BUNDLE [same options] [--mcu M --comp C --k K] [--out calldata.hex]``: the ProofIn v2
  calldata (``proveBlockFailure``) for the given block, or VerifyE's first failing unit; checked against the
  court reference model (``ledger.proofin.outcome_of``); 4:4:4 claims only. Exit 0 with calldata, 1 when there
  is no failing unit.
- ``evaluate BUNDLE --listing-id ID --seller S --salt-ref HEX [--marketplace M] [--display FILE] [--parent FILE ...]
  [--h H] [--regime EU|CN|IPTC|GMC] [--urec FILE --accept-digest HEX ...] [--ledger-ops JSON] [--level-cap 4]
  [--cap valid] [--receipt node] [ledger options]``: Evaluate on the displayed file (default:
  the bundle's listing), with an EMULATED ledger: the claim registered with the given status unless ``--ledger-ops``
  gives the builder calls (``ledger_view.snapshot_from_ops``; hop batches and ceilings live there); parents are
  fetched from the ``--parent`` files by content; U from the U record or the party's own PubSeg run (ISNet, needs
  about 2 GB). Prints ``{L, level, label, record (canonical evidence-record form), notes}``; exit 0.
- ``hop PARENT CHILD --crop Y,X,H,W [--kernel K] [--enc-class C] [--ep E] [--platform P]``: VerifyHop
  of one hop with an emulated hop batch (one leaf, the op built from the child's tables) and the parent as
  Y0 (its own tables and grid); prints the chain fields, or the ``E.HOP.*`` error and the failing rows' count.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np


def _load_image(path, side=None):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if side:
        im.thumbnail((side, side), Image.BOX)
    return np.asarray(im)


def _save_png(path, a):
    from PIL import Image
    Image.fromarray(np.asarray(a)).save(path)


def _session(d):
    with open(os.path.join(d, "session.pkl"), "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------------------ capture
def cmd_capture(a) -> int:
    from capture_emu.session import Session
    M = _load_image(a.image, a.side)
    M = np.ascontiguousarray(M[:M.shape[0] // 16 * 16, :M.shape[1] // 16 * 16])
    if a.mask:
        soft = _load_image(a.mask)[..., 0].astype(np.float32) / 255
        if soft.shape != M.shape[:2]:
            raise SystemExit(f"mask {soft.shape} != master {M.shape[:2]}")
        src = soft
    elif a.full:
        src = np.ones(M.shape[:2], np.float32)
    else:
        src = a.seg
    ses = Session(a.label, segmenter=a.seg)
    rh = ses.capture(M, src)
    ses.anchor()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "session.pkl"), "wb") as f:
        pickle.dump({"session": ses, "rec_hash": rh}, f)
    c = ses.captured[rh]
    cm = c.witness.commitment
    info = {"rec_hash": rh.hex(), "rec": c.rec_bytes.hex(), "sig_dev": c.sig64.hex(), "sig_hold": c.sig_hold.hex(),
            "batch_id": c.anchor.batch_id, "leaf_index": c.anchor.leaf_index,
            "path": [p.hex() for p in c.anchor.path], "H0": cm.H0, "W0": cm.W0, "Lmax": cm.Lmax,
            "R_cut": cm.R_cut.hex(), "R_occ": cm.R_occ.hex(), "owner": c.witness.rec.owner.hex(),
            "region_px": int(np.asarray(cm.levels[0][1] > 0).sum()), "EMULATED": True}
    with open(os.path.join(a.out, "capture.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({"rec_hash": info["rec_hash"], "H0": cm.H0, "W0": cm.W0, "Lmax": cm.Lmax}))
    return 0


# ------------------------------------------------------------------------------------------------ compile
def _program_from_json(j):
    from capture_emu.session import gain_lut
    from cutline.types import Program
    lut = bytes.fromhex(j["lut"]) if "lut" in j else gain_lut(tuple(j.get("gain", (100, 100, 100))))
    return Program(lstar=j["lstar"], crop=tuple(j["crop"]), out=tuple(j["out"]), place=tuple(j["place"]),
                   canvas=tuple(j["canvas"]), kernel=j["kernel"], sampling=j["sampling"], flags=j.get("flags", 0),
                   lut=lut, D=tuple(j.get("D", ())))


def _bg(spec, canvas):
    Hc, Wc = canvas
    if spec.startswith("gen:"):
        yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.int64)
        seed = int(spec[4:])
        return np.stack([40 + ((yy * (3 + c + seed) + xx * (5 + 2 * c)) % 256) * 3 // 4 for c in range(3)],
                        -1).astype(np.uint8)
    if spec.startswith("color:"):
        bg = np.zeros((Hc, Wc, 3), np.uint8)
        bg[:] = [int(v) for v in spec[6:].split(",")]
        return bg
    bg = _load_image(spec)
    if bg.shape[:2] != (Hc, Wc):
        raise SystemExit(f"background {bg.shape[:2]} != canvas {(Hc, Wc)}")
    return bg


def cmd_compile(a) -> int:
    from capture_emu.session import alpha_bbox, fit_program, gain_lut
    from cutline.commit.cells import occupancy, open_cells
    from cutline.compiler.compose import compile_listing, opened_set
    st = _session(a.dir)
    ses, rh = st["session"], st["rec_hash"]
    w = ses.captured[rh].witness
    if a.program:
        prog = _program_from_json(json.load(open(a.program)))
    else:
        prog = fit_program(alpha_bbox(w.commitment.levels[1][1]), a.sampling, a.kernel, gain_lut((100, 100, 100)))
    C, al, _ = w.commitment.levels[prog.lstar]
    op = opened_set(occupancy(al), prog, *al.shape)
    _, data, _ = open_cells(w.commitment, w.k_cap, prog.lstar, op)
    bg = _bg(a.bg, prog.canvas)
    comp = compile_listing(prog, {c: data[k * 1024:(k + 1) * 1024] for k, c in enumerate(op)}, al.shape[0],
                           al.shape[1], bg)
    os.makedirs(a.out, exist_ok=True)
    _save_png(os.path.join(a.out, "X.png"), comp.X)
    _save_png(os.path.join(a.out, "alpha.png"), comp.alpha)
    _save_png(os.path.join(a.out, "bg.png"), bg)
    with open(os.path.join(a.out, "program.bin"), "wb") as f:
        f.write(prog.encode())
    info = {"tHash": prog.t_hash().hex(), "canvas": list(prog.canvas), "sampling": prog.sampling,
            "opened_cells": len(op), "mcert_count": len(comp.m_cert), "sil_px": int(comp.sil.sum())}
    with open(os.path.join(a.out, "compiled.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))
    return 0


# ------------------------------------------------------------------------------------------------ encode
def cmd_encode(a) -> int:
    from cutline.jpeg.encode import encode_e0, encode_with
    X = _load_image(a.image)
    b = encode_with(a.encoder, X, a.q, a.sampling) if a.encoder else encode_e0(X, a.q, a.sampling)
    with open(a.out, "wb") as f:
        f.write(b)
    print(json.dumps({"bytes": len(b), "out": a.out}))
    return 0


# ------------------------------------------------------------------------------------------------ bundle
def cmd_bundle(a) -> int:
    from cutline import bundle as B
    from cutline.compiler.program import decode_program
    st = _session(a.dir)
    ses, rh = st["session"], st["rec_hash"]
    prog, ok = decode_program(open(os.path.join(a.compiled, "program.bin"), "rb").read())
    if not ok:
        raise SystemExit("program.bin is not a well-formed enc(pi)")
    bg = _load_image(os.path.join(a.compiled, "bg.png"))
    jpg = open(a.listing, "rb").read()
    m, core = ses.claim_members(rh, jpg, prog, bg, whole_fill=a.whole_fill, listing_id=a.listing_id,
                                seller=a.seller, marketplace=a.marketplace)
    B.write_members(a.out, m, archive=a.archive)
    raw = B.stage_a(a.out, ses.chain_id, ses.registry)
    print(json.dumps({"claim_id": raw.claim_id.hex(), "nf": raw.nf.hex(), "n_cells": len(raw.cellidx),
                      "out": a.out, "form": raw.form}))
    return 0


# ------------------------------------------------------------------------------------------------ prove-static
def cmd_prove_static(a) -> int:
    from cutline import bundle as B
    from cutline.compiler.static_rules import (LedgerState, StaticIn, encode_static_calldata, first_violation,
                                               preimages, prove_static_violation)
    from cutline.jpeg.coefs import read_coefs
    from cutline.jpeg.grammar import container_ok_v2
    from cutline.jpeg.ry import r_y_parts
    ses = _session(a.session)["session"]
    raw = B.stage_a(a.bundle, ses.chain_id, ses.registry)
    jpg = raw.members["listing.jpg"]
    g = container_ok_v2(jpg, True)
    header, r_mcu, _ = r_y_parts(g, read_coefs(jpg, g))
    c = raw.claim
    fv = first_violation(c.prog, header, raw.proof.cut_pre, c.core)
    if fv is None:
        print(json.dumps({"claim_id": raw.claim_id.hex(), "static_violation": None}))
        return 0
    rule, i, j = fv
    pre = preimages(rule, i, c.prog, header, r_mcu, c.rec, raw.proof.cut_pre)
    s = StaticIn(raw.nf, c.claimant, c.core, rule, i, j, *pre)
    st = LedgerState()
    st.register(raw.nf, c.claimant, c.core)
    assert prove_static_violation(s, st) == "StaticViolation"
    cd = encode_static_calldata(s)
    if a.out:
        with open(a.out, "w") as f:
            f.write("0x" + cd.hex() + "\n")
    print(json.dumps({"claim_id": raw.claim_id.hex(), "static_violation": [rule, i, j], "calldata_len": len(cd),
                      "out": a.out}))
    return 0


# ------------------------------------------------------------------------------------------------ verifier side
def _ledger(a):
    from cutline.registries import TierTEntry, decode_table_record
    from cutline.verify.ledger_view import StateSnapshot
    kw = {}
    if a.chain_id is not None:
        kw["chain_id"] = a.chain_id
    if a.registry is not None:
        kw["registry"] = bytes.fromhex(a.registry[2:] if a.registry.startswith("0x") else a.registry)
    led = StateSnapshot(**kw)
    for rec in a.tier_t or []:
        t = decode_table_record(bytes.fromhex(rec[2:] if rec.startswith("0x") else rec))
        led.put_tier_t(t.id, TierTEntry.from_table(t))
    return led


def _stage_a(a, led):
    from cutline import bundle as B
    return B.stage_a(a.bundle, led.chain_id, led.registry)


def _canon(v):
    from cutline.verify.record import _cv
    if isinstance(v, dict) and "dE00_p50" in v:
        from cutline.verify.record import _tone
        return _tone(v)
    return _cv(v)


def cmd_selfcheck(a) -> int:
    from cutline.verify.verify_e import selfcheck
    led = _ledger(a)
    res = selfcheck(a.bundle, led.chain_id, led.registry, led)
    print(json.dumps(res, sort_keys=True))
    return 0 if res.get("ok") else 1


def cmd_verify(a) -> int:
    from cutline.errors import CutlineError
    if not os.path.exists(a.bundle):
        print(json.dumps({"error": f"no such bundle: {a.bundle}"}))
        return 2
    led = _ledger(a)
    try:
        raw = _stage_a(a, led)
    except CutlineError as e:
        print(json.dumps({"stage": "A", "error": e.name, "g0_reason": "bundle"}))
        return 1
    if raw.claim.core.tier == 2:
        from cutline.verify.verify_t import verify_t
        r = verify_t(raw, led)
    else:
        from cutline.verify.verify_e import verify_e
        r = verify_e(raw, led, fraud=False)
    out = {"claim_id": "0x" + raw.claim_id.hex(), "tier": r.tier, "verdict": r.verdict, "g0_reason": r.g0_reason,
           "error": r.error, "static_violation": None if r.static_violation is None else list(r.static_violation),
           "n_rows": len(r.rows), "first_row": [str(v) for v in r.rows[0]] if r.rows else None,
           "facts": {k: _canon(v) for k, v in sorted(r.facts.items())}, "versions": _canon(r.versions)}
    print(json.dumps(out, sort_keys=True))
    return 0


def cmd_prove_block(a) -> int:
    from cutline.ledger.proofin import encode_proofin_calldata
    from cutline.verify.fraud import court_check, prove_block
    led = _ledger(a)
    raw = _stage_a(a, led)
    if a.mcu is not None:
        mcu, comp, k = a.mcu, a.comp, a.k
    else:
        from cutline.verify.verify_e import verify_e
        r = verify_e(raw, led, fraud=False, all_rows=False)
        if not r.rows:
            print(json.dumps({"claim_id": "0x" + raw.claim_id.hex(), "verdict": r.verdict, "g0_reason": r.g0_reason,
                              "static_violation": r.static_violation, "calldata": None}))
            return 1
        mcu, comp, _, k = r.rows[0][:4]
    p = prove_block(raw, mcu, comp, k)
    cd = encode_proofin_calldata(p)
    if a.out:
        with open(a.out, "w") as f:
            f.write("0x" + cd.hex() + "\n")
    print(json.dumps({"claim_id": "0x" + raw.claim_id.hex(), "mcu": mcu, "comp": comp, "k": k,
                      "calldata_len": len(cd), "court": court_check(p), "out": a.out}))
    return 0


def cmd_evaluate(a) -> int:
    from cutline import bundle as B
    from cutline.enc import unhx
    from cutline.types import Ctx
    from cutline.verify.evidence import evaluate
    from cutline.verify.ledger_view import snapshot_from_ops
    from cutline.verify.record import to_obj
    if a.ledger_ops:
        L = json.load(open(a.ledger_ops))
        led = snapshot_from_ops(L["chain_id"], L["registry"], L.get("height", 0), L["ops"])
    else:
        led = _ledger(a)
        raw = B.read_bundle(a.bundle, led.chain_id, led.registry)
        led.add_claim(raw.claim_id, height=0, level_cap=a.level_cap, cap=a.cap, receipt=a.receipt)
    parents = [open(p, "rb").read() for p in a.parent or []]
    import hashlib
    fm = {"sha256:" + hashlib.sha256(b).hexdigest(): b for b in parents}
    display = open(a.display, "rb").read() if a.display else B.load_members(a.bundle)[0]["listing.jpg"]
    ctx = Ctx(a.marketplace, a.listing_id, a.seller, unhx(a.salt_ref))
    urec = open(a.urec, "rb").read() if a.urec else None
    acc = [unhx(d) for d in a.accept_digest] if a.accept_digest else None
    ev = evaluate(display, a.bundle, a.h if a.h is not None else led.height, a.regime, ctx, led, fm.get, urec=urec,
                  accepted_digests=acc)
    notes = {k: ("0x" + v.hex() if isinstance(v, bytes) else v) for k, v in ev.notes.items()}
    print(json.dumps({"L": ev.L, "level": ev.level, "label": ev.label, "record": to_obj(ev.evidence),
                      "notes": notes}, sort_keys=True, default=str))
    return 0


def cmd_hop(a) -> int:
    from cutline.crypto import H
    from cutline.errors import CutlineError
    from cutline.jpeg.grammar import container_ok_v2
    from cutline.jpeg.ry import r_y_v2_bytes
    from cutline.verify import hop as HOP
    from cutline.verify.hopcodec import HopEntry, Op, encode_op, hop_leaf, op_hash
    from cutline.verify.ledger_view import StateSnapshot
    parent, child = open(a.parent, "rb").read(), open(a.child, "rb").read()
    gp = container_ok_v2(parent, False)
    try:
        J, _ = HOP.child_grammar(child)
    except CutlineError as e:
        print(json.dumps({"error": e.name}))
        return 1
    crop = tuple(int(v) for v in a.crop.split(","))
    op = Op(1, crop, (J.H, J.W), a.kernel, 0, b"".join(J.q3()), a.enc_class, a.ep, a.platform)
    led = StateSnapshot()
    if a.enc_class == 7:
        from cutline.verify import ep as EP
        led.add_ep(a.ep, EP.resolve(a.ep).registry_entry())
    bid = led.add_hop_batch([hop_leaf(r_y_v2_bytes(parent), r_y_v2_bytes(child), op_hash(op)), H(b"pad")],
                            a.platform, 1)
    e = HopEntry(encode_op(op), r_y_v2_bytes(parent), bid, 0, tuple(led.hop_proof(bid, 0)), "parent")
    st = HOP.initial_state({"q3": gp.q3(), "enc_classes": [1]},
                           {"crop": (0, 0, gp.H, gp.W), "out": (gp.H, gp.W), "canvas": (gp.H, gp.W),
                            "sampling": gp.sampling})
    try:
        s = HOP.verify_hop(parent, st, child, e, led, 1)
    except CutlineError as ex:
        print(json.dumps({"error": ex.name, "n_rows": len(getattr(ex, "rows", None) or [])}))
        return 1
    print(json.dumps({"accepted": True, "hop_classes": s.hop_classes, "chain_class": s.chain_class,
                      "lambda_chain_2p51": str(s.lambda_chain_2p51), "resampling_hops": s.resamples,
                      "excess_sum_2p30": str(s.excess_sum), "chain_cap": HOP.chain_cap(s),
                      "cum": [str(v) for v in s.cum]}))
    return 0


def _ledger_args(p):
    p.add_argument("bundle")
    p.add_argument("--chain-id", type=int)
    p.add_argument("--registry")
    p.add_argument("--tier-t", action="append", help="a 55-byte tier-T table record (hex); repeatable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cutline", description="CUTLINE reference command-line tool")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("capture")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p.add_argument("--seg", default="u2netp")
    p.add_argument("--mask")
    p.add_argument("--full", action="store_true")
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--label", default="cutline-cli")
    p.set_defaults(fn=cmd_capture)
    p = sp.add_parser("compile")
    p.add_argument("dir")
    p.add_argument("--out", required=True)
    p.add_argument("--program")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--sampling", type=int, default=0)
    p.add_argument("--kernel", type=int, default=0)
    p.add_argument("--bg", default="gen:1")
    p.set_defaults(fn=cmd_compile)
    p = sp.add_parser("encode")
    p.add_argument("image")
    p.add_argument("--out", required=True)
    p.add_argument("--q", type=int, default=85)
    p.add_argument("--sampling", type=int, default=0)
    p.add_argument("--encoder")
    p.set_defaults(fn=cmd_encode)
    p = sp.add_parser("bundle")
    p.add_argument("dir")
    p.add_argument("--compiled", required=True)
    p.add_argument("--listing", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--archive", action="store_true")
    p.add_argument("--whole-fill", action="store_true")
    p.add_argument("--listing-id", default="B0CLI00001")
    p.add_argument("--seller", default="seller-cli")
    p.add_argument("--marketplace", type=int, default=5)
    p.set_defaults(fn=cmd_bundle)
    p = sp.add_parser("prove-static")
    p.add_argument("bundle")
    p.add_argument("--session", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_prove_static)
    p = sp.add_parser("selfcheck")
    _ledger_args(p)
    p.set_defaults(fn=cmd_selfcheck)
    p = sp.add_parser("verify")
    _ledger_args(p)
    p.set_defaults(fn=cmd_verify)
    p = sp.add_parser("prove-block")
    _ledger_args(p)
    p.add_argument("--mcu", type=int)
    p.add_argument("--comp", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_prove_block)
    p = sp.add_parser("evaluate")
    _ledger_args(p)
    p.add_argument("--listing-id", required=True)
    p.add_argument("--seller", required=True)
    p.add_argument("--salt-ref", required=True)
    p.add_argument("--marketplace", type=int, default=5)
    p.add_argument("--display")
    p.add_argument("--parent", action="append")
    p.add_argument("--h", type=int)
    p.add_argument("--regime", default="EU")
    p.add_argument("--urec")
    p.add_argument("--accept-digest", action="append",
                   help="PubSeg digest(s) whose U records are accepted (default: only the local PubSeg build's digest)")
    p.add_argument("--ledger-ops")
    p.add_argument("--level-cap", type=int, default=4)
    p.add_argument("--cap", default="valid")
    p.add_argument("--receipt", default="node")
    p.set_defaults(fn=cmd_evaluate)
    p = sp.add_parser("hop")
    p.add_argument("parent")
    p.add_argument("child")
    p.add_argument("--crop", required=True)
    p.add_argument("--kernel", type=int, default=0)
    p.add_argument("--enc-class", type=int, default=1)
    p.add_argument("--ep", type=int, default=0)
    p.add_argument("--platform", type=int, default=7)
    p.set_defaults(fn=cmd_hop)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
