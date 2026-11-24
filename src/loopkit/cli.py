"""Command line surface for loopkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import shc
from .audit import CycleAudit, TimingSample
from .bitloop import assert_lossless, certify_loop
from .fdm import fdm_check
from .geometry import geometry_gate
from .nvfp4 import ans_decode, ans_encode, ans_size, dequantize_nvfp4, quantize_nvfp4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loopkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("nvfp4-roundtrip")
    q.add_argument("--file", required=True)
    q.add_argument("--block", type=int, default=16)

    c = sub.add_parser("certify")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--model", default="inkling2mini")

    f = sub.add_parser("fdm-check")
    f.add_argument("--shape", nargs=3, type=int, default=[4, 64, 64])

    a = sub.add_parser("audit")
    a.add_argument("--claim", type=float, required=True)
    a.add_argument("--baseline", required=True)
    a.add_argument("--candidate", required=True)
    a.add_argument("--loops", type=int, default=3)

    g = sub.add_parser("geometry")
    g.add_argument("--file", required=False)

    h = sub.add_parser("health", help="spectral health check from activation/output pairs")
    h.add_argument("--x", metavar="FILE.npy", help="activations (n_samples, d_in)")
    h.add_argument("--y", metavar="FILE.npy", help="outputs (n_samples, d_out)")
    h.add_argument("--batch", metavar="DIR", help="directory of *.x.npy / *.y.npy pairs")
    h.add_argument("--name", default="layer", help="layer name for a single pair")
    h.add_argument("--ridge", type=float, default=1e-3)
    return p


def _load_state(path: str) -> dict[str, np.ndarray]:
    p = Path(path)
    if p.is_file():
        return {"weights": np.load(p)}
    return {f.stem: np.load(f) for f in sorted(p.glob("*.npy"))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "nvfp4-roundtrip":
        x = np.load(args.file).astype(np.float32)
        t = quantize_nvfp4(x, args.block)
        u = ans_decode(ans_encode(t))
        ok = (np.array_equal(u.codes, t.codes) and np.array_equal(u.scales, t.scales)
              and u.shape == t.shape)
        print(json.dumps({"lossless": bool(ok), "ratio": round(ans_size(t) / max(1, x.nbytes), 4),
                          "ans_bytes": ans_size(t), "tensor_bytes": int(x.nbytes)}))
        return 0 if ok else 1

    if args.cmd == "certify":
        refs = [_load_state(args.ref)]
        cands = [_load_state(args.cand)]
        cert = certify_loop(refs, cands, model=args.model)
        Path("cert.json").write_text(json.dumps(cert.__dict__, default=str, indent=2))
        print(json.dumps({"all_exact": cert.all_exact, "root_digest": cert.root_digest}))
        try:
            assert_lossless(cert)
        except AssertionError as e:
            print(str(e))
            return 1
        return 0

    if args.cmd == "fdm-check":
        r = fdm_check(shape=tuple(args.shape))
        print(json.dumps({"path": r.path, "max_abs_err_vs_ref": r.max_abs_err_vs_ref}))
        return 0 if r.max_abs_err_vs_ref == 0.0 else 1

    if args.cmd == "audit":
        base = TimingSample(json.loads(Path(args.baseline).read_text())["name"],
                            np.array(json.loads(Path(args.baseline).read_text())["seconds"]))
        cand = TimingSample(json.loads(Path(args.candidate).read_text())["name"],
                            np.array(json.loads(Path(args.candidate).read_text())["seconds"]))
        audit = CycleAudit(loops=args.loops)
        report = audit.compare(base, cand)
        ok = audit.verdict(args.claim, report)
        print(json.dumps({**report.__dict__, "claim_valid": bool(ok)}, indent=2))
        return 0 if ok else 1

    if args.cmd == "geometry":
        if args.file:
            x = np.load(args.file).astype(np.float32)
        else:
            x = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
        bits = np.array([1.0, 2.0, 4.0])
        errors = np.array([0.5, 0.25, 0.125])
        verdict = geometry_gate(bits, errors, base_per_pass=2.0, codec_setup=1.0,
                                 codec_per_pass=0.25, loops=3, x=x)
        print(json.dumps({"go": verdict.go, "reasons": verdict.reasons,
                          "metrics": verdict.metrics}, indent=2))
        return 0 if verdict.go else 1

    if args.cmd == "health":
        if args.batch:
            samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            directory = Path(args.batch)
            for xp in sorted(directory.glob("*.x.npy")):
                name = xp.name[: -len(".x.npy")]
                yp = directory / f"{name}.y.npy"
                if not yp.is_file():
                    parser.error(f"health: missing output pair for {name}")
                samples[name] = (np.load(xp).astype(np.float64),
                                 np.load(yp).astype(np.float64))
            cards = shc.health_report(samples)
            print(json.dumps([c.__dict__ for c in cards.values()], indent=2, default=str))
            return 1 if any(c.flagged for c in cards.values()) else 0

        if args.x is None or args.y is None:
            parser.error("health: --x and --y are required unless --batch is given")
        x = np.load(args.x).astype(np.float64)
        y = np.load(args.y).astype(np.float64)
        card = shc.health_card(args.name, x, y, ridge=args.ridge)
        print(json.dumps(card.__dict__, default=str))
        return 1 if card.flagged else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
