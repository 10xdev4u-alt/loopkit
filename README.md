# loopkit

Lossless NVFP4/MXFP4 toolkit for looped MoE inference: entropy coding, bitwise
certificates, per-pass weight edits, fused dequant-GEMM staging, and a speed-claim oracle.

> **Source fact-check (2026-11-24).** Upstream llama.cpp already implements the Inkling2 looped
> architecture (`LLM_ARCH_INKLING2_LOOP`, `src/models/inkling2-loop.cpp`,
> `src/models/inkling2.cpp`, `conversion/inkling2.py`): cache slots
> `n_slot = n_phys + (n_passes-1)*(core_end-core_start)` and rank-r
> `LOOP_*_CTRL_*` gain/forget controllers, not a scalar softmax gamma. Our fork's contribution is
> union-resident MoE routing on top of that upstream graph, not building the loop graph from
> scratch. See `docs/INKLING_INTEGRATION.md`.

Target model: **Inkling 2 Mini** (Thinking Machines Lab, Apache-2.0) — 24 layers
(20 native NVFP4 + 4 MXFP8), 128-expert MoE, middle layers loop 3×, ~20 GB,
**[unverified local hypothesis; not in primary sources]** activation outlier at dimension
2027 (claimed only in our repo/config/tests; absent from TML's release post and model card).
**No weights ship with this repo.**

## What this is

- **ANS-NVFP4** — deterministic E2M1/E4M3 emulation plus entropy coding of the
  stored codes and block scales. `ans_decode(ans_encode(t))` is bit-exact on
  codes, scales, shape and block.
- **BitLoop** — SHA-256 certificates over all three loop passes. One flipped bit
  invalidates the certificate; `assert_lossless` raises with the root digest.
- **FDM-Dot** — dequant + GEMM staging that is byte-identical to the float32
  reference. No silent numerical drift behind a "fused" label.
- **CycleAudit** — bootstrap confidence intervals for speed claims. A claim is
  valid only if it is below the lower 95% bound, never rounded up.

Supporting modules: `loopedit` (immutable per-pass weight diffs) and `geometry`
(LittleBit-2 style diagnostics with explicit kill gates).

## Hardware honesty

T4 (Turing) has **no FP4 or FP8 tensor cores**. `fdm.fused_available()` returns
`False` and `fdm_dot` uses the reference matmul path. Everything on a T4 is a
numerical or staging result; no wall-clock speedup is asserted from this repo.
All real compute runs on Kaggle 2×T4, never on the local CPU.

## Install

```bash
pip install -e ".[dev]"
bash scripts/smoke.sh
```

Python 3.11+, dependencies: `numpy`, `scipy`. Optional: `torch`.

## 30-second demo

```bash
python -c "import numpy as np; np.save('/tmp/demo.npy', np.random.randn(8,256).astype('float32'))"
python -m loopkit.cli nvfp4-roundtrip --file /tmp/demo.npy
python -m loopkit.cli fdm-check
python -m loopkit.cli geometry
```

## Health check

The spectral health checker recovers weights from a pair of activation/output
arrays (saved as `.npy`) and reports a JSON health card. It needs **no model
weights**:

```bash
python -m loopkit.cli health --x FILE.npy --y FILE.npy [--name LAYER] [--ridge 1e-3]
```

The command exits 0 when the card is not flagged and 1 when it is. For a
directory of `*.x.npy` / `*.y.npy` pairs, use
`python -m loopkit.cli health --batch DIR` to print a list of cards.

## Guarantees (exactly two)

1. `ans_decode(ans_encode(t)) == t` bit-exact on stored representation.
2. `fdm_dot(a, b) == reference_dot(a, b)` bit-exact.

Everything else is measured, not asserted.

## Geometry kill gates

`geometry.geometry_gate` returns `go=False` with reasons when:
- the error scaling law has `lambda <= 0`,
- no break-even exists within the loop budget,
- FP4-aware rotation does not reduce MSE versus plain NVFP4 quantization.

Geometry is an optional module. A failed gate never affects the codec,
certificates, kernel or audit modules.

## Model config

`configs/inkling2mini.toml` holds the target constants including
`outlier_dim = 2027` **[unverified local hypothesis; not in primary sources]** and
`loop_passes = 3`. The loop fields `loop_start = 6`, `loop_end = 18` (exclusive),
`loop_passes = 3`, and `experts_per_token = 8` are verified upstream facts.

## Model planning

`loopkit.model` loads the config into a frozen `ModelPlan` with derived loop
numbers. To print the plan as JSON:

```bash
python -m loopkit.cli model [--config FILE]
```

The upstream llama.cpp formula gives
`cache_slots = n_layers + (n_passes-1)*(loop_end-loop_start)`. For the default
plan (24 physical layers, 3 passes, core layers 6..17) that is `24 + 2*12 = 48`
cache slots and 48 layer applications per token. `to_dict()` includes an
`unverified` map labelling `outlier_dim` as an unverified local hypothesis; it
is absent from primary sources and is not a stated model property.

## Tests

Three release-blocking tests: `test_ans_roundtrip_is_bit_exact`,
`test_certificate_detects_single_bit_flip`, `test_fdm_dot_matches_reference`.
CI runs all FAST tests with `pytest -q -m "not kaggle"`. The two KAGGLE-marked
geometry tests run on the T4 notebook in `kaggle/run_audit.ipynb`.

## License

Apache-2.0.
