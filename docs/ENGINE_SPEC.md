# SPEC_ENGINE — Inkling 2 Mini inference engine (C++17 / CUDA sm_75)

Status: design. Hardware: Kaggle 2×T4 (sm_75, 16 GB each, 320 GB/s HBM2, ~65 TFLOP/s FP16 WMMA,
**no FP4/FP8 tensor cores**, no NVLink, PCIe ~12 GB/s between devices). Correctness oracle and
format source of truth: `/tmp/build/loopkit` (Apache-2.0). No weights ship with loopkit; we ingest
the model's native NVFP4 codes verbatim and never re-quantize.


> **Correction — 2026-11-24 (source fact-check; supersedes premises below):**
> - **d2027 / "activation outlier at dimension 2027" is an unverified local
>   hypothesis; not in primary sources.** It appears only in our repo/config/tests
>   and is absent from Thinking Machines Lab's release post and model card. Treat
>   the FP32 outlier side-lane (ODSL) as speculative until measured on real weights.
> - **Gate: rank-r gain/forget loop controllers in the upstream implementation;
>   our earlier scalar-gamma model was a simplification.** Upstream
>   `src/models/inkling2-loop.cpp` and `src/models/inkling2.cpp` implement
>   `(1 - forget)*cur + gain*message` with `tanh` gain/forget produced from
>   `LLM_TENSOR_LOOP_Q_CTRL_*` / `_K_CTRL_*` / `_V_CTRL_*` / `_FFN_CTRL_*`
>   (`_DOWN`, `_GAIN`, `_FORGET`) at `gate_rank=64` — not a scalar softmax.
>   Cache slots: `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`; for this
>   model `24 + 2*12 = 48`, with layers 6–17 looped (not "prelude 0–11").
> - **Upstream already provides the looped graph and converter**
>   (`src/models/inkling2-loop.cpp`, `src/models/inkling2.cpp`,
>   `conversion/inkling2.py`, `gguf-py/gguf/constants.py`,
>   `LLM_ARCH_INKLING2_LOOP`). Our fork's contribution is union-resident MoE
>   routing on top of upstream, not building the loop graph from scratch.
> - The design below is retained as the original plan; every `gamma` / `d2027`
>   mention is annotated inline. See `docs/INKLING_INTEGRATION.md` for the
>   source-labelled fact inventory.

Model constants (from `configs/inkling2mini.toml`): 24 layers, 4 blocks × 6, loop_passes=3 for
blocks 2–3 (middle 12 layers), 48 layer applications/token, 26.4B total / 3.4B active / 5.4B
weight activations per token, 128 experts/layer with top-8, NVFP4 block=16 on 20 layers, MXFP8
block=32 on last 4, outlier at K-dimension **2027** **[unverified local hypothesis; not in
primary sources]**, cross-loop message = softmax mix of two previous-pass vectors with learned gate
`gamma` **[simplification — upstream uses rank-r gain/forget loop controllers]**, initialized to 0.0
in the simplified model, varloop variant 1–5 passes.
Footprint ~20 GB, so it does not fit on one 16 GB T4: **device sharding is mandatory**.

---

## 1. Three architectural decisions that exploit the loop/MoE structure

Generic "faster GEMM" work is a losing game on T4 against llama.cpp: both hit the same 320 GB/s
wall and the same WMMA peak. The wins have to come from doing *less* work that only a loop-aware
engine knows it can skip.

### Decision 1 — Route-once, pass-resident expert tiles (RORET)

**Premise to measure, not assume:** the router sees nearly the same hidden state on passes 0, 1, 2
because in the *simplified local model* the cross-loop message starts at `gamma=0` and is a small
softmax mixture **[upstream: rank-r gain/forget controllers, not a scalar gamma]**. Therefore the
top-8 expert *set* per middle layer is identical across the 3 passes for most tokens.

**Mechanism:** at the first pass of a middle layer, compute top-8 once, gather those 8 experts'
packed codes+scales from the `inkbin` weight plane into a per-request hot arena
(`ExpertTile[12 layers][8]`), and decode them to FP16 **once**. Passes 1 and 2 reuse the same
resident tiles. A per-token flag `route_dirty` handles the minority where the freshly computed
top-8 set differs; dirty tokens pay a re-gather, and the agreement rate is reported every run.

**Why it beats a stack without a loop concept:** llama.cpp/GGUF has no way to know pass 2 is the
same layer with the same experts, so it re-reads middle-layer weights from HBM three times.
5.4B weight activations/token at 4.5 bits ≈ 3.04 GB/token; the two redundant middle passes are
≈1.8B weights ≈ 1.0 GB/token. Removing them saves ≈1.0 GB and ≈3.1 ms at batch 1 (320 GB/s).
This is the single largest lever, and it is unavailable to a layer-by-layer driver.

**Honesty note:** 8 experts × 6 layers on a GPU ≈ 48 × 17.7M params ≈ 0.85B params ≈ 0.48 GB
packed / 0.95 GB FP16 — far larger than T4's 4 MB L2. The win is eliminating 2 of 3 HBM weight
streams, *not* "L2 residency". Do not claim cache-level magic.

### Decision 2 — Pass-interleaved batch scheduler with expert-major GEMM (PIBS)

**Premise:** the three passes form a depth-3 dependency chain per token. A normal batch decoder
either serializes passes (M=8 → one 16-row WMMA tile, 50% padded waste) or barrier-synchronizes the
whole batch at every pass.

**Mechanism:** a token ring holds up to 3 pass-stages per layer. Tokens that are ready at the same
(layer, expert) are folded into **one GEMM per expert**, with M = number of (token, pass) pairs
routed there — up to 8×3 = 24 rows (two m16 WMMA tiles) instead of 8. The scheduler processes all
8 tokens of a batch in lockstep per (pass, layer), so each expert weight tile is read once and
applied to all ready (token, pass) rows. KV and residuals are pass-indexed, so pass `p` never
overwrites pass `p-1` state.

**Why structural:** the extra M comes from the loop depth itself. A non-looped MoE batch decoder
can only batch across tokens (M=8 here); the loop gives us a second, orthogonal batching axis that
only exists because pass `p+1` reads pass `p`'s message rather than being an independent model.
Measured gate: top-8 set agreement ≥95% of tokens (else Decision 1 is void), and GEMM M-utilization
≥75% of the padded m16 grid at batch 8.

### Decision 3 — Turing-native dequant-to-FP16 WMMA with an FP32 outlier side-lane (ODSL)

**Premise:** T4 has FP16 tensor cores but no FP4/FP8. llama.cpp-style kernels dequantize in the
K-loop; we do too, but with the block-scale layout chosen so the 16-code block's E4M3 scale is a
single broadcast, and with a candidate d2027 spike removed from the FP16 datapath
**[d2027: unverified local hypothesis; not in primary sources]**.

**Mechanism:** decode E2M1/E4M3 via bit-exact LUTs derived from loopkit's `FP4_LEVELS` and
`_E4M3_LUT`, convert operands to FP16 for `mma.sync.m16n8k16`, and accumulate FP32. K-column 2027 **[unverified local hypothesis; not in primary sources]**
is excluded from the FP16 fragment in this prototype; its dequantized FP32 value is accumulated in a separate
`float spike[N]` per output row and added in the epilogue. Prevents FP16 overflow and keeps the
largest-magnitude dimension numerically exact.

**Why structural (conditional):** *if* the unverified local d2027 hypothesis holds (it is not in
primary sources), the outlier is a named property of *this* model. A generic engine must either
clamp (wrong) or keep the whole GEMM in FP32 (8× slower on T4). One extra FP32 FMA lane costs
~1/4096 of the K work.

---

## 2. Module layout, build, dependencies, and the key data structure

### Tree

```
/tmp/engine/
  CMakeLists.txt
  cmake/InklingConfig.cmake.in
  include/loopengine/
    config.hpp          # InklingConfig: 24/4/6/3/12, 128/top8, block 16/32, outlier 2027 [UNVERIFIED: not in primary sources]
    inkbin.hpp          # on-disk format reader (header + code plane + scale plane)
    codec.hpp           # host mirror of loopkit quantize/dequant (self-tests + converter)
    dequant.cuh         # device LUTs and __device__ inline decode (E2M1/E4M3/E8M0)
    gemm.cuh            # WMMA dequant+GEMM declarations, tile constants
    router.hpp          # top-8, agreement check, route_dirty
    loop_state.hpp      # <<< THE data structure: LoopPassCache + LoopMessage >>>
    scheduler.hpp       # PassInterleavedScheduler, 2-GPU layer partition
    engine.hpp          # InferenceEngine facade, math mode enum
    cert.hpp            # port of bitloop.state_digest/certify_loop/verify_loop
    audit.hpp           # port of CycleAudit (bootstrap CI, claim <= ci_low)
    ref.hpp             # bit-exact FP32 reference path (--math-ref)
  src/
    config.cpp inkbin.cpp codec.cpp router.cpp loop_state.cpp
    scheduler.cpp engine.cpp cert.cpp audit.cpp ref.cpp
    main.cpp            # CLI: --batch N --passes 1..5 --math-ref|--tc --certify --bench
  kernels/
    dequant_gemm.cu     # NVFP4/MXFP8 WMMA kernels + speculative d2027 side lane [unverified local hypothesis; not in primary sources]
    expert_gather.cu    # route-once gather into hot tile arena
    message_mix.cu      # scalar softmax mix / gamma gate [SIMPLIFICATION: upstream rank-r gain/forget controllers] (zero-skip at pass 0)
  tests/
    test_codec_bitwise.cpp test_dequant_bitwise.cpp test_cert.cpp
    test_scheduler.cpp test_gemm_drift.cpp
  tools/
    gen_vectors.py          # dumps loopkit codes/scales/operands as .npy
    compare_to_loopkit.py   # bit-pattern comparison, produces cert JSON
    inkbin_export.py        # offline exporter from native NVFP4 checkpoint -> inkbin
  third_party/            # intentionally empty
```

### Build

CMake ≥3.24, `project(loopengine LANGUAGES CXX CUDA)`, `CXX_STANDARD 17`,
`CUDA_STANDARD 17`, `-DCMAKE_CUDA_ARCHITECTURES=75` (T4), `-O3` in Release.
**Forbid `--use_fast_math`, `-ffast-math`, and FMA contraction in the reference target**
(`-fmad=false` for `ref.cpp`): they change FP32 bit patterns and would invalidate the certificate.
Targets: static `loopengine`, `loopengine_cli`, `loopengine_tests`, `loopkit_compat` (drives the
Python comparison). CI: `ctest --output-on-failure`; all GPU tests tagged `[kaggle]` and skipped
locally, mirroring loopkit's pytest markers.

Dependencies: CUDA Toolkit 12.x (WMMA via `<mma.h>`, `nvcuda::wmma`), C++17 STL only.
Self-contained SHA-256 (~120 LOC, FIPS 180-4) and `std::mt19937_64` bootstrap. No PyTorch, no
cuDNN, no cuBLASLt, no Eigen, no GGUF library, no protobuf, no abseil. The CLI prints JSON by hand.

### On-disk format `inkbin` (v1)

Magic `INKB`, then per-layer records: `{layer_id, tag=NVFP4|MXFP8, N, K, block, code_offset,
code_bytes, scale_offset, scale_bytes}`. Code plane is `[N][K/2]` for NVFP4 (low nibble = even
code) and `[N][K]` for MXFP8. Scale plane is **block-major** `[K/block][N]` (see §3). Exporter
copies vendor codes and E4M3/E8M0 scale bytes directly; it never calls a quantizer.

### The single most important data structure

`include/loopengine/loop_state.hpp` — the loop-pass cache is the whole design: it is where
loop-awareness, routing reuse, pass-indexed KV, and the certificate all meet.

```cpp
struct alignas(64) LoopMessage {          // one (token, pass) cross-loop handoff
  __half  prev[kHidden];                  // vector from pass p-1
  __half  curr[kHidden];                  // vector from pass p
  __half  mixed[kHidden];                 // simplified scalar mix; upstream: rank-r gain/forget controllers
  float   gamma;                          // simplified learned gate, 0.0f at pass 0 -> skip mixing
  uint32_t pass_digest;                   // rolling SHA-256 accumulator for cert.hpp
};

struct alignas(64) LoopPassCache {        // one request, all passes, all middle layers
  LoopMessage msg[kMaxBatch][kMaxPasses];                 // kMaxPasses = 5 (varloop)
  __half residual[kMaxBatch][kMiddleLayers][kMaxPasses][kHidden]; // per-pass attention residuals
  __half kv[kMaxBatch][kMiddleLayers][kMaxPasses][2][kSeq][kHeadDim];
  uint8_t route[kMaxBatch][kMiddleLayers][8];             // route-once top-8 expert ids
  uint8_t route_dirty[kMaxBatch][kMiddleLayers];          // 1 => re-gather on this pass
  float   agreement[kMiddleLayers];                       // measured top-8 set agreement
  ExpertTile tiles[kMiddleLayers][8];                     // pass-resident hot arena (GPU-local)
  uint64_t tile_lru[kMiddleLayers][128];
  size_t   bytes;                                         // instrumented bytes/token counter
};
```

With kHidden=4096, kMaxBatch=8, kMaxPasses=5, kSeq=1 (decode): messages ≈ 8×5×3×4096×2 ≈ 0.98 MB,
residuals ≈ 8×12×5×4096×2 ≈ 3.9 MB, KV ≈ 3.9 MB. Keep `kSeq=1` for decode benchmarks; prefill
uses the same type with a larger `kSeq` and is out of scope for week 1–4.

---

## 3. NVFP4 / MXFP8 dequant path on Turing

**Packed semantics (must match loopkit exactly):**
`FP4_LEVELS = {0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6}` (f32). Codes 0..15, two per
byte, low nibble first. NVFP4 scale = E4M3 code of `max(absmax/6, 2^-9)`, decoded through loopkit's
256-entry `_E4M3_LUT` (NaN code 0x7F/0xFF excluded by the encoder). MXFP8: 1-byte E4M3 code per
weight, block 32, scale byte = `exp+127` with `exp = clip(ceil(log2(absmax/448)), -126, 126)`,
decode scale as `2^(byte-127)` (exact power of two → construct FP16 exponent directly).

**Coalesced scale layout:** codes stay row-major `codes[N][K/2]`; scales are stored **block-major**
`scales[K/block][N]`. A warp handles 4 rows × one 16-code block: four 8-byte code loads at
`codes[n_i*K/2 + b*8]` plus **one** 4-byte contiguous scale load at `scales[b*N + n_base..+3]`.
The alternative row-major scale layout (`scales[n*nb + b]`) forces 4 strided single-byte loads per
warp and is 3–4× slower on T4; do not use it. MXFP8 mirrors this with 32-byte code loads and
`scales[K/32][N]`. Convert the block-major plane once in `inkbin_export.py`, not at runtime.

**FP16 tensor-core path:** per block, `fp4_lut[16]` and `e4m3_lut[256]` live in `__constant__`
memory (FP16 copies of loopkit's FP32 LUTs, with code 8 = `-0.0` → `0x8000`). Per lane:
`float v = __half2float(fp4_lut[code]) * __half2float(e4m3_lut[scale])` — the FP32 multiply is
bit-identical to numpy's float32 `vals * scales` in `dequantize_nvfp4`; only the subsequent cast
to FP16 is an approximation. Stage FP16 into shared memory, then `wmma::load_matrix_sync` →
`wmma::mma_sync` (f16 inputs, f32 accumulator) → epilogue. K-tile = 128 (8 NVFP4 blocks), N-tile
= 64, M-tile = 16; use `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. Last 4 MXFP8 layers
use K-tile = 128 (4 blocks) with the same epilogue.

**Outlier lane (speculative):** K-column 2027 **[unverified local hypothesis; not in primary
sources]** is removed from the FP16 fragment. Its FP32 dequantized value is
multiplied by the corresponding activation in FP32 and accumulated into `float spike[N]` per
output row, added after the WMMA epilogue. This prevents overflow and keeps the dominant
dimension exact. Gate: `spike` output must be bit-identical to the Python reference for that
column in `--math-ref` mode.

**What bit-exactness means here (be precise):**
1. Stored representation: codes, scales, shape, block round-trip through `inkbin` byte-for-byte —
   this is loopkit guarantee 1 (`ans_decode(ans_encode(t)) == t`).
2. Dequantized FP32 operands: our LUT+scale multiply must match `dequantize_nvfp4` / `dequantize_mxfp8`
   **bit-for-bit** (0 of 1,000,000 sampled elements may differ, including signed zero and dim 2027
   **[unverified local hypothesis; not in primary sources]**).
3. GEMM: loopkit's `fdm_dot` is byte-identical only because it calls the *same* `reference_dot`.
   A C++ FP32 GEMM cannot match numpy's internal reduction order in general, so `--math-ref` is
   validated by consuming Python-dumped FP32 operands and comparing with a documented tolerance;
   the FP32 WMMA path is a separate, measured drift budget (relative L2 ≤2e-3 on non-outlier dims,
   dim 2027 exact **[unverified local hypothesis; not in primary sources]**). Never describe
   FP16-TC output as "bit-exact" in reports.

**Certificates:** port `bitloop.py` verbatim: `state_digest` = SHA-256 over sorted tensor keys with
key + shape + dtype + raw bytes (sign bit included); `certify_pass` per pass; root digest =
`sha256(concat(pass digests))`; `verify_loop` recomputes digests. `assert_lossless` must fail on a
single flipped bit in any pass. CycleAudit semantics are preserved: a speed claim is valid only if
`claim <= ci_low` (bootstrap 95% lower bound, ≥20 repeats, ≥5 warmup); never round up.

---

## 4. Loop-aware scheduling

1. **Entry:** batch 8 tokens. Route-once for all 24 layers' MoE decisions at first use; for middle
   layers stash top-8 in `LoopPassCache::route`. Prelude (blocks 0–1, layers 0–11) runs **once**,
   as a batched M=8 GEMM. Compute and store the two message vectors `a0, b0` and `gamma=0`
   (scalar-gamma simplification; upstream uses rank-r gain/forget loop controllers).
2. **Middle loop:** for `p = 0..2` (varloop 1..5), for `l = 0..11`:
   - Skip mixing on `p==0` (`gamma=0` ⇒ `mixed = curr`) in the simplified model; upstream gates
     are rank-r gain/forget controllers.
   - Attention reads KV/residual slot `[token][l][p]` — never slot `p-1`; appends one new K/V per
     pass into its own slot. This is what "attention residuals" implies for a looped block and is
     checkable at seq=1 against the Python reference.
   - MoE: one GEMM per expert, M = count of (token, pass) pairs routed to it across currently-ready
     stages (max 24). Reuse `tiles[l][8]` from pass 0; re-gather only if `route_dirty`.
   - Write residual `[b][l][p]`, compute `a_{p+1}, b_{p+1}` and the simplified learned gate
     `gamma_{p+1}` (upstream: rank-r gain/forget controllers) into
     `msg[b][p+1]`; update `pass_digest`.
3. **GPU partition:** GPU0 owns prelude + middle block 2 (layers 12–17); GPU1 owns middle block 3
   (layers 18–23) + coda. Weights split ≈10 GB each; no NVMe spill. After each pass, transfer only
   `msg + residual + route` for 8 tokens: ≈ 8 × (3×4096×2 + 4096×2) ≈ 0.26 MB/pass, ≈0.8 MB per
   batch ⇒ ≈65 µs at 12 GB/s vs multi-ms GEMM time: negligible. **Do not expert-shard across GPUs**:
   all-to-all of 8 tokens × 4096 × 2 B × 48 layer applications would cost ~12 ms/batch.
4. **Reuse inventory:** across passes — expert tiles (Decision 1), KV base index, routing, code/scale
   plane mappings, decoder LUTs; across tokens in a batch — expert weight reads (Decision 2);
   across requests — nothing (each request owns its `LoopPassCache`).
5. **Coda:** blocks 0–1's tail layers once, then emit logits and the loop certificate root digest.

---

## 5. Four-week milestone plan

All speed numbers are **gates to measure on 2×T4 with CycleAudit** (warmup=5, repeats=50,
claim only if ≤ ci_low), not claims. Each week reports: tok/s @ batch1, tok/s @ batch8, GB/token,
and bit-exactness vs loopkit. Baseline is captured in week 1, not assumed.

| Week | Deliverable | tok/s b1 | tok/s b8 | GB/token (b1 / b8) | Bit-exactness gate |
|---|---|---|---|---|---|
| 1 | `inkbin` reader, host codec mirror, LUTs, `--math-ref` path, `cert.hpp`, `audit.hpp`, `compare_to_loopkit.py`; **no speedup claim** | baseline, measured (expected ~10–20) | baseline, measured (expected ~60–120) | baseline, expected 3.04 / 0.38 (naive 5.4B×4.5b ÷ batch) | 0/1,000,000 mismatched f32 dequant bits; ANS/inkbin roundtrip exact; `--math-ref` operands byte-identical to Python dump; 3-pass cert root matches |
| 2 | `dequant_gemm.cu` WMMA + speculative d2027 side lane (unverified local hypothesis; not in primary sources), block-major scale plane, drift test | ≥25 | ≥160 | ≤2.6 / ≤0.32 | dequant still 0 mismatches; TC relative L2 ≤2e-3; d2027 exact if hypothesis holds; report top-8 agreement % |
| 3 | RORET + PIBS: route-once, pass-resident tiles, pass-interleaved batching, 2-GPU layer split | ≥40 | ≥300 | ≤2.0 / ≤0.25 | agreement ≥95%; cert root matches for `--math-ref` on 3 passes; TC drift unchanged |
| 4 | Hardening, varloop 1–5, Metal scaffolding, CycleAudit report vs llama.cpp Q4_K_M baseline | ≥60 | ≥500 | ≤1.9 / ≤0.23 | 0 mismatches over 100 random prompt dumps; `verify_loop` true for all; **claim ≤ ci_low** and ≥1.3× vs baseline |

Week 1 exit criterion in full: (a) `compare_to_loopkit.py` reports `mismatch_count == 0` on 1e6
random NVFP4/MXFP8 codes including signed zero, block-boundary shapes (K%16≠0, K%32≠0), and
column 2027 **[unverified local hypothesis; not in primary sources]**; (b) `fdm`-equivalent probe on shape (8,256)×(256,256) shows 0.0 error in `--math-ref`;
(c) `certify_loop` detects an injected single-bit flip and `assert_lossless` raises; (d) a
CycleAudit baseline JSON with ≥50 repeats for b1 and b8 is committed. No wall-clock win is claimed.

---

## 6. STOP criteria (honest kill gates)

Stop or pivot immediately if any of these is measured, regardless of effort sunk:

1. **Route premise false:** top-8 set agreement across passes <90% on real prompts → RORET is void;
   do not build a fallback that silently gives up the structural advantage.
2. **Wrong bottleneck:** middleware-block weight re-reads are <20% of HBM traffic in the Nsight
   profile → the loop is not the win; generic kernel tuning (llama.cpp territory) is the real task.
3. **Numerics fail:** FP16-TC drift changes pass-3 top-8 for >1% of tokens, flips output argmax on
   >5% of prompts, or d2027 output (unverified local hypothesis; not in primary sources) differs
   from reference by >0 ulp in `--math-ref` → shut down the
   FP16 path.
4. **Week 2 gate missed** (b1 <25 tok/s or b8 <160 tok/s) *and* >60% of time is in dequant+GEMM →
   our kernel cannot compete with cuBLAS-backed stacks; pivot to calling a vendor library.
5. **Memory premise false:** aggregate resident set >30 GB (2×16 GB) or >10% of batches spill to
   NVMe → sharding design fails.
6. **No defensible win at week 4:** CycleAudit `ci_low` for speedup vs llama.cpp Q4_K_M is <1.0 →
   report "no measured advantage" and stop the project; do not publish a rounded-up claim.
7. **Mobile scope fails:** if the FP32 d2027 side lane (unverified local hypothesis; not in primary
   sources) exceeds 10% of decode time on Apple Silicon,
   drop the mobile milestone rather than port an uncompetitive path.
