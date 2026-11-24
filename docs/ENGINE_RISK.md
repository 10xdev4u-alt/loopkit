# SPEC_RISK.md — Skeptical risk review
## Custom C++ inference engine for Inkling 2 Mini on 2× T4

### 0. Context

Target facts (user brief + /tmp/build/loopkit/configs/inkling2mini.toml):
- Inkling 2 Mini: 24 layers (20 native NVFP4 + 4 MXFP8), 128-expert MoE, 8 active
  experts, attention residuals, learned cross-loop message gate **[upstream: rank-r gain/forget
  controllers; our scalar gamma was a simplification]**, outlier dim 2027 **[unverified local
  hypothesis; not in primary sources]**.
- Middle layers loop: 12 middle layers run 3 passes/token (48 layer applications).
- Scale: 26.4B total / 3.4B active / ~5.4B weight activations per token.
- Artifact: ~20 GB. Hardware: 2× Tesla T4 (sm_75, no FP4/FP8 tensor cores,
  16 GB each → 32 GB total, no NVLink).
- Existing stacks: llama.cpp, vLLM, SGLang, Ollama, Unsloth GGUF.

Goal as stated: a custom C++ engine that beats llama.cpp/vLLM on this hardware.

This is a skeptical review of that goal: what fails, what is not measurable on
T4, the honest benchmark protocol, the one reframe that makes the goal
achievable and verifiable in 4 weeks, and fork vs greenfield.


> **Correction — 2026-11-24 (source fact-check; supersedes premises below):**
> - **outlier dim 2027 is an unverified local hypothesis; not in primary sources** (absent from
>   TML's release post and model card; present only in our config/tests/docs). Do not treat it as
>   a source fact.
> - **Gate: rank-r gain/forget loop controllers in the upstream implementation; our earlier
>   scalar-gamma model was a simplification.** Upstream `src/models/inkling2-loop.cpp` /
>   `src/models/inkling2.cpp` use `LLM_TENSOR_LOOP_*_CTRL_{DOWN,GAIN,FORGET}` at rank 64.
> - **Upstream already provides the looped graph and converter**
>   (`src/models/inkling2-loop.cpp`, `conversion/inkling2.py`); the loop covers layers 6–17 with
>   `n_slot = n_phys + (n_passes-1)*(core_end-core_start)`. Our fork's contribution is
>   union-resident MoE routing on top of upstream, not building the loop graph from scratch.
>   Source-labelled details: `docs/INKLING_INTEGRATION.md`.

---

## 1. Top 10 reasons a custom C++ engine could FAIL to beat llama.cpp/vLLM here

### 1.1 Greenfield decode speed is a parity problem, not a win problem
- **Failure mode:** the engine ends within ~5–15% *slower* than llama.cpp at
  decode on the §3 protocol.
- **Mechanism:** T4 decode of this model is memory-bandwidth-bound (weight
  reads dominate: ~5.4B weight activations/token). llama.cpp's ggml kernels on
  Turing + cuBLAS fp16 GEMMs are already tuned for this hardware class and
  typically pull a large fraction of the HBM2 bandwidth; a greenfield engine
  that re-derives the fast path (vec_dot blocking, fp16 GEMMs) can at best
  tie. Estimates, not facts: gap of 5–15% is the likely outcome.
- **Verdict: cannot mitigate.** On T4 you cannot beat an already-tuned path in
  4 weeks; the goal must reframe to parity (§4).

### 1.2 FP4/FP8 speedups: not real on T4
- **Failure mode:** shipping the native formats as a speed win ("we're faster
  because we store 20 GB").
- **Mechanism:** sm_75 has no FP4/FP8 tensor cores. NVFP4/MXFP8 are storage
  formats — the dot kernels dequant to fp16. llama.cpp already runs this path.
  Additionally, 26.4B × 2 B = 52.8 GB > 32 GB of T4 VRAM: the fp16 materialized
  model does not fit, so there is no faster-fp16 alternative on the table.
- **Verdict: cannot mitigate (hardware boundary). Reframe to storage parity,
  not speed (§2).**

### 1.3 No NVLink: multi-GPU TP is PCIe-bound, and llama.cpp already owns it
- **Failure mode:** losing to llama.cpp on 2-GPU throughput, or wasting weeks on
  a custom TP path.
- **Mechanism:** T4↔T4 PCIe gen3 x16 is tens of GB/s in aggregate; llama.cpp
  split-mode is the tuned tensor-parallel path for this topology. A custom TP
  adds sync/gather complexity with no defensible wall-clock win.
- **Verdict: mitigable** by declaring tie-to-llama.cpp or scoping the engine to
  single-GPU; as a "beat" claim, cannot mitigate.

### 1.4 Greenfield re-implements standard MoE/attention/TP; the loop arrives late
- **Failure mode:** month 4 ships the benchmark without loop semantics verified,
  or with them bolted on unverified.
- **Mechanism:** Inkling 2 Mini's only structural delta over stock llama.cpp is
  the 3-pass loop over core layers 6–17 + cross-loop message gate. **Correction:** upstream
  llama.cpp already implements this architecture (`src/models/inkling2-loop.cpp`,
  `conversion/inkling2.py`); llama.cpp has tuned MoE
  routing, attention, and TP. A greenfield engine spends its 4 weeks on shared
  code, not the new logic.
- **Mitigation:** fork llama.cpp; weeks 1–2 port base + MoE + TP (already
  tuned), weeks 3–4 add loop + gate, verify with loopkit certificates (§5).

### 1.5 Weight traffic is the dominant term; there is no loop-caching win
- **Failure mode:** investing in a loop-caching optimization that gains
  nothing measurable.
- **Mechanism:** middle-layer weights must be re-read per pass (SMEM/register
  caching of MB-scale layers is impossible on T4). The message gate adds a
  fixed per-pass cost. The real variable is bytes/token, which llama.cpp
  already extracts near the GEMM limit.
- **Verdict: cannot mitigate as a beat; mitigate by making the protocol measure
  and report bytes/token (§2 reframe).**

### 1.6 "Fused" staging is bit-exact but not faster on T4 (loopkit FDM-Dot)
- **Failure mode:** shipping "fused" kernels on the strength of bit-exactness,
  while staying slower than llama.cpp.
- **Mechanism:** loopkit's FDM-Dot is bit-identical to the fp16 reference
  because the fused path does not exist on T4 — both are fp16 GEMMs.
  Bit-exactness is a correctness property, not a speed property.
- **Verdict: cannot mitigate as a speed claim — measure, don't assert (§3).**

### 1.7 Storage strategy: the model does not fit in 32 GB at fp16
- **Failure mode:** the engine is slower because it cannot stream the
  compressed artifact the way llama.cpp's mmap + per-layer dequant does.
- **Mechanism:** 26.4B × 2 B = 52.8 GB > 32 GB. The engine's storage plan is
  either the compressed ~20 GB artifact (llama.cpp-style mmap + per-layer
  dequant) or something that doesn't fit.
- **Verdict: mitigable by default** — adopt llama.cpp's storage plan; any speed
  claim must beat that baseline.

### 1.8 Benchmark cherry-picking: llama.cpp wins warm, custom engine wins cold
- **Failure mode:** a "win" that is first-run noise (page cache, clock ramp).
- **Mechanism:** llama.cpp on repeat runs has weights in the page cache and
  boosted clocks; a custom engine's first runs pay mmap faults + clock ramp.
  A protocol measuring the custom engine's first run shows a phantom win.
- **Verdict: fully mitigable** — the §3 protocol (warmup, alternating order,
  5 reps, day-separation, lower-95% bounds).

### 1.9 The real opponent is llama.cpp; vLLM/SGLang are trivially beaten on 2×T4
- **Failure mode:** spending the 4 weeks optimizing to beat vLLM instead of
  llama.cpp.
- **Mechanism:** vLLM/SGLang on 2×T4 without NVLink add Python/NCCL overhead;
  llama.cpp is the reference for decode tok/s here.
- **Verdict: mitigable by reframe** — set the bar to llama.cpp (§4).

### 1.10 The 4-week timeline does not fit greenfield tuning
- **Failure mode:** no window left for tuning to parity once the engine is built.
- **Mechanism:** reaching parity requires several tuning iterations on a shared
  2×T4 Kaggle box with noisy deltas; the fork path moves port work to weeks
  1–2 and tuning to weeks 3–4.
- **Mitigation:** fork first (§5); if parity is not reached by week 3, ship the
  fork engine — the spec reframes success to parity anyway (§4).

---

## 2. Claims in the "beat the rest" goal not measurable on T4 — reframes

| Claim | Measurable on T4? | Reframe |
|---|---|---|
| "Faster than llama.cpp" (end-to-end decode tok/s) | Measurable, but llama.cpp is near the bandwidth ceiling: the honest question is *parity*. | **Verified parity:** median_engine ≥ median_llama AND lower95(median_engine) ≥ median_llama (§3). |
| "FP4/FP8 speedups" | Not real: no FP4/FP8 TC on sm_75; formats dequant to fp16 in-kernel; llama.cpp runs the same path. | **Storage parity:** artifact ≈ 20 GB vs 52.8 GB fp16; both engines must store/dequant identically; the claim is about artifact size, not speed. |
| "End-to-end tok/s without weights" / weight-free steady state | Not real on T4: every decode step reads the weights (memory-bound); llama.cpp gets free weight reuse via page cache. | **Bytes of weight reads per generated token (GB/token)** — the mechanism metric; report it, don't count tokens. |
| Fused tensor-core speedups | Not real: no FP4/FP8 TC; all "fused" paths are fp16 GEMMs; both engines compete on fp16 GEMM throughput. | **Engine-specific fp16 GEMM performance**, measured per §3. |
| Multi-GPU tok/s scaling at TP=2 | Measurable but bounded by PCIe; llama.cpp already extracts a large fraction. | **Tie-to-llama.cpp** is the honest TP=2 result; single-GPU parity is the bar. |
| vLLM/SGLang comparison | Measurable, but vLLM on 2×T4/new-arch loses to llama.cpp; not the hard opponent. | **llama.cpp is the opponent**; vLLM is a footnote. |
| "Native NVFP4 speed" | Not on T4 (native speed only pays off on Blackwell, sm_100+). | Nothing claimed about native speed on T4; flag as follow-on on newer hosts. |

---

## 3. Minimum honest benchmark protocol (2× T4, llama.cpp comparison)

### Environment (both engines, both runs)
- Same box: 2× Tesla T4, 16 GB each; record driver version; verify `nvidia-smi`
  shows 0 used VRAM before each run; check GPU state is clean between engines.
- Clock lock for fairness: `nvidia-smi -lgc <max>` on both GPUs for both
  engines (record the result; skip if unavailable).
- **Precision: fp16** — the only mode with real speed on T4. llama.cpp run
  against an F16 artifact; the custom engine runs on the same artifact
  (same effective precision, so the comparison is honest).
- Weights: one artifact, e.g. `Inkling2Mini.F16.gguf` — deterministically
  dequantized from the native NVFP4/MXFP8 weights; record sha256; both engines
  use it.
- Sampling: greedy (temperature 0, no penalties); identical prompt token
  sequences for both engines.

### Metrics (exact)
1. **Decode tok/s (batch 1):** prompt P0 = 300 tokens. 50-token warmup
   discarded. N=300 decode steps timed with CUDA events around the step loop.
   tok/s = 300 / (t_end − t_start). Report median, min, p10/p90 over the 5 reps.
2. **Decode tok/s (batch ∈ {2,4})** — same, with the engine and llama.cpp at
   each batch size; report medians.
3. **Prefill/TTFT:** prompt P1 = 2048 tokens. TTFT = time to first token.
   prefill tok/s = len(P1)/TTFT.
4. **Mechanism metric:** bytes of weight reads per generated token — the engine
   instruments its GEMM loads (sum of weight bytes loaded per token); for
   llama.cpp, artifact bytes × tokens. Replaces the unverified
   "5.4B weight activations/token" constant: the protocol measures it.
5. **Peak VRAM:** engine: `cudaMemGetInfo` delta sampled during decode;
   llama.cpp: peak RSS via `/usr/bin/time -v`.

### Warmup & repetition (anti-cherry-picking)
- Per engine per prompt: 50-token warmup (discarded) + 1 extra warmup run.
- **5 repetitions; alternate order every rep** (R1: llama→engine,
  R2: engine→llama, R3: llama→engine, ...).
- **Day-separation:** repeat the whole A/B on a second session/day on the same
  box; a win needs both sessions favorable.
- llama.cpp run with defaults and documented variants (e.g., `--flash-attn on`,
  `-b 1`, `--no-mmap`); report llama.cpp's **best** median — the engine must
  beat the best, not the default.

### Win definition (CycleAudit style)
- **Win:** engine_median ≥ llama_median AND lower95(engine_median) ≥ llama_median.
- **Otherwise:** parity or trend — not a win. Report the per-rep table:
  rep | engine median | llama median | engine lower95 | outcome.

---

## 4. Single most important reframe of the goal (achievable + verifiable in 4 weeks)

**Reframe: "The goal is to ship an engine that is protocol-verified at
parity-or-better vs llama.cpp at batch 1–4 on 2×T4, with Inkling 2 Mini's loop
semantics (3 passes/token, cross-loop message gate **[rank-r gain/forget upstream]**, outlier dim
2027 **[unverified local hypothesis; not in primary sources]**) verified
bit-exact by loopkit certificates — not to 'beat llama.cpp' (a moving
target)."**

Why this is achievable and verifiable in 4 weeks:
- **Parity is the right bar:** llama.cpp is near the T4 bandwidth ceiling on
  decode, so parity is the honest 4-week deliverable; the *mechanism* claims
  (GB/token, activation reductions) are what make the engine distinct and are
  measured absolutely.
- **Verification:** the §3 protocol, run twice (two sessions on the box);
  parity at both + loop certificates = done.
- **4-week plan:** weeks 1–2 fork llama.cpp + port base + add loop (§5); week 3
  run §3 on both engines, find the real gap; week 4 close the gap or document
  parity — the deliverable is the same either way.

---

## 5. Fork vs greenfield: **fork llama.cpp** — one-paragraph argument

Fork llama.cpp because it already owns the T4 fast path the goal measures:
memory-bandwidth-bound decode via ggml's Turing-tuned kernels + cuBLAS fp16
GEMMs, MoE routing, TP split-mode over PCIe, and a tuned multi-contributor
base — so the fork gives an instant A/B (your loop engine vs the already-fast
base) and a parity baseline from week 2. Greenfield spends its 4 weeks
re-deriving MoE/attention/TP — most of what makes llama.cpp fast on T4 — and
arrives at parity only by re-copying it, while the genuinely new work (3-pass
core-layer 6–17 loop, cross-loop message gate **[rank-r gain/forget upstream]**, outlier dim 2027
**[unverified local hypothesis; not in primary sources]**, loopkit
semantics) is implemented late with no time to verify. Greenfield earns its
place only as a later, post-parity follow-on if a structural win (e.g.,
loop-aware weight reuse across the 3 passes) is proven. Additionally, fp16
materialization of 26.4B weights is 52.8 GB — larger than the 2×16 GB T4
address space — so storage strategy is already part of the speed story:
llama.cpp's mmap + compressed artifact + per-layer dequant is the fastest T4
storage plan on the table; a greenfield engine reinventing storage slows
itself down.

---

## Top 3 failure modes (blunt) + fork verdict

1. **Greenfield decode can't beat llama.cpp on T4** — llama.cpp is already at
   the T4 bandwidth ceiling for memory-bound decode (Turing-tuned ggml kernels
   + cuBLAS fp16 GEMMs); a greenfield engine can at best tie. The goal itself
   must reframe to parity (§4).
2. **FP4/FP8 speedups are unmeasurable on T4** — no FP4/FP8 tensor cores on
   sm_75; formats dequant to fp16; the ~20 GB compressed artifact is about
   storage, not speed; the fp16 model (52.8 GB) doesn't even fit 2×16 GB of
   VRAM. Claim storage parity, not speed (§2).
3. **No NVLink: TP is PCIe-bound and llama.cpp already owns it** — custom TP
   adds risk without a defensible win; scope the engine to single-GPU and
   declare tie-to-llama.cpp, or accept TP=2 tie (§1.3).

**Verdict: fork llama.cpp.**
