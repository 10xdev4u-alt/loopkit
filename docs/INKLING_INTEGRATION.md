# Inkling 2 Mini → llama.cpp fork: integration spec

Status: research only, no code changed. Date: 2026-11-24.

**Label convention used throughout**
- `[F]` = fact stated in a primary source (TML announcement/model cards, HF repo, upstream llama.cpp).
- `[I]` = inference by us from primary-source data; not stated by the source.
- `[L]` = claim from this repo only (`README.md`, `configs/inkling2mini.toml`, `docs/ENGINE_SPEC.md`); **not corroborated** by any primary source.
- `[?]` = unknown / must be verified before relying on it.

Sources: announcement `https://thinkingmachines.ai/news/inkling-2-mini/`; model card
`https://thinkingmachines.ai/model-card/inkling-2-mini/`; HF `https://huggingface.co/thinkingmachines/Inkling-2-Mini`
(model card, `config.json`, `model.safetensors.index.json`, safetensors header); varloop card
`https://thinkingmachines.ai/model-card/tml-d24-varloop-research/`; upstream llama.cpp `master`
(`gguf-py/gguf/constants.py`, `conversion/inkling2.py`, `src/models/inkling2-loop.cpp`). All reachable.

## 1. Architecture facts

**Skeleton.** `[F]` 24-layer decoder-only MoE transformer; 4 blocks × 6 layers. First and last blocks run
once, two middle blocks run three times with shared weights → 48 layer applications/token from 24 stored
layers. `[F]` Config: `core_start=6, core_end=18` (0-based → layers 6–17), `num_passes=3`, `gate_rank=64`,
`is_varloop=false`; blog 1-based "middle layers 7–18" matches. `[I]` The blog's "prelude = first block" and
our local `docs/ENGINE_SPEC.md` "prelude = layers 0–11" disagree with `core_start=6`; the config and the
controller tensor placement (below) both say layers 6–17 loop, so the local spec is wrong on this point.

**Scale.** `[F]` 26.4B total / 3.4B active; each extra pass adds ≈1.0B effective active params → ≈5.4B
weight activations/token at three passes. `[F]` Checkpoint ≈20 GB, ~19.5 GB safetensors index total size.

**MoE.** `[F]` 128 experts/layer, top-8 routed per token, `n_shared_experts=0`. Started at 384 experts,
top-8, sparsity 48; two-thirds pruned once midtraining (kept 128 most-used/layer), then healed on 1T
distilled tokens. `[F]` Config: `intermediate_size=1152`, `route_scale=8.0`, `use_gate_bias=true`,
`mlp_gate_softcap=4.0`, `mlp_up_softcap=25.0`. HF weights: `mlp.experts.w13_weight/w2_weight` plus
`_amax`, `_scale`, `_outer_scale_denominator`; `mlp.gate.weight` [128,2304], `mlp.gate.bias` [128].

**Precision / per-layer layout (exact HF header facts).** `[F]` `quantization_config`:
`quant_method=modelopt`, `quant_algo=NVFP4`. `[F]` Expert tensors in layers 0–19 are packed `U8`:
`w13_weight` [128,2304,1152], `w2_weight` [128,2304,576], `w13_weight_scale` [128,2304,144],
`w2_weight_scale` [128,2304,72], plus F32 `_amax`[128] and `_outer_scale_denominator`[1].
`[F]` Layers 20–23 use `F8_E4M3`: `w13_weight` [128,2304,2304], `w2_weight` [128,2304,1152],
scales [128,2304,144] / [128,2304,72]. `[F]` All non-expert LLM matrices are `BF16`, residual/query
banks and router biases are `F32`. `[F]` Blog: "20 layers NVFP4, 4 in MXFP8", trained and deployed at
the same precision, so no conversion loss. `[I]` Scale shape 144 = 2304/16 even for MXFP8, which sits
oddly with our local `mxfp8_block=32`; do not assume a block size without checking converter dequant.
`[F]` HF metadata lists tensor types F32/BF16/F8_E4M3/U8.

**Attention.** `[F]` Hybrid local/global; 24 query heads × 8 KV heads × head_dim 128; sliding window
1024; `d_rel=16`, `rel_extent=1024`; short conv `sconv_kernel_size=4`; low-rank projections
`wq_du`[3072,2304], `wk_dv`/`wv_dv`[1024,2304], `wr_du`[384,2304], `wo_ud`[2304,3072],
`gate_attn`[3072,2304], `attn_sink_proj`[24,2304], `rel_logits_proj.proj`[16,1024], L2 clip scales.
`[I]` `local_layer_ids` omits layers 5, 11, 17, 23 → those are the global layers (they uniquely carry
`q/k/v_input_norm` and `k/v_sconv.static_w` + `weight_proj` instead of `k/v_sconv.weight`).

**Attention residuals.** `[F]` "Instead of adding every layer's output into one running sum, each layer
reads the outputs of earlier blocks and learns its own mix"; config `n_layers_per_attn_res_block=6`,
`attn_res_normalize_values=true`. `[F]` HF banks:
`attn_res.attn_queries.0..22` (mostly [2304]; indices **4, 10, 16, 22** are [3,2304]),
`attn_res.mlp_queries.0..23` [2304], `attn_res.output_query` [2304]. Upstream converter shifts
`attn_res.attn_queries.i` → `layers.i+1.attention_input.query`. `[I]` The [3,2304] banks encode the
three-pass dimension at block tails; exact mixing math is not documented in prose.

**Cross-loop message + gate.** `[F]` Every pass starts from prelude output; passes 2 and 3 also read the
previous pass's middle-block outputs (the "message"). A learned gate merges the message with the prelude
output and with previous in-block layers' outputs; it initially blocks the message entirely (loop start =
non-looping model) and grows its gain during training. `[F]` HF cross-pass banks:
`middle_cycle_cross_pass_queries` and `middle_cycle_cross_pass_kv_queries`, both [24,2304] (24 = 2×12
core layers). Upstream converter stacks, per core layer `i`, four rows
`[q[2i], kv[2i], kv[2i+1], q[2i+1]]` into one `LOOP_MSG_QUERY` [4,2304]. `[F]` Learned controllers exist
**only on layers 6–17**: for each of `attn`, `k`, `v`, `mlp`, `down_weight`[64,4608] and
`gain_weight`/`forget_weight`[2304,64] — i.e. a rank-64 per-channel gate, not a scalar. Upstream forces
all loop/controller tensors to F32 on export. `[F]` Upstream names these
`LLM_TENSOR_LOOP_Q_CTRL_DOWN / _GAIN / _FORGET` (and `_K_CTRL_*`, `_V_CTRL_*`, `_FFN_CTRL_*`) in
`gguf-py/gguf/constants.py` / `src/models/inkling2-loop.cpp`; the graph applies
`(1-forget)*cur + gain*message` (`src/models/inkling2.cpp`). `[I]` Our local `gamma` softmax scalar
in `docs/ENGINE_SPEC.md` is a simplification; the real gate is rank-64 with gain+forget per stream.

**Context.** `[F]` `model_max_length = 1048576` (1M tokens); sliding window 1024. Natively multimodal
(ViT: 23 layers, hidden 1024, 16 heads, patch 20); `mtp_config` declares 3 MTP layers and
`target_layer_ids=[5,23,41,48]` but no MTP tensors are in the checkpoint (converter filters `mtp.*`,
`visual.*`). `[I]` 48 = number of layer applications, suggesting MTP reads pass boundaries.

**Known outlier.** `[L]` "activation outlier at dimension 2027" appears only in our `README.md`,
`configs/inkling2mini.toml` (`outlier_dim=2027`), `docs/ENGINE_SPEC.md` and tests. **none of the TML/HF
primary sources mention it.** Treat it as an **unverified local hypothesis; not in primary
sources** — a local premise only (risk #4).

**Varloop variant.** `[F]` `TML-d24-varloop-research`: same weights shape, inference-time 1–5 passes,
trained with pass ratio 1:2:6:2:1, RL at 3 passes; effective 3.4B @1, 5.4B @3, 7.4B @5. `[F]`
No-CoT horizon 16 s @1 → 37 s @3; roughly flat past 3; 10 passes did not materially degrade despite
never training past 5; serial N-hop lookup kept improving to 10. `[F]` Research-only, not for production.

## 2. What llama.cpp needs

**Rebase first — do not invent a new arch.** `[F]` Upstream `master` already has
`MODEL_ARCH.INKLING2_LOOP = "inkling2_loop"`, tensor map, `src/models/inkling2-loop.cpp`,
`src/models/inkling2.cpp`, `conversion/inkling2.py`, `common/parsers/inkling.cpp` and
`tools/mtmd/models/inkling2v.cpp`. `[I]` The correct first move for our fork is to rebase/cherry-pick
these; only add an enum if the fork is older, and then use the upstream name/ID to stay GGUF-compatible.
`[F]` Upstream allocates pass-indexed cache slots as
`n_slot = n_phys + (n_passes-1)*(core_end-core_start)` (`src/models/inkling2-loop.cpp`: 24 + 2×12 =
48 slots), sharing physical weights while each slot keeps its own KV/conv state. `[I]` Our fork's
contribution is union-resident MoE routing **on top of** this upstream loop graph/converter, not
building the loop graph from scratch.

**GGUF metadata** `[F]`: `inkling2_loop.loop.start = 6`, `.loop.end = 18`, `.loop.count = 3`,
`.loop.gate_rank = 64`, `.loop.varying = false`; also `inkling2.num_loops`,
`inkling2.skip_loop_final_norm`, and attention-residual block size 6.

**GGUF tensor names** `[F]` (upstream `MODEL_TENSOR_NAMES`; `blk.{bid}` is the 0-based block id):
`blk.N.loop_msg_query` (stacked [4,2304] above), `blk.N.loop_q_ctrl_down / _gain / _forget`,
`loop_k_ctrl_*`, `loop_v_ctrl_*`, `loop_ffn_ctrl_*` (controllers on core blocks only),
`blk.N.attention_input.query`, `blk.N.attn_res_query`, `blk.N.ffn_res_query`,
`blk.N.output_res_query`, `output_res_query`, plus standard
`token_embd`, `token_embd_norm`, `output`, `output_norm`, `attn_norm/q/k/v/out/gate/rel/...`,
`ffn_norm/gate_inp/gate_exp/up_exp/gate_up_exp/down_exp`, `ffn_post_conv`, `attn_post_conv`.
`[F]` Exact per-tensor shapes resolved by M37 from a bounded range probe of
`unsloth/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-IQ1_S.gguf` bytes 0-16777215 (sha256
87dfcdffd5f422312d54bbe180317dbab93d88a37a8bc0141125ddc8e731ba76; GGUF v3, 744 tensors,
63 KVs). 204 loop/residual tensors, all F32: 12 `loop_msg_query` [2304,4]; 144 controllers
(down [4608,64], gain/forget [64,2304] on core layers 6..17); 23 `attn_res_query`
([2304], global tails 5/11/17/23 are [2304,3]); 24 `ffn_res_query` [2304];
1 `output_res_query` [2304]. Frozen contract:
`inkengine-llama/tools/audit/GGUF_LOOP_SHAPES.json`. Measurement/contract scope only:
no weights, no engine run, no parity, no timing.

**Graph builder.** `[I]` Required replay, consistent with the facts above:
1. Build blocks 0–5 once (prelude) and cache prelude output; seed message state.
2. For pass `p = 0..2`: run layers 6–17 (blocks 2 and 3 of the 0-based 4-block split) **once per pass**
   with the *same* weights. Fetch `loop_msg_query` rows and `*_ctrl_*` gates; form the rank-64
   gain/forget mix of prelude output, previous-pass message, and previous in-block outputs. Skip mixing
   where the gate is zero (pass 0).
3. Pass-indexed KV and residual state: each pass reads/writes its own slot `[layer][pass]`, never
   `pass-1`; cross-pass KV attention uses the `kv` query rows. Budget KV/residual memory ×3.
4. Run layers 18–23 once (coda), emit logits. For varloop, read `.loop.varying` and `.loop.count`
   instead of hardcoding three.

**Cache-hook plug-in.** `[F]` The wiring is implemented opt-in/default-off in
`inkengine-llama/docs/INKLING_UPSTREAM_WIRING.md` and source-contract-gated by M36
(`tools/audit/UNION_WIRING.json`); no runtime parity or timing has been measured.
This research note was written before the fork added
`src/llama-roret.h` (union-resident MoE bookkeeping, default-off); the earlier RORET design also
lives in `docs/ENGINE_SPEC.md` as `include/loopengine/loop_state.hpp` with
`LoopPassCache`/`LoopMessage`. `[I]` Integration shape: allocate one `LoopPassCache` per request inside llama.cpp's per-sequence state
and have the graph builder pass a pointer into the kernel path, so the union-resident fields
(`msg[batch][pass]`, `residual[layer][pass]`, `kv[layer][pass][2]`, `route`, `route_dirty`, `tiles`,
`bytes`) are reused across the three passes instead of re-created. Route-once expert tiles and the
pass-indexed KV indices must be derived from the *upstream* loop pass indices, not a parallel counter.
Keep the certificate/digest hooks at pass boundaries. `[L]` The d2027 FP32 side-lane should stay
**disabled behind a flag** until risk #4 is resolved.

## 3. Conversion path

- `[F]` License is Apache-2.0 on both TML model cards and the HF repo; redistribution/conversion is
  permitted. The separate TML Acceptable Use Policy is a use restriction, not a license term.
- `[F]` Primary sources state quantized GGUF builds are available via Unsloth and llama.cpp. `[I]` This
  implies a working HF→GGUF path exists rather than needing to be written; confirmed by upstream
  `conversion/inkling2.py` (registered for `Inkling2ForConditionalGeneration` /
  `Inkling2LoopForConditionalGeneration`, `model_type` `inkling2*`), which dequantizes ModelOpt NVFP4
  expert chunks to FP32 in GGUF and filters vision/MTP tensors.
- `[F]` Converter normalizes `model.llm.` → `model.`, maps `mlp.gate.bias` → `e_score_correction_bias`,
  shifts `attn_res.attn_queries.i` to block `i+1`, validates loop config keys exactly, and writes loop
  metadata. `[I]` Do not reimplement these mappings in the fork; call upstream conversion.
- **Verify, don't assume:** (a) which quant an Unsloth GGUF actually uses (native NVFP4/MXFP8 vs
  Q4_K_M-style re-quant) and its accuracy delta; (b) whether the converter's FP32 expansion of experts
  preserves the "no conversion loss" claim bit-for-bit on real weights; (c) scale-block semantics for
  MXFP8 layers; (d) RESOLVED by M44 (synthetic CPU stdout-byte compatibility only, evidence below);
  (e) RESOLVED by M47 (frozen tokenizer/chat-template contract only, evidence below);
  (f) whether our fork's base is new
  enough to accept upstream tensor names unchanged.
- `[F]` Section-3 item (d) is resolved by M44 on a synthetic CPU fixture only: 26/26 checks
  PASS, 0 differing bytes at 918 bytes/side. Contract
  `inkengine-llama/tools/audit/UPSTREAM_CLI_COMPAT.json` (sha256
  5fb7e01e63c526844231daa6905ea7e78c9f5f37d85fbc595a0286bbae6b2ae6); summary
  `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json` (sha256
  38ab9c92f7c378e7f4ed23d834b2bd60367207b754e80f37e6f359c8a5ce8122).
  Compile/stdout-byte scope only: no real weights, no GPU, no tensor or logits parity, no timing,
  NO SPEED CLAIM. At M46 the still-open items were (b) real-weight conversion parity and
  (e) tokenizer/chat-template; (e) is resolved by M47 below. (a), (c) and (f) are unchanged.
- `[F]` Section-3 item (e) is resolved by M47 as a frozen tokenizer/chat-template
  contract gate: 26/26 checks PASS, blocker empty. Contract
  `inkengine-llama/tools/audit/TOKENIZER_CHAT_TEMPLATE.json` (sha256
  998c58d244e66d5b2dd8b09fcd33ebee9e187b38ea95adf66779463186b2d9e7); summary
  `M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json` (sha256
  3fdc5c088c4a49008bee34f7cd386b021dd89aa3e01bbdd2d35d6f06a6dc6f20).
  A bounded out-of-band GGUF probe measured byte equality of `tokenizer.chat_template`
  (6868 B, sha256 53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc)
  with HF `chat_template.jinja`; the offline checker verifies frozen-digest/length
  consistency only and cannot observe live GGUF drift. No real weights, no tokenization
  golden test, no engine run, no timing, NO SPEED CLAIM. Item (b) real-weight
  conversion parity remains open; (a), (c), (d) and (f) are unchanged.

## 4. Six highest-risk unknowns — cheap verification experiments

1. **Fork/upstream gap & enum identity** `[F]` (M29,
   `tools/audit/UPSTREAM_LOOP_CONTRACT.json` and `M29_UPSTREAM_CONTRACT_SUMMARY.json`:
   arch string, 47 tensor names, five loop KV keys and slot formula match upstream). Experiment: on a spare checkout of upstream master,
   build `gguf` py tooling and print `MODEL_TENSOR_NAMES[MODEL_ARCH.INKLING2_LOOP]`; diff against our
   fork's constants. No weights needed.
2. **Gate semantics: rank-64 gain/forget vs our scalar `gamma`** `[F]` (M30,
   `tools/audit/GATE_SEMANTICS.json` and `M30_GATE_SEMANTICS_SUMMARY.json`: down
   [64,4608], gain/forget [2304,64] F32, `(1-forget)*cur + gain*message`). Experiment: read the safetensors
   header metadata + converter `tensor_force_quant` (done: shapes [64,4608]/[2304,64], F32) and write a
   synthetic rank-64 mixing unit test with random tensors; no model weights needed. Then, if weights
   ever arrive, hook one core layer and compare gate outputs.
3. **Loop boundaries: layers 6–17 vs local "prelude 0–11"** `[F]` (M31,
   `tools/audit/LOOP_BOUNDARIES.json` and `M31_LOOP_BOUNDARIES_SUMMARY.json`: core
   layers 6..17, prelude 0..5, coda 18..23, n_slot=48). Experiment: grep HF header key list
   for `layers.N.controllers` (gives exactly 6–17) and read converter validation `core_start=6,
   core_end=18` — metadata-only, no weights. Correct `docs/ENGINE_SPEC.md` accordingly.
4. **"Outlier at dim 2027" has no primary-source support** `[L]`. Experiment: audit where 2027 is
   introduced (config/tests only); check hidden_size=2304 makes it plausible but not evidenced. Cheap
   falsification: run a public HF/SGLang trace with a forward hook on a short prompt and record the
   max-|activation| dimension; requires weights or hosted inference. Until then, flag side-lane as
   speculative.
   `[F]` (M35, cheap part only): classified LOCAL_ONLY_UNVERIFIED from primary-source
   absence; the speculative FP32 side-lane stays default-off. Real-trace
   falsification remains NO_GO_BLOCKED without weights or hosted inference
   (`tools/audit/OUTLIER_PREMISE.json`, `M35_OUTLIER_PREMISE_SUMMARY.json`).
5. **Pass-indexed KV: separate slots ×3 vs shared/cross-pass only** `[F]` (M32,
   `tools/audit/PASS_INDEXED_KV.json` and `M32_PASS_INDEXED_KV_SUMMARY.json`: 36
   distinct pass/core-layer slots, RORET key is (ip, pass), not the cache slot). Experiment: read
   `src/models/inkling2-loop.cpp` for KV slot indexing and `loop_msg_query` kv-row use; optionally
   build the graph with a tiny synthetic GGUF and dump the compute graph. No full weights needed.
6. **NVFP4/MXFP8 dequant + scale layout (16 vs 32), and Unsloth quant choice** `[F]`
   (M33/M34, `tools/audit/QUANT_LAYOUT.json`, `MXFP8_SCALE_HEADER.json` and their
   summaries: converter split-16 NVFP4 / flat-32 MXFP8 confirmed against the real
   HF header; Unsloth GGUF header shows IQ1_S/Q2_0 re-quant, not native NVFP4/MXFP8). Experiment:
   compare HF scale shapes (144 = 2304/16 for *both* precisions) against converter
   `_dequant_expert_chunk`; read the GGUF header of one small Unsloth quant by HTTP range request and
   print `general.architecture`, quantization, tensor names/shapes. Partial file only; no 20 GB download.

## 5. Facts vs inference (summary)

- **Facts from source:** all numbers in §1 except the outlier and the prelude-boundary interpretation;
  exact HF tensor names/shapes/dtypes; upstream enum, metadata keys, GGUF tensor names and converter
  behavior; the `n_slot` cache-slot formula and rank-r `LOOP_*_CTRL_*` gain/forget gate implementation;
  licenses and GGUF availability claims. Upstream owns the loop graph and converter; our fork's
  delta is union-resident MoE routing on top of upstream.
- **Inference by us:** global-layer identification from `local_layer_ids`; [3,2304] banks as pass state;
  rank-64 gate reading; MTP target ids as pass boundaries; graph-builder replay recipe; cache-hook
  integration shape; "rebase first" recommendation.
- **Local-only, unverified:** activation outlier dim 2027, scalar-softmax gate description, prelude
  layers 0–11, FP32 side-lane, and all performance/latency premises in `docs/ENGINE_SPEC.md`.
- **M38 closure note (2026-11-28):** section-4 items 1, 2, 3, 5 and 6 and both
  section-2 markers are verified within the stated contract/metadata scopes by
  M29-M37; item 4 stays LOCAL_ONLY_UNVERIFIED with real-trace falsification
  NO_GO_BLOCKED. No weights, engine run, graph execution, parity or timing.
  NO SPEED CLAIM; CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory.
- **M46 closure note (2026-11-28):** section-3 item (d) is PASS-scoped by M44: 26/26 checks,
  0 differing bytes at 918 bytes/side, synthetic CPU compile/stdout-byte scope only. Contract
  `tools/audit/UPSTREAM_CLI_COMPAT.json` and summary
  `M44_UPSTREAM_CLI_COMPAT_SUMMARY.json`. At M46 items (b) and (e) were still open; M47
  resolves (e) as a frozen contract and item (b) real-weight conversion parity remains
  open; no weights, no GPU, no tensor/logits parity, no timing, NO SPEED CLAIM.
  CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory.
- **M48 closure note (2026-11-28):** section-3 item (e) is PASS-scoped by M47 as a
  frozen tokenizer/chat-template contract gate: 26/26 checks, out-of-band byte equality
  of the 6868-byte chat template only. Contract
  `tools/audit/TOKENIZER_CHAT_TEMPLATE.json` and summary
  `M47_TOKENIZER_CHAT_TEMPLATE_SUMMARY.json`. Section-3 item (b) real-weight conversion
  parity remains NO_GO_BLOCKED; no weights, no engine run, no tokenization golden test,
  no timing, NO SPEED CLAIM. CycleAudit claim<=ci_low repeats>=20 loops=3 remains mandatory.
