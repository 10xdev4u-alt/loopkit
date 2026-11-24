# Inkling 2 Mini — format facts from primary sources

Scope: what the vendor and the published checkpoint actually say about numerics, block sizes, layer boundaries and config values. Two sections only: **PRIMARY SOURCE FACTS** and **WHAT WE MUST FIX OR VERIFY LOCALLY**. Anything not stated in a primary source is marked **[UNKNOWN]**; anything derived is marked **[DERIVED]**.

Sources consulted (all fetched 2026-11-24):

- S1 — Model card: https://thinkingmachines.ai/model-card/inkling-2-mini/
- S2 — Release post: https://thinkingmachines.ai/news/inkling-2-mini/
- S3 — HF Inkling-2-Mini model page: https://huggingface.co/thinkingmachines/Inkling-2-Mini
- S4 — HF `config.json`: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
- S5 — HF `hf_quant_config.json`: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/hf_quant_config.json
- S6 — HF `model.safetensors.index.json`: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/model.safetensors.index.json
- S7 — Varloop model card: https://huggingface.co/thinkingmachines/TML-d24-varloop-research
- S8 — Varloop `config.json`: https://huggingface.co/thinkingmachines/TML-d24-varloop-research/raw/main/config.json
- S9 — End-to-End 4-Bit Training (recipe/format definitions): https://thinkingmachines.ai/blog/end-to-end-4-bit-training/

---

## PRIMARY SOURCE FACTS

### 1. Quantization formats and block sizes

**Formats named by the release.** S1: "**Numerics support** — NVFP4 and MXFP8"; "The model is distributed as a single checkpoint of about 20GB, in mixed NVFP4 and MXFP8 precisions." S3 (HF tags) lists `other: 8-bit`, `other: modelopt`; tensor types on the HF page are `F32`, `BF16`, `F8_E4M3`, `U8`.

**Block sizes, from the format definition (S9):**

> "NVFP4 scales each block of 16 consecutive values based on the largest one in the block (in absolute terms)."

> "Blackwell GPUs also offer microscaling (MX) formats such as MXFP4 and MXFP8, which scale blocks of 32 values."

> Table: `NVFP4 | (4 × 16 + 8) ÷ 16 = 4.5` bits; `MXFP8 | (8 × 32 + 8) ÷ 32 = 8.25` bits; `BF16 | 16`.

> "NVFP4 scale factors are E4M3, making them more precise but limited in their range, while MXFP4/MXFP8 scale factors are E8M0…"

So the *formats* are NVFP4 = 16-element blocks and MXFP8 = 32-element blocks. It is **not** "both 16". Caveat: S9 describes the mainline 4-bit recipe and uses Inkling 2 Small as its worked example; S2 says of Mini "The rest of training, including quantization and the data mix, followed our mainline recipe."

**Definitive per-tensor block shapes, from the published checkpoint (S5):** these are not inferred from prose — they are values in the shipped `hf_quant_config.json`.

- NVFP4 group (experts of layers 0–19, 0-indexed):
  - `"compute_dtype": "nvfp4"`, `"output_dtype": "uint8"`
  - `"required_activation_quantization": { "amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1", "rounding": "nearest", "scale_dtype": "float8_e4m3fn", "scale_selection_rule": "fouroversix_mae" }`
  - `"weight_block_shape": [1, 16]` — **row-wise 1×16**, not 16×16 and not a flat 1-D 16 without row semantics.
  - `"weight_scale_format": "amax_denominator"`, `"weight_transform": "nvfp4_nearest"`
- MXFP8 group (experts of layers 20–23, 0-indexed):
  - `"compute_dtype": "mxfp8"`, `"output_dtype": "float8_e4m3fn"`
  - `"required_activation_quantization": { "block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0" }`
  - `"weight_block_shape": [32, 32]` — **2-D 32×32 tile**, i.e. one scale per 1024 weight elements in each expert matrix. This is the single most important nuance: activation blocks are 32 elements, but *weight* blocks are 32×32.
  - `"weight_transform": "mxfp8_nearest"`

S9 corroborates the row-wise NVFP4 choice: "we found computing scale factors over 16×16 tiles to be too coarse, hurting CE beyond what we were willing to accept. For NVFP4 training to work, we needed to quantize weights with 1×16 scale factors while avoiding the mismatch introduced during a block-scaled transpose."

**What the tensor names imply (S6).** For layers 0–19 the expert tensors include `w13_weight`, `w13_weight_amax`, `w13_weight_outer_scale_denominator`, `w13_weight_scale` (and the same four for `w2_weight`). For layers 20–23 only `w13_weight`, `w13_weight_scale`, `w2_weight`, `w2_weight_scale` appear — no `amax` and no `outer_scale_denominator`. This is exactly the S5 split: NVFP4 uses FP8 E4M3 block scales plus per-expert FP32 amax/outer-scale denominator ("amax_denominator"); MXFP8 uses E8M0 scales only. S9: "we use per-expert outer scales when quantizing weights and per-token outer scales when quantizing activations."

**[DERIVED] scale tensor shapes.** The published JSON does not list per-tensor shapes (`model.safetensors.index.json` contains only a name→file map). Given S4 dims (`hidden_size=2304`, `intermediate_size=1152`, `n_routed_experts=128`) and S5 block shapes, the implied scale shapes are:

- NVFP4 layer 0–19: `w13_weight` ~ [128, 2304, 2304] → `w13_weight_scale` ~ [128, 2304, 144] (2304/16), FP8 E4M3; `w2_weight` ~ [128, 2304, 1152] → `w2_weight_scale` ~ [128, 2304, 72]; `*_weight_amax` and `*_weight_outer_scale_denominator` are per-expert (128,) FP32.
- MXFP8 layer 20–23: `w13_weight` ~ [128, 2304, 2304] → `w13_weight_scale` ~ [128, 72, 72], UE8M0 (stored as U8); `w2_weight` ~ [128, 2304, 1152] → `w2_weight_scale` ~ [128, 72, 36].

**[UNKNOWN]** Exact numeric shapes of the scale tensors were not directly published; the above are derived from config dims plus `weight_block_shape`. Verify against the local safetensors headers if available.

**Verdict on our local assumption `nvfp4_block=16`, `mxfp8_block=32`:**

- **CONFIRMED** for activations: S5 `block_size: 16` (NVFP4) and `block_size: 32` (MXFP8); S9 format definitions agree.
- **CONFIRMED (with shape caveat)** for NVFP4 weights: `weight_block_shape [1, 16]` — 16 consecutive elements *within a row*.
- **CONTRADICTED / INCOMPLETE** for MXFP8 weights if our code treats `mxfp8_block=32` as a flat 1-D group of 32: the published weight block is **2-D [32, 32]**. A flat 32-element grouping does not describe the checkpoint's weight scales.

Also note a header inconsistency: S4 declares `"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}` (NVFP4 only), while S5 declares `"quant_method": "inkling2"` with per-tensor NVFP4 **and** MXFP8 groups. A loader that trusts S4 alone will mis-type layers 20–23.

### 2. Which layers use which precision; "no conversion loss"

**Layer precision.** S2: "Most of Inkling 2 Mini's weights were trained and are deployed natively in NVFP4, with the last four layers in MXFP8." And: "We use 4-bit NVFP4 precision in the expert layers on all forward passes from pretraining through RL, bolstered by a dedicated kernel and hardware-accelerated on NVIDIA's GB300 platform. The final 4 layers' forward passes ran in 8-bit (MXFP8)." S2: "Inkling 2 Mini's weights are deployed in the same numeric precision they were trained on—**20 layers in NVFP4 and 4 in MXFP8**—so there is **no conversion loss from quantization**."

S5 confirms exactly which tensors: NVFP4 group lists `model.llm.layers.{0..19}.mlp.experts.w13_weight` / `w2_weight`; MXFP8 group lists `model.llm.layers.{20..23}.mlp.experts.w13_weight` / `w2_weight` (0-indexed). So "20 layers NVFP4" means **the MoE expert weights of layers 0–19**, and "last four layers" means indices 20–23 = 1-indexed layers 21–24.

**Critical qualifier from the varloop card (S7):**

> "Its expert layers are NVFP4, the last four use MXFP8, and the remaining weights are BF16."

> "The model uses the same mixed NVFP4 and MXFP8 precision it was trained in, so this checkpoint is its native precision and needs no further quantization."

S5 supports this: a large BF16 group covers embeddings, norms, attention projections, gate weights and the vision tower; an FP32 group covers gate biases, `attn_res.attn_queries.*`, `attn_res.mlp_queries.*`, `middle_cycle_cross_pass*`; `unembed.weight` is `"compute_dtype": "bfloat16", "fwd_output_dtype": "float32"`. So the checkpoint is **mixed**, not "all weights in NVFP4/MXFP8". The "no conversion loss" claim is about the expert weights being deployed in the precision they were trained in; non-expert weights are BF16/FP32 and were also BF16/FP32 in training. S9: "Inkling 2 Small is the largest model we are aware of that was trained *end-to-end* with a native 4-bit forward pass… This makes training simpler and eliminates the need for calibration when a quantized checkpoint is delivered at the end of training."

**Verdict:** "no conversion loss from quantization" is an exact, published claim (S2, and equivalently S7). It does **not** imply no precision reduction relative to a hypothetical BF16 model — the model was trained quantized. It also does not imply every weight tensor is low-bit.

### 3. Layer/core boundaries — reconciled

There is **no real contradiction**; the two descriptions use different indexing conventions.

- S4/S8 `loop_config`: `{"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false/true}`. With 0-indexed layers and an **exclusive** end, this is indices 6,7,…,17 = 12 layers.
- S2: "Inkling 2 Mini has 24 layers, divided into four blocks of six layers each. The two middle blocks run three times, while the first and last blocks each run once." Blocks 2 and 3 = indices 6–17.
- S2's experiment figure uses **1-indexed** labels: "Middle layers (7–18) **1.51–1.91×**" and "Last layers (13–24) 1.24–1.36×". S1: "The middle 12 layers run three times with shared weights, giving 48 layer applications per token from 24 layers of weights."
- S2: "Blocks 2 and 3 share weights across three passes, giving 48 layer applications per token from 24 stored layers." "The prelude feeds every pass; the second and third also receive the previous pass's message." "This gate is learned during training: it initially blocks the entire message…"
- S4: `"n_layers_per_attn_res_block": 6`, `"attn_res_normalize_values": true`. S7 loop config has the same `core_start/core_end/gate_rank`, differing only in `is_varloop: true`.

So "layers 6..17 (upstream llama.cpp)" = 0-indexed inclusive start/end; "layers 7..18 (our earlier description)" = 1-indexed; both name the same 12 stored layers. The only way to get a real bug is to apply the 1-indexed numbers to 0-indexed code (range(7,19) includes layer 18 and drops layer 6).

**Gate rank:** `gate_rank: 64` is published in S4/S8. **[UNKNOWN]** The primary sources do not explain the algebra of the 64-rank gate beyond S2's prose that it "scales" the merged prelude/message/previous-layer outputs and is learned.

### 4. Context length, experts, other config values

- **Context:** S1/S3/S7: "supports a context window of up to 1M tokens." S4/S8: `"model_max_length": 1048576`, `mtp_config.max_position_embeddings: 1048576`.
- **Experts / top-k:** S1/S3/S7: "In each layer, each token is routed to 8 of 128 experts." S4/S8: `"n_routed_experts": 128`, `"num_experts_per_tok": 8`, `"n_shared_experts": 0`. S2: training began "with a sparsity of 48, with 8 experts being chosen out of 384. At a single point during midtraining, we pruned two-thirds of the experts, leaving the 128 experts that were most used in each layer." So 384/8 is the *pre-pruning* figure; 128/8 is the shipped model.
- **Parameters:** S1/S3/S7: "26.4B, 3.4B active"; S1/S3: "each extra pass through the looped layers adds about 1.0B effective active parameters. Since the model runs with three passes, there are 5.4B effective active parameters per token."
- **Architecture extras (S4/S8):** `hidden_size 2304`, `24` decoder layers, `num_attention_heads 24`, `num_key_value_heads 8`, `head_dim 128`, `intermediate_size 1152`, `local_layer_ids` = 20 of 24 (global layers are 5, 11, 17, 23 in 0-indexed), `sliding_window_size 1024`, `sconv_kernel_size 4`, `route_scale 8.0`, `use_gate_bias true`, MTP `block_size 8`, `markov_rank 256`, `target_layer_ids [5, 23, 41, 48]`. Vision tower: 23 layers, `hidden_size 1024`, `posemb_grid_size 32`.
- **Varloop (S7/S8):** same 24-layer/128-expert/1M-token shape; `is_varloop: true`; trained with 1–5 passes sampled 1:2:6:2:1, post-trained at 3.

### 5. Other primary-source statements that bear on local docs

- S2: "Attention residuals… each layer reads the outputs of the earlier blocks and learns its own mix of them." S4: `n_layers_per_attn_res_block: 6`.
- S2: "The previous loop's message is merged with the prelude's output via a gate that scales these along with the output of the previous layers in the block."
- S1/S3 warnings worth preserving in docs: "Because the middle layers run three times, compute per token is higher than the active-parameter count alone suggests"; "Setting reasoning effort to its minimum does not fully disable reasoning"; "Support for looped models in local runtimes is new".
- Release date 2026-11-20, Apache 2.0 (S1/S3/S7).

---

## WHAT WE MUST FIX OR VERIFY LOCALLY

1. **MXFP8 weight block shape is 2-D [32, 32], not a flat 32.** Our local `mxfp8_block=32` is right for activations (S5 `block_size: 32`) but wrong if the weight dequantizer groups weights into 1-D runs of 32. *Cheap test:* load `hf_quant_config.json` (or curl S5 and `jq '.model.parameter_groups[] | select(.compute_dtype=="mxfp8") | .weight_block_shape'`) → expect `[32,32]`; then assert our dequant routine's per-scale element count for `layers.20..23.mlp.experts.*_weight` is `32*32=1024`, not `32`. Print the inferred scale shape for `w13_weight` (~[128,72,72]) and diff against ours.

   **Local fix (2026-11-24):** implemented `quantize_mxfp8_weights` /
   `dequantize_mxfp8_weights` with 2-D `[32, 32]` tile scales and `layout="2d"`;
   `quantize_mxfp8` / `dequantize_mxfp8` remain the flat 1-D activation path.
   `configs/inkling2mini.toml` now records `mxfp8_weight_block_shape = [32, 32]`.

2. **Only expert weights are low-bit; the rest is BF16/FP32.** Our docs/config must not say "all weights NVFP4 except last 4 layers" — S7 says "the remaining weights are BF16", and S5 shows FP32 gate biases/attn-res queries plus BF16 attention/norm/embed/vision. Also S4's `quantization_config.quant_algo: "NVFP4"` alone would mis-type layers 20–23. *Cheap test:* enumerate tensors our quantizer touches and assert the set equals exactly `layers.{0..23}.mlp.experts.{w13_weight,w2_weight}`; assert e.g. `layers.0.attn.wq_du.weight`, `layers.0.mlp.gate.weight`, `visual.backbone.*`, `embed.weight` remain BF16 and gate biases FP32. Separately assert our loader reads S5 per-tensor groups, not only S4's top-level `quant_algo`.

3. **NVFP4 weights use [1, 16] row-wise blocks with E4M3 scales + FP32 amax/outer-scale denominator, and 4/6 scale selection.** Verify our dequant applies `scale(F8_E4M3) * amax/outer_scale_denominator` and selects scales per row of 16 (not 16×16 tiles, not a flat 16 spanning row boundaries). *Cheap test:* for `layers.0.mlp.experts.w13_weight` assert our scale count equals `numel/16` and that companion `*_amax` and `*_outer_scale_denominator` tensors exist and are consumed; check our scale-selection code matches `"fouroversix_mae"` (S9's Four Over Six) or at least that using nearest is intentional. Assert no scale is shared across two rows.

4. **Core layer boundaries: keep 0-indexed `range(6, 18)` (exclusive end).** Our earlier "7..18" is 1-indexed and describes the same layers; do not convert it naively. *Cheap test:* assert `len(list(core))==12`, `min(core)==6`, `max(core)==17`, and that layer 18 is *not* looped and layer 6 *is*. Grep docs for "layers 7–18" and confirm there is an explicit "1-indexed" label next to it.

5. **"No conversion loss" claim scope.** If our docs repeat it, qualify it as "deployed in the trained precision of the expert weights" and note non-expert weights are BF16/FP32; do not imply it means no accuracy cost versus BF16 training. *Cheap test:* grep our docs for the phrase and verify the surrounding sentence names expert weights; assert our weight conversion pipeline has no BF16→NVFP4 requantization step for expert tensors.

6. **Expert counts: shipped model is 128 routed / top-8 / 0 shared.** If any local config still carries the pre-pruning 384 experts, it is stale. *Cheap test:* assert `n_routed_experts==128`, `num_experts_per_tok==8`, `n_shared_experts==0` against S4, and check any doc mentioning 384 labels it "pre-pruning".

7. **Context length 1,048,576.** *Cheap test:* assert our `max_position_embeddings == 1048576` and that any local doc saying "1M tokens" means 2^20, not decimal 1,000,000.

8. **Gate rank 64 and FP32 gate state.** *Cheap test:* assert our loop config has `gate_rank==64`; assert gate-bias tensors load as FP32 (S5 group 1), not BF16. **[UNKNOWN]** Gate algebra is not specified in primary sources — do not invent a formula; if our implementation needs one, flag it as a local design choice rather than a published fact.

9. **Scale tensor shapes were not directly published.** **[UNKNOWN]** The derived shapes in §1 (e.g. MXFP8 `w13_weight_scale` ~ [128,72,72]) come from S4 dims + S5 block shapes. *Cheap test:* read the local safetensors headers (or `safetensors.safe_open(...).get_slice`/`.get_shape`) and print the actual shapes of `layers.20.mlp.experts.w13_weight`, `w13_weight_scale`, and `layers.0.mlp.experts.w13_weight_scale`; compare with the derived values before finalizing any dequantizer.
