# Geometry gate — definitive result (NEGATIVE)

Kernel: `jprinceraj/loopkit-geometry-gate-single-t4` (Kaggle T4, two independent
runs: before and after the E2M1 mapping fix; same verdict both times).

## What replicated

| Metric | Value | Paper reference |
|---|---|---|
| λ_mean, SVD latents | 0.4093 | ~0.54 on Llama-2 representative layer |
| λ_mean, after Joint-ITQ | 0.3633 | 0.30 |
| λ_max, SVD → ITQ | 0.695 → 0.394 | 0.88 → 0.43 |
| Rotation orthogonality | PASS (<1e-6) | required |
| Break-even γ* (SVD / rot / ITQ) | 0.3854 / 0.4586 / 0.5061 | 0.385 / 0.458 / 0.506 |

The mechanism replicates: rotation lowers distortion and the spectral
break-even bands reproduce exactly. `geometry_replication`, `rotation_orthogonal`
and `lambda_drop` gates all PASS.

## What failed

With a faithful NVFP4 quantizer (the fixed one), on a synthetic 512×2304 matrix
with a 16× outlier injected at dimension 2027:

- `fp4_mse_baseline` = 0.009946
- `fp4_mse_rotated`  = 0.010018
- `fp4_relative_gain` = **−0.0072 (−0.7%)**

Rotation *hurts* native NVFP4 by ~0.7%. The `fp4_gain` gate FAILS, so the
overall verdict is **NEGATIVE**.

## Conclusion

LittleBit-2's geometry alignment is a genuine effect for ±1 binarization of
FP16-origin low-rank factors, but it does **not** transfer to a natively
NVFP4-trained model. This matches the skeptic's prediction: the bimodal ±1
target collapses FP4's 16-level codebook, and "free rotation" only exists
inside a low-rank factorization. We keep the diagnostic tooling
(`geometry`, `shc`) and drop the idea of applying Joint-ITQ to Inkling 2 Mini's
weights.

Artifacts: `geometry_gate.json` in the kernel output; notebooks in
`kaggle_geom/` and `kaggle_geom_single/`.
