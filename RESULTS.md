# Results

## Kaggle 2×T4 audit run — PASS

Kernel: `jprinceraj/loopkit-t4-audit` (GPU T4 x2, 2026-11-23).
Artifacts: `device_check.json`, `summary.json` in the kernel output.

Verified:

| Item | Value |
|---|---|
| `torch.cuda.device_count()` | 2 (`Tesla T4`, `Tesla T4`) |
| FP16 matmul on each GPU | OK (both checksums nonzero) |
| Bit-exact ANS roundtrip on 1024×1024 tensor, GPU 0 | PASS, 4,194,304 B → 565,073 B (13.5%) |
| Bit-exact ANS roundtrip on 1024×1024 tensor, GPU 1 | PASS, 4,194,304 B → 564,669 B (13.5%) |
| Wall time per GPU (codec + matmul + sync) | 0.313 s / 0.267 s |
| Spectral break-even γ* (SVD / rotation / ITQ) | 0.3854 / 0.4586 / 0.5061 |

The γ* values reproduce the LittleBit-2 paper's finite-N reference bands
(0.385 / 0.458 / 0.506). This validates the geometry diagnostic, **not** a
claim that LittleBit-2 transfers to Inkling 2 Mini's native NVFP4 — the FP4
track is a separate kill gate.

## Local CI

`pytest -q -m "not kaggle"`: 18 passed, 1 deselected.
`bash scripts/smoke.sh`: 3 gating tests passed, CLI `fdm-check` clean.
GitHub Actions `fast-tests`: success on `master`.

## Hardware honesty

T4 has no FP4/FP8 tensor cores. All numbers above are bit-exactness, size and
spectral results. No wall-clock speedup is claimed from this repo.
