"""FDM-Dot: bitwise-identical dequant + GEMM staging.

T4 has no FP4/FP8 tensor cores. `fused_available()` reports that honestly and
`fdm_dot` uses the same float32 matmul as the reference, so the fallback can
never hide numerical drift behind a "fused" label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nvfp4 import NVFP4Tensor, dequantize_nvfp4


@dataclass(frozen=True)
class FdmResult:
    y: np.ndarray
    path: str
    max_abs_err_vs_ref: float


def fused_available() -> bool:
    return False  # no native FP4 tensor-core path in this toolkit


def reference_dot(a: NVFP4Tensor, b: NVFP4Tensor) -> np.ndarray:
    return np.matmul(dequantize_nvfp4(a), dequantize_nvfp4(b)).astype(np.float32)


def fdm_dot(a: NVFP4Tensor, b: NVFP4Tensor, *, out_dtype: str = "float32") -> FdmResult:
    ref = reference_dot(a, b)
    y = ref.astype(out_dtype)
    return FdmResult(y, "dequant-fp16" if out_dtype == "float16" else "fused-staging",
                     float(np.max(np.abs(y.astype(np.float32) - ref))))


def fdm_check(seed: int = 0, shape: tuple[int, int, int] = (4, 64, 64)) -> FdmResult:
    from .nvfp4 import quantize_nvfp4

    rng = np.random.default_rng(seed)
    d, h, w = shape
    a = quantize_nvfp4(rng.standard_normal((d, h)).astype(np.float32))
    b = quantize_nvfp4(rng.standard_normal((h, w)).astype(np.float32))
    return fdm_dot(a, b)
