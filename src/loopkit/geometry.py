"""Geometry diagnostics with explicit kill gates. This module is optional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
svd = np.linalg.svd


@dataclass(frozen=True)
class GoNoGo:
    go: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def lambda_law(bits: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Fit err = c * 2**(-lam * bits); returns (lam, c)."""
    b = np.asarray(bits, dtype=float)
    e = np.maximum(np.asarray(errors, dtype=float), 1e-30)
    A = np.column_stack([-b, np.ones_like(b)])
    lam, logc = np.linalg.lstsq(A, np.log2(e), rcond=None)[0]
    return float(lam), float(2 ** logc)


def predict_error(bits: float, law: tuple[float, float]) -> float:
    lam, c = law
    return float(c * 2 ** (-lam * bits))


def break_even_passes(base_per_pass: float, codec_setup: float,
                      codec_per_pass: float) -> int | None:
    """Smallest n >= 1 where n passes of base work beat codec setup + n*cost."""
    if base_per_pass <= codec_per_pass:
        return None
    n = int(np.floor(codec_setup / (base_per_pass - codec_per_pass))) + 1
    return max(1, n)


def spectral_break_even(distortion: float, n: int = 4096, rank: int = 256,
                        lo: float = 0.05, hi: float = 0.95) -> float:
    """Finite-N gamma* where low-rank tail error equals binarization distortion."""
    k = np.arange(1, n + 1, dtype=np.float64)

    def tail(g: float) -> float:
        s = np.cumsum(k ** (-2 * g))
        return 1.0 - s[rank - 1] / s[-1]

    for _ in range(80):
        mid = (lo + hi) / 2
        if tail(mid) > distortion:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def joint_itq(passes: Sequence[np.ndarray], *, iters: int = 50,
              seed: int = 0) -> np.ndarray:
    """One shared orthogonal rotation from the summed covariance of loop passes."""
    z = np.concatenate([np.asarray(p, dtype=np.float64).reshape(len(p), -1) for p in passes], axis=0)
    z = z - z.mean(axis=0, keepdims=True)
    cov = z.T @ z
    U, _, Vt = svd(cov)
    R = U @ Vt
    for _ in range(max(0, iters - 1)):  # refine against binary targets
        b = np.where(z @ R >= 0, 1.0, -1.0)
        U, _, Vt = svd(b.T @ z)
        R = U @ Vt
    return R


def fp4_itq(x: np.ndarray, *, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rotate x toward the NVFP4 E2M1 codebook; returns (R, codes)."""
    from .nvfp4 import quantize_nvfp4

    z = np.asarray(x, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((z.shape[1], z.shape[1])))
    R = q
    best = float("inf")
    codes = None
    for _ in range(max(1, iters)):
        rotated = z @ R
        from .nvfp4 import dequantize_nvfp4

        t = quantize_nvfp4(rotated.astype(np.float32))
        codes = t.codes
        err = float(np.mean((rotated - dequantize_nvfp4(t).astype(np.float64)) ** 2))
        if err >= best:
            break
        best = err
        b = np.where(rotated >= 0, 1.0, -1.0)
        U, _, Vt = svd(b.T @ z)
        R = U @ Vt
    return R, codes


def geometry_gate(bits: np.ndarray, errors: np.ndarray, *,
                  base_per_pass: float, codec_setup: float, codec_per_pass: float,
                  loops: int = 3, x: np.ndarray | None = None) -> GoNoGo:
    reasons: list[str] = []
    lam, c = lambda_law(bits, errors)
    metrics: dict[str, float] = {"lambda": lam, "c": c}
    if lam <= 0:
        reasons.append("no scaling law")
    n = break_even_passes(base_per_pass, codec_setup, codec_per_pass)
    metrics["break_even_passes"] = float(n) if n else -1.0
    if n is None or n > loops:
        reasons.append("no break-even within loop budget")
    if x is not None:
        from .nvfp4 import quantize_nvfp4, dequantize_nvfp4

        baseline = dequantize_nvfp4(quantize_nvfp4(np.asarray(x, dtype=np.float32)))
        R, _ = fp4_itq(x)
        rotated = np.asarray(x, dtype=np.float64) @ R
        rq = dequantize_nvfp4(quantize_nvfp4(rotated.astype(np.float32)))
        m_rot = float(np.mean((rotated - rq) ** 2))
        m_base = float(np.mean((baseline - np.asarray(x, dtype=np.float32)) ** 2))
        metrics["fp4_rotated_mse"] = m_rot
        metrics["fp4_baseline_mse"] = m_base
        if m_rot >= m_base:
            reasons.append("rotation does not help FP4")
    return GoNoGo(not reasons, tuple(reasons), metrics)
