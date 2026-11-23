"""Spectral Health Checker — diagnostics for ridge-recovered layer weights.

A health card is computed from synthetic activations and their outputs: the
weights are recovered by ridge least squares, then their singular-value decay,
participation ratio and kurtosis are summarized. No model weights are needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HealthCard:
    layer: str
    n_samples: int
    gamma_hat: float
    kurtosis: float
    lambda_max: float
    lambda_mean: float
    condition_number: float
    flagged: bool


def recover_weights(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Estimate W in Y ~ X @ W with (X^T X + ridge I)^-1 X^T Y.

    Returns a d_in x d_out float64 array. When n_samples <= d_in the dual
    identity (X^T X + r I)^-1 X^T = X^T (X X^T + r I)^-1 solves the smaller
    system; this is the same estimator and keeps the underdetermined case
    well-posed through the ridge term.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be (n_samples, d_in) and y must be (n_samples, d_out)")
    n, d_in = x.shape
    ridge = float(ridge)

    if n <= d_in:
        gram = x @ x.T
        gram += ridge * np.eye(n)
        return x.T @ np.linalg.solve(gram, y)

    gram = x.T @ x
    gram += ridge * np.eye(d_in)
    return np.linalg.solve(gram, x.T @ y)


def _gamma_from_spectrum(s: np.ndarray) -> float:
    """Fit log(s_k) = a - gamma log(k) on the largest singular values."""
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if s.size < 2:
        return 0.0
    m = min(256, s.size)
    log_s = np.log(s[:m])
    log_k = np.log(np.arange(1, m + 1, dtype=np.float64))
    design = np.column_stack((log_k, np.ones(m)))
    slope = np.linalg.lstsq(design, log_s, rcond=None)[0][0]
    return float(-slope)


def _lambda_stats(w: np.ndarray) -> tuple[float, float]:
    """Participation ratio per weight column: 1 - (L1/L2)^2 / d_in."""
    d_in = w.shape[0]
    l1 = np.abs(w).sum(axis=0)
    l2 = np.sqrt(np.square(w).sum(axis=0))
    ratio = np.zeros_like(l1)
    np.divide(l1, l2, out=ratio, where=l2 > 0)
    lam = 1.0 - (ratio * ratio) / d_in
    return float(lam.max(initial=0.0)), float(lam.mean())


def _kurtosis(w: np.ndarray) -> float:
    z = np.asarray(w, dtype=np.float64).ravel()
    if z.size == 0:
        return 0.0
    z = z - z.mean()
    var = float(np.mean(z * z))
    if var <= 0.0:
        return 0.0
    return float(np.mean(z**4) / (var * var))


def health_card(name: str, x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-3,
                lambda_threshold: float = 0.6, kurtosis_threshold: float = 8.0) -> HealthCard:
    """Summarize the spectral health of one layer from (activations, outputs)."""
    w = recover_weights(x, y, ridge)
    s = np.linalg.svd(w, compute_uv=False)
    lambda_max, lambda_mean = _lambda_stats(w)
    kurtosis = _kurtosis(w)
    s_max = float(s[0]) if s.size else 0.0
    s_min = float(s[-1]) if s.size else 0.0
    condition_number = s_max / s_min if s_min > 0.0 else float("inf")
    return HealthCard(
        layer=name,
        n_samples=int(np.asarray(x).shape[0]),
        gamma_hat=_gamma_from_spectrum(s),
        kurtosis=kurtosis,
        lambda_max=lambda_max,
        lambda_mean=lambda_mean,
        condition_number=float(condition_number),
        flagged=bool(lambda_max > lambda_threshold or kurtosis > kurtosis_threshold),
    )


def health_report(samples: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, HealthCard]:
    """Build one health card per named (x, y) sample pair."""
    return {name: health_card(name, x, y) for name, (x, y) in samples.items()}
