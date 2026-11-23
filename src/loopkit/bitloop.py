"""Bit-exact certificates for looped inference state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PassCertificate:
    pass_index: int
    layer: str
    tensor: str
    n_elements: int
    n_mismatch: int
    max_abs_err: float
    exact: bool
    digest: str


@dataclass(frozen=True)
class LoopCertificate:
    model: str
    n_passes: int
    passes: tuple[PassCertificate, ...]
    all_exact: bool
    root_digest: str


def _bytes(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def state_digest(state: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        a = np.asarray(state[key])
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(_bytes(a))
    return h.hexdigest()


def certify_pass(ref: Mapping[str, np.ndarray], cand: Mapping[str, np.ndarray], *,
                 pass_index: int, layer: str = "") -> PassCertificate:
    keys = sorted(set(ref) | set(cand))
    mismatches = 0
    max_err = 0.0
    n = 0
    for key in keys:
        a, b = ref.get(key), cand.get(key)
        if a is None or b is None:
            mismatches += 1
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatches += 1
            n += a.size
            continue
        n += a.size
        if not np.array_equal(a, b):
            mismatches += int(np.count_nonzero(a != b))
            if a.dtype.kind in "fiu b":
                max_err = max(max_err, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
            else:
                max_err = float("inf")
    return PassCertificate(
        pass_index=pass_index, layer=layer, tensor="*", n_elements=n,
        n_mismatch=mismatches, max_abs_err=max_err,
        exact=mismatches == 0, digest=state_digest(cand),
    )


def certify_loop(refs: Sequence[Mapping[str, np.ndarray]],
                 cands: Sequence[Mapping[str, np.ndarray]], *,
                 model: str = "inkling2mini") -> LoopCertificate:
    if len(refs) != len(cands):
        raise ValueError("reference and candidate must have the same number of passes")
    passes = tuple(certify_pass(r, c, pass_index=i) for i, (r, c) in enumerate(zip(refs, cands)))
    root = hashlib.sha256("".join(p.digest for p in passes).encode()).hexdigest()
    return LoopCertificate(model, len(passes), passes,
                           all(p.exact for p in passes), root)


def assert_lossless(cert: LoopCertificate) -> None:
    if not cert.all_exact:
        bad = [p for p in cert.passes if not p.exact]
        raise AssertionError(
            f"non-exact loop {cert.root_digest}: "
            + ", ".join(f"pass {p.pass_index} ({p.n_mismatch} mismatches)" for p in bad)
        )


def verify_loop(cert: LoopCertificate,
                cands: Sequence[Mapping[str, np.ndarray]]) -> bool:
    if len(cands) != cert.n_passes:
        return False
    return all(p.digest == state_digest(c) for p, c in zip(cert.passes, cands))
