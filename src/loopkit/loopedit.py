"""Immutable per-pass weight edits with digest verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .bitloop import state_digest


@dataclass(frozen=True)
class WeightEdit:
    pass_index: int
    key: str
    delta: np.ndarray
    old_digest: str
    new_digest: str
    n_changed: int


def diff_passes(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], *,
                pass_index: int) -> tuple[WeightEdit, ...]:
    if set(a) != set(b):
        raise ValueError("pass states must have identical keys")
    edits = []
    for key in sorted(a):
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"shape/dtype mismatch for {key}")
        delta = y.astype(np.int64) - x.astype(np.int64) if x.dtype.kind in "iu" else y - x
        delta = delta.astype(x.dtype)
        n_changed = int(np.count_nonzero(y != x))
        edits.append(WeightEdit(pass_index, key, delta,
                                hashlib.sha256(x.tobytes()).hexdigest(),
                                hashlib.sha256(y.tobytes()).hexdigest(), n_changed))
    return tuple(edits)


def apply_edits(state: Mapping[str, np.ndarray],
                edits: Sequence[WeightEdit]) -> dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in state.items()}
    for edit in edits:
        if edit.key not in out:
            raise KeyError(edit.key)
        cur = np.ascontiguousarray(out[edit.key])
        if hashlib.sha256(cur.tobytes()).hexdigest() != edit.old_digest:
            raise ValueError(f"edit applied to a different state than it was diffed from: {edit.key}")
        new = np.asarray(cur) + edit.delta
        if hashlib.sha256(np.ascontiguousarray(new).tobytes()).hexdigest() != edit.new_digest:
            raise ValueError(f"edit result digest mismatch: {edit.key}")
        out[edit.key] = new
    return out
