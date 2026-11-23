"""CycleAudit — speed claims must be earned by measurement."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TimingSample:
    name: str
    seconds: np.ndarray
    cycles: int = 0
    setup_seconds: float = 0.0

    def __post_init__(self):
        if not self.cycles:
            object.__setattr__(self, "cycles", int(np.asarray(self.seconds).size))

    def median(self) -> float:
        return float(np.median(self.seconds)) + self.setup_seconds / max(1, self.cycles)


@dataclass(frozen=True)
class CycleReport:
    baseline: str
    candidate: str
    loops: int
    speedup: float
    ci_low: float
    ci_high: float
    claim_valid: bool
    reason: str


class CycleAudit:
    def __init__(self, *, loops: int = 3, warmup: int = 5,
                 repeats: int = 50, seed: int = 0) -> None:
        self.loops = loops
        self.warmup = warmup
        self.repeats = repeats
        self.seed = seed

    def measure(self, name: str, fn: Callable[[], object], *,
                setup: Callable[[], object] | None = None) -> TimingSample:
        t0 = time.perf_counter()
        if setup is not None:
            setup()
        setup_s = time.perf_counter() - t0
        for _ in range(self.warmup):
            fn()
        samples = np.empty(self.repeats, dtype=np.float64)
        for i in range(self.repeats):
            t = time.perf_counter()
            fn()
            samples[i] = time.perf_counter() - t
        return TimingSample(name, samples, self.repeats, setup_s)

    def compare(self, base: TimingSample, cand: TimingSample) -> CycleReport:
        rng = np.random.default_rng(self.seed)
        b = rng.choice(base.seconds, size=(2000, base.cycles), replace=True).mean(axis=1) + base.setup_seconds / base.cycles
        c = rng.choice(cand.seconds, size=(2000, cand.cycles), replace=True).mean(axis=1) + cand.setup_seconds / cand.cycles
        ratios = b / np.maximum(c, 1e-12)
        speedup = base.median() / max(cand.median(), 1e-12)
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        return CycleReport(base.name, cand.name, self.loops, float(speedup),
                           float(lo), float(hi), False, "")

    def verdict(self, claim_speedup: float, report: CycleReport) -> bool:
        if self.repeats < 20 or self.loops < 1:
            return False
        return claim_speedup <= report.ci_low

    def to_json(self, report: CycleReport) -> str:
        d = {**report.__dict__, "ci_low": report.ci_low, "ci_high": report.ci_high}
        return json.dumps(d, indent=2, sort_keys=True)
