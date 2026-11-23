import numpy as np
from loopkit.audit import CycleAudit, TimingSample


def test_oracle_rejects_inflated_claim():
    audit = CycleAudit(repeats=50, seed=1)
    base = TimingSample("base", np.full(50, 0.010))
    cand = TimingSample("cand", np.full(50, 0.008))
    report = audit.compare(base, cand)
    assert 1.2 <= report.speedup <= 1.3
    assert not audit.verdict(2.0, report)
    assert audit.verdict(1.1, report)


def test_oracle_rejects_too_few_repeats():
    audit = CycleAudit(repeats=5)
    base = TimingSample("base", np.full(5, 0.01))
    assert not audit.verdict(1.0, audit.compare(base, base))
