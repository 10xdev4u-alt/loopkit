"""Executable load-count claims for union-resident MoE routing.

Without loop awareness, a token routes (and loads) ``top_k`` experts on each
of ``n_passes`` passes: ``loads_off = n_passes * top_k``. Route-once keeps a
single union of experts across the passes: ``loads_on = union_experts``. A
fractional union is the per-token average (e.g. 8.48 experts when most tokens
agree but some are dirty).

These are structural counts of expert loads, not measured wall-clock savings.
No speedup is asserted here: a reduction in loads is only an upper bound on
what a real engine could save.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficReport:
    """Expert-load accounting for one routing policy."""

    n_passes: int
    top_k: int
    loads_off: int
    loads_on: int
    reduction: float
    union_experts: int
    msg_scale: float

    def to_dict(self) -> dict:
        """JSON-ready view with all report fields."""
        return {
            "n_passes": self.n_passes,
            "top_k": self.top_k,
            "loads_off": self.loads_off,
            "loads_on": self.loads_on,
            "reduction": self.reduction,
            "union_experts": self.union_experts,
            "msg_scale": self.msg_scale,
        }


def union_report(
    union_experts: int,
    *,
    n_passes: int = 3,
    top_k: int = 8,
    msg_scale: float = 0.1,
) -> TrafficReport:
    """Compare union-resident routing with per-pass routing.

    ``reduction = 1 - loads_on / loads_off``, rounded to three decimals.
    Raises ``ValueError`` when the union is smaller than one pass's top-k
    (impossible) or larger than the no-loop total (no saving to report).
    """
    if n_passes < 1 or top_k < 1:
        raise ValueError("n_passes and top_k must be >= 1")
    loads_off = n_passes * top_k
    if union_experts < top_k:
        raise ValueError(f"union_experts={union_experts} is below top_k={top_k}")
    if union_experts > loads_off:
        raise ValueError(f"union_experts={union_experts} exceeds {loads_off} loads")

    return TrafficReport(
        n_passes=n_passes,
        top_k=top_k,
        loads_off=loads_off,
        loads_on=union_experts,
        reduction=round(1 - union_experts / loads_off, 3),
        union_experts=union_experts,
        msg_scale=msg_scale,
    )


def sweep_report(samples: dict[str, int]) -> dict[str, dict]:
    """Report load counts for a ``{scale: union_experts}`` sweep.

    Keys are the message scale as given (JSON object keys are strings);
    values are :meth:`TrafficReport.to_dict` for that scale.
    """
    return {
        str(scale): union_report(union, msg_scale=float(scale)).to_dict()
        for scale, union in samples.items()
    }
