"""Model planning from the verified Inkling2 architecture facts.

The loop constants here come from upstream llama.cpp
(``src/models/inkling2-loop.cpp``): 24 physical layers, 3 passes, and a loop
core over layers 6..17 (``core_start=6, core_end=18``). Pass-indexed cache
slots follow the upstream formula ``n_phys + (n_passes-1)*(loop_end-loop_start)``
(24 + 2*12 = 48), which is also the number of layer applications per token.

``outlier_dim`` is an UNVERIFIED local hypothesis (current value 2027). It is
absent from every primary source -- TML's release post and model card -- and is
kept only as a local test knob. It is not a stated model property.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "inkling2mini.toml"

_UNVERIFIED_OUTLIER = (
    "UNVERIFIED local hypothesis (absent from primary sources); "
    "not a stated model property"
)


@dataclass(frozen=True)
class ModelPlan:
    """Planning view of the Inkling2 looped MoE architecture.

    ``n_layers`` is the number of *physical* stored layers (24 upstream).
    ``loop_start``/``loop_end`` bound the loop core, with ``loop_end``
    exclusive (6 and 18 upstream, i.e. layers 6..17). Each extra pass
    re-applies those core layers, so cache slots and layer applications
    share the same upstream formula.

    ``outlier_dim`` is an UNVERIFIED local hypothesis, not a verified
    architecture fact: it appears only in our config/tests and not in any
    primary source. Treat it as a local test knob, never as a model property.
    """

    name: str
    n_layers: int
    loop_start: int
    loop_end: int
    n_passes: int
    experts: int
    experts_per_token: int
    nvfp4_layers: int
    mxfp8_layers: int
    outlier_dim: int | None

    @property
    def cache_slots(self) -> int:
        """Pass-indexed slots: n_phys + (n_passes-1)*core length."""
        return self.n_layers + (self.n_passes - 1) * (self.loop_end - self.loop_start)

    @property
    def layer_applications(self) -> int:
        """Layer applications per token, using the same loop formula."""
        return self.n_layers + (self.n_passes - 1) * (self.loop_end - self.loop_start)

    def to_dict(self) -> dict:
        """JSON-ready view; the nested ``unverified`` map labels local claims."""
        return {
            "name": self.name,
            "n_layers": self.n_layers,
            "loop_start": self.loop_start,
            "loop_end": self.loop_end,
            "n_passes": self.n_passes,
            "experts": self.experts,
            "experts_per_token": self.experts_per_token,
            "nvfp4_layers": self.nvfp4_layers,
            "mxfp8_layers": self.mxfp8_layers,
            "outlier_dim": self.outlier_dim,
            "cache_slots": self.cache_slots,
            "layer_applications": self.layer_applications,
            "unverified": {"outlier_dim": _UNVERIFIED_OUTLIER},
        }


def load_plan(path: str | None = None) -> ModelPlan:
    """Load a plan from TOML, defaulting to ``configs/inkling2mini.toml``."""
    target = Path(path) if path is not None else _DEFAULT_CONFIG
    with target.open("rb") as fh:
        cfg = tomllib.load(fh)
    model = cfg["model"]
    codec = cfg["codec"]
    return ModelPlan(
        name=model["name"],
        n_layers=model["layers"],
        loop_start=model["loop_start"],
        loop_end=model["loop_end"],
        n_passes=model["loop_passes"],
        experts=model["experts"],
        experts_per_token=model["experts_per_token"],
        nvfp4_layers=codec["nvfp4_layers"],
        mxfp8_layers=codec["mxfp8_layers"],
        outlier_dim=model.get("outlier_dim"),
    )
