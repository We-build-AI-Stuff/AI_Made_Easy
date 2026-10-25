"""Training components: losses, optimizers, LR schedulers, metrics, trainer.

One table per component kind is the single source of truth: it registers the
canvas blocks *and* drives code generation for PyTorch and Keras 3. Each entry
renders a framework expression from the block's resolved params; ``keras`` is
``None`` when Keras has no faithful equivalent (reported as a clear error).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition
from ai_made_easy.core.training.train_helpers import TRAIN_HELPERS  # noqa: F401  (re-exported)


Render = Callable[[dict], str]


@dataclass(frozen=True)
class Component:
    type_id: str
    name: str
    params: tuple = ()
    torch: Render | None = None
    keras: Render | None = None
    desc: str = ""
    meta: dict = field(default_factory=dict)
    checks: Callable | None = None


def _floats(value) -> list[float]:
    return [float(v) for v in str(value).replace(";", ",").split(",") if v.strip()]


def _ints(value) -> list[int]:
    return [int(float(v)) for v in str(value).replace(";", ",").split(",") if v.strip()]


# by area (imported here, after what they share above)
from ai_made_easy.core.training.catalog.losses import (  # noqa: E402
    LOSSES,
    METRIC_LOSS_IDS,
    PAIR_LOSS_IDS,
)
from ai_made_easy.core.training.catalog.optimizers import (  # noqa: E402
    OPTIMIZERS,
)
from ai_made_easy.core.training.catalog.schedulers import (  # noqa: E402
    SCHEDULERS,
)
from ai_made_easy.core.training.catalog.metrics import (  # noqa: E402
    METRICS,
)
from ai_made_easy.core.training.catalog.trainer import (  # noqa: E402
    EXPLAIN,
    EXPLAIN_ATTENTION,
    EXPLAIN_GRADCAM,
    EXPLAIN_IG,
    EXPLAIN_SALIENCY,
    EXPLAIN_SHAP,
    KFOLD,
    PICTURE_EXPLAIN_IDS,
    PREDICT,
    TRAINER,
)


def _losses_by_id() -> dict[str, Component]:
    return {c.type_id: c for c in LOSSES}


LOSS_IDS = tuple(c.type_id for c in LOSSES)
OPTIMIZER_IDS = tuple(c.type_id for c in OPTIMIZERS)
SCHEDULER_IDS = tuple(c.type_id for c in SCHEDULERS)
METRIC_IDS = tuple(c.type_id for c in METRICS)
COMPONENTS: dict[str, Component] = {
    c.type_id: c for c in (*LOSSES, *OPTIMIZERS, *SCHEDULERS, *METRICS, TRAINER, KFOLD, PREDICT,
                           EXPLAIN, EXPLAIN_IG, EXPLAIN_SHAP, EXPLAIN_GRADCAM,
                           EXPLAIN_ATTENTION, EXPLAIN_SALIENCY)
}

_CATEGORY = {
    **{t: ("Loss", "training") for t in LOSS_IDS},
    **{t: ("Optimizer", "training") for t in OPTIMIZER_IDS},
    **{t: ("Scheduler", "training") for t in SCHEDULER_IDS},
    **{t: ("Metrics", "evaluation") for t in METRIC_IDS},
    "train.trainer": ("Training", "training"),
    "train.kfold": ("Training", "training"),
    "eval.predict": ("Metrics", "evaluation"),
    "eval.permutation_importance": ("Metrics", "evaluation"),
    "eval.integrated_gradients": ("Metrics", "evaluation"),
    "eval.shap_values": ("Metrics", "evaluation"),
    "eval.grad_cam": ("Metrics", "evaluation"),
    "eval.attention_maps": ("Metrics", "evaluation"),
    "eval.audio_saliency": ("Metrics", "evaluation"),
}
EXPLAIN_IDS = ("eval.permutation_importance", "eval.integrated_gradients", "eval.shap_values")
EXPLAIN_METHODS = {"eval.permutation_importance": "permutation",
                   "eval.integrated_gradients": "integrated_gradients",
                   "eval.shap_values": "shap"}


def _libraries(c: Component) -> str:
    if c.type_id in (*EXPLAIN_IDS, *PICTURE_EXPLAIN_IDS):
        return "PyTorch"
    if c.type_id in METRIC_IDS or c.type_id in ("train.trainer", "train.kfold", "eval.predict"):
        return "PyTorch · Keras"
    return " · ".join(lib for lib, ok in (("PyTorch", c.torch), ("Keras", c.keras)) if ok)


def register_all() -> None:
    reg = get_registry()
    for type_id, comp in COMPONENTS.items():
        category, family = _CATEGORY[type_id]
        reg.register(BlockDefinition(
            type_id=type_id, display_name=comp.name, category=category,
            color=family_color(family), params=tuple(comp.params),
            checks_fn=comp.checks, description=comp.desc, library=_libraries(comp),
            meta={"kind": category.lower(), **{k: v for k, v in comp.meta.items()
                                               if not callable(v)}},
            requires=tuple(comp.meta.get("requires", ())), extra=comp.meta.get("extra", ""),
        ))


def render(comp_id: str, params: dict, framework: str) -> str | None:
    comp = COMPONENTS[comp_id]
    fn = comp.torch if framework == "pytorch" else comp.keras
    return fn(params) if fn else None


def resolved(comp_id: str, params: dict | None) -> dict[str, Any]:
    comp = COMPONENTS[comp_id]
    out = {p.name: p.default for p in comp.params}
    out.update(params or {})
    return out

__all__ = [
    "COMPONENTS",
    "Component",
    "EXPLAIN",
    "EXPLAIN_ATTENTION",
    "EXPLAIN_GRADCAM",
    "EXPLAIN_IDS",
    "EXPLAIN_IG",
    "EXPLAIN_METHODS",
    "EXPLAIN_SALIENCY",
    "EXPLAIN_SHAP",
    "KFOLD",
    "LOSSES",
    "LOSS_IDS",
    "METRICS",
    "METRIC_IDS",
    "METRIC_LOSS_IDS",
    "OPTIMIZERS",
    "OPTIMIZER_IDS",
    "PAIR_LOSS_IDS",
    "PICTURE_EXPLAIN_IDS",
    "PREDICT",
    "Render",
    "SCHEDULERS",
    "SCHEDULER_IDS",
    "TRAINER",
    "TRAIN_HELPERS",
    "register_all",
    "render",
    "resolved",
]
