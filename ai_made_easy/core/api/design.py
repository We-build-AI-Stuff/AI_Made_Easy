"""Designs: validation, fixes, code, summaries."""
from __future__ import annotations

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph


def validate(graph: dict | Graph) -> dict:
    """Issues (with Quick Fix labels where one exists) and inferred shapes."""
    from ai_made_easy.core.fixes import fix_for_issue

    g = _graph(graph)
    issues = g.validate()
    shapes, _ = g.infer_shapes_detailed()
    rows = []
    for i in issues:
        try:
            fix = fix_for_issue(g, i)
        except Exception:  # noqa: BLE001 — a fix is optional
            fix = None
        rows.append({"severity": i.severity, "node": i.node_id, "message": i.message,
                     "fix": {"label": fix[0], "description": fix[1]} if fix else None})
    out = {"valid": not any(i.severity == "error" for i in issues),
           "issues": rows,
           "shapes": {k: list(v) for k, v in shapes.items()}}
    if any(n.type_id.startswith("pipeline.") for n in g.nodes.values()):
        out["designs"] = stage_designs(g)
    return out


def stage_designs(graph: dict | Graph, base: str | None = None) -> dict:
    """Each pipeline stage's design as blocks to draw inline on the canvas."""
    from ai_made_easy.core.pipelines.designs import inline

    return inline(_graph(graph), base)


def apply_fix(graph: dict | Graph, issue_index: int) -> dict:
    """Apply the Quick Fix of ``validate(graph)["issues"][issue_index]``."""
    from ai_made_easy.core.fixes import fix_for_issue

    g = _graph(graph)
    issues = g.validate()
    if not 0 <= issue_index < len(issues):
        raise ApiError("no issue at that index (the graph changed?)")
    fix = fix_for_issue(g, issues[issue_index])
    if fix is None:
        raise ApiError("this issue has no automatic fix")
    return {"label": fix[0], "graph": fix[2].to_dict()}


def targets() -> dict:
    from ai_made_easy.core.targets import RENDERERS, target_label

    return {"targets": [{"id": t, "label": target_label(t)} for t in RENDERERS]}


def generate(graph: dict | Graph, target: str = "pytorch_model") -> str:
    from ai_made_easy.core.targets import RENDERERS

    if target not in RENDERERS:
        raise ApiError(f"unknown target {target!r}; one of {sorted(RENDERERS)}")
    return RENDERERS[target](_graph(graph))


def summarize(graph: dict | Graph) -> dict:
    from ai_made_easy.core import budget
    from ai_made_easy.core.summary import summarize as _summarize

    g = _graph(graph)
    s = _summarize(g)
    try:
        costs = budget.estimate(g).layers
    except Exception:  # noqa: BLE001 — the summary never fails on an estimate
        costs = []
    flops = {i: c.flops for i, c in enumerate(costs)} if len(costs) == len(s.layers) else {}
    return {"total_params": s.total_params,
            "total_params_display": s.total_params_display,
            "total_flops": sum(flops.values()),
            "layers": [{"name": L.name, "type": L.type_id,
                        "output_shape": L.output_shape, "params": L.params,
                        "flops": flops.get(i, 0)}
                       for i, L in enumerate(s.layers)]}


def list_devices() -> dict:
    from ai_made_easy.core import budget

    return {"devices": [d.to_dict() for d in budget.devices().values()]}


def estimate_budget(graph: dict | Graph, device: str | None = None,
                    limits: dict | None = None) -> dict:
    """FLOPs / memory / latency estimate and the project's budget checks.

    ``device`` and ``limits`` override the budget saved in the project.
    """
    from ai_made_easy.core import budget

    g = _graph(graph)
    if device is not None or limits:
        try:
            budget.set_budget(g, **({"device": device} if device is not None else {}),
                              **(limits or {}))
        except (KeyError, ValueError) as exc:
            raise ApiError(str(exc).strip("'\"")) from exc
    return budget.budget_report(g)


def expand(graph: dict | Graph, node_id: str) -> dict:
    from ai_made_easy.core.composites import expand_in_graph

    expanded = expand_in_graph(_graph(graph), node_id)
    return {"graph": expanded.to_dict(), "node_count": len(expanded.nodes)}
