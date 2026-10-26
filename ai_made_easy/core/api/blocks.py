"""Blocks, samples and the catalog."""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, SAMPLES_DIR, _graph


def list_blocks(category: str | None = None) -> dict:
    from ai_made_easy.core.registry import get_registry
    from ai_made_easy.core.spec import missing_requirements

    blocks = []
    for defn in get_registry().all():
        row = defn.to_dict()
        row["missing"] = missing_requirements(defn)  # in this (server) environment
        blocks.append(row)
    if category:
        blocks = [b for b in blocks if b["category"].lower() == category.lower()]
    return {"count": len(blocks), "blocks": blocks}


def list_families() -> dict:
    from ai_made_easy.core.families import all_families

    return {"families": [f.to_dict() for f in all_families()]}


def list_tasks(family: str | None = None) -> dict:
    from ai_made_easy.core.tasks import all_tasks

    return {"tasks": [t.to_dict() for t in all_tasks(family)]}


def describe_design(graph: dict | Graph) -> dict:
    """Family, task and trainable frameworks of a design: ``frameworks`` are the ones
    every block of this design has a form for, ``framework`` the one it trains with."""
    from ai_made_easy.core.families import family_of, resolve_framework, usable_frameworks
    from ai_made_easy.core.tasks import task_of

    g = _graph(graph)
    family = family_of(g)
    try:
        task = task_of(g)
    except Exception:  # noqa: BLE001 — incomplete designs have no task yet
        task = None
    usable = usable_frameworks(g)
    try:
        chosen = resolve_framework(g)
    except ValueError:
        chosen = None
    return {"family": family.to_dict(), "task": task.to_dict() if task else None,
            "frameworks": usable, "framework": chosen}


def table_layout(graph: dict | Graph, node_id: str) -> dict:
    """Rows, columns and values for a table parameter's grid editor (probability tables)."""
    from ai_made_easy.core.pgm.network import layout

    g = _graph(graph)
    if node_id not in g.nodes:
        raise ApiError(f"no block {node_id!r} in the design")
    return layout(g, node_id)


def list_samples() -> dict:
    names = sorted(p.name for p in SAMPLES_DIR.glob("*.json"))
    return {"count": len(names), "samples": names}


def read_sample(name: str) -> dict:
    import json

    path = SAMPLES_DIR / Path(name).name
    if not path.exists():
        raise ApiError(f"no sample named {name!r}")
    return json.loads(path.read_text(encoding="utf-8"))
