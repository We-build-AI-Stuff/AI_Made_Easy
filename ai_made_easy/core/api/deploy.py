"""Serving packages and the wizard's recipes."""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, _sweeps, manager, sweep_store


def _facts(facts: dict | None = None, path: str | None = None, target: str = "",
           task: str = ""):  # noqa: ANN202 — DataFacts
    from ai_made_easy.core.recipes import DataFacts, detect

    if path:
        path = _local(path)
        found = detect(path, target=target, task=task)
        if found.error:
            raise ApiError(found.error)
        return found
    return DataFacts.from_dict(facts or {})


def wizard_tasks() -> dict:
    """Every task with the data modalities it has recipes for and AutoML support."""
    from ai_made_easy.core.automl import TASK_METRICS
    from ai_made_easy.core.families import get_family
    from ai_made_easy.core.recipes import all_recipes, modalities_for
    from ai_made_easy.core.tasks import all_tasks

    all_recipes()
    rows = []
    for t in all_tasks():
        recipes = [r for r in all_recipes() if t.id in r.tasks]
        if not recipes:
            continue
        kinds = sorted({k for r in recipes for k in r.data_kinds if k != "demo"})
        rows.append({**t.to_dict(), "family_label": get_family(t.family).label,
                     "modalities": list(dict.fromkeys(r.modality for r in recipes)),
                     "demo_modalities": modalities_for(t.id),
                     "data_kinds": kinds, "recipes": len(recipes),
                     "automl": t.id in TASK_METRICS,
                     "metric": TASK_METRICS.get(t.id, ("", ""))[0]})
    return {"tasks": rows}


def _local(path: str) -> str:
    """A Kaggle / Hub source (``kaggle:owner/name``, ``hf:repo``) becomes its downloaded
    file or folder; anything else is returned unchanged."""
    from ai_made_easy.core.integrations import datasets

    if not datasets.is_source(path):
        return path
    try:
        return str(datasets.fetch(path))
    except datasets.SourceError as exc:
        raise ApiError(str(exc)) from exc


def detect_data(path: str, target: str = "", task: str = "") -> dict:
    """Format, size, target, classes and the tasks the data at ``path`` fits. ``path``
    may also be a Kaggle or Hub source (``kaggle:owner/name``, ``hf:repo``), downloaded
    into the cache first."""
    from ai_made_easy.core.recipes import detect

    path = _local(path)
    if not Path(path).expanduser().exists():
        raise ApiError(f"{path} does not exist")
    return detect(path, target=target, task=task).to_dict()


def list_recipes(task: str | None = None) -> dict:
    from ai_made_easy.core.recipes import all_recipes

    return {"recipes": [r.to_dict() for r in all_recipes() if task is None or task in r.tasks]}


def recommend_recipes(task: str, *, facts: dict | None = None, path: str | None = None,
                      target: str = "", budget: dict | None = None, modality: str = "",
                      limit: int | None = None) -> dict:
    """Recipes for ``task`` ranked for the data (``path`` or detected ``facts``; none:
    demo data) and the budget, each with reasons, a validated graph and its costs."""
    from ai_made_easy.core import recipes

    found = _facts(facts, path, target, task)
    try:
        ranked = recipes.recommend(task, found, budget or None, modality=modality or None,
                                   limit=limit)
    except KeyError as exc:
        raise ApiError(f"unknown task {task!r}") from exc
    return {"facts": found.to_dict(), "suggestions": [s.to_dict() for s in ranked]}


def build_recipe(recipe_id: str, task: str = "", *, facts: dict | None = None,
                 path: str | None = None, target: str = "", knobs: dict | None = None,
                 budget: dict | None = None) -> dict:
    from ai_made_easy.core import recipes

    found = _facts(facts, path, target, task)
    try:
        return recipes.build(recipe_id, task or None, found, knobs, budget or None).to_dict()
    except recipes.RecipeError as exc:
        raise ApiError(str(exc)) from exc


def explain_design(graph: dict | Graph) -> dict:
    """Why each block is in the design (recipe notes, else the block's description)."""
    from ai_made_easy.core.recipes import explain

    return explain(_graph(graph))


def start_automl(spec: dict, project: str = "", listener=None) -> dict:  # noqa: ANN001
    """Search recipes and their settings for ``spec["task"]``; returns a sweep_id
    (poll get_sweep / automl_leaderboard)."""
    from ai_made_easy.core import automl

    spec = dict(spec)
    if spec.get("path"):
        spec["facts"] = _facts(None, spec.pop("path"), spec.pop("target", ""),
                               spec.get("task", "")).to_dict()
    try:
        runner = automl.start(manager(), sweep_store(), spec, project=project,
                              listener=listener)
    except (automl.AutoMLError, TypeError) as exc:
        raise ApiError(str(exc)) from exc
    _sweeps[runner.sweep_id] = runner
    return {"sweep_id": runner.sweep_id, "candidates": [r.id for r in runner.recipes],
            "metric": runner.spec.metric, "direction": runner.spec.direction,
            "objectives": runner.automl.objectives, "device": runner.automl.target_device()}


def automl_leaderboard(sweep_id: str) -> dict:
    from ai_made_easy.core.automl import leaderboard

    record = sweep_store().get(sweep_id)
    return {"sweep_id": sweep_id, "state": record.state, "metric": record.spec.get("metric"),
            "direction": record.spec.get("direction"), "best": record.best,
            "objectives": record.spec.get("objectives") or [],
            "device": record.spec.get("device"), "trials": leaderboard(record)}
