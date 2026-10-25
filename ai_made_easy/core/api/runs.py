"""Training runs: start, follow, stop, resume, results."""
from __future__ import annotations

from typing import Any

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, manager


def start_training(graph: dict | Graph, framework: str = "auto", project: str = "",
                   tags: list[str] | None = None, on: str | None = None) -> dict:
    """Train the design here, or ``on`` a backend (``ssh:<host>``, ``kaggle``, …)."""
    from ai_made_easy.core.runner.backends import BackendError

    g = _graph(graph)
    errors = [i for i in g.validate() if i.severity == "error"]
    if errors:
        raise ApiError("the graph has errors: " + "; ".join(i.message for i in errors[:5]))
    try:
        run_id = manager().start(g, framework, project=project, tags=tags, on=on)
    except BackendError as exc:
        raise ApiError(str(exc)) from exc
    return {"run_id": run_id, "backend": on or "local"}


def list_backends() -> dict:
    """Where runs can train: this computer, saved backends and the built-in kinds."""
    from ai_made_easy.core.runner import backends

    return {"backends": backends.listing(), "kinds": list(backends.KINDS),
            "options": {kind: cls.options_help
                        for kind, cls in backends._backend_classes().items()}}


def test_backend(on: str) -> dict:
    """Check a backend works (connection, sign-in) and what accelerator it has."""
    from ai_made_easy.core.runner import backends

    try:
        backend = backends.resolve(on)
    except backends.BackendError as exc:
        raise ApiError(str(exc)) from exc
    return {"backend": backend.spec, "kind": backend.kind, **backend.test()}


def save_backend(name: str, kind: str, options: dict | None = None) -> dict:
    """Save a backend under a name (options only: hosts, GPU types; never secrets)."""
    from ai_made_easy.core.runner import backends

    try:
        return backends.save(name, kind, **(options or {}))
    except backends.BackendError as exc:
        raise ApiError(str(exc)) from exc


def delete_backend(name: str) -> dict:
    from ai_made_easy.core.runner import backends

    try:
        backends.remove(name)
    except backends.BackendError as exc:
        raise ApiError(str(exc)) from exc
    return {"deleted": name}


def attach_run(run_id: str) -> dict:
    """Follow a remote run again (after a restart or a lost connection)."""
    from ai_made_easy.core.runner.backends import BackendError

    try:
        return manager().attach(run_id)
    except (BackendError, KeyError) as exc:
        raise ApiError(str(exc).strip("'\"")) from exc


def import_run_results(path: str) -> dict:
    """Complete a Colab hand-off run from the results zip its notebook downloaded."""
    try:
        return manager().import_results(path)
    except (ValueError, KeyError, OSError) as exc:
        raise ApiError(str(exc).strip("'\"")) from exc


def run_status(run_id: str) -> dict:
    return manager().status(run_id)


def run_metrics(run_id: str) -> dict:
    return {"run_id": run_id, "epochs": manager().metrics(run_id)}


def stop_run(run_id: str) -> dict:
    return manager().stop(run_id)


def list_runs(project: str | None = None, parent: str | None = None) -> dict:
    from ai_made_easy.core.train_more import boosted

    records = manager().history.list(project=project, parent=parent)
    rows = []
    for r in records:
        row = r.to_dict()
        row["boosted"] = r.framework == "sklearn" and boosted(r.graph)   # Train more: trees
        row.pop("graph", None)
        row.pop("params", None)
        rows.append(row)
    return {"count": len(rows), "runs": rows}


def get_run(run_id: str) -> dict:
    data = manager().history.get(run_id).to_dict()
    data["epochs"] = manager().history.epochs(run_id)
    data["artifacts"] = sorted(p.name for p in manager().history.path(run_id).iterdir())
    return data


def diff_designs(before: dict, after: dict) -> dict:
    """What changed from ``before`` to ``after`` (blocks, parameters, wires)."""
    from ai_made_easy.core import diff

    return diff.diff(before, after)


def diff_run(run_id: str, design: dict) -> dict:
    """The current ``design`` compared with the design a run trained."""
    record = manager().history.get(run_id)
    return {"run_id": run_id, **diff_designs(record.graph or {}, design)}


def run_explanation(run_id: str) -> dict:
    """The feature importance a run saved (``eval.permutation_importance``), else empty."""
    import json as _json

    try:
        data = _json.loads((manager().history.path(run_id) / "explain.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"run_id": run_id, "features": []}
    return {"run_id": run_id, **data}


def run_samples(run_id: str, limit: int = 12) -> dict:
    """Sample images (PNG data URLs) and texts a run saved, with per-class scores."""
    import base64
    import json as _json

    folder = manager().history.path(run_id)
    files = sorted((folder / "eval_samples").glob("*.png"))[:limit]
    metrics, classes = {}, []
    for name, default in (("metrics.json", {}), ("classes.json", [])):
        try:
            value = _json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = default
        if name == "metrics.json":
            metrics = value
        else:
            classes = value
    per_class = metrics.get("per_class_ap") or metrics.get("per_class_iou") or []
    texts = [{"name": f.stem, "text": f.read_text(encoding="utf-8", errors="replace")[:20000]}
             for f in sorted((folder / "eval_samples").glob("*.txt"))[:limit]]
    return {"run_id": run_id, "classes": classes, "per_class": per_class,
            "per_class_metric": ("AP" if "per_class_ap" in metrics else
                                 "IoU" if "per_class_iou" in metrics else ""),
            "samples": [{"name": f.stem, "data_url": "data:image/png;base64,"
                         + base64.b64encode(f.read_bytes()).decode()} for f in files],
            "texts": texts}


def update_run(run_id: str, *, tags: list[str] | None = None, note: str | None = None) -> dict:
    fields: dict[str, Any] = {}
    if tags is not None:
        fields["tags"] = [str(t) for t in tags]
    if note is not None:
        fields["note"] = str(note)
    return manager().history.update(run_id, **fields).to_dict()


def delete_run(run_id: str) -> dict:
    mgr = manager()
    if mgr.is_live(run_id) and mgr.status(run_id)["state"] == "running":
        raise ApiError("stop the run before deleting it")
    mgr.history.delete(run_id)
    return {"deleted": run_id}


def compare_runs(run_ids: list[str]) -> dict:
    from ai_made_easy.core.runs.history import compare

    if len(run_ids) < 2:
        raise ApiError("select at least two runs to compare")
    history = manager().history
    result = compare([history.get(r) for r in run_ids])
    result["epochs"] = {r: history.epochs(r) for r in run_ids}
    return result
