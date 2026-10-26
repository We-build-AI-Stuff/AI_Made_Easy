"""Hyperparameter sweeps."""
from __future__ import annotations

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, _sweeps, manager, sweep_store


def sweepable_params(graph: dict | Graph) -> dict:
    from ai_made_easy.core.sweeps import sweepable_params as _params

    return {"params": _params(_graph(graph))}


def start_sweep(graph: dict | Graph, spec: dict, project: str = "",
                listener=None) -> dict:  # noqa: ANN001
    from ai_made_easy.core.sweeps import SweepError, SweepRunner, SweepSpec

    g = _graph(graph)
    errors = [i for i in g.validate() if i.severity == "error"]
    if errors:
        raise ApiError("the graph has errors: " + "; ".join(i.message for i in errors[:5]))
    try:
        runner = SweepRunner(manager(), sweep_store(), g, SweepSpec.from_dict(spec),
                             project=project, listener=listener)
    except (SweepError, TypeError) as exc:
        raise ApiError(str(exc)) from exc
    _sweeps[runner.sweep_id] = runner
    runner.start()
    return {"sweep_id": runner.sweep_id}


def get_sweep(sweep_id: str) -> dict:
    return sweep_store().get(sweep_id).to_dict()


def list_sweeps(project: str | None = None) -> dict:
    rows = []
    for rec in sweep_store().list(project):
        row = rec.to_dict()
        row.pop("graph", None)
        rows.append(row)
    return {"count": len(rows), "sweeps": rows}


def stop_sweep(sweep_id: str) -> dict:
    runner = _sweeps.get(sweep_id)
    if runner is None:
        raise ApiError("that sweep is not running in this process")
    runner.stop()
    return {"sweep_id": sweep_id, "stopping": True}


def wait_sweep(sweep_id: str, timeout: float = 3600.0) -> dict:
    runner = _sweeps.get(sweep_id)
    if runner is not None:
        runner.wait(timeout)
    return get_sweep(sweep_id)


def sweep_analysis(sweep_id: str) -> dict:
    """Which settings mattered: parameter importance and the parallel-coordinates data."""
    from ai_made_easy.core.sweep_analysis import analyze

    return {"sweep_id": sweep_id, **analyze(sweep_store().get(sweep_id))}


def sweep_again(sweep_id: str) -> dict:
    """A follow-up sweep narrowed around the best trials (3.7): the spec, the design and
    what changed, for review before starting it."""
    from ai_made_easy.core.sweep_analysis import narrowed

    try:
        return {"sweep_id": sweep_id, **narrowed(sweep_store().get(sweep_id))}
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


def sweep_best_graph(sweep_id: str) -> dict:
    from ai_made_easy.core.sweeps import SweepError, best_graph

    try:
        return best_graph(sweep_store().get(sweep_id))
    except SweepError as exc:
        raise ApiError(str(exc)) from exc
