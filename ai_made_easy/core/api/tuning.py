"""Tune and inspect a run (3.8): learning rates, train more, results by group."""
from __future__ import annotations

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, manager


def find_learning_rate(graph: dict | Graph, *, start: float | None = None,
                       end: float | None = None, steps: int | None = None,
                       project: str = "", framework: str = "auto") -> dict:
    """Start a learning-rate range test on the design (a short run on this computer), with
    the framework it trains with (PyTorch or Keras)."""
    from ai_made_easy.core import lr_find

    try:
        run_id = lr_find.start(manager(), _graph(graph), start=start or lr_find.START,
                               end=end or lr_find.END, steps=steps or lr_find.STEPS,
                               project=project, framework=framework or "auto")
    except (lr_find.LRFindError, ValueError) as exc:
        raise ApiError(str(exc)) from exc
    return {"run_id": run_id}


def learning_rate_result(run_id: str) -> dict:
    """The range test's state, its loss curve and the suggested learning rate."""
    from ai_made_easy.core import lr_find

    return lr_find.result(manager(), run_id)


def apply_learning_rate(graph: dict | Graph, lr: float) -> dict:
    """The design with its optimizer's learning rate set."""
    from ai_made_easy.core import lr_find

    try:
        return lr_find.apply(_graph(graph), float(lr)).to_dict()
    except (lr_find.LRFindError, TypeError, ValueError) as exc:
        raise ApiError(str(exc)) from exc


def train_more(run_id: str, epochs: int, lr_scale: float = 1.0,
               trees: int | None = None) -> dict:
    """Continue a finished network for more epochs, or a boosted model for more trees, as a
    new run."""
    from ai_made_easy.core import train_more as tm

    try:
        return {"run_id": tm.start(manager(), run_id, int(epochs), float(lr_scale),
                                   None if trees is None else int(trees)),
                "continues": run_id}
    except (tm.TrainMoreError, FileNotFoundError) as exc:
        raise ApiError(str(exc)) from exc


def resume_info(run_id: str) -> dict:
    """Whether a run can resume, and after which epoch (``reason`` says why not)."""
    from ai_made_easy.core import resume

    try:
        return {"run_id": run_id, "resumable": True,
                "after_epoch": resume.check(manager(), run_id)}
    except resume.ResumeError as exc:
        return {"run_id": run_id, "resumable": False, "reason": str(exc)}


def resume_run(run_id: str) -> dict:
    """Continue a stopped, failed or interrupted run after its last saved epoch."""
    from ai_made_easy.core import resume

    try:
        return {"run_id": run_id, "after_epoch": resume.start(manager(), run_id)}
    except resume.ResumeError as exc:
        raise ApiError(str(exc)) from exc


def group_columns(run_id: str) -> dict:
    """The columns of a finished run's table that results can be split by."""
    from ai_made_easy.core import groups

    try:
        return groups.columns(manager(), run_id)
    except (groups.GroupsError, FileNotFoundError) as exc:
        raise ApiError(str(exc)) from exc


def results_by_group(run_id: str, column: str) -> dict:
    """Score a finished run's test rows for each value of ``column`` (a new short run)."""
    from ai_made_easy.core import groups

    try:
        return {"run_id": groups.start(manager(), run_id, column), "source": run_id,
                "column": column}
    except (groups.GroupsError, FileNotFoundError) as exc:
        raise ApiError(str(exc)) from exc


def group_result(run_id: str) -> dict:
    """Per-group metrics, each group's gap to the whole test set, and flagged groups."""
    from ai_made_easy.core import groups

    return groups.result(manager(), run_id)
