"""Headless application API: the one surface the CLI, the MCP server and the
web server call. Graphs travel as JSON dicts (the project-file format);
every function returns JSON-serialisable data. Pure Python, no Qt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_made_easy.core.graph import Graph

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

_manager = None


class ApiError(ValueError):
    """A request the API cannot honour (unknown target, bad input …)."""


def manager():
    """Process-wide RunManager (lazy, so importing the API stays cheap)."""
    global _manager
    if _manager is None:
        from ai_made_easy.core.runner.manager import RunManager

        _manager = RunManager()
    return _manager


def set_manager(mgr) -> None:  # noqa: ANN001 — tests and servers inject their own
    global _manager
    _manager = mgr


def _graph(graph: dict | Graph) -> Graph:
    if isinstance(graph, Graph):
        return graph
    if not isinstance(graph, dict):
        raise ApiError("graph must be a project JSON object")
    return Graph.from_dict(graph)


# ---------------------------------------------------------------- blocks

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
    from ai_made_easy.core.families import family_of, resolve_framework
    from ai_made_easy.core.tasks import task_of

    g = _graph(graph)
    family = family_of(g)
    try:
        task = task_of(g)
    except Exception:  # noqa: BLE001 — incomplete designs have no task yet
        task = None
    layers = []   # blocks that become code in a framework (not data, training, I/O)
    for node in g.nodes.values():
        try:
            supports = node.definition().supports
        except Exception:  # noqa: BLE001 — unknown blocks make no claim either way
            continue
        if supports("pytorch") or supports("keras"):
            layers.append(supports)
    usable = [fw for fw in family.frameworks if all(s(fw) for s in layers)]
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


# ----------------------------------------------------------------- graph

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


# ------------------------------------------------------------------ runs

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


# ---------------------------------------------------------------- sweeps

_sweeps: dict = {}
_sweep_store = None


def sweep_store():
    global _sweep_store
    from ai_made_easy.core.sweeps import SweepStore

    if _sweep_store is None or _sweep_store.root != SweepStore().root:
        _sweep_store = SweepStore()
    return _sweep_store


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


# ---------------------------------------------------------------- 3.8: tune and inspect

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


# ---------------------------------------------------------------- deploy

# ------------------------------------------------------------------ wizard / recipes

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


# ------------------------------------------------------------------ pipelines

_pipelines: dict = {}
_pipeline_store = None


def pipeline_store():
    global _pipeline_store
    from ai_made_easy.core.pipelines.runner import PipelineStore

    if _pipeline_store is None or _pipeline_store.root != PipelineStore().root:
        _pipeline_store = PipelineStore()
    return _pipeline_store


def start_pipeline(graph: dict | Graph, project: str = "", base: str | None = None,
                   listener=None, on: str = "") -> dict:  # noqa: ANN001
    """Run a pipeline's stages in order (finished, unchanged stages are reused);
    ``on`` sends its training stages to a backend."""
    from ai_made_easy.core.pipelines.runner import PipelineError, PipelineRunner

    try:
        runner = PipelineRunner(manager(), pipeline_store(), _graph(graph), project, base,
                                listener, on=on)
    except PipelineError as exc:
        raise ApiError(str(exc)) from exc
    _pipelines[runner.pipeline_id] = runner
    runner.start()
    return {"pipeline_id": runner.pipeline_id, "stages": runner.record.order}


def get_pipeline(pipeline_id: str) -> dict:
    return pipeline_store().get(pipeline_id).to_dict()


def list_pipelines(project: str | None = None) -> dict:
    rows = []
    for rec in pipeline_store().list(project):
        row = rec.to_dict()
        row.pop("graph", None)
        rows.append(row)
    return {"count": len(rows), "pipelines": rows}


def stop_pipeline(pipeline_id: str) -> dict:
    runner = _pipelines.get(pipeline_id)
    if runner is None:
        raise ApiError("that pipeline is not running in this process")
    runner.stop()
    return {"pipeline_id": pipeline_id, "stopping": True}


def resume_pipeline(pipeline_id: str, base: str | None = None, listener=None) -> dict:  # noqa: ANN001
    """Run a pipeline again from its last finished stage."""
    from ai_made_easy.core.pipelines.runner import PipelineError, resume

    try:
        runner = resume(manager(), pipeline_store(), pipeline_id, base, listener)
    except PipelineError as exc:
        raise ApiError(str(exc)) from exc
    _pipelines[runner.pipeline_id] = runner
    return {"pipeline_id": runner.pipeline_id, "resumed_from": pipeline_id}


def wait_pipeline(pipeline_id: str, timeout: float = 3600.0) -> dict:
    runner = _pipelines.get(pipeline_id)
    if runner is not None:
        runner.wait(timeout)
    return get_pipeline(pipeline_id)


def model_registry():
    from ai_made_easy.core.deploy import ModelRegistry

    return ModelRegistry()


def deploy_formats() -> dict:
    from ai_made_easy.core.deploy import FORMAT_LABELS, FORMATS

    return {fw: [{"id": f, "label": FORMAT_LABELS[f]} for f in fmts]
            for fw, fmts in FORMATS.items()}


def export_options(run_id: str | None = None, *, model: str | None = None,
                   version: str | int = "latest", graph: dict | None = None,
                   framework: str | None = None) -> dict:
    """Export formats for a run, a registered model or a design: each with
    ``available`` / ``reason`` (what the design can't export) and ``suggested`` (what
    the project's budget device runs)."""
    from ai_made_easy.core.deploy import exportability
    from ai_made_easy.core.graph import Graph

    if run_id:
        rows = exportability.options_for_run(manager().history.path(run_id))
    elif model:
        registry = model_registry()
        mv = registry.get(model, version)
        rows = exportability.options_for_run(registry.path(model, mv.version))
    elif graph is not None:
        rows = exportability.options(Graph.from_dict(graph), framework)
    else:
        raise ApiError("give a run, a registered model or a design")
    return {"formats": rows}


def _package(source: Path, out_dir: str, formats, name=None, version="1") -> dict:  # noqa: ANN001
    from ai_made_easy.core.deploy import DeployError, build_package

    try:
        return build_package(source, out_dir, formats=tuple(formats or ()), name=name,
                             version=str(version), python=manager().python).to_dict()
    except DeployError as exc:
        raise ApiError(str(exc)) from exc


def deploy_run(run_id: str, out_dir: str, formats: list[str] | None = None,
               name: str | None = None) -> dict:
    """Build a model-server package (FastAPI + Dockerfile) from a finished run."""
    return _package(manager().history.path(run_id), out_dir, formats, name)


def register_model(run_id: str, name: str, description: str = "",
                   also: list[str] | None = None) -> dict:
    """Register a finished run; ``also`` mirrors the version to ``mlflow`` / ``wandb``."""
    from ai_made_easy.core.deploy import RegistryError

    registry = model_registry()
    try:
        mv = registry.register(manager().history.path(run_id), name, description)
        if also:
            mv = registry.mirror(name, mv.version, also)
        return mv.to_dict()
    except RegistryError as exc:
        raise ApiError(str(exc)) from exc


def list_models() -> dict:
    rows = [v.to_dict() for v in model_registry().list()]
    return {"count": len(rows), "models": rows}


def set_model_stage(name: str, version: int, stage: str,
                    also: list[str] | None = None) -> dict:
    """Promote a version; ``also`` registers it in MLflow / W&B with the same stage."""
    from ai_made_easy.core.deploy import RegistryError

    registry = model_registry()
    try:
        mv = registry.set_stage(name, int(version), stage)
        if also:
            mv = registry.mirror(name, mv.version, also)
        return mv.to_dict()
    except RegistryError as exc:
        raise ApiError(str(exc)) from exc


def mirror_model(name: str, version: str | int, to: list[str]) -> dict:
    """Register an existing version in MLflow's model registry and / or W&B."""
    from ai_made_easy.core.deploy import RegistryError

    registry = model_registry()
    try:
        mv = registry.get(name, version)
        return registry.mirror(name, mv.version, to).to_dict()
    except RegistryError as exc:
        raise ApiError(str(exc)) from exc


# ---------------------------------------------------------------- tools you use

def tracking_status(graph: dict | None = None) -> dict:
    """Which trackers a design (or every run, ``$AIME_TRACKING``) mirrors to, and
    whether their packages are installed."""
    import importlib.util
    import os

    from ai_made_easy.core import tracking

    meta = (graph or {}).get("meta") or {}
    try:
        chosen = tracking.selected(meta)
    except tracking.TrackingError as exc:
        raise ApiError(str(exc)) from exc
    return {"selected": chosen, "env": os.environ.get("AIME_TRACKING", ""),
            "mlflow_uri": tracking.mlflow_uri(),
            "trackers": [{"id": t, "label": tracking.LABELS[t],
                          "installed": importlib.util.find_spec(t) is not None}
                         for t in tracking.TRACKERS]}


def export_runs(out: str, *, project: str | None = None, fmt: str = "",
                epochs: bool = False) -> dict:
    """Write run records (or every epoch, ``epochs=True``) as CSV / Parquet."""
    from ai_made_easy.core.runs.export import export_runs as _export

    try:
        return _export(manager().history, out, project=project, fmt=fmt, epochs=epochs)
    except ValueError as exc:
        raise ApiError(str(exc)) from exc


def open_mlflow_model(uri: str, input_shape: list[int] | None = None, name: str = "") -> dict:
    """Open an MLflow PyTorch / ONNX model as a design (through the importers)."""
    from ai_made_easy.core import tracking

    try:
        kind, file = tracking.open_mlflow_model(uri)
    except tracking.TrackingError as exc:
        raise ApiError(str(exc)) from exc
    return import_model("onnx" if kind == "onnx" else "pytorch", str(file),
                        input_shape=input_shape, name=name or uri.rstrip("/").split("/")[-1])


def _source_dir(run_id: str | None, model: str | None, version: str | int) -> Path:
    if run_id:
        return manager().history.path(run_id)
    if model:
        registry = model_registry()
        mv = registry.get(model, version)
        return registry.path(model, mv.version)
    raise ApiError("give a run or a registered model")


def push_to_hub(repo_id: str, *, run_id: str | None = None, model: str | None = None,
                version: str | int = "latest", private: bool = True) -> dict:
    """Push a trained run (or a registered version) to the Hugging Face Hub as a model
    repo with its model card. Sign-in: ``huggingface-cli login``."""
    from ai_made_easy.core.integrations import hub

    try:
        return hub.push_model(_source_dir(run_id, model, version), repo_id, private=private)
    except hub.HubError as exc:
        raise ApiError(str(exc)) from exc


def pull_from_hub(repo_id: str, name: str = "", revision: str | None = None) -> dict:
    """Download a model AI Made Easy pushed to the Hub and register it locally."""
    from ai_made_easy.core.integrations import hub

    try:
        folder = hub.pull_model(repo_id, revision=revision)
        mv = model_registry().register(folder, name or hub.model_name(repo_id),
                                       f"from the Hugging Face Hub: {repo_id}")
    except (hub.HubError, ValueError) as exc:
        raise ApiError(str(exc)) from exc
    return mv.to_dict()


def deploy_to_space(space_id: str, *, run_id: str | None = None, model: str | None = None,
                    version: str | int = "latest", formats: list[str] | None = None,
                    private: bool = True) -> dict:
    """Build the FastAPI package and deploy it as a Hugging Face Docker Space."""
    import tempfile

    from ai_made_easy.core.integrations import hub

    source = _source_dir(run_id, model, version)
    with tempfile.TemporaryDirectory(prefix="aime_space_") as tmp:
        built = _package(source, str(Path(tmp) / "package"), formats)
        try:
            return {**hub.deploy_space(Path(built["path"]), space_id, private=private),
                    "package": built}
        except hub.HubError as exc:
            raise ApiError(str(exc)) from exc


def fetch_dataset(source: str, *, force: bool = False) -> dict:
    """Download a Kaggle dataset / competition file or a Hub dataset into the cache
    (``kaggle:owner/name``, ``kaggle-competition:name[/file]``, ``hf:repo[:split]``)
    and detect it like a local folder."""
    from ai_made_easy.core.integrations import datasets

    try:
        path = datasets.fetch(source, force=force)
    except datasets.SourceError as exc:
        raise ApiError(str(exc)) from exc
    return {"source": source, "path": str(path), "facts": detect_data(str(path))}


def store_upload(path: str, filename: str) -> dict:
    """A data file uploaded through the web UI: stored, then detected like a local path."""
    from ai_made_easy.core.data import uploads
    from ai_made_easy.core.runner.manager import max_unpacked_bytes

    try:
        stored = uploads.store(Path(path), filename, max_unpacked=max_unpacked_bytes())
    except uploads.UploadError as exc:
        raise ApiError(str(exc)) from exc
    return {"source": filename, "path": str(stored), "facts": detect_data(str(stored))}


def submit_to_kaggle(competition: str, file: str, message: str, *,
                     confirm: bool = False) -> dict:
    """Submit predictions to a Kaggle competition. Every caller must ask the user first
    and pass ``confirm=True``: a submission counts against the daily limit."""
    from ai_made_easy.core.integrations import datasets

    if not confirm:
        raise ApiError("submitting to Kaggle needs confirmation (confirm=True) each time")
    try:
        return datasets.submit(competition, file, message)
    except datasets.SourceError as exc:
        raise ApiError(str(exc)) from exc


def delete_model(name: str, version: int) -> dict:
    model_registry().delete(name, int(version))
    return {"deleted": f"{name}/{version}"}


def deploy_model(name: str, version: str | int, out_dir: str,
                 formats: list[str] | None = None) -> dict:
    """Package a registered model version ('latest', 'production' or a number)."""
    registry = model_registry()
    mv = registry.get(name, version)
    return _package(registry.path(name, mv.version), out_dir, formats, name, mv.version)


# ---------------------------------------------------------------- import

def import_model(kind: str, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32", kwargs: dict | None = None, name: str = "") -> dict:
    """Import a PyTorch module, ONNX file or Keras model as an editable graph."""
    from ai_made_easy.core.importers import ModelImportError
    from ai_made_easy.core.importers import import_model as _import

    try:
        return _import(kind, source=source, attr=attr, input_shape=input_shape, dtype=dtype,
                       kwargs=kwargs, name=name, python=manager().python).to_dict()
    except ModelImportError as exc:
        raise ApiError(str(exc)) from exc


# ---------------------------------------------------------------- data

_profiles = None


def profile_cache():
    global _profiles
    if _profiles is None:
        from ai_made_easy.core.data.lints import ProfileCache

        _profiles = ProfileCache()
    return _profiles


def _dataset_node(graph: Graph, node_id: str | None):
    from ai_made_easy.core.data.lints import dataset_nodes

    nodes = dataset_nodes(graph)
    if node_id:
        nodes = [n for n in nodes if n[0] == node_id]
    if not nodes:
        raise ApiError(f"no dataset block {node_id!r}" if node_id else
                       "the design has no dataset block")
    return nodes[0]


def profile_data(graph: dict | Graph | None = None, node_id: str | None = None, *,
                 path: str | None = None, target: str | None = None,
                 base: str | None = None) -> dict:
    """Profile the dataset behind a dataset block, or any table / folder ``path``."""
    from ai_made_easy.core.data.profile import profile_path

    if path:
        if not Path(path).expanduser().exists():
            raise ApiError(f"{path} does not exist")
        return profile_path(path, target=target).to_dict()
    if graph is None:
        raise ApiError("pass a graph or a path")
    _nid, type_id, params = _dataset_node(_graph(graph), node_id)
    return profile_cache().profile(type_id, params, base).to_dict()


def data_issues(graph: dict | Graph, base: str | None = None) -> dict:
    """Data warnings (missing values, leakage, imbalance, ...) for the design's datasets."""
    from ai_made_easy.core.data.lints import data_issues as _issues

    issues = _issues(_graph(graph), profile_cache(), base)
    return {"issues": [{"severity": i.severity, "message": i.message, "node_id": i.node_id}
                       for i in issues]}


def split_preview(graph: dict | Graph, base: str | None = None) -> dict:
    """Samples per class in train / validation / test, as the training script splits."""
    from ai_made_easy.core.data.lints import task_of
    from ai_made_easy.core.data.splits import split_preview as _preview
    from ai_made_easy.core.training import data_catalog as dcat

    g = _graph(graph)
    _nid, type_id, params = _dataset_node(g, None)
    split = next((dict(n.resolved_params()) for n in g.nodes.values()
                  if n.type_id == "prep.split"), None)
    if split is None:
        split = {p.name: p.default for p in dcat.BLOCKS["prep.split"].params}
    result = _preview(type_id, params, split, base=base, task=task_of(g) or "multiclass",
                      modality=dcat.BLOCKS[type_id].modality or "")
    if isinstance(result, str):
        raise ApiError(result)
    return result.to_dict()


def data_fingerprint(graph: dict | Graph, base: str | None = None) -> dict:
    from ai_made_easy.core.data.fingerprint import fingerprint

    _nid, type_id, params = _dataset_node(_graph(graph), None)
    return {"fingerprint": fingerprint(type_id, params, base)}


def augmentation_preview(graph: dict | Graph, out_dir: str, images: int = 4,
                         variants: int = 6, base: str | None = None) -> dict:
    """Render evaluation and random training views of dataset images (PNG files)."""
    from ai_made_easy.core.data.augment import PreviewError
    from ai_made_easy.core.data.augment import augmentation_preview as _preview

    try:
        return _preview(_graph(graph), out_dir, base=base, python=manager().python,
                        images=images, variants=variants)
    except PreviewError as exc:
        raise ApiError(str(exc)) from exc


# ------------------------------------------------------------------ try a model (3.5)

_try = None


def try_service():
    """The loaded predictors of recently tried runs (lazy: nothing loads until a try)."""
    global _try
    if _try is None:
        import atexit

        from ai_made_easy.core.tryit import TryService

        _try = TryService()
        atexit.register(_try.close)   # no predictor outlives the app or server
    return _try


def try_describe(run_id: str) -> dict:
    """What a finished run takes, to build a form: the input kind and help, an example, and
    the signature (input shape, classes, table columns). Loads the run's predictor."""
    from ai_made_easy.core import tryit

    try:
        return tryit.describe(try_service(), manager().history, run_id)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc


def try_run(run_id: str, inputs: list, explain: bool = False) -> dict:
    """Predictions of a finished run for ``inputs`` (the same requests as its model server:
    rows, texts, base64 images, WAV files...), computed on this machine. ``explain`` adds a
    Grad-CAM heatmap to an image classifier's predictions."""
    from ai_made_easy.core import tryit

    try:
        return tryit.predict(try_service(), manager().history, run_id, inputs, explain=explain)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc


def try_file(run_id: str, filename: str, data: bytes) -> dict:
    """Predictions of a finished run for every input in a file (a CSV table, a text file,
    or a zip of images or WAV clips) as CSV text: ``{"csv", "rows", "filename"}``."""
    from ai_made_easy.core import tryit

    try:
        return tryit.predict_file(try_service(), manager().history, run_id, filename, data)
    except tryit.TryError as exc:
        raise ApiError(str(exc)) from exc
