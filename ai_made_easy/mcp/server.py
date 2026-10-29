"""MCP server: expose the AI Made Easy engine as Model Context Protocol tools.

Any MCP client (Claude Desktop, ZCode, ...) can design, inspect, generate
code for, and train models by driving the exact same pure-Python core the
desktop UI uses. Graphs travel as JSON — the same files the app saves.

Run:    aime-mcp            (stdio transport, for MCP client configs)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_made_easy.core import api

mcp = FastMCP("ai-made-easy")

SAMPLES_DIR = api.SAMPLES_DIR


# ------------------------------------------------------------ pure impls
# Thin JSON adapters over core.api (kept as functions so tests and other
# hosts can call them without the MCP transport).

def _err(exc: Exception) -> str:
    return json.dumps({"error": str(exc)})


def _impl_list_blocks(category: str | None = None) -> str:
    return json.dumps(api.list_blocks(category))


def _impl_list_samples() -> str:
    return json.dumps(api.list_samples())


def _impl_read_sample(name: str) -> str:
    try:
        return json.dumps(api.read_sample(name))
    except api.ApiError as exc:
        return _err(exc)


def _impl_validate(graph: dict) -> dict:
    return api.validate(graph)


def _impl_generate(graph: dict, target: str = "pytorch_model") -> str:
    try:
        return api.generate(graph, target)
    except api.ApiError as exc:
        return _err(exc)


def _impl_summarize(graph: dict) -> dict:
    return api.summarize(graph)


def _impl_expand(graph: dict, node_id: str) -> dict:
    return api.expand(graph, node_id)


def _impl_start_training(graph: dict, framework: str = "auto", on: str = "") -> dict:
    result = api.start_training(graph, framework, on=on or None)
    result["hint"] = ("poll get_run_status(run_id); metrics arrive per epoch"
                      + ("" if not on or on == "local" else
                         " (Kaggle and Colab: when the remote run reports back)"))
    return result


def _impl_run_status(run_id: str) -> dict:
    return api.run_status(run_id)


def _impl_run_metrics(run_id: str) -> dict:
    return api.run_metrics(run_id)


def _impl_stop_run(run_id: str) -> dict:
    return api.stop_run(run_id)


# ------------------------------------------------------------- MCP tools

@mcp.tool()
def list_blocks(category: str | None = None) -> str:
    """Every registered block with its full JSON schema (params, ports,
    category, composite flag). Filter by category optionally. Start here to
    learn the vocabulary before building graphs."""
    return _impl_list_blocks(category)


@mcp.tool()
def list_samples() -> str:
    """Sample project graphs shipped with the app — good starting points."""
    return _impl_list_samples()


@mcp.tool()
def read_sample(name: str) -> str:
    """Load one sample graph as JSON (pass a name from list_samples)."""
    return _impl_read_sample(name)


@mcp.tool()
def validate_graph(graph: dict[str, Any]) -> str:
    """Validate a block graph: structure, wiring, and shape inference.
    Returns issues with severities; 'valid' is true when no errors exist."""
    return json.dumps(_impl_validate(graph))


@mcp.tool()
def generate_code(graph: dict[str, Any], target: str = "pytorch_model") -> str:
    """Generate code for a graph. Targets: pytorch_model, keras_model,
    pytorch_train, keras_train, sklearn_train (runnable training scripts), llm (LLM workflow
    script from llm.* blocks)."""
    return _impl_generate(graph, target)


@mcp.tool()
def summarize_model(graph: dict[str, Any]) -> str:
    """Analytic model summary: per-layer output shapes and parameter counts
    (no torch needed)."""
    return json.dumps(_impl_summarize(graph))


@mcp.tool()
def expand_architecture(graph: dict[str, Any], node_id: str) -> str:
    """Expand a composite architecture block (arch.*) into primitive blocks.
    Returns the updated graph JSON — codegen needs expanded graphs."""
    return json.dumps(_impl_expand(graph, node_id))


@mcp.tool()
def start_training(graph: dict[str, Any], framework: str = "auto", on: str = "") -> str:
    """Start training the graph (framework: auto, pytorch, keras or sklearn). ``on`` picks
    where: "" / local (this computer), ssh:<host from ~/.ssh/config>, kaggle, colab, modal,
    container or a saved backend's name (see list_backends). Returns a run_id; poll
    get_run_status / get_run_metrics; stop with stop_run. Every run is kept in the run
    history."""
    try:
        return json.dumps(_impl_start_training(graph, framework, on))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def list_backends() -> str:
    """Where runs can train (this computer, SSH hosts, Kaggle, Colab, Modal, containers),
    the saved backends and every kind's options. Sign-in is each tool's own (~/.ssh,
    ~/.kaggle/kaggle.json, modal token, Google Drive for desktop): never pass secrets."""
    return json.dumps(api.list_backends())


@mcp.tool()
def test_backend(on: str) -> str:
    """Check a backend works (connection / sign-in) and report its accelerator and the
    device profile budget checks will use."""
    try:
        return json.dumps(api.test_backend(on))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def save_backend(name: str, kind: str, options: dict[str, Any] | None = None) -> str:
    """Save a backend under a name, e.g. kind "ssh" with {"host": "gpu-box"}, or "kaggle"
    with {"accelerator": "P100"}. Options only (hosts, GPU types); never secrets."""
    try:
        return json.dumps(api.save_backend(name, kind, options))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def attach_run(run_id: str) -> str:
    """Follow a remote run again after the app restarted or the connection dropped."""
    try:
        return json.dumps(api.attach_run(run_id), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def import_run_results(path: str) -> str:
    """Complete a Colab hand-off run from the results zip its notebook downloaded."""
    try:
        return json.dumps(api.import_run_results(path), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def get_run_status(run_id: str) -> str:
    """Current state of a training run (state, epochs done, latest metrics,
    log tail)."""
    return json.dumps(_impl_run_status(run_id))


@mcp.tool()
def get_run_metrics(run_id: str) -> str:
    """All epoch events of a training run (losses + scores per epoch)."""
    return json.dumps(_impl_run_metrics(run_id))


@mcp.tool()
def stop_run(run_id: str) -> str:
    """Stop a running training run."""
    return json.dumps(_impl_stop_run(run_id))


@mcp.tool()
def list_runs(project: str | None = None) -> str:
    """Run history (newest first): status, final and best metrics, duration."""
    return json.dumps(api.list_runs(project))


@mcp.tool()
def compare_runs(run_ids: list[str]) -> str:
    """Compare two or more runs: metrics side by side and the parameters that
    differ between them."""
    try:
        return json.dumps(api.compare_runs(run_ids), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def sweepable_parameters(graph: dict[str, Any]) -> str:
    """Parameters of a graph a hyperparameter sweep can vary, with suggested
    ranges (keys look like 'opt.lr')."""
    return json.dumps(api.sweepable_params(graph), default=str)


@mcp.tool()
def start_sweep(graph: dict[str, Any], spec: dict[str, Any]) -> str:
    """Start a hyperparameter sweep. spec = {"dimensions": [{"node", "param",
    "kind": float|int|choice, "low", "high", "log", "values"}], "metric",
    "direction": min|max, "strategy": tpe|random|grid, "max_trials"}.
    Returns a sweep_id; poll get_sweep."""
    try:
        return json.dumps(api.start_sweep(graph, spec))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def get_sweep(sweep_id: str) -> str:
    """Sweep state, every trial (values, score, run_id) and the best trial."""
    try:
        record = api.get_sweep(sweep_id)
        record.pop("graph", None)
        return json.dumps(record, default=str)
    except KeyError as exc:
        return _err(exc)


@mcp.tool()
def wizard_tasks() -> str:
    """Tasks the new-project wizard supports: their data modalities, the data kinds
    their recipes read and whether AutoML can search them."""
    return json.dumps(api.wizard_tasks(), default=str)


@mcp.tool()
def detect_data(path: str, target: str = "", task: str = "") -> str:
    """Inspect a table file or dataset folder: format, rows, target, classes,
    features and the tasks it fits (best first)."""
    try:
        return json.dumps(api.detect_data(path, target, task), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def recommend_recipes(task: str, path: str = "", target: str = "",
                      budget: dict[str, Any] | None = None, modality: str = "",
                      limit: int = 5) -> str:
    """Rank design recipes for a task on the data at path (empty: demo data) and an
    optional budget ({"device", "max_train_memory_gb", "max_latency_ms",
    "max_params_m"}). Each suggestion has reasons, cautions, a validated graph and
    cost estimates; open the graph or pass a recipe id to build_recipe."""
    try:
        return json.dumps(api.recommend_recipes(task, path=path or None, target=target,
                                                budget=budget, modality=modality,
                                                limit=limit), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def build_recipe(recipe: str, task: str = "", path: str = "", target: str = "",
                 knobs: dict[str, Any] | None = None) -> str:
    """Build one recipe's design (optionally with settings such as {"lr": 0.001,
    "width": 128}) for the data at path; returns the graph."""
    try:
        return json.dumps(api.build_recipe(recipe, task, path=path or None, target=target,
                                           knobs=knobs), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def explain_design(graph: dict[str, Any]) -> str:
    """Why every block is in a design: the recipe's notes, else the block's own
    description, plus the recipe and task it implements."""
    return json.dumps(api.explain_design(graph), default=str)


@mcp.tool()
def start_automl(task: str, path: str = "", target: str = "", max_trials: int = 12,
                 epochs: int = 0, recipes: list[str] | None = None,
                 budget: dict[str, Any] | None = None, objectives: list[str] | None = None,
                 device: str = "", on: str = "", concurrency: int = 1) -> str:
    """AutoML: train every candidate recipe for the task once, then tune recipes and
    their settings with TPE, pruning weak trials early and skipping designs over the
    budget. ``objectives`` (latency_ms, params_m, model_mb, estimated on ``device``)
    makes it multi-objective: the leaderboard then marks the Pareto front (neural and
    classic recipes together; classic costs are measured after training). ``on`` trains
    the trials on a backend (ssh:<host>, kaggle, modal, ...), ``concurrency`` at once.
    Returns a sweep_id; poll automl_leaderboard."""
    spec = {"task": task, "path": path or None, "target": target, "max_trials": max_trials,
            "epochs": epochs, "recipes": recipes or [], "budget": budget or {},
            "objectives": objectives or [], "device": device, "on": on,
            "concurrency": concurrency}
    try:
        return json.dumps(api.start_automl({k: v for k, v in spec.items() if v is not None}))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def automl_leaderboard(sweep_id: str) -> str:
    """An AutoML search's trials, best first, with their recipe, settings and score
    (multi-objective: the Pareto front first, with each trial's costs)."""
    try:
        return json.dumps(api.automl_leaderboard(sweep_id), default=str)
    except KeyError as exc:
        return _err(exc)


@mcp.tool()
def start_pipeline(graph: dict[str, Any], project: str = "", on: str = "") -> str:
    """Run a pipeline design (Train → Fine-tune / Distill / Prune / Quantize → Evaluate →
    Export / Register stages). Finished, unchanged stages from earlier attempts are
    reused; ``on`` sends the training stages to a backend. Returns a pipeline_id; poll
    get_pipeline."""
    try:
        return json.dumps(api.start_pipeline(graph, project, on=on))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def get_pipeline(pipeline_id: str) -> str:
    """A pipeline's state and every stage: state, child run id, metrics and outputs."""
    try:
        record = api.get_pipeline(pipeline_id)
        record.pop("graph", None)
        return json.dumps(record, default=str)
    except KeyError as exc:
        return _err(exc)


@mcp.tool()
def resume_pipeline(pipeline_id: str) -> str:
    """Run a stopped or failed pipeline again from its last finished stage."""
    try:
        return json.dumps(api.resume_pipeline(pipeline_id))
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def export_formats(run_id: str) -> str:
    """The export formats of a finished run: which the design can use (with the reason
    when not) and which the project's budget device runs (suggested)."""
    try:
        return json.dumps(api.export_options(run_id))
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def what_would_change(run_id: str, row: dict, desired: str | None = None,
                      target: float | None = None, locked: list[str] | None = None) -> str:
    """Counterfactuals for a finished run trained on a table: given one row ({column:
    value}), the smallest changes that make it predict ``desired`` (a class; default the
    runner-up) or reach ``target`` (a regressor's value). ``locked`` columns stay fixed.
    Values tried come from the run's table; the run's own model answers."""
    try:
        return json.dumps(api.what_would_change(run_id, row, desired, target, locked),
                          default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def deploy_run(run_id: str, out_dir: str, formats: list[str] | None = None) -> str:
    """Build a model-server package (FastAPI app, Dockerfile, pinned requirements) from a
    finished training run, with optional exports: onnx, onnx_int8, torchscript, coreml,
    openvino, executorch, executorch_int8, gguf, gguf_q8_0, gguf_q4_0 (PyTorch); onnx,
    saved_model, litert, litert_int8 (Keras). Each export is checked against the model (see
    the result's formats)."""
    try:
        return json.dumps(api.deploy_run(run_id, out_dir, formats), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def use_group_thresholds(groups_run_id: str) -> str:
    """Use the thresholds per group a Results-by-group run fitted (two-class classifiers)
    for the run it scored: its Try it and deployment packages then decide each row's class
    with its group's threshold (requests include the group column)."""
    try:
        return json.dumps(api.use_group_thresholds(groups_run_id), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def stop_group_thresholds(run_id: str) -> str:
    """Stop using thresholds per group for a run: one threshold (0.5) for every row."""
    try:
        return json.dumps(api.stop_group_thresholds(run_id), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def gguf_run(run_id: str, out_dir: str, quant: str = "") -> str:
    """A finished LoRA fine-tune of a Hugging Face model (Llama, Qwen2, Phi-3 or Gemma 3) as
    one GGUF file for llama.cpp, Ollama and LM Studio: the adapter merged into the base model,
    float32 or quant q8_0 / q4_0, checked against transformers with llama.cpp."""
    try:
        return json.dumps(api.gguf_run(run_id, out_dir, quant), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def register_model(run_id: str, name: str) -> str:
    """Copy a finished run into the model registry as a new version."""
    try:
        return json.dumps(api.register_model(run_id, name), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def mirror_model(name: str, version: str, to: list[str]) -> str:
    """Also register a model version in MLflow's model registry and / or as a W&B model
    artifact (to = ["mlflow", "wandb"]), with its stage."""
    try:
        return json.dumps(api.mirror_model(name, version, to), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def tracking_status() -> str:
    """Experiment tracking: which trackers runs mirror to (AIME_TRACKING, or a project's
    meta.tracking), the MLflow store, and whether mlflow / wandb are installed."""
    return json.dumps(api.tracking_status())


@mcp.tool()
def export_runs(out: str, project: str | None = None, epochs: bool = False) -> str:
    """Write the run history as a CSV or Parquet table (by the file's suffix) for
    notebooks: one row per run, or one per epoch with epochs=true."""
    try:
        return json.dumps(api.export_runs(out, project=project, epochs=epochs))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def fetch_dataset(source: str) -> str:
    """Download a dataset and detect it like a local folder: kaggle:owner/name,
    kaggle-competition:name[/file] or hf:owner/repo[:split]. Returns the local path (use
    it as the data path of recommend_recipes / build_recipe) and what was detected."""
    try:
        return json.dumps(api.fetch_dataset(source), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def open_mlflow_model(uri: str, input_shape: list[int] | None = None) -> str:
    """Open an MLflow PyTorch or ONNX model (models:/name/1, runs:/id/model) as an
    editable design, through the importers."""
    try:
        return json.dumps(api.open_mlflow_model(uri, input_shape=input_shape), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def push_to_hub(repo_id: str, run_id: str = "", model: str = "",
                version: str = "latest") -> str:
    """Push a finished run (or a registered model version) to the Hugging Face Hub as a
    private model repo with its model card. Only when the user asked to publish; sign-in
    is the user's own `hf auth login`."""
    try:
        return json.dumps(api.push_to_hub(repo_id, run_id=run_id or None, model=model or None,
                                          version=version), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def list_models() -> str:
    """Registered model versions with stage and metrics."""
    return json.dumps(api.list_models(), default=str)


@mcp.tool()
def import_model(kind: str, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32") -> str:
    """Import an existing model as an editable graph: kind = pytorch (source = .py
    file or module, attr = class / factory, input_shape required), onnx (.onnx)
    or keras (.keras / .h5). Returns the graph and a fidelity report."""
    try:
        return json.dumps(api.import_model(kind, source, attr, input_shape, dtype), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def list_tasks(family: str = "") -> str:
    """Tasks a design can be trained for (losses, metrics, output role, serving schema),
    and the model families (neural, classic, llm, ...)."""
    return json.dumps({**api.list_tasks(family or None), **api.list_families()})


@mcp.tool()
def estimate_budget(graph: dict, device: str = "", max_train_memory_gb: float = 0,
                    max_latency_ms: float = 0, max_params_m: float = 0) -> str:
    """Estimate FLOPs, training memory, model size and inference latency of a neural
    design on a target device (see ``devices``: rtx_3060, t4, a100_40, apple_m_series,
    iphone_ane, raspberry_pi_5, ...) and check it against the given limits (0 = none)."""
    limits = {k: v for k, v in {"max_train_memory_gb": max_train_memory_gb,
                                "max_latency_ms": max_latency_ms,
                                "max_params_m": max_params_m}.items() if v}
    try:
        report = api.estimate_budget(graph, device or None, limits)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    report["devices"] = [d["id"] for d in api.list_devices()["devices"]]
    return json.dumps(report)


@mcp.tool()
def profile_data(graph: dict | None = None, path: str = "", target: str = "") -> str:
    """Profile a dataset: the dataset block of ``graph``, or any table / class-folder
    ``path``. Returns rows, column statistics, class counts and findings (missing
    values, constant / ID-like columns, target leakage, duplicates, imbalance)."""
    try:
        return json.dumps(api.profile_data(graph, path=path or None, target=target or None),
                          default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def data_issues(graph: dict) -> str:
    """Data warnings for the design's dataset in the context of its pipeline."""
    try:
        return json.dumps(api.data_issues(graph))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def split_preview(graph: dict) -> str:
    """Samples per class in train / validation / test, exactly as training splits them."""
    try:
        return json.dumps(api.split_preview(graph))
    except api.ApiError as exc:
        return _err(exc)


def _prepare() -> None:
    """Before serving: load the block registry (and numpy with it), and read requests
    from a private copy of the stdin pipe while standard input itself reads nothing.

    On Windows, a thread blocked reading the stdin pipe stalled other work in the
    process: the first tool call, which loaded numpy's extension, never returned. With
    the imports done up front and the pipe moved off standard input (child processes
    such as training workers no longer inherit it), requests can't block each other.
    """
    from ai_made_easy.core.registry import get_registry

    get_registry()
    pipe = os.dup(0)                       # not inheritable
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdin = open(pipe, encoding="utf-8", errors="replace", closefd=True)  # noqa: SIM115


def main() -> None:
    _prepare()
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
