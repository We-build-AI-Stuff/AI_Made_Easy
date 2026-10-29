"""Multi-stage pipelines."""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.api import ApiError, _graph, _pipelines, manager, pipeline_store


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


def gguf_run(run_id: str, out_dir: str, quant: str = "") -> dict:
    """A finished LoRA fine-tune as one GGUF file (llama.cpp, Ollama, LM Studio): the
    adapter merged into its Hugging Face base model (Llama, Qwen2, Phi-3 or Gemma 3),
    float32 or ``quant`` q8_0 / q4_0, checked against transformers with llama.cpp."""
    from ai_made_easy.core.deploy import hf_gguf

    try:
        return hf_gguf.export_run(manager().history.path(run_id), Path(out_dir), quant)
    except (hf_gguf.HfGgufError, OSError) as exc:
        raise ApiError(str(exc)) from exc


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
