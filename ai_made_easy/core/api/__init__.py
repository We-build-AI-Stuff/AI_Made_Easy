"""Headless application API: the one surface the CLI, the MCP server and the
web server call. Graphs travel as JSON dicts (the project-file format);
every function returns JSON-serialisable data. Pure Python, no Qt.

The functions live in submodules by area; they are all importable from here.
"""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.graph import Graph

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"
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


_sweeps: dict = {}
_sweep_store = None


def sweep_store():
    global _sweep_store
    from ai_made_easy.core.sweeps import SweepStore

    if _sweep_store is None or _sweep_store.root != SweepStore().root:
        _sweep_store = SweepStore()
    return _sweep_store


_pipelines: dict = {}
_pipeline_store = None


def pipeline_store():
    global _pipeline_store
    from ai_made_easy.core.pipelines.runner import PipelineStore

    if _pipeline_store is None or _pipeline_store.root != PipelineStore().root:
        _pipeline_store = PipelineStore()
    return _pipeline_store


_profiles = None


def profile_cache():
    global _profiles
    if _profiles is None:
        from ai_made_easy.core.data.lints import ProfileCache

        _profiles = ProfileCache()
    return _profiles


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


# the API, by area (imported last: they use the state above)
from ai_made_easy.core.api.blocks import (  # noqa: E402
    describe_design,
    list_blocks,
    list_families,
    list_samples,
    list_tasks,
    read_sample,
    table_layout,
)
from ai_made_easy.core.api.design import (  # noqa: E402
    apply_fix,
    estimate_budget,
    expand,
    generate,
    list_devices,
    stage_designs,
    summarize,
    targets,
    validate,
)
from ai_made_easy.core.api.runs import (  # noqa: E402
    attach_run,
    compare_runs,
    delete_backend,
    delete_run,
    diff_designs,
    diff_run,
    get_run,
    import_run_results,
    list_backends,
    list_runs,
    run_explanation,
    run_metrics,
    run_samples,
    run_status,
    save_backend,
    start_training,
    stop_run,
    test_backend,
    update_run,
)
from ai_made_easy.core.api.sweeps import (  # noqa: E402
    get_sweep,
    list_sweeps,
    start_sweep,
    stop_sweep,
    sweep_again,
    sweep_analysis,
    sweep_best_graph,
    sweepable_params,
    wait_sweep,
)
from ai_made_easy.core.api.tuning import (  # noqa: E402
    apply_learning_rate,
    find_learning_rate,
    group_columns,
    group_result,
    group_thresholds,
    learning_rate_result,
    results_by_group,
    stop_group_thresholds,
    use_group_thresholds,
    resume_info,
    resume_run,
    train_more,
)
from ai_made_easy.core.api.deploy import (  # noqa: E402
    _facts,
    _local,
    automl_leaderboard,
    build_recipe,
    detect_data,
    explain_design,
    list_recipes,
    recommend_recipes,
    start_automl,
    wizard_tasks,
)
from ai_made_easy.core.api.pipelines import (  # noqa: E402
    _package,
    deploy_formats,
    deploy_run,
    gguf_run,
    export_options,
    get_pipeline,
    list_models,
    list_pipelines,
    mirror_model,
    model_registry,
    register_model,
    resume_pipeline,
    set_model_stage,
    start_pipeline,
    stop_pipeline,
    wait_pipeline,
)
from ai_made_easy.core.api.tools import (  # noqa: E402
    _source_dir,
    delete_model,
    deploy_model,
    deploy_to_space,
    export_runs,
    fetch_dataset,
    open_mlflow_model,
    pull_from_hub,
    push_to_hub,
    store_upload,
    submit_to_kaggle,
    tracking_status,
)
from ai_made_easy.core.api.importing import (  # noqa: E402
    import_model,
)
from ai_made_easy.core.api.data import (  # noqa: E402
    _dataset_node,
    augmentation_preview,
    data_fingerprint,
    data_issues,
    profile_data,
    split_preview,
)
from ai_made_easy.core.api.trying import (  # noqa: E402
    try_describe,
    try_file,
    try_run,
    what_would_change,
)

__all__ = [
    "ApiError",
    "SAMPLES_DIR",
    "_dataset_node",
    "_facts",
    "_graph",
    "_local",
    "_manager",
    "_package",
    "_pipeline_store",
    "_pipelines",
    "_profiles",
    "_source_dir",
    "_sweep_store",
    "_sweeps",
    "_try",
    "apply_fix",
    "apply_learning_rate",
    "attach_run",
    "augmentation_preview",
    "automl_leaderboard",
    "build_recipe",
    "compare_runs",
    "data_fingerprint",
    "data_issues",
    "delete_backend",
    "delete_model",
    "delete_run",
    "deploy_formats",
    "deploy_model",
    "deploy_run",
    "gguf_run",
    "deploy_to_space",
    "describe_design",
    "detect_data",
    "diff_designs",
    "diff_run",
    "estimate_budget",
    "expand",
    "explain_design",
    "export_options",
    "export_runs",
    "fetch_dataset",
    "find_learning_rate",
    "generate",
    "get_pipeline",
    "get_run",
    "get_sweep",
    "group_columns",
    "group_result",
    "group_thresholds",
    "import_model",
    "import_run_results",
    "learning_rate_result",
    "list_backends",
    "list_blocks",
    "list_devices",
    "list_families",
    "list_models",
    "list_pipelines",
    "list_recipes",
    "list_runs",
    "list_samples",
    "list_sweeps",
    "list_tasks",
    "manager",
    "mirror_model",
    "model_registry",
    "open_mlflow_model",
    "pipeline_store",
    "profile_cache",
    "profile_data",
    "pull_from_hub",
    "push_to_hub",
    "read_sample",
    "recommend_recipes",
    "register_model",
    "results_by_group",
    "stop_group_thresholds",
    "use_group_thresholds",
    "resume_info",
    "resume_pipeline",
    "resume_run",
    "run_explanation",
    "run_metrics",
    "run_samples",
    "run_status",
    "save_backend",
    "set_manager",
    "set_model_stage",
    "split_preview",
    "stage_designs",
    "start_automl",
    "start_pipeline",
    "start_sweep",
    "start_training",
    "stop_pipeline",
    "stop_run",
    "stop_sweep",
    "store_upload",
    "submit_to_kaggle",
    "summarize",
    "sweep_again",
    "sweep_analysis",
    "sweep_best_graph",
    "sweep_store",
    "sweepable_params",
    "table_layout",
    "targets",
    "test_backend",
    "tracking_status",
    "train_more",
    "try_describe",
    "try_file",
    "try_run",
    "try_service",
    "update_run",
    "validate",
    "wait_pipeline",
    "wait_sweep",
    "what_would_change",
    "wizard_tasks",
]
