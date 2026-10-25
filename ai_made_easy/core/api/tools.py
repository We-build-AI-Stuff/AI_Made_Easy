"""The tools you use: tracking, the Hub, Kaggle, backends."""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.api import ApiError, manager
from ai_made_easy.core.api.deploy import detect_data
from ai_made_easy.core.api.importing import import_model
from ai_made_easy.core.api.pipelines import _package, model_registry


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
