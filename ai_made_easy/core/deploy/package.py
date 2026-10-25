"""Deployment packages: a trained run → a self-contained model server.

``build_package(run_dir, out_dir)`` writes::

    app.py              FastAPI server (/health, /metadata, /predict[/image])
    model/              the generated training script (as model_def.py), the
                        trained weights and the fitted preprocessing state
    metadata.json       task, classes, input description, metrics, provenance
    requirements.txt    pinned to the versions the model was trained with
    Dockerfile          CPU image running uvicorn as a non-root user
    README.md
    export_formats.py   verification + optional ONNX / TorchScript / Core ML /
                        ExecuTorch / LiteRT exports, each checked against the model
                        (run in the training Python environment)
    exporters.py        the converters export_formats.py uses (exporters.py here)

Pure Python; the verification step runs as a subprocess.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ai_made_easy.core.deploy import exporters, templates

FORMATS = exporters.FORMATS
FORMAT_LABELS = exporters.LABELS
_SERVING_DROP = {"datasets", "torchaudio"}  # only needed to load training data
_PIP_NAMES = {"pillow": "pillow", "scikit-learn": "scikit-learn", "keras>=3": "keras"}
_INPUT_HELP = {
    "records": "a list of records, e.g. [{\"column\": value, ...}]",
    "texts": "a list of strings",
    "images": "a list of base64-encoded images (or use POST /predict/image)",
    "audio": "a list of mono waveforms (float arrays at the training sample rate)",
    "speech": "a list of clips: base64 WAV files, float waveforms at the training sample rate, "
              "or {\"waveform\": [...], \"sample_rate\": n} (resampled)",
    "windows": "a list of windows, each [time steps][feature columns]",
    "generation": "a list of requests: a count, or {\"n\": 4, \"seed\": 0, \"class\": "
                  "\"name\", \"steps\": 50, \"guidance\": 2.0}; returns base64 PNG images",
    "text_to_image": "a list of prompts, or {\"prompt\", \"steps\", \"guidance\", \"seed\"}; "
                     "returns base64 PNG images",
    "prompt": "a list of prompts (language model: text or {\"prompt\", \"max_new_tokens\", "
              "\"temperature\", \"top_k\", \"seed\"}) or source texts (seq2seq)",
    "evidence": "a list of evidence dicts {\"Variable\": \"state\"}, or {\"evidence\": {...}, "
                "\"variables\": [...], \"kind\": \"marginal\" | \"map\"}; returns posteriors",
    "sequences": "a list of observation sequences ([[value, ...], ...] per step); returns the "
                 "hidden state of every step",
    "series": "a list of series: {\"history\": [values], \"past_covariates\": [[...]], "
              "\"future_covariates\": [[...]] (history + horizon rows)} or a plain list of "
              "values",
    "arrays": "a list of samples shaped like the model input",
    "graph": "a list of graphs {\"x\": [[node features]], \"edges\": [[u, v], ...]}; "
             "node models also take {\"nodes\": [index, ...]} for the training graph, link "
             "models {\"pairs\": [[u, v], ...]}",
    "ranking": "a list of requests {\"user\": id, \"k\": 10, \"exclude_seen\": true} -> "
               "the top items, or {\"user\": id, \"items\": [ids]} -> their scores",
    "observations": "a list of observations ([values], or {\"observation\": [...], "
                    "\"deterministic\": true}) -> the action to take",
    "density": "a list of points ([values] or {\"column\": value}) -> log-density, or "
               "{\"sample\": n, \"seed\": 0} -> points drawn from the model",
}


class DeployError(RuntimeError):
    pass


@dataclass
class PackageResult:
    path: Path
    metadata: dict
    formats: dict = field(default_factory=dict)
    log: str = ""
    verified: bool = False

    def to_dict(self) -> dict:
        return {"path": str(self.path), "metadata": self.metadata, "formats": self.formats,
                "verified": self.verified, "log": self.log[-4000:]}


def _artifacts(run_dir: Path, framework: str) -> tuple[Path, list[Path]]:
    script = next(iter(sorted(run_dir.glob(f"*_train_{framework}.py"))), None)
    if script is None:
        raise DeployError(f"no {framework} training script in {run_dir}")
    patterns = {"pgmpy": ("*_model.pkl",), "statsmodels": ("*_ssm.pkl",),
                "pymc": ("*_draws.npz", "*_state.json"),
                "gpytorch": ("*_gp.pt", "inference_state.json"),
                "pytorch": ("*_best.pt", "inference_state.pkl"),
                "keras": ("*_best.keras", "inference_state.pkl"),
                "sklearn": ("*_model.joblib", "inference.json")}[framework]
    files = []
    if framework == "pytorch" and (run_dir / "lora").is_dir():
        # Stable Diffusion LoRA: the adapters (a diffusers folder) instead of a checkpoint,
        # plus the random base model when it was trained offline (weights = none)
        patterns = ("inference_state.pkl",)
        files += [run_dir / "lora", *sorted(run_dir.glob("*_tiny_base.pt"))]
    for pattern in patterns:
        found = sorted(run_dir.glob(pattern))
        if not found:
            raise DeployError(
                f"the run has no {pattern} — it did not finish training, was trained with "
                "k-fold cross-validation only, or predates AI Made Easy 2.0 (retrain it)")
        files.extend(found)
    for optional in ("classes.json", "metrics.json", "calibration.npz", "qat_int8.pt2"):
        if (run_dir / optional).exists():
            files.append(run_dir / optional)
    return script, files


def _describe(graph_dict: dict) -> dict:
    """Task, modality and input kind from the run's design."""
    from ai_made_easy.core.classic.generate import collect_classic, is_classic
    from ai_made_easy.core.graph import Graph

    graph = Graph.from_dict(graph_dict)
    from ai_made_easy.core.tasks import task_of

    task = None if is_classic(graph) else task_of(graph)
    if task is not None and task.family == "gp":
        return {"task": task.id, "modality": "tabular", "dataset": "", "input_kind": "records",
                "input_help": "a list of rows {\"feature\": value}; returns the predictive "
                "mean, std and a 95% interval (or a class probability)", "response": task.serving}
    if task is not None and task.family == "ppl":
        from ai_made_easy.core.ppl.tasks import dataset_of as ppl_data

        data = ppl_data(graph)
        return {"task": task.id, "modality": "tabular", "dataset": data.type_id if data else "",
                "input_kind": "records", "input_help": "a list of rows {\"column\": value} "
                "(predictor and group columns); returns the posterior predictive with a 94% "
                "interval", "response": task.serving}
    if task is not None and task.family == "pgm":
        from ai_made_easy.core.pgm.tasks import dataset_of as pgm_data

        data = pgm_data(graph)
        kind = "sequences" if task.id == "regime_detection" else "evidence"
        return {"task": task.id, "modality": "tabular", "dataset": data.type_id if data else "",
                "input_kind": kind, "input_help": _INPUT_HELP[kind], "response": task.serving}
    # image-in tasks (detection, segmentation); generated images are described below
    if task is not None and "image" in task.modalities and task.trainer_kind not in (
            "supervised", "vae", "adversarial", "diffusion", "text_to_image", "flow"):
        from ai_made_easy.core.vision.tasks import dataset_of

        data = dataset_of(graph)
        out = {"task": task.id, "modality": "image",
               "dataset": data.type_id if data else "data.synthetic_shapes",
               "input_kind": "images", "input_help": _INPUT_HELP["images"],
               "response": task.serving}
        if task.trainer_kind == "detection":
            out["export_note"] = ("ONNX / TorchScript exports take one image at the training "
                                  "size [1, C, H, W] in [0, 1] and return padded detections "
                                  "[1, D, 6]: x1, y1, x2, y2, score, label (score 0 = padding)")
        return out
    if task is not None and task.family == "ssm":
        from ai_made_easy.core.ssm.template import dataset_of as ssm_data

        data = ssm_data(graph)
        return {"task": task.id, "modality": "timeseries",
                "dataset": data.type_id if data else "data.structural_series",
                "input_kind": "series", "input_help": _INPUT_HELP["series"],
                "response": task.serving}
    if task is not None and task.trainer_kind == "forecasting":
        from ai_made_easy.core.forecast.tasks import dataset_of as forecast_data

        data = forecast_data(graph)
        return {"task": task.id, "modality": "timeseries",
                "dataset": data.type_id if data else "data.synthetic_series",
                "input_kind": "series", "input_help": _INPUT_HELP["series"],
                "response": task.serving,
                "export_note": "forecasters are served by the FastAPI package; covariates and "
                               "scaling run in Python before the model"}
    if task is not None and task.trainer_kind in ("vae", "adversarial", "diffusion",
                                                  "text_to_image", "language_model",
                                                  "seq2seq"):
        from ai_made_easy.core.generative.tasks import dataset_of as generative_data

        data = generative_data(graph)
        kind = "generation" if task.trainer_kind in ("vae", "adversarial", "diffusion") \
            else "text_to_image" if task.trainer_kind == "text_to_image" else "prompt"
        return {"task": task.id, "modality": "text" if kind == "prompt" else "image",
                "dataset": data.type_id if data else "",
                "input_kind": kind, "input_help": _INPUT_HELP[kind], "response": task.serving}
    if task is not None and task.trainer_kind == "graph":
        from ai_made_easy.core.gnn.tasks import dataset_of as graph_data

        data = graph_data(graph)
        return {"task": task.id, "modality": "graph",
                "dataset": data.type_id if data else "data.synthetic_graph",
                "input_kind": "graph", "input_help": _INPUT_HELP["graph"],
                "response": task.serving,
                "export_note": "the ONNX export takes node features x [N, F] and edges "
                               "edge_index [2, E] (any N and E) and returns one row per node "
                               "(per graph for graph classification; node embeddings for link "
                               "prediction)"}
    if task is not None and task.trainer_kind == "recommendation":
        from ai_made_easy.core.recsys.tasks import dataset_of as rec_data

        data = rec_data(graph)
        return {"task": task.id, "modality": "interactions",
                "dataset": data.type_id if data else "data.synthetic_interactions",
                "input_kind": "ranking", "input_help": _INPUT_HELP["ranking"],
                "response": task.serving}
    if task is not None and task.trainer_kind == "rl":
        from ai_made_easy.core.rl.tasks import env_node

        env = env_node(graph)
        env_id = env.resolved_params()["env_id"] if env is not None else "CartPole-v1"
        return {"task": task.id, "modality": "environment", "dataset": env_id,
                "input_kind": "observations", "input_help": _INPUT_HELP["observations"],
                "response": task.serving,
                "export_note": "the exported ONNX / TorchScript model is the policy network "
                               "(action scores); serving uses stable-baselines3"}
    if task is not None and task.trainer_kind == "flow":
        from ai_made_easy.core.flows.tasks import dataset_of as flow_data

        data = flow_data(graph)
        return {"task": task.id, "modality": "tabular",
                "dataset": data.type_id if data else "data.density_2d",
                "input_kind": "density", "input_help": _INPUT_HELP["density"],
                "response": task.serving}
    if task is not None and task.trainer_kind == "speech":
        from ai_made_easy.core.speech.tasks import dataset_of as speech_data

        data = speech_data(graph)
        return {"task": task.id, "modality": "audio",
                "dataset": data.type_id if data else "data.synthetic_speech",
                "input_kind": "speech", "input_help": _INPUT_HELP["speech"],
                "response": task.serving}
    if is_classic(graph):
        spec = collect_classic(graph)
        task, modality, block = spec.task, spec.modality, spec.dataset["block"]
        tabular_frame = block in ("data.csv", "data.sklearn")
    else:
        from ai_made_easy.core.training.spec import collect_spec

        spec = collect_spec(graph)
        task, modality, block = spec.task, spec.modality, spec.dataset["block"]
        tabular_frame = block == "data.csv"
    if tabular_frame:
        kind = "records"
    elif modality == "text":
        kind = "texts"
    elif modality == "image":
        kind = "images"
    elif modality == "audio":
        kind = "audio"
    elif modality == "timeseries":
        kind = "windows"
    else:
        kind = "arrays"
    return {"task": task, "modality": modality, "dataset": block, "input_kind": kind,
            "input_help": _INPUT_HELP[kind]}


def _export_kind(graph_dict: dict) -> str:
    from ai_made_easy.core.deploy.exportability import export_kind
    from ai_made_easy.core.graph import Graph

    try:
        return export_kind(Graph.from_dict(graph_dict)) if graph_dict else "tensor"
    except Exception:  # noqa: BLE001 — an old or partial design: one input tensor
        return "tensor"


def _blocked(graph_dict: dict, framework: str) -> dict[str, str]:
    """format -> why the design can't be exported to it (skipped without trying)."""
    from ai_made_easy.core.deploy.exportability import blockers
    from ai_made_easy.core.graph import Graph

    try:
        graph = Graph.from_dict(graph_dict) if graph_dict else None
    except Exception:  # noqa: BLE001
        return {}
    return {f: r for f, (r, _n) in blockers(graph, framework).items()} if graph else {}


def _requirements(script_text: str, framework: str, env: dict, input_kind: str) -> list[str]:
    match = re.search(r"^Needs:\s*(.+)$", script_text, re.M)
    names = [n.strip() for n in (match.group(1) if match else "").split(",") if n.strip()]
    names = [_PIP_NAMES.get(n, n) for n in names if n not in _SERVING_DROP]
    if framework == "keras" and "torch" not in names and env.get("keras_backend", "torch") == "torch":
        names.append("torch")
    if input_kind == "images" and "pillow" not in names:
        names.append("pillow")
    if input_kind == "records" and "pandas" not in names:
        names.append("pandas")
    if framework != "sklearn" and "transformers" in script_text and "AutoTokenizer" in script_text:
        names.append("transformers")
    packages = env.get("packages", {})
    pinned = []
    for name in dict.fromkeys(names):
        base = re.split(r"[<>=]", name)[0]
        version = packages.get(base)
        pinned.append(f"{base}=={version}" if version else name)
    pinned += ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
    if input_kind == "images":
        pinned.append("python-multipart>=0.0.9")
    return pinned


def _example(meta: dict, signature: dict) -> str:
    kind = meta["input_kind"]
    if kind == "records" and signature.get("columns"):
        return json.dumps({"inputs": [{c: 0 for c in signature["columns"][:6]}]})
    if kind == "texts":
        return json.dumps({"inputs": ["an example text"]})
    if kind == "images":
        return '{"inputs": ["<base64 image>"]}'
    if kind == "evidence":
        return json.dumps({"inputs": [{"evidence": {}, "kind": "marginal"}]})
    if kind == "sequences":
        return json.dumps({"inputs": [[0.1, 2.0, 2.2]]})
    if kind == "generation":
        return json.dumps({"inputs": [{"n": 4, "seed": 0}]})
    if kind == "prompt":
        return json.dumps({"inputs": ["the cat"]})
    if kind == "text_to_image":
        return json.dumps({"inputs": [{"prompt": "a photo of a circle", "steps": 30}]})
    if kind == "observations":
        shape = signature.get("input_shape") or [4]
        return json.dumps({"inputs": [[0.0] * int(shape[0])]})
    if kind == "ranking":
        return json.dumps({"inputs": [{"user": "1", "k": 5}]})
    if kind == "graph":
        return json.dumps({"inputs": [{"nodes": [0, 1, 2]}]})
    if kind == "density":
        shape = signature.get("input_shape") or [2]
        return json.dumps({"inputs": [[0.0] * int(shape[0]), {"sample": 3, "seed": 0}]})
    if kind == "speech":
        return '{"inputs": ["<base64 WAV file>"]}'
    if kind == "series":
        return json.dumps({"inputs": [{"history": [12.0, 15.0, 14.0, 18.0, 21.0]}]})
    shape = signature.get("input_shape") or []
    if kind == "arrays" and len(shape) == 1 and shape[0] <= 16:
        return json.dumps({"inputs": [[0.0] * int(shape[0])]})
    return '{"inputs": [...]}'


def build_package(run_dir: Path | str, out_dir: Path | str, *, formats: tuple[str, ...] = (),
                  python: str | None = None, name: str | None = None, version: str = "1",
                  record: dict | None = None, verify: bool = True,
                  timeout: float = 900) -> PackageResult:
    """Build a deployment package from a finished run folder."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if record is None:
        run_file = run_dir / "run.json"
        if not run_file.exists():
            raise DeployError(f"{run_dir} is not a run folder (no run.json)")
        record = json.loads(run_file.read_text(encoding="utf-8"))
    if record.get("status") not in (None, "finished"):
        raise DeployError(f"the run is {record.get('status')}; only finished runs can be deployed")
    framework = record.get("framework", "pytorch")
    formats = tuple(dict.fromkeys(exporters.ALIASES.get(f, f) for f in formats))
    unknown = [f for f in formats if f not in FORMATS.get(framework, ())]
    if unknown:
        raise DeployError(f"{', '.join(unknown)} not available for {framework} models; "
                          f"choose from {', '.join(FORMATS.get(framework, ())) or 'none'}")
    script, files = _artifacts(run_dir, framework)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise DeployError(f"{out_dir} is not empty")
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    script_text = script.read_text(encoding="utf-8")
    (model_dir / "model_def.py").write_text(script_text, encoding="utf-8")
    for file in files:
        if file.is_dir():
            shutil.copytree(file, model_dir / file.name)
        else:
            shutil.copy2(file, model_dir / file.name)

    name = name or record.get("name") or "model"
    image = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "model"
    env = record.get("env") or {}
    meta = {
        "name": name, "version": str(version), "framework": framework,
        **_describe(record.get("graph") or {}),
        "run_id": record.get("run_id", ""), "project": record.get("project", ""),
        "metrics": record.get("final_metrics") or {},
        "trained_with": {"python": env.get("python"), "packages": env.get("packages", {})},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "AI Made Easy",
    }
    py = str(env.get("python") or "3.12")
    py_minor = ".".join(py.split(".")[:2])
    keras_env = ""
    if framework == "keras":
        meta["keras_backend"] = env.get("keras_backend") or "torch"
        keras_env = f" \\\n    KERAS_BACKEND={meta['keras_backend']}"
    reqs = _requirements(script_text, framework, env, meta["input_kind"])
    (out_dir / "requirements.txt").write_text("\n".join(reqs) + "\n", encoding="utf-8")
    (out_dir / "app.py").write_text(templates.APP.format(
        name=name, image=image,
        image_route=(templates.IMAGE_ROUTE if meta["input_kind"] == "images" else
                     templates.GENERATE_ROUTE if meta["input_kind"] in ("generation", "prompt",
                                                                       "text_to_image", "density")
                     else "")), encoding="utf-8")
    (out_dir / "Dockerfile").write_text(templates.DOCKERFILE.format(
        name=name, python=py_minor, env=keras_env), encoding="utf-8")
    (out_dir / ".dockerignore").write_text(templates.DOCKERIGNORE, encoding="utf-8")
    graph_dict = record.get("graph") or {}
    kind = _export_kind(graph_dict)
    meta["export_kind"] = kind
    blocked = _blocked(graph_dict, framework)
    (out_dir / "export_formats.py").write_text(templates.EXPORT_SCRIPT.format(
        framework=framework, formats=tuple(f for f in formats if f not in blocked),
        kind=kind), encoding="utf-8")
    shutil.copy2(Path(exporters.__file__), out_dir / "exporters.py")

    result = PackageResult(out_dir, meta)
    signature: dict = {}
    if verify:
        import os

        run_env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        if framework == "keras":
            run_env["KERAS_BACKEND"] = meta["keras_backend"]
        from ai_made_easy.core.runner.manager import python_executable

        proc = subprocess.run([python or python_executable(), "export_formats.py"], cwd=out_dir,
                              capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env=run_env)
        result.log = proc.stdout + proc.stderr
        if "EXPORT-DONE" not in proc.stdout:
            raise DeployError("the packaged model could not be loaded:\n" + result.log[-3000:])
        report = json.loads((out_dir / "export_report.json").read_text(encoding="utf-8"))
        signature = report.get("signature", {})
        result.formats = report.get("formats", {})
        result.verified = True
    for fmt in formats:
        if fmt in blocked:
            result.formats[fmt] = {"ok": False, "error": blocked[fmt]}
    meta["signature"] = signature
    # relative paths with / on every OS: the package moves between machines
    meta["exports"] = {k: Path(v["file"]).as_posix() if v.get("file") else v.get("file")
                       for k, v in result.formats.items() if v.get("ok")}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    metrics_rows = "".join(f"| {k} | {v:.4g} |\n" for k, v in list(meta["metrics"].items())[:8]
                           if isinstance(v, (int, float)))
    exports = "".join(f"- {FORMAT_LABELS.get(k, k)}: `{v['file']}`\n"
                      for k, v in result.formats.items() if v.get("ok"))
    (out_dir / "README.md").write_text(templates.README.format(
        name=name, run_id=meta["run_id"], task=meta["task"], framework=framework,
        input_help=meta["input_help"], metrics_rows=metrics_rows, image=image,
        image_endpoint=("- `POST /predict/image` — multipart image files\n"
                        if meta["input_kind"] == "images" else
                        "- `POST /generate` — the same requests as `/predict`\n"
                        if meta["input_kind"] in ("generation", "prompt", "text_to_image")
                        else ""),
        example=_example(meta, signature).replace("'", "\\'"),
        exports_section=("\n## Exported formats\n\n" + exports) if exports else ""), encoding="utf-8")
    return result
