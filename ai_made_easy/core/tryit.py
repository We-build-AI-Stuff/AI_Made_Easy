"""Try a finished run (3.5): its predictor is loaded once in a worker process, in the run's
own folder and interpreter, and answers predictions from both apps. Nothing leaves this
machine: no package is built and no server is started.

The worker imports the run's training script (``main()`` is not run), calls its
``load_predictor`` and then answers one JSON request per line with ``infer``. The two most
recently tried runs stay loaded.

3.6: a whole file of inputs at once (a table, a text file, or a zip of images or WAV clips)
comes back as a CSV of predictions, and an image classifier's prediction can come with its
Grad-CAM heatmap, computed in the worker from the loaded network (so runs trained before
3.6 explain too).
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import queue
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

_MARK = "AIME-TRY "
_KEEP = 2

_WORKER = r'''
import importlib.util, json, sys
from pathlib import Path

run_dir, script = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(run_dir))
out = sys.stdout
sys.stdout = sys.stderr   # the script's own prints must not mix with the replies


def reply(obj):
    plain = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)  # noqa: E731
    out.write("AIME-TRY " + json.dumps(obj, default=plain) + "\n")
    out.flush()


try:
    spec = importlib.util.spec_from_file_location("model_def", script)
    model_def = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_def)
    model_def.load_predictor(run_dir, device="cpu")
except Exception as exc:  # noqa: BLE001
    reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
    sys.exit(1)
state = getattr(model_def, "INFERENCE_STATE", {}) or {}


def conv_layer():
    """The last Conv2d of a PyTorch image model (Grad-CAM's layer), or None."""
    try:
        import torch.nn as nn
    except ImportError:
        return None
    model = getattr(model_def, "_PREDICTOR", None)
    if not isinstance(model, nn.Module) or not hasattr(model_def, "open_image"):
        return None
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else None


def explain(raw):
    """Grad-CAM (Selvaraju et al., 2017) of the predicted class, drawn over each image."""
    import base64
    import io

    import numpy as np
    import torch
    from PIL import Image

    model, layer = model_def._PREDICTOR, conv_layer()
    if layer is None:
        raise ValueError("Grad-CAM needs a PyTorch image model with a convolution")
    out = []
    for item in raw:
        x = torch.from_numpy(np.ascontiguousarray(model_def.prepare_inputs([item])))
        x = x.to(next(model.parameters()).device)
        kept = {}
        hooks = [layer.register_forward_hook(lambda m, i, o: kept.__setitem__("a", o)),
                 layer.register_full_backward_hook(lambda m, gi, go: kept.__setitem__("g", go[0]))]
        try:
            model.eval()
            with torch.enable_grad():
                scores = model(x.requires_grad_(True)).float()
                k = int(scores.argmax(1)[0]) if scores.shape[1] > 1 else 0
                model.zero_grad()
                scores[0, k].backward()
        finally:
            for h in hooks:
                h.remove()
        weights = kept["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * kept["a"]).sum(1))[0].detach().cpu().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        image = model_def.open_image(item).convert("RGB")
        heat = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
        h = np.asarray(heat, dtype=np.float32)[..., None] / 255
        colour = np.concatenate([h, np.zeros_like(h), 1 - h], axis=-1) * 255
        blend = np.asarray(image, dtype=np.float32) * 0.55 + colour * 0.45
        buf = io.BytesIO()
        Image.fromarray(np.uint8(np.clip(blend, 0, 255))).save(buf, format="PNG")
        out.append({"class_index": k, "gradcam": base64.b64encode(buf.getvalue()).decode()})
    return out


reply({"ready": True, "signature": {
    "input_shape": list(getattr(model_def, "INPUT_SHAPE", ()) or ()),
    "classes": [str(c) for c in (getattr(model_def, "CLASS_NAMES", None) or [])] or None,
    "columns": ((state.get("table") or {}).get("columns")
                or getattr(model_def, "_COLUMNS", None)),
    "sample_rate": getattr(model_def, "SAMPLE_RATE", None),
    "explain": conv_layer() is not None,
}})
for line in sys.stdin:
    try:
        request = json.loads(line)
        if isinstance(request, dict) and "explain" in request:
            reply({"ok": True, "predictions": explain(request["explain"])})
            continue
        reply({"ok": True, "predictions": model_def.infer(request)})
    except Exception as exc:  # noqa: BLE001 — reported to the app, the worker carries on
        reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
'''


class TryError(RuntimeError):
    pass


class _Worker:
    def __init__(self, run_dir: Path, script: Path, python: str, env: dict) -> None:
        self.proc = subprocess.Popen(
            [python, "-u", "-c", _WORKER, str(run_dir), str(script)], cwd=run_dir,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
            env=env)
        self._replies: queue.Queue = queue.Queue()
        self._err: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.lock = threading.Lock()
        self.signature: dict = {}

    def _read(self) -> None:
        for line in self.proc.stdout:
            if line.startswith(_MARK):
                self._replies.put(json.loads(line[len(_MARK):]))
        self._replies.put(None)   # the worker exited

    def _drain(self) -> None:
        for line in self.proc.stderr:
            self._err = (self._err + [line])[-40:]

    def reply(self, timeout: float) -> dict:
        try:
            got = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TryError(f"the model did not answer within {timeout:.0f} s") from None
        if got is None:
            raise TryError("the model stopped:\n" + "".join(self._err)[-2000:])
        return got

    def ask(self, inputs: list | dict, timeout: float) -> list:
        with self.lock:
            self.proc.stdin.write(json.dumps(inputs) + "\n")
            self.proc.stdin.flush()
            got = self.reply(timeout)
        if not got.get("ok"):
            raise TryError(got.get("error") or "the prediction failed")
        return got["predictions"]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.alive():
            self.proc.kill()
        self.proc.wait(timeout=10)


class TryService:
    """Loaded predictors by run id (the most recent few)."""

    def __init__(self, keep: int = _KEEP) -> None:
        self._workers: OrderedDict[str, _Worker] = OrderedDict()
        self._keep = keep
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Lock] = {}

    def worker(self, run_id: str, run_dir: Path, record: dict, *,
               load_timeout: float = 300) -> _Worker:
        from ai_made_easy.core.deploy.package import _artifacts
        from ai_made_easy.core.runner.manager import python_executable

        with self._lock:
            got = self._workers.get(run_id)
            if got is not None and got.alive():
                self._workers.move_to_end(run_id)
                return got
            loading = self._loading.setdefault(run_id, threading.Lock())
        # one load per run at a time; loading a model (up to minutes) never blocks the
        # other runs' predictions
        with loading:
            with self._lock:
                got = self._workers.get(run_id)
                if got is not None and got.alive():
                    return got
            worker = self._load(run_dir, record, python_executable, _artifacts, load_timeout)
            with self._lock:
                self._workers[run_id] = worker
                self._workers.move_to_end(run_id)
                while len(self._workers) > self._keep:
                    _, old = self._workers.popitem(last=False)
                    old.close()
            return worker

    @staticmethod
    def _load(run_dir: Path, record: dict, python_executable, artifacts,  # noqa: ANN001
              load_timeout: float) -> _Worker:
        if record.get("status") != "finished":
            raise TryError(f"the run is {record.get('status') or 'not finished'}; only "
                           "finished runs can be tried")
        framework = record.get("framework", "pytorch")
        try:
            script, _ = artifacts(run_dir, framework)
        except Exception as exc:  # noqa: BLE001 — DeployError: what is missing
            raise TryError(str(exc)) from exc
        env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        if framework == "keras":
            env["KERAS_BACKEND"] = (record.get("env") or {}).get("keras_backend") or "torch"
        worker = _Worker(run_dir, script, python_executable(), env)
        first = worker.reply(load_timeout)
        if not first.get("ready"):
            worker.close()
            raise TryError("the model could not be loaded: " + str(first.get("error")))
        worker.signature = first.get("signature") or {}
        return worker

    def forget(self, run_id: str) -> None:
        """Close a run's worker, so the next prediction loads what the run has now (its
        thresholds per group, say)."""
        with self._lock:
            worker = self._workers.pop(run_id, None)
        if worker is not None:
            worker.close()

    def close(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.close()
            self._workers.clear()


def _target(service: TryService, history, run_id: str):  # noqa: ANN001, ANN202
    from ai_made_easy.core.deploy.package import _describe

    history.get(run_id)   # an unknown run raises here
    folder = history.path(run_id)
    record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    return service.worker(run_id, folder, record), _describe(record.get("graph") or {})


def describe(service: TryService, history, run_id: str) -> dict:  # noqa: ANN001
    """What a finished run takes: input kind and help, an example, its signature."""
    from ai_made_easy.core.deploy.package import _example

    worker, meta = _target(service, history, run_id)
    try:
        example = json.loads(_example(meta, worker.signature)).get("inputs")
    except ValueError:   # a placeholder such as "<base64 image>"
        example = None
    adopted = history.path(run_id) / "group_thresholds.json"
    groups = json.loads(adopted.read_text(encoding="utf-8")) if adopted.is_file() else None
    return {"run_id": run_id, "task": meta.get("task"), "modality": meta.get("modality"),
            "input_kind": meta["input_kind"], "input_help": meta.get("input_help", ""),
            "example": example, "signature": worker.signature, "group_thresholds": groups,
            "can_explain": bool(worker.signature.get("explain")),
            "batch_formats": BATCH_FORMATS.get(meta["input_kind"], "")}


def predict(service: TryService, history, run_id: str, inputs: list,  # noqa: ANN001
            explain: bool = False) -> dict:
    """The run's predictions for ``inputs`` (the same requests as its model server); with
    ``explain``, an image classifier's predictions also carry a Grad-CAM heatmap (base64
    PNG) of the predicted class."""
    import time

    if not isinstance(inputs, list) or not inputs:
        raise TryError("inputs must be a non-empty list")
    if len(inputs) > 64:
        raise TryError("try at most 64 inputs at a time")
    worker, meta = _target(service, history, run_id)
    kind = meta["input_kind"]
    if kind == "images":
        inputs = strip_data_urls(inputs)
    elif kind == "audio":
        inputs = wav_paths(inputs, temp_folder())
    if explain and not worker.signature.get("explain"):
        raise TryError("Grad-CAM explains PyTorch image models with a convolution")
    started = time.perf_counter()
    predictions = worker.ask(inputs, timeout=120)
    if explain:
        cams = worker.ask({"explain": inputs}, timeout=120)
        predictions = [{**p, "gradcam": c["gradcam"]} if isinstance(p, dict) else p
                       for p, c in zip(predictions, cams, strict=True)]
    return {"run_id": run_id, "input_kind": kind, "predictions": predictions,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


# ------------------------------------------------------------------ a file of inputs (3.6)

MAX_BATCH_ROWS = 10_000
_CHUNK = 64
BATCH_FORMATS = {"records": ".csv", "arrays": ".csv", "texts": ".csv,.txt", "prompt": ".csv,.txt",
                 "images": ".zip", "audio": ".zip", "speech": ".zip"}
_MAX_UNZIPPED = 2 * 2**30
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def _number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _cell(text: str):
    """A CSV cell as the model's server would get it in JSON: numbers as numbers."""
    try:
        if not text.strip() or text.strip().lower() in ("nan", "na", "null"):
            return None                           # a missing value
        return float(text)
    except ValueError:
        return text


def batch_items(kind: str, filename: str, data: bytes, folder: Path,
                signature: dict | None = None) -> tuple[list, list]:
    """(inputs, a name per input) from an uploaded file, for an input kind."""
    name = filename.lower()
    if kind == "arrays":
        shape = (signature or {}).get("input_shape") or []
        if len(shape) != 1 or not name.endswith(".csv"):
            raise TryError("a file of inputs works for models that take a list of numbers: "
                           "a CSV with one input per row")
        rows = [r for r in csv.reader(io.StringIO(data.decode("utf-8-sig"))) if r]
        if rows and any(not _number(v) for v in rows[0]):
            rows = rows[1:]                          # a header row
        bad = next((i for i, r in enumerate(rows) if len(r) != shape[0]), None)
        if bad is not None:
            raise TryError(f"row {bad + 1} has {len(rows[bad])} values; the model takes "
                           f"{shape[0]}")
        try:
            return [[float(v) for v in r] for r in rows], [str(i + 1) for i in range(len(rows))]
        except ValueError as exc:
            raise TryError(f"every value must be a number: {exc}") from None
    if kind == "records":
        if not name.endswith((".csv", ".tsv")):
            raise TryError("a table model predicts a CSV file: one row per input, with the "
                           "columns it was trained on")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")),
                                   delimiter="\t" if name.endswith(".tsv") else ","))
        return ([{k: _cell(v or "") for k, v in r.items() if k is not None} for r in rows],
                [str(i + 1) for i in range(len(rows))])
    if kind in ("texts", "prompt"):
        text = data.decode("utf-8-sig")
        if name.endswith(".csv"):
            rows = list(csv.DictReader(io.StringIO(text)))
            column = next((c for c in (rows[0].keys() if rows else ())
                           if c and c.lower() in ("text", "prompt", "sentence")), None)
            if column is None:
                raise TryError("the CSV needs a 'text' column")
            texts = [r[column] or "" for r in rows]
        else:
            texts = [line for line in text.splitlines() if line.strip()]
        return texts, [str(i + 1) for i in range(len(texts))]
    if kind in ("images", "audio", "speech"):
        import zipfile

        if not name.endswith(".zip"):
            raise TryError(f"{'images' if kind == 'images' else 'clips'} come as a .zip file")
        wanted = _IMAGE_EXT if kind == "images" else (".wav",)
        inputs, names, total = [], [], 0
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                base = Path(info.filename).name
                if info.is_dir() or base.startswith(".") or not base.lower().endswith(wanted):
                    continue
                if info.file_size > 50 * 2**20:
                    raise TryError(f"{info.filename} is larger than 50 MB")
                total += info.file_size
                if total > _MAX_UNZIPPED:
                    raise TryError(f"the zip unpacks to more than {_MAX_UNZIPPED >> 20} MB")
                blob = archive.read(info)
                if kind == "images":
                    inputs.append(base64.b64encode(blob).decode())
                else:
                    path = folder / f"clip_{len(inputs)}.wav"
                    path.write_bytes(blob)
                    inputs.append(str(path))
                names.append(info.filename)
        if not inputs:
            raise TryError(f"the zip has no {'images' if kind == 'images' else 'WAV files'}")
        return inputs, names
    raise TryError(f"this model ({kind}) takes JSON inputs; try them one list at a time")


def _flat(prediction) -> dict:
    """One CSV row from a prediction: scalars as they are, lists joined, probabilities as
    one column per class."""
    if not isinstance(prediction, dict):
        return {"prediction": prediction}
    row = {}
    for key, value in prediction.items():
        if key == "probabilities" and isinstance(value, dict):
            row.update({f"p({cls})": v for cls, v in value.items()})
        elif isinstance(value, (list, tuple)):
            row[key] = " ".join(str(v) for v in value) if len(value) <= 64 else f"[{len(value)}]"
        elif isinstance(value, dict):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


def predict_file(service: TryService, history, run_id: str, filename: str,  # noqa: ANN001
                 data: bytes, changes: bool = False) -> dict:
    """Predictions for every input in an uploaded file, as CSV text (the inputs' names or
    row numbers first, then the prediction columns). With ``changes``, a table's rows also
    get the smallest change that predicts another class (the first rows; see
    counterfactual.MAX_ROWS)."""
    import time

    worker, meta = _target(service, history, run_id)
    kind = meta["input_kind"]
    inputs, names = batch_items(kind, filename, data, temp_folder(), worker.signature)
    if len(inputs) > MAX_BATCH_ROWS:
        raise TryError(f"at most {MAX_BATCH_ROWS} inputs per file (this one has {len(inputs)})")
    if not inputs:
        raise TryError("the file has no inputs")
    started = time.perf_counter()
    rows = []
    for at in range(0, len(inputs), _CHUNK):
        rows += [_flat(p) for p in worker.ask(inputs[at:at + _CHUNK], timeout=300)]
    if changes:
        from ai_made_easy.core import counterfactual

        if kind != "records":
            raise TryError("what would change it is worked out for models that read table "
                           "rows")
        try:
            extra = counterfactual.for_rows(service, history, run_id, inputs)
        except counterfactual.CounterfactualError as exc:
            raise TryError(str(exc)) from exc
        rows = [{**r, **e} for r, e in zip(rows, extra, strict=True)]
    key = "file" if kind in ("images", "audio", "speech") else "row"
    columns = [key] + list(dict.fromkeys(c for r in rows for c in r))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for name, row in zip(names, rows, strict=True):
        writer.writerow({key: name, **row})
    return {"run_id": run_id, "rows": len(rows), "csv": out.getvalue(),
            "filename": f"{Path(filename).stem}_predictions.csv",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


def wav_paths(inputs: list, folder: Path) -> list:
    """Uploaded WAV files (base64, or a data: URL) become paths the audio predictor reads."""
    out = []
    for i, item in enumerate(inputs):
        if isinstance(item, str) and not Path(item).exists():
            data = item.split(",", 1)[1] if item.startswith("data:") else item
            path = folder / f"clip_{i}.wav"
            path.write_bytes(base64.b64decode(data))
            out.append(str(path))
        else:
            out.append(item)
    return out


def strip_data_urls(inputs: list) -> list:
    """Images from a browser come as data: URLs; the predictor takes the base64 part."""
    return [i.split(",", 1)[1] if isinstance(i, str) and i.startswith("data:") else i
            for i in inputs]


def temp_folder() -> Path:
    return Path(tempfile.mkdtemp(prefix="aime_try_"))
