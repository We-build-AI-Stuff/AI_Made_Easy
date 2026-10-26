"""Model exporters, each checked against the model it came from.

PyTorch: ONNX, ONNX int8, TorchScript, Core ML, OpenVINO, ExecuTorch and ExecuTorch int8.
Keras: ONNX, TensorFlow SavedModel, LiteRT and LiteRT int8. scikit-learn: ONNX.

This file is copied into every serving package (next to ``export_formats.py``) and the
app's tests call it directly, so it imports only the standard library and NumPy at the
top; each converter is imported when its format is asked for.

Three kinds of model:

- ``tensor``: one input tensor with a batch dimension (most designs)
- ``detector``: one image in [1, C, H, W], padded detections [1, D, 6] out (x1, y1, x2,
  y2, score, label); exported at the training image size with post-processing (NMS)
  included
- ``graph``: node features [N, F] and edges [2, E] in; the ONNX file takes any number of
  nodes and edges

Every export runs the source model and the exported file on the same inputs (real
samples from the run when it saved ``calibration.npz``, random ones otherwise). Float
formats must match within ``FLOAT_TOL`` (relative to the output's scale); int8 formats
report how often their top-1 agrees with the float model and, with labelled samples,
both accuracies.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

FLOAT_TOL = 1e-4
HALF_TOL = 5e-2             # Core ML programs compute in float16 by default: deep networks
                            # (a small GPT, a diffusion U-Net) drift 2-3% of their output scale
ARM_TOL = 1e-2              # OpenVINO on ARM CPUs: its kernels drift even in float32 and
                            # its accuracy mode; over Mamba-2's scan, up to 5.3e-3 of the
                            # output scale (200 random initialisations; most stay below 1e-3)
CLASS_TOL = 2e-3            # per-pixel outputs: or every class agrees and drift stays below
DETECTION_MATCH = 0.9       # float detector exports: share of detections that must match
INT8_FORMATS = ("onnx_int8", "executorch_int8", "litert_int8")
ALIASES = {"tflite": "litert"}
FORMATS = {
    "pytorch": ("onnx", "onnx_int8", "torchscript", "coreml", "openvino", "executorch",
                "executorch_int8", "gguf"),
    "keras": ("onnx", "saved_model", "litert", "litert_int8"),
    "sklearn": ("onnx",),
}
LABELS = {
    "onnx": "ONNX", "onnx_int8": "ONNX (int8 quantized)", "torchscript": "TorchScript",
    "coreml": "Core ML", "openvino": "OpenVINO (IR: .xml + .bin)",
    "executorch": "ExecuTorch (.pte)",
    "executorch_int8": "ExecuTorch int8 (.pte, XNNPACK)",
    "saved_model": "TensorFlow SavedModel", "litert": "LiteRT (.tflite)",
    "litert_int8": "LiteRT int8 (.tflite)",
    "gguf": "GGUF (llama.cpp, Ollama, LM Studio)",
}
SHORT = {"onnx": "ONNX", "onnx_int8": "ONNX int8", "torchscript": "TorchScript",
         "coreml": "Core ML", "openvino": "OpenVINO", "executorch": "ExecuTorch", "executorch_int8": "ExecuTorch int8",
         "gguf": "GGUF",
         "saved_model": "SavedModel", "litert": "LiteRT", "litert_int8": "LiteRT int8"}
_SOURCE = {"pytorch": "PyTorch", "keras": "Keras", "sklearn": "scikit-learn"}
_MAX_CASES = 16          # inputs every export is checked on
_MAX_CALIBRATION = 64    # inputs int8 calibration sees
_MAX_EVAL = 256          # labelled inputs for the accuracy comparison



def _arm() -> bool:
    import platform

    return platform.machine().lower() in ("arm64", "aarch64")


def _tolerance(fmt: str) -> float:
    """How close an export must come to the framework's own outputs."""
    if fmt == "coreml":
        return HALF_TOL
    if fmt == "openvino" and _arm():
        return ARM_TOL
    return FLOAT_TOL

class Skipped(Exception):
    """The format does not apply to this model; the message says why."""


# ------------------------------------------------------------------ samples

@dataclass
class Samples:
    """Inputs for checks and calibration. Every case is a tuple of arrays for one
    model call with batch size 1 (detectors and ExecuTorch / LiteRT programs have a
    fixed batch of one)."""

    kind: str
    cases: list
    calibration: list
    eval_cases: list = field(default_factory=list)
    labels: np.ndarray | None = None
    source: str = "random inputs"
    qat: Path | None = None   # the run's quantization-aware trained int8 program

    def example(self, batch: int = 1) -> tuple:
        """The tracing example: ``batch`` rows for tensor models (2 keeps the batch
        dimension dynamic), the first case otherwise."""
        if self.kind != "tensor" or batch == 1 or len(self.cases) < 2:
            return self.cases[0]
        return tuple(np.concatenate([c[i] for c in self.cases[:batch]])
                     for i in range(len(self.cases[0])))


def _rows(x: np.ndarray) -> list:
    return [(x[i:i + 1],) for i in range(len(x))]


def samples_from_arrays(kind: str, x: np.ndarray, x_eval: np.ndarray | None = None,
                        y_eval: np.ndarray | None = None, source: str = "") -> Samples:
    """Samples for tensor models and detectors from arrays of inputs."""
    calibration = _rows(x[:_MAX_CALIBRATION])
    evals = _rows(x_eval[:_MAX_EVAL]) if x_eval is not None and len(x_eval) else []
    labels = None
    if y_eval is not None and len(evals) and np.asarray(y_eval).ndim == 1:
        y = np.asarray(y_eval[:_MAX_EVAL])
        if np.issubdtype(y.dtype, np.integer) or np.all(np.mod(y, 1) == 0):
            labels = y.astype(np.int64)
    cases = (evals or calibration)[:_MAX_CASES]
    return Samples(kind, cases, calibration, evals, labels,
                   source or "samples saved by the training run")


def random_samples(kind: str, input_shape, *, integer: bool = False, vocab: int = 2,
                   n: int = 8, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    shape = (n, *[int(d) for d in input_shape])
    if integer:
        x = rng.integers(0, max(vocab, 1), shape).astype(np.int64)
    elif kind == "detector":
        x = rng.random(shape, dtype=np.float32)
    else:
        x = rng.standard_normal(shape).astype(np.float32)
    return Samples(kind, _rows(x), _rows(x), source="random inputs")


def graph_samples(x: np.ndarray, edge_index: np.ndarray, seed: int = 0) -> Samples:
    """The graph itself plus a smaller induced subgraph (so the check also covers a
    different number of nodes and edges)."""
    x = np.asarray(x, np.float32)
    edge_index = np.asarray(edge_index, np.int64).reshape(2, -1)
    cases = [(x, edge_index)]
    n = len(x)
    if n >= 4:
        keep = n * 2 // 3
        mask = (edge_index[0] < keep) & (edge_index[1] < keep)
        cases.append((x[:keep], edge_index[:, mask]))
    rng = np.random.default_rng(seed)
    noisy = (x + rng.normal(0, 0.1, x.shape)).astype(np.float32)
    cases.append((noisy, edge_index))
    return Samples("graph", cases, cases, source="the training graph and a subgraph")


def random_graph(features: int, nodes: int = 24, edges: int = 60, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((nodes, features)).astype(np.float32)
    ei = rng.integers(0, nodes, (2, edges)).astype(np.int64)
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    out = graph_samples(x, ei, seed)
    out.source = "a random graph"
    return out


def load_samples(kind: str, folder: Path, *, input_shape=(), integer: bool = False,
                 graph: tuple | None = None, features: int = 0) -> Samples:
    """Samples for a packaged model: the run's ``calibration.npz`` when present."""
    if kind == "graph":
        if graph is not None:
            return graph_samples(*graph)
        return random_graph(features or (int(input_shape[-1]) if input_shape else 8))
    path = Path(folder) / "calibration.npz"
    qat = Path(folder) / "qat_int8.pt2"
    if path.exists():
        with np.load(path) as data:
            x = data["x"]
            if len(x):
                found = samples_from_arrays(kind, x, data["x_eval"] if "x_eval" in data else None,
                                            data["y_eval"] if "y_eval" in data else None)
                found.qat = qat if qat.exists() else None
                return found
    if not input_shape:
        raise Skipped("the model's input shape is unknown")
    return random_samples(kind, input_shape, integer=integer)


# ------------------------------------------------------------------ comparison

def _scale(ref: np.ndarray) -> float:
    return max(1.0, float(np.abs(ref).max())) if ref.size else 1.0


def _classes(out: np.ndarray, kind: str) -> np.ndarray | None:
    """Predicted classes when the output is class scores (rows of K > 1 scores, or
    [1, K, H, W] per-pixel scores)."""
    if kind == "detector":
        return None
    if out.ndim == 2 and out.shape[-1] > 1:
        return out.argmax(-1)
    if out.ndim == 4 and out.shape[1] > 1:
        return out.argmax(1)
    return None


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def match_detections(ref: np.ndarray, got: np.ndarray) -> tuple[int, int, float]:
    """(matched, compared, score diff) for padded [D, 6] detections. A reference
    detection matches an exported one with the same label, a score within 0.02 and IoU >=
    0.9 (or every corner within half a pixel, for zero-area boxes). Detections tied with the last kept score are not compared: which of
    equal scores survive the cut is arbitrary."""
    ref = ref[ref[:, 4] > 0]
    got = got[got[:, 4] > 0]
    if not len(ref):
        return (0, 0, 0.0) if not len(got) else (0, len(got), float(got[:, 4].max()))
    ref = ref[np.argsort(-ref[:, 4])]
    got = got[np.argsort(-got[:, 4])] if len(got) else got
    n = min(len(ref), len(got))
    score_diff = float(np.abs(ref[:n, 4] - got[:n, 4]).max()) if n else 1.0
    cutoff = ref[-1, 4]
    decisive = ref[ref[:, 4] > cutoff + 1e-3] if len(ref) > 1 else ref
    if not len(decisive):
        return len(ref), len(ref), score_diff
    if not len(got):
        return 0, len(decisive), score_diff
    ious = _iou(decisive[:, :4], got[:, :4])
    near = np.abs(decisive[:, None, :4] - got[None, :, :4]).max(-1) <= 0.5  # zero-area boxes
    same = (decisive[:, None, 5] == got[None, :, 5]) & \
        (np.abs(decisive[:, None, 4] - got[None, :, 4]) <= 0.02)
    matched = int((((ious >= 0.9) | near) & same).any(1).sum())
    return matched, len(decisive), score_diff


@dataclass
class Check:
    """How an exported model compares with the source model."""

    metrics: dict
    ok: bool
    note: str


def compare(kind: str, int8: bool, refs: list, outs: list, *, labels=None,
            eval_refs=None, eval_outs=None, source: str = "PyTorch",
            tol: float = FLOAT_TOL) -> Check:
    metrics: dict = {}
    if kind == "detector":
        matched = compared = 0
        score_diff = 0.0
        for r, g in zip(refs, outs, strict=True):
            m, c, s = match_detections(r.reshape(-1, 6), g.reshape(-1, 6))
            matched, compared = matched + m, compared + c
            score_diff = max(score_diff, s)
        rate = matched / compared if compared else 1.0
        metrics.update(matched_detections=round(rate, 4), max_score_diff=score_diff)
        ok = int8 or rate >= DETECTION_MATCH
        note = (f"{matched}/{compared} detections match {source}" if compared
                else "no detections to compare")
        return Check(metrics, ok, note)
    diffs, agree, total = [], 0, 0
    worst = 0.0
    for r, g in zip(refs, outs, strict=True):
        g = np.asarray(g, np.float64).reshape(r.shape)
        diff = float(np.abs(g - r).max()) if r.size else 0.0
        diffs.append(diff)
        worst = max(worst, diff / _scale(r))
        cr, cg = _classes(r, kind), _classes(g, kind)
        if cr is not None:
            agree += int((cr == cg).sum())
            total += cr.size
    metrics["max_diff"] = max(diffs) if diffs else 0.0
    if total:
        metrics["top1_agreement"] = round(agree / total, 4)
    if labels is not None and eval_refs and eval_outs:
        acc_ref = _accuracy(eval_refs, labels, kind)
        acc_got = _accuracy(eval_outs, labels, kind)
        if acc_ref is not None and acc_got is not None:
            metrics.update(accuracy_source=round(acc_ref, 4), accuracy=round(acc_got, 4))
    if not int8:
        if worst <= tol:
            return Check(metrics, True, f"matches {source} (max diff {metrics['max_diff']:.1e})")
        per_pixel = all(r.ndim == 4 for r in refs)
        if per_pixel and metrics.get("top1_agreement") == 1.0 and worst <= CLASS_TOL:
            # deep networks (e.g. Mask2Former) drift a little more in float rounding
            return Check(metrics, True, f"predicts the same class as {source} everywhere "
                                        f"(max diff {metrics['max_diff']:.1e})")
        return Check(metrics, False,
                     f"output differs from {source} by {metrics['max_diff']:.2e}")
    parts = []
    if "top1_agreement" in metrics:
        parts.append(f"top-1 agrees with float on {metrics['top1_agreement']:.1%}")
    else:
        parts.append(f"max diff {metrics['max_diff']:.2e}")
    if "accuracy" in metrics:
        parts.append(f"accuracy {metrics['accuracy_source']:.1%} -> {metrics['accuracy']:.1%}")
        if metrics["accuracy_source"] - metrics["accuracy"] > 0.01:
            parts.append("more than 1 point lower: try more calibration data or QAT")
    return Check(metrics, True, "; ".join(parts))


def _accuracy(outs: list, labels: np.ndarray, kind: str) -> float | None:
    preds = []
    for out in outs:
        c = _classes(np.asarray(out), kind)
        if c is None or c.size != 1:
            return None
        preds.append(int(c.reshape(-1)[0]))
    n = min(len(preds), len(labels))
    return float((np.asarray(preds[:n]) == labels[:n]).mean()) if n else None


# ------------------------------------------------------------------ export driver

def size(path: Path) -> int:
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() \
        else path.stat().st_size


Runner = Callable[[tuple], np.ndarray]



@contextlib.contextmanager
def _coreml_numpy_compat():
    """coremltools 9.0 casts a shape held in a one-element array with ``int(array)``,
    which NumPy 2.5 rejects ("only 0-dimensional arrays can be converted"): every traced
    reshape that reads the input's size failed. During a conversion its cast reduces the
    value to a scalar first. Left alone if coremltools' internals look different."""
    try:
        from coremltools.converters.mil.frontend.torch import ops
        original = ops._cast
        get_inputs, mb = ops._get_inputs, ops.mb
    except (ImportError, AttributeError):
        yield
        return

    def cast(context, node, dtype, dtype_name):  # noqa: ANN001
        x = get_inputs(context, node, expected=1)[0]
        if not x.can_be_folded_to_const() or isinstance(x.val, dtype):
            return original(context, node, dtype, dtype_name)
        value = dtype(np.asarray(x.val).reshape(-1)[0])
        context.add(mb.const(val=value, name=node.name), node.name)

    ops._cast = cast
    try:
        yield
    finally:
        ops._cast = original

class Exporter:
    """Runs the requested formats for one model and collects a report."""

    def __init__(self, model, framework: str, samples: Samples, out_dir: Path, *,  # noqa: ANN001
                 log: Callable[[str], None] = print, merges: list | None = None) -> None:
        self.model, self.framework, self.samples = model, framework, samples
        self.merges = merges   # a language model's BPE merges (None: one token per byte)
        self.kind = samples.kind
        self.out = Path(out_dir)
        self.log = log
        self._refs: list | None = None
        self._eval_refs: list | None = None
        self.report: dict = {}

    # -- the source model
    def reference(self, case: tuple) -> np.ndarray:
        if self.framework == "pytorch":
            import torch

            with torch.no_grad():
                out = self.model(*[torch.from_numpy(np.ascontiguousarray(a)) for a in case])
            return out.float().cpu().numpy()
        if self.framework == "keras":
            return np.asarray(self.model.predict(case[0], verbose=0), np.float64)
        return np.asarray(self.model.predict_proba(case[0]) if hasattr(
            self.model, "predict_proba") else self.model.predict(case[0]), np.float64)

    def refs(self) -> list:
        if self._refs is None:
            self._refs = [self.reference(c) for c in self.samples.cases]
        return self._refs

    def eval_refs(self) -> list:
        if self._eval_refs is None:
            self._eval_refs = [self.reference(c) for c in self.samples.eval_cases]
        return self._eval_refs

    # -- the loop
    def run(self, formats) -> dict:  # noqa: ANN001
        wanted = [ALIASES.get(f, f) for f in formats]
        if wanted:
            self.out.mkdir(parents=True, exist_ok=True)
        table = {"pytorch": self._torch_table, "keras": self._keras_table,
                 "sklearn": self._sklearn_table}.get(self.framework, dict)()
        for fmt in dict.fromkeys(wanted):
            if fmt not in table:
                self.report[fmt] = {"ok": False,
                                    "error": f"not available for {self.framework} models"}
                self.log(f"{fmt}: not available for {self.framework} models")
                continue
            self._one(fmt, table[fmt])
        return self.report

    def _one(self, fmt: str, builds) -> None:  # noqa: ANN001
        """Try the format's ways of exporting in order; the first that passes its check
        wins (e.g. a dynamic batch first, then a fixed batch of one)."""
        builds = builds if isinstance(builds, (list, tuple)) else [builds]
        started = time.time()
        errors: list[str] = []
        for build in builds:
            try:
                path, runner, extra = build()
                entry = {"ok": True, "file": str(path), "bytes": size(path),
                         "note": extra or ""}
                if runner is not None:
                    check = self._check(fmt, runner)
                    entry["metrics"] = check.metrics
                    if not check.ok:
                        raise RuntimeError(check.note)
                    entry["note"] = "; ".join(x for x in (check.note, extra) if x)
                elif not entry["note"]:
                    entry["note"] = "not checked here (no runtime for this format)"
            except Skipped as exc:
                self.report[fmt] = {"ok": False, "error": str(exc)}
                self.log(f"{fmt}: skipped: {exc}")
                return
            except ImportError as exc:
                name = getattr(exc, "name", None)
                error = f"not installed: {name}" if name else str(exc)
                self.report[fmt] = {"ok": False, "error": error}
                self.log(f"{fmt}: skipped: {error}")
                return
            except Exception as exc:  # noqa: BLE001 — reported per format
                errors.append(_short(exc))
                if len(errors) == len(builds):
                    traceback.print_exc()
                continue
            entry["seconds"] = round(time.time() - started, 1)
            self.report[fmt] = entry
            self.log(f"{fmt}: {Path(path).name} ({entry['bytes'] / 1e6:.2f} MB) "
                     f"{entry['note']}".rstrip())
            return
        self.report[fmt] = {"ok": False, "error": errors[0]}
        self.log(f"{fmt}: failed: {errors[0]}")

    def stochastic(self) -> bool:
        """True when the model samples at inference (MC dropout, Bayesian layers)."""
        if not hasattr(self, "_stochastic"):
            case = self.samples.cases[0]
            a, b = self.reference(case), self.reference(case)
            self._stochastic = bool(a.shape == b.shape and a.size and
                                    not np.allclose(a, b, rtol=0, atol=1e-6))
        return self._stochastic

    def _check(self, fmt: str, runner: Runner) -> Check:
        outs = [np.asarray(runner(c)) for c in self.samples.cases]
        if self.stochastic():
            for r, g in zip(self.refs(), outs, strict=True):
                if r.size != g.size:
                    return Check({}, False, f"output shape {g.shape} differs from {r.shape}")
            return Check({"stochastic": 1}, True,
                         "the model samples at inference (dropout or Bayesian layers stay "
                         "on), so outputs were checked for shape, not value")
        int8 = fmt in INT8_FORMATS
        labels = self.samples.labels
        eval_outs = None
        if labels is not None and self.samples.eval_cases:
            eval_outs = [np.asarray(runner(c)) for c in self.samples.eval_cases]
        return compare(self.kind, int8, self.refs(), outs, labels=labels,
                       eval_refs=self.eval_refs() if eval_outs else None, eval_outs=eval_outs,
                       source=_SOURCE.get(self.framework, self.framework),
                       tol=_tolerance(fmt))

    # ------------------------------------------------------------ PyTorch
    def _torch_table(self) -> dict:
        onnx = [lambda: self.torch_onnx(2, False), lambda: self.torch_onnx(2, True)]
        script = [lambda: self.torchscript(2)]
        if self.kind == "tensor":   # models that fix the batch size while tracing
            onnx.append(lambda: self.torch_onnx(1, False))
            script.append(lambda: self.torchscript(1))
        return {"onnx": onnx, "onnx_int8": self.torch_onnx_int8, "torchscript": script,
                "coreml": self.coreml,
                "openvino": [lambda: self.openvino(2), lambda: self.openvino(1)],
                "executorch": lambda: self.executorch(False),
                "executorch_int8": lambda: self.executorch(True),
                "gguf": self.gguf}

    def _tensors(self, case: tuple) -> tuple:
        import torch

        return tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in case)

    def _onnx_layout(self, batch: int) -> tuple[list, dict | None, int]:
        if self.kind == "graph":
            return (["x", "edge_index"],
                    {"x": {0: "nodes"}, "edge_index": {1: "edges"}, "output": {0: "rows"}}, 18)
        if self.kind == "detector":
            return ["input"], None, 17
        names = [f"input_{i}" for i in range(len(self.samples.cases[0]))] \
            if len(self.samples.cases[0]) > 1 else ["input"]
        if batch == 1:
            return names, None, 18
        axes = {n: {0: "batch"} for n in names}
        axes["output"] = {0: "batch"}
        return names, axes, 18

    def _ort(self, path: Path) -> Runner | None:
        try:
            import onnxruntime as ort
        except ImportError:
            return None
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        return lambda case: session.run(None, dict(zip(names, case, strict=True)))[0]

    def torch_onnx(self, batch: int = 2, dynamo: bool = False) -> tuple:
        import torch

        path = self.out / "model.onnx"
        names, axes, opset = self._onnx_layout(batch)
        example = self.samples.example(batch)
        kwargs = {"dynamo": dynamo, "opset_version": opset, "input_names": names,
                  "output_names": ["output"]}
        if axes:
            kwargs["dynamic_axes"] = axes
        if dynamo:
            kwargs.update(external_data=False, verbose=False)
        with torch.no_grad(), _plain_attention():
            torch.onnx.export(self.model, self._tensors(example), str(path), **kwargs)
        import onnx

        onnx.checker.check_model(onnx.load(str(path)))
        runner = self._ort(path)
        notes = [] if runner else ["install onnxruntime to check its outputs"]
        if self.kind == "tensor" and batch == 1:
            notes.append("batch size 1 (the model fixes its batch size)")
        return path, runner, "; ".join(notes)

    def torch_onnx_int8(self) -> tuple:
        """Dynamic int8 quantization of the float ONNX file (int8 weights; activations
        are quantized on the fly, so no calibration is needed)."""
        source = self.out / "model.onnx"
        if "onnx" not in self.report:
            self._one("onnx", self._torch_table()["onnx"])
        if not self.report["onnx"].get("ok"):
            raise RuntimeError("the float ONNX export failed: "
                               + self.report["onnx"].get("error", ""))
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        model = onnx.load(str(source))
        del model.graph.value_info[:]   # stale shape notes from the exporter confuse it
        _unshare_initializers(model)
        clean = self.out / "model.float-for-int8.onnx"
        onnx.save(model, str(clean))
        path = self.out / "model.int8.onnx"
        try:
            quantize_dynamic(str(clean), str(path), weight_type=QuantType.QInt8)
        finally:
            clean.unlink(missing_ok=True)
        return path, self._ort(path), f"{size(path) / max(size(source), 1):.0%} of the float size"

    def torchscript(self, batch: int = 2) -> tuple:
        import torch

        path = self.out / "model.torchscript.pt"
        with torch.no_grad(), _plain_attention():
            traced = torch.jit.trace(self.model, self._tensors(self.samples.example(batch)),
                                     check_trace=False)
        traced.save(str(path))
        loaded = torch.jit.load(str(path))

        def run(case: tuple) -> np.ndarray:
            with torch.no_grad():
                return loaded(*self._tensors(case)).float().numpy()
        return path, run, "batch size 1 (the model fixes its batch size)" \
            if self.kind == "tensor" and batch == 1 else ""

    def coreml(self) -> tuple:
        if self.kind != "tensor":
            raise Skipped("Core ML export is for single-tensor models; use ONNX")
        import coremltools as ct
        import torch

        example = self._tensors(self.samples.example(1))
        path = self.out / "model.mlpackage"
        with torch.no_grad(), _plain_attention():
            traced = torch.jit.trace(self.model, example)
        names = ["input" if len(example) == 1 else f"input_{i}" for i in range(len(example))]
        inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in zip(names, example,
                                                                         strict=True)]
        with _coreml_numpy_compat():
            converted = ct.convert(traced, inputs=inputs, convert_to="mlprogram")
        converted.save(str(path))
        if sys.platform != "darwin":
            return path, None, "not checked here (Core ML runs on Apple devices)"
        output = converted.get_spec().description.output[0].name

        def run(case: tuple) -> np.ndarray:
            arrays = [np.asarray(a, np.int32 if np.issubdtype(np.asarray(a).dtype, np.integer)
                                 else np.float32) for a in case]
            return converted.predict(dict(zip(names, arrays, strict=True)))[output]

        return path, run, "float16, as on Apple devices"

    def openvino(self, batch: int = 2) -> tuple:
        """OpenVINO's IR (a folder: ``model.xml`` and its weights ``model.bin``), any
        batch size (1 when tracing fixed it), run on this computer's CPU to check it."""
        if self.kind != "tensor":
            raise Skipped("OpenVINO export is for single-tensor models; use ONNX")
        import openvino as ov
        import torch

        example = self._tensors(self.samples.example(batch))
        folder = self.out / "model.openvino"
        folder.mkdir(exist_ok=True)
        with torch.no_grad(), _plain_attention():
            converted = ov.convert_model(self.model, example_input=example[0]
                                         if len(example) == 1 else example)
        ov.save_model(converted, str(folder / "model.xml"), compress_to_fp16=False)
        # float32 (ARM CPUs would compute in float16 by default), without faster but less
        # exact kernels
        compiled = ov.Core().compile_model(str(folder / "model.xml"), "CPU",
                                           {"INFERENCE_PRECISION_HINT": "f32",
                                            "EXECUTION_MODE_HINT": "ACCURACY"})

        def run(case: tuple) -> np.ndarray:
            return np.asarray(compiled([np.ascontiguousarray(a) for a in case])[0])

        return folder, run, "float32, on this computer's CPU" + (
            f" (an ARM CPU: within {ARM_TOL:g})" if _arm() else "") + (
            "; batch size 1 (the model fixes its batch size)" if batch == 1 else "")

    def executorch(self, int8: bool) -> tuple:
        if self.kind == "graph":
            raise Skipped("graph networks take a different number of nodes and edges per "
                          "request; ExecuTorch programs have fixed shapes (use ONNX)")
        import torch
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge_transform_and_lower

        example = self._tensors(self.samples.example(1))
        model = self.model
        qat = int8 and self.framework == "pytorch" and self.samples.qat is not None
        if qat:   # trained with int8 rounding simulated: its own scales, no calibration
            model = torch.export.load(str(self.samples.qat)).module()
        elif int8:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

            with _plain_attention():
                captured = torch.export.export(model, example).module()
            quantizer = XNNPACKQuantizer()
            quantizer.set_global(get_symmetric_quantization_config())
            prepared = prepare_pt2e(captured, quantizer)
            with torch.no_grad():
                for case in self.samples.calibration:
                    prepared(*self._tensors(case))
            model = convert_pt2e(prepared)
        with torch.no_grad(), _plain_attention():
            program = torch.export.export(model, example)
        lowered = to_edge_transform_and_lower(program, partitioner=[XnnpackPartitioner()])
        path = self.out / ("model.int8.pte" if int8 else "model.pte")
        path.write_bytes(lowered.to_executorch().buffer)
        note = "batch size 1"
        if qat:
            note += "; quantization-aware trained"
        elif int8:
            note += f"; calibrated on {len(self.samples.calibration)} {self.samples.source}"
        try:
            from executorch.runtime import Runtime
        except ImportError:
            return path, None, note + "; install the ExecuTorch runtime to check it"
        method = Runtime.get().load_program(str(path)).load_method("forward")

        def run(case: tuple) -> np.ndarray:
            return method.execute(list(self._tensors(case)))[0].float().numpy()
        return path, run, note

    def gguf(self) -> tuple:
        """A Llama-style decoder as one GGUF file (float32) for llama.cpp, Ollama and LM
        Studio, with its byte-level tokenizer; checked token by token against PyTorch
        with llama-cpp-python when it is installed."""
        import gguf
        import torch

        layout = llama_layout(self.model)
        context = int(np.asarray(self.samples.cases[0][0]).shape[-1])
        path = self.out / "model.gguf"
        writer = gguf.GGUFWriter(str(path), "llama")
        writer.add_name("AI Made Easy decoder")
        writer.add_context_length(context)
        writer.add_embedding_length(layout["dim"])
        writer.add_block_count(len(layout["blocks"]))
        writer.add_feed_forward_length(layout["hidden"])
        writer.add_head_count(layout["heads"])
        writer.add_head_count_kv(layout["kv_heads"])
        writer.add_layer_norm_rms_eps(layout["eps"])
        writer.add_rope_freq_base(10000.0)
        writer.add_rope_dimension_count(layout["head_dim"])
        writer.add_vocab_size(layout["vocab"])
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        tokens, rules = gguf_vocabulary(layout["vocab"], self.merges)
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("gpt-2")
        writer.add_token_list(tokens)
        writer.add_token_types([gguf.TokenType.NORMAL] * len(tokens))
        # llama.cpp needs merges and gguf can't write none: a byte model gets one merge
        # of two-byte pieces, which never occur (text starts as single bytes)
        writer.add_token_merges(rules or [f"{tokens[0] * 2} {tokens[0] * 2}"])
        writer.add_bos_token_id(0)            # no special tokens: the model reads raw text
        writer.add_eos_token_id(0)
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)

        def put(name: str, tensor) -> None:  # noqa: ANN001
            writer.add_tensor(name, tensor.detach().float().cpu().numpy().astype(np.float32))

        kv = layout["kv_heads"] * layout["head_dim"]
        with torch.no_grad():
            put("token_embd.weight", layout["embed"].weight)
            for i, block in enumerate(layout["blocks"]):
                k, v = block.attn.kv.weight[:kv], block.attn.kv.weight[kv:]
                gate, up = block.ffn.gate_up.weight.chunk(2, dim=0)
                for name, tensor in (("attn_norm", block.norm1.weight),
                                     ("attn_q", block.attn.q.weight), ("attn_k", k),
                                     ("attn_v", v), ("attn_output", block.attn.out.weight),
                                     ("ffn_norm", block.norm2.weight), ("ffn_gate", gate),
                                     ("ffn_up", up), ("ffn_down", block.ffn.down.weight)):
                    put(f"blk.{i}.{name}.weight", tensor)
            norm = getattr(layout["norm"], "weight", None)
            put("output_norm.weight", norm if norm is not None else torch.ones(layout["dim"]))
            put("output.weight", layout["head"].weight)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        note = f"the run's {len(rules)} BPE merges" if rules else "one token per byte"
        runner = self._llama_cpp(path, context)
        if runner is None:
            note += "; not checked here (pip install llama-cpp-python to check it)"
        return path, runner, note

    def _llama_cpp(self, path: Path, context: int) -> Runner | None:
        try:
            from llama_cpp import Llama
        except ImportError:
            return None
        # on the CPU with a float32 cache (llama.cpp keeps keys and values in float16 by
        # default, which alone moves the logits by ~5e-4)
        llm = Llama(model_path=str(path), n_ctx=context, logits_all=True, verbose=False,
                    n_gpu_layers=0, type_k=0, type_v=0, flash_attn=False,
                    n_threads=max(1, min(4, os.cpu_count() or 1)))

        def run(case: tuple) -> np.ndarray:
            rows = []
            for ids in np.asarray(case[0]).reshape(-1, context):
                llm.reset()
                llm.eval([int(t) for t in ids])
                rows.append(np.asarray(llm.scores[:context], np.float64))
            return np.stack(rows)
        return run

    # ------------------------------------------------------------ Keras
    def _keras_table(self) -> dict:
        return {"onnx": self.keras_onnx, "saved_model": self.saved_model,
                "litert": lambda: self.litert(False), "litert_int8": lambda: self.litert(True)}

    def keras_onnx(self) -> tuple:
        path = self.out / "model.onnx"
        self.model.export(str(path), format="onnx", verbose=False)
        runner = self._ort(path)
        return path, runner, "" if runner else "install onnxruntime to check its outputs"

    def saved_model(self) -> tuple:
        path = self.out / "saved_model"
        self.model.export(str(path), format="tf_saved_model", verbose=False)
        import tensorflow as tf

        loaded = tf.saved_model.load(str(path))
        return path, lambda case: np.asarray(loaded.serve(tf.constant(case[0]))), ""

    def litert(self, int8: bool) -> tuple:
        import keras

        path = self.out / ("model.int8.tflite" if int8 else "model.tflite")
        example = self.samples.example(1)[0]
        spec = [keras.InputSpec(shape=example.shape, dtype=str(example.dtype))]
        kwargs = {}
        backend = keras.backend.backend()
        if int8:
            if backend != "tensorflow":
                raise Skipped("LiteRT int8 needs the TensorFlow backend "
                              "(pip install tensorflow)")
            import tensorflow as tf

            calibration = self.samples.calibration

            def representative():
                for case in calibration:
                    yield [case[0].astype(example.dtype)]
            kwargs = {"optimizations": [tf.lite.Optimize.DEFAULT],
                      "representative_dataset": representative}
        self.model.export(str(path), format="litert", input_signature=spec, verbose=False,
                          **kwargs)
        runner = _litert_runner(path)
        note = "batch size 1"
        if int8:
            note += f"; calibrated on {len(self.samples.calibration)} {self.samples.source}"
        if runner is None:
            note += "; install ai-edge-litert to check it"
        return path, runner, note

    # ------------------------------------------------------------ scikit-learn
    def _sklearn_table(self) -> dict:
        return {"onnx": self.sklearn_onnx}

    def sklearn_onnx(self) -> tuple:
        from skl2onnx import to_onnx

        path = self.out / "model.onnx"
        n_features = self.model.n_features_in_
        onx = to_onnx(self.model, np.zeros((1, n_features), dtype=np.float32))
        path.write_bytes(onx.SerializeToString())
        return path, None, ""


def _unshare_initializers(model) -> None:  # noqa: ANN001 — onnx.ModelProto
    """Give every node its own copy of a weight. The exporter merges identical weights
    (e.g. several zero-initialised layers) and the int8 quantizer would then transpose
    the shared one once per user."""
    import onnx

    inits = {t.name: t for t in model.graph.initializer}
    users: dict[str, list] = {}
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name in inits:
                users.setdefault(name, []).append((node, i))
    for name, refs in users.items():
        for k, (node, i) in enumerate(refs[1:], 1):
            copy = onnx.TensorProto()
            copy.CopyFrom(inits[name])
            copy.name = f"{name}__copy{k}"
            model.graph.initializer.append(copy)
            node.input[i] = copy.name


class _plain_attention:  # noqa: N801 — a context manager
    """Turn off PyTorch's fused fast path for transformer layers while exporting: the
    fused kernel has no ONNX / ExecuTorch equivalent, the plain one does (same math)."""

    def __enter__(self) -> None:
        import torch

        mha = getattr(torch.backends, "mha", None)
        self._was = mha.get_fastpath_enabled() if mha else None
        if mha:
            mha.set_fastpath_enabled(False)

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        import torch

        if self._was is not None:
            torch.backends.mha.set_fastpath_enabled(self._was)


def llama_layout(model) -> dict:  # noqa: ANN001 — torch.nn.Module
    """A Llama-style decoder's sizes: Embedding, Llama blocks, RMSNorm and a Linear head
    without bias, in that order. Raises Skipped for any other model."""
    import math

    import torch

    reason = ("GGUF holds Llama-style decoders (Embedding → Llama Decoder Blocks → RMSNorm "
              "→ Dense without bias, as in the Tiny Llama recipe)")
    parts = list(model.children())
    names = [type(p).__name__ for p in parts]
    if len(parts) < 4 or names[0] != "Embedding" or names[-2] != "RMSNorm" \
            or names[-1] != "Linear" or any(n != "LlamaBlock" for n in names[1:-2]):
        raise Skipped(f"{reason}; this model is {' → '.join(names) or 'empty'}")
    embed, blocks, norm, head = parts[0], parts[1:-2], parts[-2], parts[-1]
    first = blocks[0]
    attn, ffn = first.attn, first.ffn
    hidden = ffn.down.in_features
    out = {"vocab": embed.num_embeddings, "dim": embed.embedding_dim, "blocks": blocks,
           "heads": attn.heads, "kv_heads": attn.kv_heads, "head_dim": attn.head_dim,
           "hidden": hidden, "eps": float(first.norm1.eps), "embed": embed, "norm": norm,
           "head": head}
    if head.bias is not None or head.out_features != out["vocab"] \
            or head.in_features != out["dim"]:
        raise Skipped(f"{reason}; the last Dense must map {out['dim']} features to the "
                      f"{out['vocab']} tokens, without bias")
    for block in blocks:
        a, f = block.attn, block.ffn
        same = (a.heads, a.kv_heads, a.head_dim, f.down.in_features) == (
            out["heads"], out["kv_heads"], out["head_dim"], hidden)
        if not same or not a.causal or a.rope is None or not isinstance(f.act, torch.nn.SiLU):
            raise Skipped(f"{reason}; every block needs the same sizes, causal attention, "
                          "rotary positions and SwiGLU")
        inv = 1.0 / 10000.0 ** (torch.arange(0, a.head_dim, 2).float() / a.head_dim)
        if not torch.allclose(a.rope.inv_freq.float().cpu(), inv):
            raise Skipped(f"{reason}; rotary positions must use base 10000")
    final_eps = getattr(norm, "eps", None)
    if final_eps is not None and not math.isclose(final_eps, out["eps"], rel_tol=1e-6):
        raise Skipped(f"{reason}; llama.cpp uses one RMSNorm epsilon: the blocks have "
                      f"{out['eps']}, the last norm {final_eps}")
    return out


def _byte_text() -> dict[int, str]:
    """GPT-2's printable stand-in for each byte, which byte-level BPE vocabularies use."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    out, extra = {}, 0
    for b in range(256):
        if b in keep:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + extra)
            extra += 1
    return out


def gguf_vocabulary(vocab: int, merges: list | None) -> tuple[list[str], list[str]]:
    """The tokens (ids 0-255 are bytes, then one per BPE merge) and the merges, written
    the way llama.cpp's byte-level BPE (\"gpt2\") tokenizer reads them."""
    text = _byte_text()
    pieces = {i: bytes([i]) for i in range(256)}
    rules = []
    for i, (a, b) in enumerate(merges or []):
        pieces[256 + i] = pieces[int(a)] + pieces[int(b)]
        rules.append(f"{''.join(text[x] for x in pieces[int(a)])} "
                     f"{''.join(text[x] for x in pieces[int(b)])}")
    if len(pieces) != vocab:
        raise Skipped(f"the model has {vocab} tokens but its tokenizer {len(pieces)}")
    tokens = ["".join(text[x] for x in pieces[i]) for i in range(vocab)]
    return tokens, rules


def _short(exc: Exception) -> str:
    """The useful line of an exporter error (some put the whole TorchScript graph in)."""
    text = str(exc).strip() or type(exc).__name__
    first = text.split("\n\n", 1)[0].split("Torch IR graph at exception", 1)[0].strip()
    return first[-600:]


def _litert_runner(path: Path) -> Runner | None:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter  # noqa: N806
        except ImportError:
            return None
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    output = interpreter.get_output_details()[0]

    def run(case: tuple) -> np.ndarray:
        for detail, value in zip(inputs, case, strict=True):
            interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))
        interpreter.invoke()
        return np.asarray(interpreter.get_tensor(output["index"]), np.float64)
    return run


def export_all(model, framework: str, formats, out_dir: Path, samples: Samples, *,  # noqa: ANN001
               log: Callable[[str], None] = print, merges: list | None = None) -> dict:
    """Export ``model`` to every format in ``formats``; the per-format report. A language
    model's BPE ``merges`` go into its GGUF tokenizer."""
    return Exporter(model, framework, samples, out_dir, log=log, merges=merges).run(formats)


def summary(report: dict) -> dict:
    """Flat metrics for a pipeline stage: ``<format>_<metric>`` and sizes in MB."""
    out = {}
    for fmt, entry in report.items():
        if not entry.get("ok"):
            continue
        out[f"{fmt}_mb"] = round(entry.get("bytes", 0) / 1e6, 4)
        for key, value in (entry.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                out[f"{fmt}_{key}"] = value
    return out


if __name__ == "__main__":  # pragma: no cover — a quick look at a report
    import sys

    print(json.dumps(summary(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["formats"]), indent=1))
