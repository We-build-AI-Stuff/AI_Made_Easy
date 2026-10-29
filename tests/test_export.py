"""3.2 Phase 2: export everywhere.

Every sample design exports to ONNX, TorchScript and ExecuTorch (float and int8) and the
exported file matches PyTorch, or the design rule names why it can't. Packages built
from trained runs check their exports on the run's own samples. Keras -> LiteRT runs
where TensorFlow is installed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core.deploy import exporters as ex
from ai_made_easy.core.deploy.exportability import (
    blockers,
    export_kind,
    options,
    options_for_run,
    suggested_formats,
)
from ai_made_easy.core.graph import Graph

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "ai_made_easy" / "samples"
HAS = {m: importlib.util.find_spec(m) is not None
       for m in ("torch", "torchvision", "onnxruntime", "executorch", "torch_geometric",
                 "transformers", "tensorflow", "ai_edge_litert", "fastapi", "coremltools",
                 "openvino")}
needs_torch = pytest.mark.skipif(not HAS["torch"], reason="needs torch")


# ------------------------------------------------------------------ comparisons

def test_detection_matching_ignores_ties_and_handles_flat_boxes():
    ref = np.zeros((5, 6), np.float32)
    ref[:3] = [[0, 0, 10, 10, 0.9, 1], [5, 5, 20, 20, 0.8, 0], [3, 3, 3, 9, 0.7, 2]]
    got = ref.copy()
    assert ex.match_detections(ref, got)[:2] == (2, 2)   # the last kept score is not compared
    got[1, 5] = 2                                         # a different label
    assert ex.match_detections(ref, got)[0] == 1
    tied = np.zeros((3, 6), np.float32)
    tied[:3] = [[0, 0, 4, 4, 0.25, 1], [8, 8, 9, 9, 0.25, 1], [1, 1, 2, 2, 0.25, 1]]
    other = tied[[2, 0, 1]].copy()
    other[0, :4] = [20, 20, 30, 30]   # which of the tied boxes survive is arbitrary
    assert ex.match_detections(tied, other)[:2] == (3, 3)
    assert ex.match_detections(np.zeros((4, 6)), np.zeros((4, 6))) == (0, 0, 0.0)


def test_compare_float_int8_and_accuracy():
    refs = [np.array([[2.0, 1.0, 0.0]]), np.array([[0.0, 3.0, 1.0]])]
    close = [r + 1e-6 for r in refs]
    assert ex.compare("tensor", False, refs, close).ok
    off = [r + 0.01 for r in refs]
    check = ex.compare("tensor", False, refs, off)
    assert not check.ok and "differs" in check.note
    # int8: never fails, reports agreement and the accuracy change
    flipped = [refs[0], np.array([[3.0, 0.0, 1.0]])]
    check = ex.compare("tensor", True, refs, flipped, labels=np.array([0, 1]),
                       eval_refs=refs, eval_outs=flipped)
    assert check.ok and check.metrics["top1_agreement"] == 0.5
    assert check.metrics["accuracy_source"] == 1.0 and check.metrics["accuracy"] == 0.5
    assert "more than 1 point lower" in check.note
    # segmentation-like outputs: per-pixel classes; small drift with equal classes passes
    seg = [np.random.default_rng(0).normal(size=(1, 3, 4, 4))]
    assert ex.compare("tensor", False, seg, [seg[0] + 5e-4]).ok


def test_samples_from_a_run_and_random():
    x = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    s = ex.samples_from_arrays("tensor", x, x[:10], np.arange(10) % 3)
    assert len(s.cases) == 10 and s.cases[0][0].shape == (1, 4)
    assert s.labels.tolist()[:3] == [0, 1, 2] and s.example(2)[0].shape == (2, 4)
    r = ex.random_samples("tensor", (3, 5), integer=True, vocab=7)
    assert r.cases[0][0].dtype == np.int64 and r.cases[0][0].max() < 7
    g = ex.random_graph(6)
    assert g.kind == "graph" and len(g.cases) == 3
    assert g.cases[1][0].shape[0] < g.cases[0][0].shape[0]   # a smaller subgraph too


# ------------------------------------------------------------------ rules and options

def _sample(name: str) -> Graph:
    return Graph.from_dict(json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8")))


def test_export_kinds_and_blockers():
    assert export_kind(_sample("shapes_detection")) == "detector"
    assert export_kind(_sample("graph_node_classification")) == "graph"
    assert export_kind(_sample("mnist_cnn")) == "tensor"
    assert set(blockers(_sample("mnist_cnn"))) == {"gguf", "gguf_q8_0", "gguf_q4_0"}   # LMs only
    assert "Llama-style" in blockers(_sample("mnist_cnn"))["gguf"][0]
    assert "gguf" not in blockers(_sample("tiny_llama"))
    det = blockers(_sample("shapes_detection"))
    assert {"executorch", "executorch_int8", "coreml"} <= set(det) and "onnx" not in det
    assert "NMS" in det["executorch"][0]
    speech = blockers(_sample("keyword_spotting"))
    assert speech["executorch"][1] == "mel" and "STFT" in speech["executorch"][0]
    assert set(blockers(_sample("stable_diffusion_lora"))) == set(ex.FORMATS["pytorch"])
    assert blockers(_sample("mnist_cnn"), "keras") == {}


def test_device_suggestions_and_options(isolated_home):
    from ai_made_easy.core import budget

    assert suggested_formats("raspberry_pi_5", "pytorch")[0] == "executorch_int8"
    assert suggested_formats("raspberry_pi_5", "keras")[0] == "litert_int8"
    assert suggested_formats("iphone_ane", "pytorch")[0] == "coreml"
    assert budget.get_device("android_phone").kind == "mobile"
    assert suggested_formats("", "pytorch") == [] and suggested_formats("nope", "keras") == []
    graph = _sample("shapes_detection")
    budget.set_budget(graph, device="raspberry_pi_5")
    rows = {r["id"]: r for r in options(graph)}
    assert not rows["executorch_int8"]["available"] and not rows["executorch_int8"]["suggested"]
    assert rows["onnx_int8"]["suggested"] and rows["onnx_int8"]["device"].startswith("Raspberry")
    assert rows["onnx"]["available"] and not rows["onnx"]["suggested"]


def test_export_design_rule():
    from ai_made_easy.core import budget

    graph = _sample("shapes_detection")
    assert not [i for i in graph.validate() if "export" in i.message]
    budget.set_budget(graph, device="android_phone")
    issues = [i for i in graph.validate() if "can't be exported" in i.message]
    assert issues and issues[0].severity == "warning"
    assert "ExecuTorch int8" in issues[0].message and "ONNX int8 still works" in issues[0].message
    speech = _sample("keyword_spotting")
    budget.set_budget(speech, device="cpu_laptop")   # runs ONNX: nothing blocked there
    assert not [i for i in speech.validate() if "export" in i.message]
    pooled = json.loads((SAMPLES / "graph_classification_gin.json").read_text(encoding="utf-8"))
    node = next(n for n in pooled["nodes"] if n["id"] == "gin_1")
    node["type"], node["params"] = "graph.topk_pool", {"ratio": 0.5}
    issues = Graph.from_dict(pooled).validate()
    assert any(i.severity == "info" and i.message.startswith("ONNX export: TopK")
               and i.node_id == "gin_1" for i in issues)


# ------------------------------------------------------------------ every sample

def _pytorch_samples() -> list[str]:
    from ai_made_easy.core.deploy.exportability import framework_of

    out = []
    for path in sorted(SAMPLES.glob("*.json")):
        graph = Graph.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if framework_of(graph) == "pytorch" and len(blockers(graph)) < len(ex.FORMATS["pytorch"]):
            out.append(path.stem)
    return out


_NEEDS = {"graph_": "torch_geometric", "shapes_": "torchvision", "yolo_": "torchvision",
          "cifar10_transfer": "torchvision"}
_FORMAT_NEEDS = {"onnx": "onnxruntime", "onnx_int8": "onnxruntime",
                 "executorch": "executorch", "executorch_int8": "executorch",
                 "coreml": "coremltools", "openvino": "openvino"}
_SLOW = {"speech_recognition_ctc", "lstm_classifier", "shapes_diffusion"}


def _build(graph: Graph):  # noqa: ANN202
    from ai_made_easy.core.codegen import emit_graph, generate

    os.environ["AIME_SKIP_PRETRAINED"] = "1"
    try:
        ns: dict = {}
        exec(compile(generate(graph, "pytorch"), "<sample>", "exec"), ns)  # noqa: S102
        model = ns["build_model"]().eval()
    finally:
        os.environ.pop("AIME_SKIP_PRETRAINED", None)
    plan = emit_graph(graph)
    kind = export_kind(graph)
    if kind == "graph":
        return model, ex.random_graph(int(plan.input_shape[-1]))
    return model, ex.random_samples(kind, plan.input_shape, n=3,
                                    integer=plan.input_dtype.startswith("int"))


@needs_torch
@pytest.mark.timeout(900)
@pytest.mark.parametrize("fmt", ["onnx", "torchscript", "onnx_int8", "executorch",
                                 "executorch_int8", "coreml", "openvino"])
@pytest.mark.parametrize("name", _pytorch_samples() if HAS["torch"] else [])
def test_every_sample_round_trips(name, fmt, tmp_path):
    """Exports match PyTorch (float within 1e-4 of the output scale; int8 runs and is
    compared), or the design's export rule names why the format can't be used."""
    for prefix, module in _NEEDS.items():
        if name.startswith(prefix) and not HAS[module]:
            pytest.skip(f"needs {module}")
    if _FORMAT_NEEDS.get(fmt) and not HAS[_FORMAT_NEEDS[fmt]]:
        pytest.skip(f"needs {_FORMAT_NEEDS[fmt]}")
    if fmt.startswith("executorch") and name in _SLOW and not os.environ.get("AIME_SLOW"):
        pytest.skip("slow ExecuTorch lowering (set AIME_SLOW=1)")
    graph = Graph.from_dict(json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8")))
    blocked = blockers(graph)
    if fmt in blocked:
        assert blocked[fmt][0], "a blocked format needs a reason"
        pytest.skip(f"blocked by design rule: {blocked[fmt][0][:80]}")
    model, samples = _build(graph)
    report = ex.export_all(model, "pytorch", [fmt], tmp_path, samples, log=lambda _m: None)
    entry = report[fmt]
    assert entry["ok"], entry.get("error")   # float formats: the exporter enforced 1e-4


@pytest.mark.skipif(not (HAS["torchvision"] and HAS["transformers"] and HAS["onnxruntime"]),
                    reason="needs the vision-tasks extra and onnxruntime")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("head", [
    {"type": "vision.instance_segmenter", "params": {"num_classes": 3, "weights": "none",
                                                     "max_detections": 10}},
    {"type": "vision.keypoint_detector", "params": {"num_classes": 3, "num_keypoints": 3,
                                                    "weights": "none", "max_detections": 10}},
    {"type": "vision.detector", "params": {"arch": "fcos_resnet50_fpn", "num_classes": 3,
                                           "weights": "none", "max_detections": 10}},
    {"type": "vision.hf_detector", "params": {"arch": "detr", "num_classes": 3,
                                              "max_detections": 10}},
    {"type": "vision.hf_segmenter", "params": {"num_classes": 4}},
    {"type": "vision.hf_mask2former", "params": {"num_classes": 4}},
], ids=["mask_rcnn", "keypoint_rcnn", "fcos", "detr", "segformer", "mask2former"])
def test_vision_heads_export_to_onnx(head, tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_vision import design

    graph = Graph.from_dict(design(head))
    model, samples = _build(graph)
    report = ex.export_all(model, "pytorch", ["onnx", "torchscript"], tmp_path, samples,
                           log=lambda _m: None)
    for fmt in ("onnx", "torchscript"):
        assert report[fmt]["ok"], report[fmt].get("error")
    if export_kind(graph) == "detector":
        assert report["onnx"]["metrics"]["matched_detections"] >= ex.DETECTION_MATCH


# ------------------------------------------------------------------ packages

@pytest.fixture()
def mgr(tmp_path):
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    return RunManager(RunHistory(tmp_path / "runs"))


def _train(mgr, data: dict, framework: str = "auto") -> Path:  # noqa: ANN001
    run_id = mgr.start(Graph.from_dict(data), framework)
    status = mgr.wait(run_id, 900)
    assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
    return mgr.history.path(run_id)


@needs_torch
@pytest.mark.skipif(not HAS["onnxruntime"], reason="needs onnxruntime")
@pytest.mark.timeout(900)
def test_package_checks_exports_on_the_runs_samples(mgr, tmp_path):
    from ai_made_easy.core.deploy import build_package

    run = _train(mgr, tiny_classifier_dict(epochs=8))
    with np.load(run / "calibration.npz") as calib:
        assert calib["x"].shape[1:] == (16,) and len(calib["y_eval"]) == len(calib["x_eval"])
    formats = ["onnx", "onnx_int8", "torchscript", "tflite"]   # tflite: an old Keras name
    if HAS["executorch"]:
        formats += ["executorch", "executorch_int8"]
    with pytest.raises(Exception, match="litert not available for pytorch"):
        build_package(run, tmp_path / "bad", formats=tuple(formats))
    formats.remove("tflite")
    result = build_package(run, tmp_path / "pkg", formats=tuple(formats))
    assert result.verified and (tmp_path / "pkg/exporters.py").exists()
    assert (tmp_path / "pkg/model/calibration.npz").exists()
    assert "samples saved by the training run" in result.log
    onnx = result.formats["onnx"]
    assert onnx["ok"] and onnx["metrics"]["max_diff"] < 1e-4
    assert onnx["metrics"]["top1_agreement"] == 1.0 and "accuracy" in onnx["metrics"]
    q = result.formats["onnx_int8"]
    assert q["ok"] and q["metrics"]["top1_agreement"] >= 0.9
    meta = json.loads((tmp_path / "pkg/metadata.json").read_text(encoding="utf-8"))
    assert meta["export_kind"] == "tensor" and meta["exports"]["onnx"] == "exports/model.onnx"
    if HAS["executorch"]:
        et, et8 = result.formats["executorch"], result.formats["executorch_int8"]
        assert et["ok"] and et["metrics"]["max_diff"] < 1e-4
        assert et8["ok"] and "calibrated on" in et8["note"]
        # int8 keeps top-1 accuracy within a point of float
        assert et8["metrics"]["accuracy_source"] - et8["metrics"]["accuracy"] <= 0.01 + 1e-9
        assert et8["metrics"]["max_diff"] > 0      # really quantized (tiny: same file size)
    rows = {r["id"]: r for r in options_for_run(run)}
    assert rows["onnx"]["available"] and not rows["onnx"]["suggested"]


@pytest.mark.skipif(not (HAS["torchvision"] and HAS["onnxruntime"]),
                    reason="needs torchvision and onnxruntime")
@pytest.mark.timeout(900)
def test_detector_package_exports_onnx(mgr, tmp_path):
    from ai_made_easy.core.deploy import build_package

    sys.path.insert(0, str(ROOT / "tests"))
    from test_vision import DETECTOR, design

    run = _train(mgr, design(DETECTOR, name="det_export"))
    result = build_package(run, tmp_path / "pkg", formats=("onnx", "executorch"))
    onnx = result.formats["onnx"]
    assert onnx["ok"], onnx.get("error")
    assert onnx["metrics"]["matched_detections"] >= ex.DETECTION_MATCH
    assert "NMS" in result.formats["executorch"]["error"]     # skipped by the design rule
    meta = json.loads((tmp_path / "pkg/metadata.json").read_text(encoding="utf-8"))
    assert meta["export_kind"] == "detector" and "[1, D, 6]" in meta["export_note"]
    import onnxruntime as ort

    session = ort.InferenceSession(str(tmp_path / "pkg/exports/model.onnx"))
    (shape,) = [i.shape for i in session.get_inputs()]
    assert shape == [1, 3, 64, 64]


@pytest.mark.skipif(not (HAS["torch_geometric"] and HAS["onnxruntime"]),
                    reason="needs the graph extra and onnxruntime")
@pytest.mark.timeout(900)
def test_graph_package_exports_onnx_for_any_graph_size(mgr, tmp_path):
    from ai_made_easy.core.deploy import build_package

    data = json.loads((SAMPLES / "graph_node_classification.json").read_text(encoding="utf-8"))
    next(n for n in data["nodes"] if n["type"] == "train.trainer")["params"]["epochs"] = 2
    run = _train(mgr, data)
    result = build_package(run, tmp_path / "pkg", formats=("onnx", "torchscript"))
    assert result.formats["onnx"]["ok"], result.formats["onnx"].get("error")
    assert "the training graph and a subgraph" in result.log
    import onnxruntime as ort

    session = ort.InferenceSession(str(tmp_path / "pkg/exports/model.onnx"))
    x, ei = (i.shape for i in session.get_inputs())
    assert isinstance(x[0], str) and isinstance(ei[1], str)   # nodes and edges vary
    rng = np.random.default_rng(0)
    out = session.run(None, {"x": rng.normal(size=(7, x[1])).astype(np.float32),
                             "edge_index": np.array([[0, 1, 2], [1, 2, 3]], np.int64)})[0]
    assert out.shape[0] == 7


@pytest.mark.skipif(not (HAS["tensorflow"] and HAS["ai_edge_litert"]),
                    reason="needs tensorflow and ai-edge-litert")
@pytest.mark.timeout(1200)
def test_keras_package_exports_litert(mgr, tmp_path):
    """Keras -> LiteRT (float and int8 calibrated on the run's samples). A run trained on
    the torch backend is converted in a TensorFlow-backend process."""
    from ai_made_easy.core.deploy import build_package

    run = _train(mgr, tiny_classifier_dict(epochs=8, name="keras_tiny"), "keras")
    result = build_package(run, tmp_path / "pkg", formats=("litert", "litert_int8"))
    lite, lite8 = result.formats["litert"], result.formats["litert_int8"]
    assert lite["ok"], lite.get("error")
    assert lite["metrics"]["max_diff"] < 1e-4
    assert lite8["ok"], lite8.get("error")
    assert lite8["metrics"]["top1_agreement"] >= 0.9 and "calibrated on" in lite8["note"]
    assert lite8["metrics"]["max_diff"] > 0     # really quantized (tiny: overhead dominates)


@needs_torch
@pytest.mark.skipif(not HAS["fastapi"], reason="needs fastapi")
def test_format_routes(tmp_path, isolated_home):
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    graph = json.loads((SAMPLES / "shapes_detection.json").read_text(encoding="utf-8"))
    graph.setdefault("meta", {})["budget"] = {"device": "raspberry_pi_5"}
    with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
        rows = {r["id"]: r for r in client.post("/api/export/formats",
                                                json={"graph": graph}).json()["formats"]}
        assert not rows["executorch"]["available"] and "NMS" in rows["executorch"]["reason"]
        assert rows["onnx_int8"]["suggested"]
        assert client.get("/api/runs/missing/formats").status_code == 404


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="needs PySide6")
def test_desktop_deploy_dialog_shows_reasons_and_suggestions(qtbot, tmp_path, isolated_home):
    from ai_made_easy.ui.features.deploy import DeployDialog

    graph = json.loads((SAMPLES / "shapes_detection.json").read_text(encoding="utf-8"))
    graph.setdefault("meta", {})["budget"] = {"device": "raspberry_pi_5"}
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "framework": "pytorch",
                                              "graph": graph, "status": "finished"}), encoding="utf-8")
    dialog = DeployDialog(None, run, "pytorch", "det")
    qtbot.addWidget(dialog)
    checks = dialog.format_checks
    assert not checks["executorch"].isEnabled() and "NMS" in checks["executorch"].toolTip()
    assert checks["onnx_int8"].isChecked() and "Raspberry Pi 5" in checks["onnx_int8"].text()
    assert not checks["onnx"].isChecked()
    assert "executorch" not in dialog.selected_formats()
