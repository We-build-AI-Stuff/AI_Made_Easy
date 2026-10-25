"""Deployment packages and the model registry, end to end (train → package → serve)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from ai_made_easy.core.deploy import (  # noqa: E402
    DeployError,
    ModelRegistry,
    RegistryError,
    build_package,
)
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402

_CLIENT = r"""
import json, sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
import app
client = TestClient(app.app)
out = []
for method, path, body, headers in json.loads(sys.argv[1]):
    if method == "get":
        r = client.get(path, headers=headers)
    elif method == "files":
        r = client.post(path, files=[("files", (f, open(f, "rb").read(), "image/png"))
                                     for f in body], headers=headers)
    else:
        r = client.post(path, json=body, headers=headers)
    out.append({"status": r.status_code, "body": r.json()})
print("RESPONSES " + json.dumps(out))
"""


def call(pkg: Path, requests: list, env: dict | None = None) -> list[dict]:
    """Run requests against the package's app in a fresh interpreter."""
    proc = subprocess.run([sys.executable, "-c", _CLIENT, json.dumps(requests)], cwd=pkg,
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, "KERAS_BACKEND": "torch", **(env or {})})
    assert "RESPONSES " in proc.stdout, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(proc.stdout.split("RESPONSES ", 1)[1])


def train(mgr: RunManager, graph: Graph, framework: str = "auto") -> Path:
    run_id = mgr.start(graph, framework)
    status = mgr.wait(run_id, 600)
    assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
    return mgr.history.path(run_id)


@pytest.fixture()
def mgr(tmp_path):
    return RunManager(RunHistory(tmp_path / "runs"))


def test_package_serves_predictions_and_exports(mgr, tmp_path):
    pytest.importorskip("onnxruntime")
    run = train(mgr, Graph.from_dict(tiny_classifier_dict()))
    pkg = tmp_path / "pkg"
    result = build_package(run, pkg, formats=("onnx", "onnx_int8", "torchscript"))
    assert result.verified
    assert result.formats["onnx"]["ok"] and "matches PyTorch" in result.formats["onnx"]["note"]
    assert result.formats["torchscript"]["ok"] and result.formats["onnx_int8"]["ok"]
    for name in ("app.py", "Dockerfile", "requirements.txt", "README.md", "metadata.json",
                 "model/model_def.py", "model/inference_state.pkl", "exports/model.onnx"):
        assert (pkg / name).exists(), name
    meta = json.loads((pkg / "metadata.json").read_text(encoding="utf-8"))
    assert meta["task"] == "multiclass" and meta["input_kind"] == "arrays"
    assert meta["signature"]["input_shape"] == [16]
    reqs = (pkg / "requirements.txt").read_text(encoding="utf-8")
    assert "torch==" in reqs and "fastapi" in reqs
    sample = [[0.5] * 16, [-1.0] * 16]
    health, metadata, ok, bad, empty = call(pkg, [
        ["get", "/health", None, {}], ["get", "/metadata", None, {}],
        ["post", "/predict", {"inputs": sample}, {}],
        ["post", "/predict", {"inputs": [[1.0] * 3]}, {}],
        ["post", "/predict", {"inputs": []}, {}]])
    assert health["status"] == 200 and metadata["body"]["name"] == "tiny"
    preds = ok["body"]["predictions"]
    assert ok["status"] == 200 and len(preds) == 2
    assert set(preds[0]) == {"label", "class_index", "confidence", "probabilities"}
    assert abs(sum(preds[0]["probabilities"].values()) - 1) < 1e-4
    assert bad["status"] == 422 and "expects" in bad["body"]["detail"]
    assert empty["status"] == 422

    # identical to the training script's own inference
    sys.path.insert(0, str(pkg / "model"))
    try:
        import importlib

        model_def = importlib.import_module("model_def")
        model_def.load_predictor(pkg / "model")
        assert model_def.infer(sample) == preds
    finally:
        sys.path.remove(str(pkg / "model"))
        sys.modules.pop("model_def", None)


def test_api_key_is_enforced(mgr, tmp_path):
    run = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1)))
    pkg = tmp_path / "pkg"
    build_package(run, pkg)
    body = {"inputs": [[0.0] * 16]}
    denied, allowed, health = call(pkg, [
        ["post", "/predict", body, {}], ["post", "/predict", body, {"x-api-key": "s3cret"}],
        ["get", "/health", None, {}]], env={"AIME_API_KEY": "s3cret"})
    assert denied["status"] == 401 and allowed["status"] == 200 and health["status"] == 200


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_package_runs_under_uvicorn(mgr, tmp_path):
    pytest.importorskip("uvicorn")
    import urllib.request

    run = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1)))
    pkg = tmp_path / "pkg"
    build_package(run, pkg)
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port",
                               str(port)], cwd=pkg, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 120
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                assert time.time() < deadline, "the server never answered /health"
                time.sleep(0.3)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict", method="POST",
            data=json.dumps({"inputs": [[0.2] * 16]}).encode(),
            headers={"content-type": "application/json"})
        reply = json.loads(urllib.request.urlopen(request, timeout=30).read())
        assert reply["predictions"][0]["label"] in ("0", "1", "2")
        assert reply["latency_ms"] >= 0
    finally:
        server.terminate()
        server.wait(10)


@pytest.mark.parametrize("scenario,framework,make_inputs", [
    ("table", "pytorch", "records"),
    ("table", "keras", "records"),
    ("text", "pytorch", "texts"),
    ("images", "pytorch", "images"),
    ("series_lstm", "pytorch", "windows"),
])
def test_modalities_package_and_serve(mgr, tmp_path, scenario, framework, make_inputs):
    pytest.importorskip("pandas")
    if framework == "keras":
        pytest.importorskip("keras")
    if scenario == "images":
        pytest.importorskip("torchvision")
    from verify_training import SCENARIOS

    sc = next(s for s in SCENARIOS if s.name == scenario)
    data = tmp_path / "data"
    data.mkdir()
    run = train(mgr, sc.build(data), framework)
    pkg = tmp_path / "pkg"
    meta = build_package(run, pkg).metadata
    assert meta["input_kind"] == make_inputs
    if make_inputs == "records":
        inputs = [{"color": "red", "size": 12.0, "weight": 40.0},
                  {"color": "blue", "size": None, "weight": 70.0, "extra": 1}]
        requests = [["post", "/predict", {"inputs": inputs}, {}]]
    elif make_inputs == "texts":
        requests = [["post", "/predict", {"inputs": ["loved it, superb", "boring plot"]}, {}]]
    elif make_inputs == "windows":
        import numpy as np

        window = np.zeros((16, 2)).tolist()
        requests = [["post", "/predict", {"inputs": [window]}, {}]]
    else:
        import base64

        files = sorted(str(p) for p in (data / "images").rglob("*.png"))[:2]
        encoded = base64.b64encode(Path(files[0]).read_bytes()).decode()
        requests = [["post", "/predict", {"inputs": [encoded]}, {}],
                    ["files", "/predict/image", files, {}]]
    for reply in call(pkg, requests):
        assert reply["status"] == 200, reply
        first = reply["body"]["predictions"][0]
        assert ("label" in first) or ("prediction" in first)
    if make_inputs == "windows":
        assert isinstance(first["prediction"], list) and len(first["prediction"]) == 2


def test_classic_pipeline_package(mgr, tmp_path):
    pytest.importorskip("sklearn")
    from verify_classic import SCENARIOS

    data = tmp_path / "data"
    data.mkdir()
    graph = SCENARIOS["table"][0](data)
    run = train(mgr, graph)
    pkg = tmp_path / "pkg"
    result = build_package(run, pkg)
    assert result.metadata["framework"] == "sklearn"
    assert result.metadata["signature"]["columns"] == ["color", "size", "weight"]  # id dropped
    assert "scikit-learn==" in (pkg / "requirements.txt").read_text(encoding="utf-8")
    (reply,) = call(pkg, [["post", "/predict", {"inputs": [
        {"color": "red", "size": 3.0, "weight": 10.0, "id": 5}]}, {}]])
    assert reply["status"] == 200 and reply["body"]["predictions"][0]["label"] in ("yes", "no")


def test_package_errors(mgr, tmp_path):
    run = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1)))
    with pytest.raises(DeployError, match="not available"):
        build_package(run, tmp_path / "a", formats=("tflite",))
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x", encoding="utf-8")
    with pytest.raises(DeployError, match="not empty"):
        build_package(run, tmp_path / "busy")
    with pytest.raises(DeployError, match="not a run folder"):
        build_package(tmp_path, tmp_path / "b")
    rec = mgr.history.create(tiny_classifier_dict())
    mgr.history.finalize(rec.run_id, "failed", 1)
    with pytest.raises(DeployError, match="failed"):
        build_package(mgr.history.path(rec.run_id), tmp_path / "c")


def test_registry_versions_stages_and_deploy(mgr, tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    run_a = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1)))
    run_b = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1, lr=0.1)))
    v1 = registry.register(run_a, "churn-model", "baseline")
    v2 = registry.register(run_b, "churn-model")
    assert (v1.version, v2.version) == (1, 2) and v1.task == "multiclass"
    with pytest.raises(RegistryError):
        registry.register(run_a, "bad name!")
    registry.set_stage("churn-model", 1, "production")
    registry.set_stage("churn-model", 2, "production")
    stages = {v.version: v.stage for v in registry.versions("churn-model")}
    assert stages == {1: "archived", 2: "production"}
    assert registry.get("churn-model", "production").version == 2
    assert registry.get("churn-model").version == 2
    # the registry copy is self-contained: the original run can go
    import shutil

    shutil.rmtree(run_b)
    pkg = tmp_path / "pkg"
    build_package(registry.path("churn-model", 2), pkg, name="churn-model", version="2")
    (reply,) = call(pkg, [["post", "/predict", {"inputs": [[0.0] * 16]}, {}]])
    assert reply["status"] == 200 and reply["body"]["version"] == "2"
    registry.delete("churn-model", 1)
    assert [v.version for v in registry.list()] == [2]


def test_cli_deploy_and_models(tmp_path, isolated_home):
    env = {**os.environ, "AIME_HOME": str(isolated_home)}
    mgr = RunManager(RunHistory(isolated_home / "runs"))
    run = train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1)))
    cli = [sys.executable, "-m", "ai_made_easy.cli"]
    out = subprocess.run(cli + ["deploy", run.name, "-o", str(tmp_path / "pkg"),
                                "--formats", "torchscript"],
                         capture_output=True, text=True, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "torchscript" in out.stdout and (tmp_path / "pkg" / "app.py").exists()
    out = subprocess.run(cli + ["models", "register", run.name, "demo"], capture_output=True,
                         text=True, env=env, timeout=120)
    assert "registered demo v1" in out.stdout, out.stderr
    out = subprocess.run(cli + ["models", "stage", "demo", "1", "production"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert "production" in out.stdout
    out = subprocess.run(cli + ["models"], capture_output=True, text=True, env=env, timeout=120)
    assert "demo" in out.stdout and "production" in out.stdout
