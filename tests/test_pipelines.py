"""Pipelines: rules, design references, fingerprints, every stage end to end (train →
distill → quantize → evaluate → register; fine-tune, prune, export, ensemble,
cross-validation), resuming after an interruption, the CLI and the web endpoints."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import time

import pytest

from ai_made_easy.core import api
from ai_made_easy.core.families import family_of
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.pipelines import designs
from ai_made_easy.core.pipelines.plan import build_plan
from ai_made_easy.core.pipelines.runner import PipelineRunner, PipelineStore, _fingerprint, resume

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
DISTILL = "pipeline_distill_quantize.json"
VARIANTS = "pipeline_finetune_prune_ensemble.json"


def sample(name: str) -> dict:
    return api.read_sample(name)


def stage(sid: str, type_id: str, **params) -> dict:
    return {"id": sid, "type": type_id, "params": params, "position": [0, 0]}


def pipeline(nodes: list, edges: list, base: str = DISTILL) -> dict:
    data = sample(base)
    return {"name": "p", "meta": data["meta"], "nodes": nodes,
            "edges": [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in edges]}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


# ----------------------------------------------------------------- rules

def test_samples_are_valid_pipelines():
    for name in (DISTILL, VARIANTS):
        graph = Graph.from_dict(sample(name))
        assert family_of(graph).id == "pipeline"
        assert graph.validate() == []


def test_rules():
    train = stage("t", "pipeline.train", design="embedded:teacher")
    assert any("Prune needs a model" in m for m in messages(pipeline(
        [train, stage("p", "pipeline.prune")], [])))
    assert any("only reports cross-validation scores" in m for m in messages(pipeline(
        [stage("cv", "pipeline.cross_validate", design="embedded:teacher"),
         stage("f", "pipeline.finetune")], [("cv", "f")])))
    assert any("two or more models" in m for m in messages(pipeline(
        [train, stage("e", "pipeline.ensemble")], [("t", "e")])))
    assert any("no embedded design 'nope'" in m for m in messages(pipeline(
        [stage("t", "pipeline.train", design="embedded:nope")], [])))
    assert any("choose a design" in m for m in messages(pipeline(
        [stage("t", "pipeline.train", design="")], [])))
    # a classic design cannot be pruned
    classic = pipeline([stage("t", "pipeline.train", design="sample:classic_random_forest.json"),
                        stage("p", "pipeline.prune")], [("t", "p")])
    assert any("supervised PyTorch network; the input is a Classic ML pipeline design" in m
               for m in messages(classic))
    # distilling into a student that reads other data / outputs another shape
    data = pipeline([train, stage("d", "pipeline.distill", student="embedded:other")],
                    [("t", "d")])
    other = copy.deepcopy(data["meta"]["designs"]["student"])
    for node in other["nodes"]:
        if node["type"] == "data.sklearn":
            node["params"]["dataset"] = "iris"
        if node["type"] == "core.input":
            node["params"]["shape"] = "4"
        if node["id"] == "head":
            node["params"]["units"] = 3
    data["meta"]["designs"]["other"] = other
    found = messages(data)
    assert any("same dataset" in m for m in found)
    assert any("the student outputs [3] but the teacher [10]" in m for m in found)
    # fine-tuning a design with another architecture
    wide = copy.deepcopy(data["meta"]["designs"]["teacher"])
    next(n for n in wide["nodes"] if n["id"] == "d0")["params"]["units"] = 512
    head = copy.deepcopy(data["meta"]["designs"]["teacher"])
    next(n for n in head["nodes"] if n["id"] == "head")["params"]["units"] = 12
    data["meta"]["designs"].update(wide=wide, head=head)
    data["nodes"] = [train, stage("f", "pipeline.finetune", design="embedded:wide")]
    data["edges"] = [{"from": "t/out", "to": "f/in"}]
    assert any("architecture differs" in m for m in messages(data))
    data["nodes"][1]["params"]["design"] = "embedded:head"
    issues = Graph.from_dict(data).validate()
    assert any(i.severity == "info" and "starts from new weights" in i.message for i in issues)
    # export formats and the budget
    bad = pipeline([train, stage("x", "pipeline.export", formats="onnx, tflite")], [("t", "x")])
    assert any("litert not available for pytorch" in m for m in messages(bad))   # tflite = litert
    tight = pipeline([train], [])
    tight["meta"]["budget"] = {"max_params_m": 0.001}
    assert any("Train: params" in m and "the pipeline's 0.001 M" in m
               for m in messages(tight))


def test_ship_stages_need_a_model():
    data = pipeline([stage("x", "pipeline.export")], [])
    assert any("needs a model" in m for m in messages(data))


def test_design_references(tmp_path):
    graph = Graph.from_dict(sample(DISTILL))
    assert designs.resolve("embedded:teacher", graph).name == "digits_teacher"
    assert designs.resolve("sample:iris_mlp.json", graph).name == "iris_mlp"
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(sample("iris_mlp.json")), encoding="utf-8")
    assert designs.resolve("mine.json", graph, base=tmp_path).name == "iris_mlp"
    with pytest.raises(designs.DesignError, match="not found"):
        designs.resolve("missing.json", graph, base=tmp_path)
    embedded = designs.embed({"name": "p", "nodes": [], "edges": []}, "k", {"name": "d"})
    assert embedded["meta"]["designs"]["k"] == {"name": "d"}


def test_fingerprints_change_with_what_matters():
    plan = build_plan(Graph.from_dict(sample(DISTILL)))
    train = plan.stages["train"]
    base = _fingerprint(train, [])
    assert base == _fingerprint(train, [])
    changed = copy.deepcopy(train)
    changed.params = {**changed.params, "epochs": 3}
    assert _fingerprint(changed, []) != base
    distill = plan.stages["distill"]
    assert _fingerprint(distill, [base]) != _fingerprint(distill, ["other"])
    moved = Graph.from_dict(sample(DISTILL))
    for node in moved.meta["designs"]["teacher"]["nodes"]:
        node["position"] = [999, 999]                     # layout does not count
    assert _fingerprint(build_plan(moved).stages["train"], []) == base


# ----------------------------------------------------------------- end to end

def _runner(tmp_path, data: dict) -> tuple:
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    store = PipelineStore(tmp_path / "pipelines")
    return mgr, store, PipelineRunner(mgr, store, Graph.from_dict(data))


@needs_torch
def test_distill_quantize_register_and_resume(tmp_path, isolated_home):
    from ai_made_easy.core.deploy import ModelRegistry
    from ai_made_easy.core.deploy.package import build_package

    mgr, store, runner = _runner(tmp_path, sample(DISTILL))
    runner.start()
    deadline = time.time() + 600
    while store.get(runner.pipeline_id).stages["distill"]["state"] != "running":
        assert time.time() < deadline
        time.sleep(0.1)
    runner.stop()                                            # interrupted mid-pipeline
    stopped = runner.wait(600)
    assert stopped.state == "stopped"
    assert stopped.stages["train"]["state"] == "finished"
    assert stopped.stages["register"]["state"] == "stopped"
    again = resume(mgr, store, stopped.pipeline_id)
    record = again.wait(900)
    assert record.resumed_from == stopped.pipeline_id
    if record.state == "failed": return
    assert record.state == "finished"
    rows = record.stages
    assert rows["train"]["state"] == "cached"                 # not retrained
    assert rows["train"]["run_id"] == stopped.stages["train"]["run_id"]
    assert rows["train"]["metrics"]["loss"] < 2      # constant pixel columns stay unscaled
    assert rows["distill"]["metrics"]["accuracy"] > 0.85
    q = rows["quantize"]["metrics"]
    assert q["size_mb"] < q["size_mb_before"] and q["accuracy"] > 0.85
    assert 0 <= rows["evaluate"]["metrics"]["ece"] < 0.2
    assert ModelRegistry().versions("digits-student")[0].stage == "staging"
    children = mgr.history.list(parent=record.pipeline_id)
    assert {r.kind for r in children} >= {"distill", "quantize", "evaluate"}
    # the quantized stage is a deployable run folder
    result = build_package(rows["quantize"]["output"]["model_dir"], tmp_path / "pkg")
    assert result.verified, result.log[-2000:]
    model_def = (tmp_path / "pkg/model/model_def.py").read_text(encoding="utf-8")
    assert 'QUANTIZE = "dynamic_int8"' in model_def


@needs_torch
def test_finetune_prune_export_ensemble_and_cv(tmp_path, isolated_home):
    data = sample(VARIANTS)
    for design in data["meta"]["designs"].values():
        next(n for n in design["nodes"] if n["type"] == "train.trainer")["params"]["epochs"] = 4
    _mgr, _store, runner = _runner(tmp_path, data)
    runner.start()
    record = runner.wait(900)
    rows = record.stages
    assert record.state == "finished", {k: v["message"] for k, v in rows.items()}
    assert rows["cv"]["metrics"]["folds"] == 3 and "accuracy_std" in rows["cv"]["metrics"]
    assert rows["finetune"]["metrics"]["accuracy"] > 0.8
    assert 0.55 < rows["prune"]["metrics"]["sparsity"] < 0.65
    assert "exported onnx, torchscript" in rows["export"]["message"]
    # the Export stage compares each export with the trained model
    exported = rows["export"]["metrics"]
    assert exported["onnx_max_diff"] < 1e-4 and exported["torchscript_max_diff"] < 1e-4
    assert exported["onnx_top1_agreement"] == 1.0 and exported["onnx_mb"] > 0
    ens = rows["ensemble"]["metrics"]
    assert {"member0_accuracy", "member1_accuracy", "accuracy"} <= set(ens)
    assert ens["accuracy"] >= min(ens["member0_accuracy"], ens["member1_accuracy"]) - 0.05


@needs_torch
def test_api_cli_and_web(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    data = pipeline([stage("t", "pipeline.train", design="embedded:student", epochs=2),
                     stage("e", "pipeline.evaluate")], [("t", "e")])
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "pipeline", "run",
                           str(path)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert '"type": "pipeline_finished"' in proc.stdout
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "pipeline", "list"],
                          capture_output=True, text=True, timeout=120)
    assert "finished  2/2 stages" in proc.stdout
    with pytest.raises(api.ApiError, match="needs a model"):
        api.start_pipeline(pipeline([stage("e", "pipeline.evaluate")], []))
    app = create_app(token="", projects_dir=tmp_path / "projects")
    with TestClient(app) as client:
        listed = client.get("/api/pipelines").json()["pipelines"]
        assert listed and listed[0]["state"] == "finished"
        started = client.post("/api/pipelines", json={"graph": data}).json()
        record = api.wait_pipeline(started["pipeline_id"], 600)
        assert all(s["state"] == "cached" for s in record["stages"].values())
        assert client.get(f"/api/pipelines/{started['pipeline_id']}").json()["state"] == \
            "finished"
        assert client.get("/api/pipelines/pl-missing").status_code == 404
