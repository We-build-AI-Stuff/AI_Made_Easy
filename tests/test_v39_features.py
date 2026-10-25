"""3.9 features: resume an interrupted run, train boosted trees more, find a learning rate
for Keras designs."""
from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api, resume
from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import RunManager, python_executable
from ai_made_easy.core.runs.history import RunHistory


def _resumable_design(epochs: int = 4) -> dict:
    """A design whose training draws random numbers (dropout, shuffling) and carries
    state between epochs (momentum, a step schedule): what a resume has to restore."""
    graph = tiny_classifier_dict(epochs=epochs)
    graph["nodes"].insert(2, {"id": "drop", "type": "core.dropout", "params": {"p": 0.3},
                              "position": [0, 0]})
    graph["nodes"].append({"id": "sched", "type": "train.step_lr",
                           "params": {"step_size": 1, "gamma": 0.7}, "position": [0, 0]})
    graph["edges"] = [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in
                      zip(["in", "d1", "act", "drop", "d2"], ["d1", "act", "drop", "d2", "out"],
                          strict=True)]
    return graph


def _run(folder: Path, script: Path, before_main: str = "") -> subprocess.CompletedProcess:
    driver = (f"import importlib.util, sys\n"
              f"spec = importlib.util.spec_from_file_location('m', {str(script)!r})\n"
              f"m = importlib.util.module_from_spec(spec); sys.modules['m'] = m\n"
              f"spec.loader.exec_module(m)\n{before_main}\nm.main()\n")
    return subprocess.run([python_executable(), "-c", driver], cwd=folder, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=600)


@pytest.mark.slow
def test_a_resumed_run_ends_exactly_where_an_uninterrupted_one_does(tmp_path):
    torch = pytest.importorskip("torch")
    graph = Graph.from_dict(_resumable_design())
    straight, broken = tmp_path / "straight", tmp_path / "broken"
    for folder in (straight, broken):
        folder.mkdir()
    done = _run(straight, export_training(graph, "pytorch", straight))
    assert done.returncode == 0, done.stderr[-2000:]
    script = export_training(graph, "pytorch", broken)
    crash = ("def hook(epoch, model):\n    if epoch == 3:\n"
             "        raise SystemExit('power cut')\nm.EPOCH_HOOK = hook")
    cut = _run(broken, script, crash)
    assert cut.returncode != 0 and "epoch 2/4" in cut.stdout and "epoch 3/4" not in cut.stdout
    assert json.loads((broken / "resume.json").read_text(encoding="utf-8"))["epoch"] == 2
    resumed = _run(broken, script, "m.RESUME = True")
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert "resuming after epoch 2" in resumed.stdout and "epoch 1/4" not in resumed.stdout
    a = torch.load(next(straight.glob("*_best.pt")))
    b = torch.load(next(broken.glob("*_best.pt")))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert json.loads((straight / "metrics.json").read_text(encoding="utf-8")) == json.loads(
        (broken / "metrics.json").read_text(encoding="utf-8"))
    assert not (broken / "resume.pt").exists() and not (broken / "resume.json").exists()


@pytest.mark.slow
def test_a_stopped_run_resumes_as_the_same_run(isolated_home, tmp_path):
    pytest.importorskip("torch")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(_resumable_design(epochs=6)))
        assert mgr.wait(run_id, 600)["state"] == "finished"
        with pytest.raises(api.ApiError, match="only a stopped, failed or interrupted"):
            api.resume_run(run_id)
        # make it look stopped after epoch 2: the files a stop would have left
        folder = mgr.history.path(run_id)
        _run(folder, next(folder.glob("*_train_pytorch.py")),
             "def hook(epoch, model):\n    if epoch == 3:\n        raise SystemExit(1)\n"
             "m.EPOCH_HOOK = hook")
        mgr.history.update(run_id, status="stopped")
        assert resume.saved_epoch(mgr, run_id) == 2
        assert api.resume_run(run_id) == {"run_id": run_id, "after_epoch": 2}
        assert mgr.wait(run_id, 600)["state"] == "finished"
        record = mgr.history.get(run_id)
        assert record.status == "finished" and record.epochs_done == 6
        assert [e["epoch"] for e in mgr.history.epochs(run_id)] == [1, 2, 3, 4, 5, 6]
        assert resume.saved_epoch(mgr, run_id) is None
    finally:
        api.set_manager(None)


def test_resume_says_why_a_run_cannot(isolated_home, tmp_path):
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        record = mgr.history.create(tiny_classifier_dict(), framework="pytorch")
        mgr.history.finalize(record.run_id, "stopped", 1)
        with pytest.raises(api.ApiError, match="PyTorch"):
            api.resume_run(record.run_id)
        export_training(Graph.from_dict(tiny_classifier_dict()), "pytorch",
                        mgr.history.path(record.run_id))
        with pytest.raises(api.ApiError, match="no epoch to resume"):
            api.resume_run(record.run_id)
        mgr.history.update(record.run_id, backend="ssh:gpu-box")
        with pytest.raises(api.ApiError, match="did not come back from ssh:gpu-box"):
            api.resume_run(record.run_id)
        mgr.history.update(record.run_id, backend="a-removed-backend")
        with pytest.raises(api.ApiError, match="not set up any more"):
            api.resume_run(record.run_id)
        with pytest.raises(KeyError):
            api.resume_run("nope")
    finally:
        api.set_manager(None)


def test_keep_epochs_recomputes_the_record(tmp_path):
    history = RunHistory(tmp_path / "runs")
    record = history.create(tiny_classifier_dict(), framework="pytorch")
    for epoch, loss in enumerate([0.9, 0.5, 0.7, 0.2], start=1):
        history.append_epoch(record.run_id, {"epoch": epoch, "metrics": {"val_loss": loss}})
    history.keep_epochs(record.run_id, 3)
    kept = history.get(record.run_id)
    assert kept.epochs_done == 3 and kept.best_metrics["val_loss"] == 0.5
    assert [e["epoch"] for e in history.epochs(record.run_id)] == [1, 2, 3]


def test_the_template_saves_and_clears_its_resume_state():
    from ai_made_easy.core.training.generate import generate_training

    source = generate_training(Graph.from_dict(tiny_classifier_dict()), "pytorch")
    assert "RESUME = False" in source and "save_resume(epoch, resumable" in source
    assert "Path(RESUME_FILE).unlink(missing_ok=True)" in source


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_resume_restores_rng_states_mapped_to_an_accelerator(tmp_path, monkeypatch, device):
    torch = pytest.importorskip("torch")
    import numpy as np

    script = export_training(Graph.from_dict(tiny_classifier_dict()), "pytorch", tmp_path)
    generated = runpy.run_path(str(script))

    class MappedState:
        """Stand in for a tensor moved to an accelerator by map_location."""

        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self.value

    saved = [torch.Generator().manual_seed(seed).get_state() for seed in range(5)]
    numpy_rng = np.random.RandomState(42)
    state = {
        "model": object(), "shuffle": MappedState(saved[0]), "epoch": 2,
        "best": (0.5, {}, 1),
        "rng": {"torch": MappedState(saved[1]), "numpy": numpy_rng.get_state(),
                "cuda": [MappedState(s) for s in saved[2:4]], "mps": MappedState(saved[4])},
    }
    load = Mock(return_value=state)
    monkeypatch.setattr(torch, "load", load)
    shuffle, cpu_rng, mps_rng = (torch.Generator() for _ in range(3))
    cuda_rngs = [torch.Generator(), torch.Generator()]
    restored_numpy = np.random.RandomState()
    monkeypatch.setattr(torch, "set_rng_state", cpu_rng.set_state)
    monkeypatch.setattr(np.random, "set_state", restored_numpy.set_state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def restore_cuda(states):
        for rng, value in zip(cuda_rngs, states, strict=True):
            rng.set_state(value)

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore_cuda)
    monkeypatch.setattr(torch.mps, "set_rng_state", mps_rng.set_state)
    model = Mock()
    objects = {"model": model, "shuffle": generated["GeneratorState"](shuffle)}
    assert generated["load_resume"](objects, device) == (2, state["best"])
    load.assert_called_once_with("resume.pt", map_location=device, weights_only=False)
    model.load_state_dict.assert_called_once_with(state["model"])
    for rng, expected in zip([shuffle, cpu_rng, *cuda_rngs, mps_rng], saved, strict=True):
        assert torch.equal(rng.get_state(), expected)
    np.testing.assert_array_equal(restored_numpy.random_sample(10), numpy_rng.random_sample(10))


# ------------------------------------------------------------------ train boosted trees more
BOOSTERS = [
    ("ml.xgb_classifier", {"n_estimators": 20}),
    ("ml.lgbm_classifier", {"n_estimators": 20}),
    ("ml.catboost_classifier", {"iterations": 20}),
    ("ml.gradient_boosting_classifier", {"n_estimators": 20}),
    ("ml.hist_gradient_boosting_classifier", {"max_iter": 20}),
]


def _boosted_design(table: Path, type_id: str, params: dict) -> Graph:
    from ai_made_easy.core import recipes

    facts = recipes.detect(table)
    graph = recipes.build("gradient_boosting", facts.task, facts, {}).to_dict()
    graph["nodes"] = [n for n in graph["nodes"] if n["type"] != "train.kfold"]
    model = next(n for n in graph["nodes"] if n["id"] == "model")
    model["type"], model["params"] = type_id, params
    return Graph.from_dict(graph)


@pytest.mark.slow
@pytest.mark.parametrize("type_id,params", BOOSTERS, ids=[b[0] for b in BOOSTERS])
def test_a_boosted_model_trains_more_trees(type_id, params, isolated_home, tmp_path):
    """The model keeps its 20 trees and grows 15 more. The count comes from the run (the
    boosting library stays out of this process: two OpenMP runtimes can crash it)."""
    pytest.importorskip("sklearn")
    pytest.importorskip({"ml.xgb": "xgboost", "ml.lgb": "lightgbm", "ml.cat": "catboost"}.get(
        type_id[:6], "sklearn"))
    from test_v38_features import _region_table

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(_boosted_design(_region_table(tmp_path / "t.csv"), type_id, params))
        assert mgr.wait(run_id, 600)["state"] == "finished", mgr.history.get(run_id).error
        more = api.train_more(run_id, 10, trees=15)["run_id"]
        status = mgr.wait(more, 600)
        assert status["state"] == "finished", mgr.history.get(more).error[-2000:]
        record = mgr.history.get(more)
        assert "15 more trees" in record.note and "accuracy" in record.final_metrics
        assert record.final_metrics["trees"] == 35
    finally:
        api.set_manager(None)


# ------------------------------------------------------------------ learning-rate finder (Keras)
@pytest.mark.slow
def test_the_finder_runs_a_keras_design(isolated_home, tmp_path):
    pytest.importorskip("keras")
    from ai_made_easy.core import lr_find

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = api.find_learning_rate(tiny_classifier_dict(), steps=30,
                                        framework="keras")["run_id"]
        assert mgr.wait(run_id, 600)["state"] == "finished", mgr.history.get(run_id).error
        out = api.learning_rate_result(run_id)
        assert len(out["lrs"]) >= 10 and len(out["lrs"]) == len(out["losses"])
        assert out["lrs"][0] == pytest.approx(lr_find.START)
        assert all(b > a for a, b in zip(out["lrs"], out["lrs"][1:], strict=False))
        assert out["suggested"] and out["start"] <= out["suggested"] <= out["end"]
        assert mgr.history.get(run_id).framework == "keras"
        assert not list(mgr.history.path(run_id).glob("*.keras"))     # no model saved
        first = out["raw"][:3]
        assert all(v > 0 for v in first)    # per-batch losses recovered from Keras' means
    finally:
        api.set_manager(None)


def test_the_finder_needs_a_framework_that_trains(isolated_home, tmp_path):
    api.set_manager(RunManager(RunHistory(tmp_path / "runs")))
    try:
        with pytest.raises(api.ApiError):
            api.find_learning_rate(tiny_classifier_dict(), framework="sklearn")
    finally:
        api.set_manager(None)


@pytest.mark.slow
def test_the_cli_finds_and_applies_a_learning_rate(isolated_home, tmp_path):
    pytest.importorskip("torch")
    from ai_made_easy.cli import main

    project = tmp_path / "tiny.json"
    project.write_text(json.dumps(tiny_classifier_dict(lr=0.5)), encoding="utf-8")
    api.set_manager(None)
    try:
        assert main(["lr-find", str(project), "--steps", "30", "--apply"]) == 0
    finally:
        api.manager().stop_all()
        api.set_manager(None)
    saved = json.loads(project.read_text(encoding="utf-8"))
    lr = next(n["params"]["lr"] for n in saved["nodes"] if n["type"] == "train.sgd")
    assert lr != 0.5 and 1e-7 <= lr <= 10
