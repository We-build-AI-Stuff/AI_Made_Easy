"""3.11 features. Phase 0: several GPUs on Windows, with k-fold and metric losses; Keras
compiles."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict
from test_v39_features import _run

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import python_executable


def _design(**trainer) -> dict:
    design = tiny_classifier_dict(epochs=3)
    node = next(n for n in design["nodes"] if n["type"] == "train.trainer")
    node["params"].update(device="cpu", **trainer)
    return design


def _script(design: dict, folder: Path, framework: str = "pytorch") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return export_training(Graph.from_dict(design), framework, folder)


def _ok(done: subprocess.CompletedProcess) -> str:
    assert done.returncode == 0, (done.stdout[-2000:], done.stderr[-3000:])
    return done.stdout


# ------------------------------------------------------------ Phase 0: several GPUs
@pytest.mark.slow
def test_spawned_processes_train_together_through_the_worker(tmp_path):
    """Windows can't fork: the other processes are spawned from the script, get the
    settings changed since, and this process (the worker's, whose output becomes the
    run's events) trains as the first one."""
    pytest.importorskip("torch")
    script = _script(_design(gpus=2), tmp_path)
    source = script.read_text(encoding="utf-8")
    script.write_text(source.replace('START_METHOD = "spawn" if sys.platform == "win32" '
                                     'else "fork"', 'START_METHOD = "spawn"'),
                      encoding="utf-8")
    assert 'START_METHOD = "spawn"\n' in script.read_text(encoding="utf-8")
    done = subprocess.run([python_executable(), "-m", "ai_made_easy.worker.train_worker",
                           str(script)], cwd=tmp_path, capture_output=True, encoding="utf-8",
                          errors="replace", timeout=600)
    out = _ok(done)
    assert "training on 2 CPU processes (DistributedDataParallel)" in out
    epochs = [json.loads(line) for line in out.splitlines()
              if line.startswith("{") and '"type": "epoch"' in line]
    assert [e["epoch"] for e in epochs] == [1, 2, 3]          # one report, the first's
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["accuracy"] > 0.5


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="forks; the spawn test covers Windows")
def test_k_fold_trains_across_processes(tmp_path):
    pytest.importorskip("torch")
    design = _design(gpus=2)
    design["nodes"].append({"id": "kf", "type": "train.kfold", "params": {"k": 3},
                            "position": [0, 0]})
    out = _ok(_run(tmp_path, _script(design, tmp_path)))
    assert "training on 2 CPU processes" in out
    assert [line.split(":")[0] for line in out.splitlines() if line.startswith("fold ")] == \
        ["fold 1/3", "fold 2/3", "fold 3/3"]               # scored once, by the first
    import numpy as np

    oof = np.load(tmp_path / "oof.npz")
    assert len(oof["targets"]) == len(oof["outputs"]) and set(oof["fold"]) == {0, 1, 2}


_CENTRES = ("import torch\n"
            "real = m.fit\n"
            "def fit(*args, **kwargs):\n"
            "    model, loss_fn = real(*args, **kwargs)\n"
            "    torch.save(loss_fn.state_dict(), f'centres{m.RANK}.pt')\n"
            "    return model, loss_fn\n"
            "m.fit = fit")


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="forks; the spawn test covers Windows")
def test_metric_loss_centres_stay_the_same_in_every_process(tmp_path):
    """ArcFace's class centres live in the loss, outside DistributedDataParallel: their
    gradients are averaged too, so each process ends with the same centres."""
    torch = pytest.importorskip("torch")
    design = _design(gpus=2)
    loss = next(n for n in design["nodes"] if n["id"] == "loss")
    loss.update(type="train.loss_arcface", params={"classes": 3})
    _ok(_run(tmp_path, _script(design, tmp_path), _CENTRES))
    first, second = (torch.load(tmp_path / f"centres{r}.pt") for r in (0, 1))
    assert first.keys() == second.keys() and first
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_k_fold_and_metric_losses_keep_their_gpus():
    from ai_made_easy.core.training.generate import generate_training

    design = _design(gpus=2)
    design["nodes"].append({"id": "kf", "type": "train.kfold", "params": {"k": 3},
                            "position": [0, 0]})
    source = generate_training(Graph.from_dict(design), "pytorch")
    assert "GPUS = 2" in source and "several GPUs train networks without" not in source


# ------------------------------------------------------------ Phase 0: Keras compiles
@pytest.mark.slow
def test_keras_compiles_when_asked(tmp_path):
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("checked on the torch backend")
    design = _design(compile=True)
    script = _script(design, tmp_path, "keras")
    assert "COMPILE = True" in script.read_text(encoding="utf-8")
    out = _ok(_run(tmp_path, script))
    assert "compiling with jit_compile" in out
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["accuracy"] > 0.4


@pytest.mark.slow
def test_keras_trains_without_compiling_when_it_fails(tmp_path):
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("checked on the torch backend")
    script = _script(_design(compile=True, epochs=2), tmp_path, "keras")
    broken = ("import keras\n"
              "real = keras.Model.fit\n"
              "def fit(self, *args, **kwargs):\n"
              "    if self.jit_compile:\n"
              "        raise RuntimeError('cannot compile this model')\n"
              "    return real(self, *args, **kwargs)\n"
              "keras.Model.fit = fit")
    out = _ok(_run(tmp_path, script, broken))
    assert "jit_compile did not work (RuntimeError: cannot compile this model): " \
           "training without" in out
    assert (tmp_path / "metrics.json").is_file()


_WORKER_BREAKS = ("def broken(*args):\n"
                  "    raise SystemExit(3)\n"
                  "m._one_of_many = broken")
_FIRST_BREAKS = ("real = m.main\n"
                 "def main():\n"
                 "    if m.RANK == 0 and m.WORLD > 1:\n"
                 "        raise RuntimeError('the first process broke')\n"
                 "    real()\n"
                 "m.main = main")


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="forks; the spawn test covers Windows")
@pytest.mark.parametrize("breaks", [_WORKER_BREAKS, _FIRST_BREAKS], ids=["worker", "first"])
def test_a_failing_process_stops_the_run(tmp_path, breaks):
    """A process that fails leaves the others waiting for it in the next exchange: the
    run stops instead of hanging until the backend's timeout."""
    import time

    pytest.importorskip("torch")
    started = time.time()
    done = _run(tmp_path, _script(_design(gpus=2, epochs=50), tmp_path), breaks)
    assert done.returncode != 0
    assert time.time() - started < 120
    if breaks is _WORKER_BREAKS:
        assert "a training process stopped (exit code 3): stopping the run" in done.stdout
    else:
        assert "the first process broke" in done.stderr


def test_spawned_processes_warn_about_settings_they_cant_get(tmp_path):
    pytest.importorskip("torch")
    script = _script(_design(gpus=2), tmp_path)
    done = _run(tmp_path, script, "m.START_METHOD = 'spawn'\n"
                                  "m.INIT_HOOK = lambda model: None\n"
                                  "m._train_as = lambda *a: None\n"
                                  "import torch.multiprocessing as mp\n"
                                  "class Fake:\n"
                                  "    processes = []\n"
                                  "    def join(self): return True\n"
                                  "mp.start_processes = lambda *a, **k: Fake()")
    out = _ok(done)
    assert ("warning: INIT_HOOK can't be passed to the other processes here (they start "
            "from this file): they train without them") in out


@pytest.mark.slow
def test_keras_retries_from_the_state_before_compiling(tmp_path):
    """A failed jit_compile that already took a step: the retry starts from the weights
    and optimizer state from before, so the run ends as one that never compiled."""
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("checked on the torch backend")
    plain = tmp_path / "plain"
    _ok(_run(plain, _script(_design(epochs=2), plain, "keras")))
    retried = tmp_path / "retried"
    broken = ("import keras\n"
              "real = keras.Model.fit\n"
              "def fit(self, x, y, *args, **kwargs):\n"
              "    if self.jit_compile:\n"
              "        self.compile(optimizer=self.optimizer, loss=self.loss, jit_compile=False)\n"
              "        self.train_on_batch(x[:8], y[:8])     # a step, then the failure\n"
              "        self.jit_compile = True\n"
              "        raise RuntimeError('cannot compile this model')\n"
              "    import numpy as np\n"
              "    moved = sum((float(np.abs(np.array(v)).sum()) for v in self.optimizer.variables\n"
              "                 if 'momentum' in v.path), 0.0)\n"
              "    print(f'retry: step {int(self.optimizer.iterations)}, momentum {moved}', flush=True)\n"
              "    return real(self, x, y, *args, **kwargs)\n"
              "keras.Model.fit = fit")
    out = _ok(_run(retried, _script(_design(epochs=2, compile=True), retried, "keras"),
                   broken))
    assert "jit_compile did not work" in out
    assert "retry: step 0, momentum 0.0" in out       # the failed step's momentum is gone
    read = lambda f: json.loads((f / "metrics.json").read_text(encoding="utf-8"))  # noqa: E731
    assert read(retried) == read(plain)


# ------------------------------------------------------------ Phase 1: bigger models
def test_bfloat16_needs_no_loss_scaling():
    from ai_made_easy.core.training.generate import generate_training

    source = generate_training(Graph.from_dict(_design(precision="bfloat16",
                                                       mixed_precision=True)), "pytorch")
    assert 'PRECISION = "bfloat16"' in source
    assert "and amp_dtype == torch.float16)" in source        # the scaler only for float16
    plain = generate_training(Graph.from_dict(_design()), "pytorch")
    assert 'PRECISION = "auto"' in plain and "SHARDED = False" in plain


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="forks; the spawn test covers Windows")
@pytest.mark.parametrize("checkpointing", [False, True])
def test_sharded_processes_train_and_save_a_whole_model(tmp_path, checkpointing):
    torch = pytest.importorskip("torch")
    design = _design(gpus=2, sharding=True, epochs=4, gradient_checkpointing=checkpointing)
    out = _ok(_run(tmp_path, _script(design, tmp_path)))
    assert "training on 2 CPU processes (FSDP: the model is split between them)" in out
    assert [line.split()[1] for line in out.splitlines() if line.startswith("epoch")] == \
        ["1/4", "2/4", "3/4", "4/4"]
    weights = torch.load(next(tmp_path.glob("*_best.pt")))
    assert all(not hasattr(w, "placements") for w in weights.values())   # plain tensors
    assert weights["dense_1.weight"].shape == (8, 16)                     # whole, not a shard
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["accuracy"] > 0.5
    assert not (tmp_path / "resume.pt").exists()                 # sharded runs don't resume


def test_sharding_checks_and_memory():
    from ai_made_easy.core import budget
    from ai_made_easy.core.training.catalog import TRAINER

    messages = [m for _, m in TRAINER.checks({"sharding": True, "gpus": 1})]
    assert any("set GPUs to 2 or more" in m for m in messages)
    assert ("error", "sharded training keeps no averaged weights: turn off EMA / SWA, or "
                     "sharding") in TRAINER.checks({"sharding": True, "gpus": 2,
                                                    "ema_decay": 0.99})
    whole = budget.estimate(Graph.from_dict(_design(gpus=4)))
    sharded = budget.estimate(Graph.from_dict(_design(gpus=4, sharding=True)))
    assert sharded.sharded and not whole.sharded
    model_part = lambda e: e.train_memory_bytes(batch_size=0) - (e.device.overhead_gb * budget.GB  # noqa: E731
                                                               if e.device else 0)
    assert model_part(sharded) < model_part(whole)


def test_the_fix_for_memory_shards_a_model_on_several_gpus(monkeypatch):
    from ai_made_easy.core import budget, fixes

    graph = Graph.from_dict(_design(gpus=4, mixed_precision=True))   # already on
    est = budget.estimate(graph)
    whole, sharded_bytes = est.train_memory_bytes(), None
    est.sharded = True
    sharded_bytes = est.train_memory_bytes()
    limit = (whole + sharded_bytes) / 2 / budget.GB              # fits only when sharded
    monkeypatch.setattr(budget, "check", lambda g: [type("C", (), {"kind": "train_memory",
                                                                   "limit": limit})()])
    fixed = fixes._fit_training_memory(graph)
    assert fixed and fixed[0] == "Shard the model"
    trainer = next(n for n in fixed[2].nodes.values() if n.type_id == "train.trainer")
    assert trainer.params["sharding"] is True


@pytest.mark.slow
def test_quantization_aware_training_feeds_the_int8_export(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("executorch")
    pytest.importorskip("torchao")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    run_id = mgr.start(Graph.from_dict(_design(qat_epochs=2)), "pytorch")
    status = mgr.wait(run_id, 600)
    assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
    run = mgr.history.path(run_id)
    logs = "\n".join(mgr.get(run_id).logs)
    assert "quantization-aware epoch 2/2" in logs and "int8 (quantization-aware) test:" in logs
    assert (run / "qat_int8.pt2").is_file()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["int8_accuracy"] > 0.5
    result = build_package(run, tmp_path / "pkg", formats=("executorch_int8",))
    entry = result.formats["executorch_int8"]
    assert entry["ok"], entry
    assert "quantization-aware trained" in entry["note"]


def test_quantization_aware_training_skips_what_it_cant_quantize():
    from ai_made_easy.core.training.generate import generate_training

    design = _design(qat_epochs=1)
    loss = next(n for n in design["nodes"] if n["id"] == "loss")
    loss.update(type="train.loss_arcface", params={"classes": 3})
    source = generate_training(Graph.from_dict(design), "pytorch")
    assert "QAT_EPOCHS = 0" in source
    assert "quantization-aware training is for models with one input and output" in source


# ------------------------------------------------------------ Phase 2: fairness mitigation
def _weights_code() -> dict:
    import numpy as np

    from ai_made_easy.core.training.data_template import GROUP_WEIGHTS

    ns = {"np": np}
    exec(GROUP_WEIGHTS, ns)  # noqa: S102 — the generated helper itself
    return ns


def test_group_weights_make_the_group_say_nothing_about_the_class():
    import numpy as np

    weights = _weights_code()["group_weights"]
    groups = np.array(["n"] * 80 + ["s"] * 20)
    labels = np.array([1] * 60 + [0] * 20 + [1] * 2 + [0] * 18)   # n: 75% positive, s: 10%
    w = weights(groups, labels)
    overall = (w * labels).sum() / w.sum()
    for g in ("n", "s"):
        mine = groups == g
        assert (w[mine] * labels[mine]).sum() / w[mine].sum() == pytest.approx(overall)
        assert w[mine].sum() == pytest.approx(mine.sum())        # each group keeps its size


def _shifted_table(path: Path, n: int = 1200) -> Path:
    """east's features are measured low, so a model trained without the region
    under-scores it: its positives are missed far more often."""
    import random

    rng = random.Random(0)
    rows = ["a,b,region,label"]
    for i in range(n):
        a, b = rng.random(), rng.random()
        region = ("north", "south", "east")[i % 3]
        label = int(a + b > 1)
        if region == "east":
            a, b = a * 0.6, b * 0.6
        rows.append(f"{a:.3f},{b:.3f},{region},{label}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _weighted(design: dict, **params) -> dict:
    design["nodes"].append({"id": "cb", "type": "prep.class_balance", "position": [0, 0],
                            "params": {"strategy": "group weights", **params}})
    return design


def test_group_weights_need_a_column_a_table_and_one_split(tmp_path):
    from test_v38_features import _network

    from ai_made_easy.core.codegen import CodegenError
    from ai_made_easy.core.training.generate import generate_training

    network = _network(_shifted_table(tmp_path / "t.csv", 30)).to_dict()
    with pytest.raises(CodegenError, match="set Class Balancing's group column"):
        generate_training(Graph.from_dict(_weighted(json.loads(json.dumps(network)))),
                          "pytorch")
    with pytest.raises(CodegenError, match="is the target"):
        generate_training(Graph.from_dict(_weighted(json.loads(json.dumps(network)),
                                                    group_column="label")), "pytorch")
    folds = _weighted(json.loads(json.dumps(network)), group_column="region")
    folds["nodes"].append({"id": "kf", "type": "train.kfold", "params": {"k": 3},
                           "position": [0, 0]})
    with pytest.raises(CodegenError, match="remove K-Fold"):
        generate_training(Graph.from_dict(folds), "pytorch")
    with pytest.raises(CodegenError, match="CSV / Parquet table"):
        generate_training(Graph.from_dict(_weighted(_design(), group_column="region")),
                          "pytorch")


_SAMPLED = ("import numpy as np\n"
            "real = m.group_weights\n"
            "def spy(groups, labels):\n"
            "    w = real(groups, labels)\n"
            "    np.save('weights.npy', w); np.save('groups.npy', np.asarray(groups).astype(str))\n"
            "    return w\n"
            "m.group_weights = spy")


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["pytorch", "keras", "sklearn"])
def test_group_weights_train_in_every_framework(tmp_path, framework):
    """The training rows are weighted by their group and class: the weighted rows of
    each region are positive as often as all of them."""
    import numpy as np

    if framework == "keras":
        keras = pytest.importorskip("keras")
        if keras.backend.backend() != "torch":
            pytest.skip("checked on the torch backend")
    else:
        pytest.importorskip("torch" if framework == "pytorch" else "sklearn")
    from test_v38_features import _network

    from ai_made_easy.core import recipes

    table = _shifted_table(tmp_path / "t.csv", 300)
    if framework == "sklearn":
        facts = recipes.detect(table)
        design = recipes.build("random_forest", facts.task, facts, {"n_estimators": 50}).to_dict()
    else:
        design = _network(table).to_dict()
    script = _script(_weighted(design, group_column="region"), tmp_path / "run", framework)
    _ok(_run(tmp_path / "run", script, _SAMPLED))
    w, groups = np.load(tmp_path / "run" / "weights.npy"), np.load(tmp_path / "run" / "groups.npy")
    assert len(w) == len(groups) and set(groups) == {"north", "south", "east"}
    assert not np.allclose(w, 1.0)


@pytest.mark.slow
def test_a_threshold_per_group_evens_out_a_shifted_group(isolated_home, tmp_path):
    """east is under-scored: its own, lower threshold (fitted on the validation rows)
    finds its positives again, judged on the test rows."""
    pytest.importorskip("torch")
    from test_v38_features import _network

    from ai_made_easy.core import api
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = api.start_training(_network(_shifted_table(tmp_path / "t.csv")))["run_id"]
        assert mgr.wait(run_id, 600)["state"] == "finished"
        scoring = api.results_by_group(run_id, "region")["run_id"]
        assert mgr.wait(scoring, 600)["state"] == "finished"
        out = api.group_result(scoring)
        found = out["mitigation"]
        assert "east" in out["fairness"]["flagged"]
        assert found["fitted_on"] == "validation rows" and found["helps"]
        assert found["thresholds"]["east"] < min(found["thresholds"]["north"],
                                                 found["thresholds"]["south"])
        assert found["fairness_after"]["odds_gap"] < found["fairness_before"]["odds_gap"] / 2
        assert found["after"]["overall"]["accuracy"] > found["before"]["overall"]["accuracy"]
        saved = json.loads((mgr.history.path(scoring) / "group_thresholds.json")
                           .read_text(encoding="utf-8"))
        assert saved["column"] == "region" and saved["thresholds"] == found["thresholds"]
    finally:
        api.set_manager(None)


@pytest.mark.slow
def test_a_classic_classifier_says_when_thresholds_dont_help(isolated_home, tmp_path):
    """The forest sees the region and learns east's shift: there is nothing to even out,
    and the summary says so instead of claiming a gain."""
    pytest.importorskip("sklearn")
    from ai_made_easy.core import api, recipes
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        facts = recipes.detect(_shifted_table(tmp_path / "t.csv"))
        run_id = mgr.start(recipes.build("random_forest", facts.task, facts,
                                         {"n_estimators": 50}))
        assert mgr.wait(run_id, 600)["state"] == "finished"
        scoring = api.results_by_group(run_id, "region")["run_id"]
        assert mgr.wait(scoring, 600)["state"] == "finished"
        found = api.group_result(scoring)["mitigation"]
        assert found["fitted_on"].startswith("half of the test rows")
        assert found["before"]["overall"]["count"] == found["after"]["overall"]["count"]
        if not found["helps"]:
            assert found["summary"].startswith("A threshold per group doesn't help here")
    finally:
        api.set_manager(None)


# ------------------------------------------------------------ Phase 2: GGUF
def _llama_model(vocab: int = 256):
    import types

    import torch

    from ai_made_easy.core import api
    from ai_made_easy.core.codegen import generate

    design = api.read_sample("tiny_llama.json")
    for node in design["nodes"]:
        if node["id"] == "embed":
            node["params"]["num_embeddings"] = vocab
        if node["id"] == "lm_head":
            node["params"]["units"] = vocab
    if vocab > 256:
        design["nodes"].append({"id": "tok", "type": "gen.bpe_tokenizer", "position": [0, 0],
                                "params": {"vocab_size": vocab}})
    module = types.ModuleType("llama_model")
    exec(generate(Graph.from_dict(design), "pytorch"), module.__dict__)  # noqa: S102
    torch.manual_seed(0)
    model = module.build_model().eval()
    with torch.no_grad():
        for p in model.parameters():          # away from the initial ones (norms of 1)
            p.add_(torch.randn_like(p) * 0.05)
    return model


def test_gguf_takes_llama_style_decoders_only():
    pytest.importorskip("torch")
    import torch

    from ai_made_easy.core.deploy import exporters

    layout = exporters.llama_layout(_llama_model())
    assert (layout["vocab"], layout["dim"], len(layout["blocks"]), layout["heads"],
            layout["kv_heads"], layout["head_dim"]) == (256, 128, 2, 4, 2, 32)
    with pytest.raises(exporters.Skipped, match="GGUF holds Llama-style decoders"):
        exporters.llama_layout(torch.nn.Sequential(torch.nn.Linear(4, 2)))
    tokens, rules = exporters.gguf_vocabulary(258, [(116, 104), (256, 101)])   # "th", "the"
    assert len(tokens) == 258 and tokens[32] == "Ġ" and tokens[257] == "the"
    assert rules == ["t h", "th e"]
    with pytest.raises(exporters.Skipped, match="has 300 tokens but its tokenizer 258"):
        exporters.gguf_vocabulary(300, [(116, 104), (256, 101)])


def _text_tools() -> dict:
    import heapq
    import re
    from collections import Counter, defaultdict

    import numpy as np

    from ai_made_easy.core.generative import runtime

    ns = {"np": np, "re": re, "heapq": heapq, "Counter": Counter, "defaultdict": defaultdict}
    exec(runtime.DATA_CODE, ns)  # noqa: S102 — the run's own text helpers
    exec(runtime.TOKENIZER_CODE, ns)  # noqa: S102
    return ns


@pytest.mark.parametrize("bpe", [False, True], ids=["bytes", "bpe"])
def test_gguf_matches_pytorch_and_the_runs_tokens(tmp_path, bpe):
    """llama.cpp's logits are PyTorch's, token by token, and it splits text into the
    same tokens as the run (bytes, or the run's BPE merges)."""
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    llama_cpp = pytest.importorskip("llama_cpp")
    from ai_made_easy.core.deploy import exporters

    tools = _text_tools()
    text = tools["synthetic_text"](400, 0)
    merges = tools["train_bpe"](text, 300) if bpe else None
    vocab = 256 + len(merges or [])
    samples = exporters.random_samples("tensor", (64,), integer=True, vocab=vocab)
    report = exporters.export_all(_llama_model(vocab), "pytorch", ["gguf"], tmp_path, samples,
                                  merges=merges)["gguf"]
    assert report["ok"], report
    assert report["note"].startswith("matches PyTorch") and report["metrics"]["max_diff"] < 1e-4
    assert ("BPE merges" in report["note"]) == bpe
    llm = llama_cpp.Llama(model_path=str(tmp_path / "model.gguf"), vocab_only=True,
                          verbose=False)
    for sample in (text[:2000], "The robot's map isn't here: 1234 apples, naïve café!\n\n"
                                "  spaced   out \x00\x00 end"):
        expected = tools["BPE"](merges).encode(sample) if bpe else tools["encode_bytes"](sample)
        assert llm.tokenize(sample.encode(), add_bos=False, special=False) == list(expected)
        assert llm.detokenize(list(expected)).decode() == sample


@pytest.mark.slow
def test_a_trained_llama_exports_to_gguf_from_its_package(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    from test_modern_blocks import _llama_design

    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    design = _llama_design()
    for node in design["nodes"]:
        if node["type"] == "core.embedding":
            node["params"]["num_embeddings"] = 300
        if node["type"] == "core.dense":
            node["params"].update(units=300, bias=False)
    design["nodes"].append({"id": "tok", "type": "gen.bpe_tokenizer", "position": [0, 0],
                            "params": {"vocab_size": 300}})
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    run_id = mgr.start(Graph.from_dict(design))
    assert mgr.wait(run_id, 600)["state"] == "finished", mgr.history.get(run_id).error[-2000:]
    result = build_package(mgr.history.path(run_id), tmp_path / "pkg", formats=("gguf",))
    entry = result.formats["gguf"]
    assert entry["ok"], entry
    assert "BPE merges" in entry["note"] and entry["note"].startswith("matches PyTorch")
    assert (tmp_path / "pkg" / entry["file"]).is_file()
