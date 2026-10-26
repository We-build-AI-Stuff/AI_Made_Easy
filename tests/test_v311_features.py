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
                 "        print(f'breaking after {time.time() - began:.1f}s', file=sys.stderr)\n"
                 "        raise RuntimeError('the first process broke')\n"
                 "    real()\n"
                 "m.main = main")
# where a slow stop spends its time: every thread's stack, if it's still going after 40 s
_TIMED = ("import faulthandler, sys, time\n"
          "began = time.time()\n"
          "faulthandler.dump_traceback_later(40, exit=False)\n")


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="forks; the spawn test covers Windows")
@pytest.mark.parametrize("breaks", [_WORKER_BREAKS, _FIRST_BREAKS], ids=["worker", "first"])
def test_a_failing_process_stops_the_run(tmp_path, breaks):
    """A process that fails leaves the others waiting for it in the next exchange: the
    run stops instead of hanging until the backend's timeout."""
    import time

    pytest.importorskip("torch")
    started = time.time()
    done = _run(tmp_path, _script(_design(gpus=2, epochs=50), tmp_path), _TIMED + breaks)
    assert done.returncode != 0
    took = time.time() - started
    assert took < 120, f"{took:.0f}s\n{done.stdout[-1500:]}\n{done.stderr[-8000:]}"
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
@pytest.mark.parametrize("mixed", [False, True], ids=["float32", "mixed_float16"])
def test_keras_retries_from_the_state_before_compiling(tmp_path, mixed):
    """A failed jit_compile that already took a step: the retry starts from the weights
    and optimizer state from before, so the run ends as one that never compiled. With
    mixed precision the optimizer compiled is the loss-scaling wrapper, whose state counts
    too."""
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("checked on the torch backend")
    plain = tmp_path / "plain"
    _ok(_run(plain, _script(_dropout(_design(epochs=2, mixed_precision=mixed)), plain,
                            "keras")))
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
    out = _ok(_run(retried, _script(_dropout(_design(epochs=2, compile=True,
                                                    mixed_precision=mixed)), retried, "keras"), broken))
    assert "jit_compile did not work" in out
    assert "retry: step 0, momentum 0.0" in out       # the failed step's momentum is gone
    for name in ("metrics.json", "predictions.json"):   # and dropout draws as if never compiled
        read = lambda f: json.loads((f / name).read_text(encoding="utf-8"))  # noqa: E731
        assert read(retried) == read(plain), name


def _dropout(design: dict) -> dict:
    """The tiny design with dropout after its hidden layer (a layer that draws randomly)."""
    design["nodes"].append({"id": "drop", "type": "core.dropout", "params": {"p": 0.3},
                            "position": [0, 0]})
    design["edges"] = [e for e in design["edges"] if e["from"] != "act/out"]
    design["edges"] += [{"from": "act/out", "to": "drop/in"}, {"from": "drop/out", "to": "d2/in"}]
    return design


# ------------------------------------------------------------ Phase 1: bigger models
def test_bfloat16_needs_no_loss_scaling():
    from ai_made_easy.core.training.generate import generate_training

    source = generate_training(Graph.from_dict(_design(precision="bfloat16",
                                                       mixed_precision=True)), "pytorch")
    assert 'PRECISION = "bfloat16"' in source
    assert 'device.type == "cuda" and amp_dtype == torch.float16' in source   # float16 only
    # sharded, each process holds a shard of the gradients: they agree on skipping a step
    assert 'ShardedGradScaler("cuda", enabled=scaling)' in source
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
    # each GPU holds the whole weights once, before they are split
    assert model_part(sharded) >= sharded.params * 4


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
    averaged = Graph.from_dict(_design(gpus=4, mixed_precision=True, ema_decay=0.99))
    fixed = fixes._fit_training_memory(averaged)                 # sharding keeps no EMA
    assert not fixed or fixed[0] != "Shard the model"


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


_QAT_FAILS_LATE = ("import torch\n"
                   "def broken(*args, **kwargs):\n"
                   "    raise RuntimeError('disk full')\n"
                   "torch.export.save = broken")


@pytest.mark.slow
def test_quantization_aware_training_failing_late_keeps_the_float_results(tmp_path):
    """QAT is optional: a failure after fine-tuning (here saving the program) is reported
    and the run still writes its metrics."""
    pytest.importorskip("torch")
    pytest.importorskip("executorch")
    pytest.importorskip("torchao")
    out = _ok(_run(tmp_path, _script(_design(qat_epochs=1), tmp_path), _QAT_FAILS_LATE))
    assert "quantization-aware training skipped: RuntimeError: disk full" in out
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "accuracy" in metrics and "int8_accuracy" not in metrics
    assert not (tmp_path / "qat_int8.pt2").exists()


def test_quantization_aware_training_skips_what_it_cant_quantize():
    from ai_made_easy.core.training.generate import generate_training

    design = _design(qat_epochs=1)
    loss = next(n for n in design["nodes"] if n["id"] == "loss")
    loss.update(type="train.loss_arcface", params={"classes": 3})
    source = generate_training(Graph.from_dict(design), "pytorch")
    assert "QAT_EPOCHS = 0" in source
    assert "quantization-aware training is for models with one input and output" in source


_COPIES = ("import copy\n"
           "import torch\n"
           "model = m.Tiny()\n"
           "assert m.checkpoint_layers(model) > 0\n"
           "twin = copy.deepcopy(model)\n"
           "with torch.no_grad():\n"
           "    for p in twin.parameters():\n"
           "        p.zero_()\n"
           "x = torch.randn(4, 16)\n"
           "for mode in ('train', 'eval'):\n"
           "    getattr(twin, mode)(); getattr(model, mode)()\n"
           "    assert torch.equal(twin(x), torch.zeros(4, 3)), mode   # its own (zero) weights\n"
           "    assert not torch.equal(model(x), torch.zeros(4, 3)), mode\n"
           "print('copies run their own layers')\n"
           "raise SystemExit(0)")


def test_a_copy_of_a_checkpointed_model_runs_its_own_layers(tmp_path):
    """EMA, SWA and quantization-aware training copy the model: the copy's checkpointing
    wrappers must call the copy's layers, not the original's."""
    pytest.importorskip("torch")
    script = _script(_design(gradient_checkpointing=True), tmp_path)
    assert "copies run their own layers" in _ok(_run(tmp_path, script, _COPIES))


# ------------------------------------------------------------ fixed along the way
_CUT_SHORT = ("import pathlib\n"
              "real = pathlib.Path.write_text\n"
              "def write_text(self, data, *args, **kwargs):\n"
              "    if self.name.startswith('resume.json') and '\"epoch\": 3' in data:\n"
              "        real(self, data[:5], *args, **kwargs)\n"
              "        raise SystemExit('stopped while saving')\n"
              "    return real(self, data, *args, **kwargs)\n"
              "pathlib.Path.write_text = write_text")


@pytest.mark.slow
@pytest.mark.parametrize("framework", ["pytorch", "keras"])
def test_a_run_stopped_while_saving_still_resumes(tmp_path, framework):
    """A stop in the middle of writing resume.json (a fast run writes it many times a
    second) leaves the previous epoch's, whole, so the run still resumes."""
    from test_v39_features import _resumable_design

    if framework == "keras":
        keras = pytest.importorskip("keras")
        if keras.backend.backend() != "torch":
            pytest.skip("checked on the torch backend")
    else:
        pytest.importorskip("torch")
    cut = _run(tmp_path, _script(_resumable_design(), tmp_path, framework), _CUT_SHORT)
    assert cut.returncode != 0 and "stopped while saving" in cut.stderr, cut.stderr[-2000:]
    assert json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))["epoch"] == 2


_STOP_BETWEEN = ("import os, pathlib, signal\n"
                 "real = pathlib.Path.replace\n"
                 "saved = []\n"
                 "def replace(self, target):\n"
                 "    if pathlib.Path(target).name in ('resume.pt', 'resume.weights.h5'):\n"
                 "        saved.append(1)\n"
                 "        if len(saved) == 3:\n"
                 "            os.kill(os.getpid(), signal.SIGTERM)   # a stop between the files\n"
                 "    return real(self, target)\n"
                 "pathlib.Path.replace = replace")


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="a stop there can't be held (TerminateProcess)")
@pytest.mark.parametrize("framework", ["pytorch", "keras"])
def test_a_stop_between_the_saved_files_waits_for_the_last(tmp_path, framework):
    """The saved state and resume.json describe the same epoch: a stop that arrives
    between them waits until both are in place, so the app says the epoch the run resumes
    after."""
    import pickle

    from test_v39_features import _resumable_design

    if framework == "keras":
        keras = pytest.importorskip("keras")
        if keras.backend.backend() != "torch":
            pytest.skip("checked on the torch backend")
    torch = pytest.importorskip("torch")
    stopped = _run(tmp_path, _script(_resumable_design(), tmp_path, framework), _STOP_BETWEEN)
    assert stopped.returncode == -15, stopped.stderr[-2000:]       # stopped, by the SIGTERM
    assert json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))["epoch"] == 3
    if framework == "pytorch":
        assert torch.load(tmp_path / "resume.pt", weights_only=False)["epoch"] == 3
    else:
        assert pickle.loads((tmp_path / "resume.extra.pkl").read_bytes())["epoch"] == 3
