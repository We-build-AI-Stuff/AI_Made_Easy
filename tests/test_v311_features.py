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
