"""Train anywhere (3.2 Phase 1): run backends.

SSH and Docker run for real against stand-in clients (``tests/fakes``) that execute
the "remote" side on this machine, so the launcher, follower, upload, download and
reconnect paths are the shipped code. Kaggle and Modal are faked at their SDK
boundary; Colab syncs through a folder standing in for Google Drive for desktop.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner import backends
from ai_made_easy.core.runner.backends import bundle as bundle_mod
from ai_made_easy.core.runner.backends.kaggle import KaggleBackend
from ai_made_easy.core.runner.backends.modal_backend import ModalBackend
from ai_made_easy.core.runner.manager import RunManager

FAKES = Path(__file__).parent / "fakes"


@pytest.fixture()
def manager(isolated_home):
    return RunManager()


@pytest.fixture()
def fake_ssh(tmp_path, monkeypatch):
    """ssh to any host lands here; the "remote" python3 is this interpreter."""
    import os

    if os.name == "nt":
        pytest.skip("the fake SSH host runs POSIX shell scripts (real hosts are Linux)")

    bin_dir = tmp_path / "remote-bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    (bin_dir / "python3").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AIME_SSH", f"{sys.executable} {FAKES / 'fake_ssh.py'}")
    monkeypatch.setenv("FAKE_SSH_HOME", str(tmp_path / "remote-home"))
    return tmp_path / "remote-home"


def _design(epochs: int = 2, samples: int = 0) -> Graph:
    graph = tiny_classifier_dict(epochs=epochs)
    if samples:  # a slower run, for tests that stop it part-way
        next(n for n in graph["nodes"] if n["id"] == "data")["params"]["n_samples"] = samples
    return Graph.from_dict(graph)


def _csv_design(tmp_path: Path) -> Graph:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 4))
    frame = pd.DataFrame(x, columns=list("abcd"))
    frame["y"] = (x[:, 0] + x[:, 1] > 0).astype(int)
    (tmp_path / "local data").mkdir()
    path = tmp_path / "local data" / "table.csv"
    frame.to_csv(path, index=False)
    graph = tiny_classifier_dict(epochs=1)
    for node in graph["nodes"]:
        if node["id"] == "data":
            node.update(type="data.csv", params={"path": str(path), "format": "csv",
                                                  "target_column": "y"})
        if node["id"] == "in":
            node["params"] = {"shape": "4"}
        if node["id"] == "d2":
            node["params"] = {"units": 2}
    return Graph.from_dict(graph)


def _final(manager: RunManager, run_id: str, timeout: float = 180) -> dict:
    status = manager.wait(run_id, timeout=timeout)
    return {"status": status, "record": manager.history.get(run_id),
            "epochs": manager.history.epochs(run_id)}


# ------------------------------------------------------------------ resolve / store

def test_resolve_kinds_and_saved_backends(isolated_home):
    assert backends.resolve(None).kind == "local"
    assert backends.resolve("ssh:gpu-box").host == "gpu-box"
    assert backends.resolve("kaggle").kind == "kaggle"
    with pytest.raises(backends.BackendError, match="unknown backend"):
        backends.resolve("nowhere")
    with pytest.raises(backends.BackendError, match="no option"):
        backends.save("box", "ssh", host="h", password="secret")
    backends.save("box", "ssh", host="gpu-box", python="/opt/venv/bin/python")
    resolved = backends.resolve("box")
    assert (resolved.kind, resolved.host, resolved.python) == ("ssh", "gpu-box",
                                                               "/opt/venv/bin/python")
    names = [b["name"] for b in backends.listing()]
    assert names[:2] == ["local", "box"] and {"kaggle", "colab", "modal"} <= set(names)
    stored = (isolated_home / "backends.json").read_text(encoding="utf-8")
    assert "secret" not in stored
    backends.remove("box")
    assert backends.saved() == []


def test_gpu_names_pick_device_profiles():
    ids = {d["id"] for d in api.list_devices()["devices"]}
    for name, device in (("Tesla T4", "t4"), ("Tesla P100-PCIE-16GB", "p100"),
                         ("NVIDIA A100-SXM4-80GB", "a100_80"), ("NVIDIA A100-SXM4-40GB",
                                                                "a100_40"),
                         ("NVIDIA L4", "l4"), ("NVIDIA A10G", "a10g"), ("H100", "h100")):
        assert backends.device_for_gpu(name) == device
        assert device in ids
    assert backends.device_for_gpu("Unknown GPU") == ""


# ------------------------------------------------------------------ bundles

def test_bundle_relocates_local_data(tmp_path, isolated_home):
    graph = _csv_design(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = bundle_mod.prepare(graph, "pytorch", "r1", run_dir)
    script = (run_dir / bundle.script).read_text(encoding="utf-8")
    assert str(tmp_path) not in script, "no local path may reach the remote script"
    assert "data/data_path_table.csv" in script
    with tarfile.open(fileobj=io.BytesIO(bundle.tar_bytes()), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert {bundle.script, "_aime_worker.py", "bundle.json", "requirements.txt",
            "data/data_path_table.csv"} <= names
    assert "torch" in bundle.requirements
    # the original design is untouched
    assert graph.nodes["data"].params["path"].endswith("table.csv")
    assert Path(graph.nodes["data"].params["path"]).is_absolute()


def test_embedded_bundle_has_a_size_limit(tmp_path, monkeypatch, isolated_home):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = bundle_mod.prepare(_design(), "pytorch", "r1", run_dir)
    assert "BUNDLE = \"" in bundle.launcher(embed=True)
    monkeypatch.setattr(bundle_mod, "MAX_EMBEDDED_BYTES", 100)
    with pytest.raises(ValueError, match="Kaggle dataset"):
        bundle.launcher(embed=True)


def test_launcher_runs_a_bundle_and_reports_failures(tmp_path, isolated_home):
    import subprocess

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bundle = bundle_mod.prepare(_design(epochs=1), "pytorch", "r1", run_dir)
    launcher = tmp_path / "aime_launch.py"
    launcher.write_text(bundle.launcher(embed=True, args=[str(tmp_path / "far")]), encoding="utf-8")
    out = subprocess.run([sys.executable, str(launcher)], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    events = [json.loads(line) for line in (tmp_path / "far" / "events.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    assert [e["type"] for e in events if e["type"] in ("epoch", "done")] == ["epoch", "done"]
    assert (tmp_path / "far" / "exit_code").read_text(encoding="utf-8") == "0"
    # a broken bundle still leaves an exit code and the reason
    bad = subprocess.run([sys.executable, str(bundle_mod.launcher_path()),
                          str(tmp_path / "bad"), "--bundle", str(tmp_path / "missing.tgz")],
                         capture_output=True, text=True)
    assert bad.returncode == 1
    assert (tmp_path / "bad" / "exit_code").read_text(encoding="utf-8") == "1"
    assert "missing.tgz" in (tmp_path / "bad" / "events.jsonl").read_text(encoding="utf-8")


# ------------------------------------------------------------------ SSH

def test_ssh_run_matches_a_local_run(manager, fake_ssh):
    local = _final(manager, manager.start(_design()))
    remote_id = manager.start(_design(), on="ssh:gpu-box")
    remote = _final(manager, remote_id)
    assert remote["record"].status == "finished", remote["record"].error
    assert remote["record"].backend == "ssh:gpu-box"
    assert [e["metrics"] for e in remote["epochs"]] == [e["metrics"] for e in local["epochs"]]
    assert remote["record"].final_metrics == local["record"].final_metrics
    assert remote["record"].remote["host"] == "gpu-box"
    # the checkpoint came back; the launcher's own files did not
    workdir = manager.history.path(remote_id)
    assert list(workdir.glob("*.pt")), sorted(p.name for p in workdir.iterdir())
    assert not (workdir / "bundle.tar.gz").exists()
    assert any("connected" in line for line in manager.get(remote_id).logs)


def test_ssh_run_survives_a_dropped_connection(manager, fake_ssh, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SSH_DROP", str(tmp_path / "dropped"))
    run_id = manager.start(_design(epochs=3), on="ssh:gpu-box")
    result = _final(manager, run_id)
    assert (tmp_path / "dropped").exists(), "the follower connection was never dropped"
    assert result["record"].status == "finished", result["record"].error
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3]  # none lost, none twice
    assert any("reconnecting" in line for line in manager.get(run_id).logs)


def test_ssh_unreachable_host_fails_with_a_reason(manager, fake_ssh):
    run_id = manager.start(_design(), on="ssh:unreachable")
    result = _final(manager, run_id, timeout=60)
    assert result["record"].status == "failed"
    assert "Connection refused" in result["record"].error


def test_ssh_stop_and_reattach(manager, fake_ssh, isolated_home):
    run_id = manager.start(_design(epochs=500, samples=40000), on="ssh:gpu-box")
    deadline = time.time() + 120
    while not manager.history.epochs(run_id) and time.time() < deadline:
        time.sleep(0.2)
    assert manager.history.epochs(run_id), "no epoch recorded after 120s"
    # the app "restarts": the old follower no longer writes the record, and a new
    # manager follows the still-running remote run again
    manager.get(run_id).history = None
    later = RunManager()
    assert later.history.mark_interrupted() == 0  # remote runs are not marked failed
    later.attach(run_id)
    deadline = time.time() + 60
    while not later.get(run_id).epochs and time.time() < deadline:
        time.sleep(0.2)
    assert later.get(run_id).epochs, "the reattached follower saw no epochs"
    later.stop(run_id)
    status = later.wait(run_id, timeout=60)
    assert status["state"] == "stopped"
    assert later.history.get(run_id).status == "stopped"


def test_ssh_probe_reports_python_and_accelerator(fake_ssh):
    result = backends.resolve("ssh:gpu-box").test()
    assert result["ok"] and result["message"].startswith("gpu-box: Python 3")
    assert result["accelerator"]["device"] in ("cpu_laptop", "apple_m_series")
    assert not backends.resolve("ssh:unreachable").test()["ok"]
    parsed = backends.resolve("ssh:x")._parse_accelerator(["Tesla T4, 15360 MiB", "Linux x86"])
    assert parsed == {"gpu": "Tesla T4", "memory_gb": 15.0, "device": "t4"}


# ------------------------------------------------------------------ container

def test_container_run(manager, tmp_path, monkeypatch):
    log = tmp_path / "docker.log"
    monkeypatch.setenv("AIME_DOCKER", f"{sys.executable} {FAKES / 'fake_docker.py'}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    run_id = manager.start(_design(epochs=1), on="container")
    result = _final(manager, run_id)
    assert result["record"].status == "finished", result["record"].error
    assert len(result["epochs"]) == 1 and result["record"].final_metrics
    calls = log.read_text(encoding="utf-8").splitlines()
    assert any(c.startswith("build -t aime-train:") for c in calls)
    assert any(c.startswith("run --rm --name aime-") for c in calls)
    assert not (manager.history.path(run_id) / "container").exists()


# ------------------------------------------------------------------ Kaggle

class FakeKaggle:
    """The Kaggle SDK boundary: push stores the kernel, the next status polls run it."""

    instances: list[FakeKaggle] = []

    def __init__(self) -> None:
        self.pushed: dict = {}
        self.polls = 0
        FakeKaggle.instances.append(self)

    def username(self) -> str:
        return "ada"

    def push(self, folder: str) -> str:
        self.pushed = {"meta": json.loads(Path(folder, "kernel-metadata.json").read_text(encoding="utf-8")),
                       "code": Path(folder, "aime_kernel.py").read_text(encoding="utf-8")}
        return "https://www.kaggle.com/code/ada/kernel"

    def status(self, slug: str) -> str:
        self.polls += 1
        return "running" if self.polls < 2 else "complete"

    def output(self, slug: str, dest: str) -> None:
        import subprocess

        # what Kaggle does: run the kernel; /kaggle/working becomes the outputs
        # a forward-slash path: pasted into the kernel's source, C:\U… would be an escape
        code = self.pushed["code"].replace("/kaggle/working", Path(dest).as_posix())
        Path(dest, "kernel.py").write_text(code.replace(", '--install'", ""), encoding="utf-8")
        subprocess.run([sys.executable, str(Path(dest, "kernel.py"))], check=False,
                       capture_output=True, timeout=300)
        Path(dest, "kernel.py").unlink()

    def cancel(self, slug: str) -> bool:
        return True


def test_kaggle_run_completes_with_its_record(manager, monkeypatch):
    monkeypatch.setattr(KaggleBackend, "client_factory", FakeKaggle)
    backends.save("kag", "kaggle", accelerator="P100", poll_seconds=0.05,
                  datasets=["ada/shapes"])
    run_id = manager.start(_design(epochs=2), on="kag")
    result = _final(manager, run_id)
    assert result["record"].status == "finished", result["record"].error
    assert [e["epoch"] for e in result["epochs"]] == [1, 2]
    assert result["record"].final_metrics
    meta = FakeKaggle.instances[-1].pushed["meta"]
    assert meta["id"] == f"ada/aime-{run_id}" and meta["is_private"] is True
    assert meta["machine_shape"] == "NvidiaTeslaP100" and meta["dataset_sources"] == [
        "ada/shapes"]
    assert result["record"].remote["accelerator"]["device"] == "p100"
    assert result["status"]["live"] is False
    logs = manager.get(run_id).logs
    assert any("kernel running" in line for line in logs)


# ------------------------------------------------------------------ Modal

class FakeModal:
    """The Modal SDK boundary: ``run`` executes the launcher locally, streaming."""

    def __init__(self) -> None:
        self.folder: Path | None = None
        self.code = 1

    def run(self, run_id, bundle, launcher, *, gpu, requirements, timeout):  # noqa: ANN001
        import subprocess
        import tempfile

        self.folder = Path(tempfile.mkdtemp())
        (self.folder / "bundle.tar.gz").write_bytes(bundle)
        proc = subprocess.Popen([sys.executable, "-c", launcher, str(self.folder / run_id),
                                 "--bundle", str(self.folder / "bundle.tar.gz")],
                                stdout=subprocess.PIPE, text=True)
        yield from proc.stdout
        self.code = proc.wait()

    def returncode(self) -> int:
        return self.code

    def download(self, run_id, dest):  # noqa: ANN001
        import shutil

        for item in (self.folder / run_id).iterdir():
            if item.name not in ("pid", "exit_code"):
                shutil.copy2(item, dest / item.name)

    def terminate(self) -> None:
        pass


def test_modal_run_streams_live(manager, monkeypatch):
    monkeypatch.setattr(ModalBackend, "client_factory", FakeModal)
    design = _design(epochs=2)
    design.meta["budget"] = {"max_params_m": 0.0001}  # 163 parameters: over
    run_id = manager.start(design, on="modal")
    result = _final(manager, run_id)
    # the budget is checked on the backend's GPU, and the run still goes ahead
    assert any("warning: estimated parameters" in line and "T4" in line
               for line in manager.get(run_id).logs), manager.get(run_id).logs[:5]
    assert result["record"].status == "finished", result["record"].error
    assert len(result["epochs"]) == 2 and result["record"].final_metrics
    assert result["record"].remote["accelerator"]["device"] == "t4"
    assert list(manager.history.path(run_id).glob("*.pt"))


def test_modal_asks_for_several_gpus(manager, monkeypatch):
    asked = []

    class Counting(FakeModal):
        def run(self, *args, gpu, **kwargs):  # noqa: ANN002, ANN003
            asked.append(gpu)
            yield from super().run(*args, gpu=gpu, **kwargs)

    monkeypatch.setattr(ModalBackend, "client_factory", Counting)
    backends.save("modal-pair", "modal", gpu="A100", gpu_count=2)
    run_id = manager.start(_design(epochs=1), on="modal-pair")
    result = _final(manager, run_id)
    assert asked == ["A100:2"] and result["record"].status == "finished"
    assert result["record"].remote["accelerator"] == {"gpu": "A100:2", "count": 2,
                                                      "device": "a100_40"}
    for bad in ({"gpu_count": 9}, {"gpu_count": "two"}, {"gpu": "none", "gpu_count": 2}):
        with pytest.raises(backends.BackendError, match="gpu_count must be 1 to 8"):
            ModalBackend("m", **bad)


# ------------------------------------------------------------------ Colab

def _run_notebook(book: dict, drive: Path) -> None:
    """Execute a hand-off notebook the way Colab would (Drive mounted at ``drive``)."""
    import subprocess

    code = "".join(book["cells"][2]["source"])
    run = code.split("RUN = ", 1)[1].splitlines()[0]
    run_dir = eval(run).replace("/content/drive/MyDrive", str(drive))  # noqa: S307
    subprocess.run([sys.executable, f"{run_dir}/aime_launch.py", run_dir, "--bundle",
                    f"{run_dir}/bundle.tar.gz"], check=True, capture_output=True, timeout=300)


def test_colab_through_drive_for_desktop(manager, tmp_path):
    drive = tmp_path / "My Drive"
    drive.mkdir()
    backends.save("colab-drive", "colab", drive_folder=str(drive), poll_seconds=0.1)
    run_id = manager.start(_design(epochs=2), on="colab-drive")
    book_path = drive / "AI Made Easy" / f"{run_id}.ipynb"
    deadline = time.time() + 30
    while not book_path.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert book_path.exists(), "no notebook after 30s"
    book = json.loads(book_path.read_text(encoding="utf-8"))
    assert "drive.mount" in "".join(book["cells"][1]["source"])
    assert manager.status(run_id)["state"] == "running"
    _run_notebook(book, drive)
    result = _final(manager, run_id, timeout=60)
    assert result["record"].status == "finished", result["record"].error
    assert [e["epoch"] for e in result["epochs"]] == [1, 2]
    assert result["record"].final_metrics


def test_colab_clears_a_previous_attempt_in_its_drive_folder(manager, tmp_path, monkeypatch):
    """A resumed Colab run reuses its Drive folder: the last attempt's exit code and events
    are removed, or the follow would end the new attempt before the notebook runs."""
    from ai_made_easy.core.runs import history

    monkeypatch.setattr(history, "new_run_id", lambda: "20260101-000000-abcd")
    drive = tmp_path / "My Drive"
    folder = drive / "AI Made Easy" / "runs" / "20260101-000000-abcd"
    folder.mkdir(parents=True)
    (folder / "exit_code").write_text("0", encoding="utf-8")
    (folder / "events.jsonl").write_text('{"type": "epoch", "epoch": 1}\n', encoding="utf-8")
    backends.save("colab-drive", "colab", drive_folder=str(drive), poll_seconds=0.1)
    run_id = manager.start(_design(epochs=2), on="colab-drive")
    assert run_id == "20260101-000000-abcd"
    deadline = time.time() + 30
    while not (drive / "AI Made Easy" / f"{run_id}.ipynb").exists() and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(1)                      # several polls of the folder
    assert not (folder / "exit_code").exists() and not (folder / "events.jsonl").exists()
    assert manager.status(run_id)["state"] == "running"
    manager.stop(run_id)


def test_colab_manual_hand_off_and_import(manager, tmp_path):
    import shutil
    import subprocess

    backends.save("colab-zip", "colab", manual=True)
    run_id = manager.start(_design(epochs=2), on="colab-zip")
    book_path = manager.history.path(run_id) / f"{run_id}_colab.ipynb"
    deadline = time.time() + 30
    while not book_path.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert book_path.exists(), "no notebook after 30s"
    book = json.loads(book_path.read_text(encoding="utf-8"))
    # in Colab: the %%writefile cell, the run cell, then the zip download
    source = "".join(book["cells"][1]["source"]).split("\n", 1)[1]
    (tmp_path / "content").mkdir()
    (tmp_path / "content" / "aime_launch.py").write_text(source, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "content" / "aime_launch.py"),
                    str(tmp_path / "content" / "aime_run")], check=True, capture_output=True,
                   timeout=300)
    zip_path = shutil.make_archive(str(tmp_path / f"aime_run_{run_id}"), "zip",
                                   str(tmp_path / "content" / "aime_run"))
    assert manager.status(run_id)["state"] == "running"
    later = RunManager()  # the import can happen after a restart
    status = later.import_results(zip_path)
    assert status["state"] == "finished"
    record = later.history.get(run_id)
    assert record.status == "finished" and record.final_metrics
    assert len(later.history.epochs(run_id)) == 2
    with pytest.raises(ValueError, match="already finished"):
        later.import_results(zip_path)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        later.import_results(bad)


# ------------------------------------------------------------------ API surface

def test_api_lists_tests_and_starts_on_backends(manager, fake_ssh):
    api.set_manager(manager)
    try:
        listed = api.list_backends()
        assert listed["backends"][0]["name"] == "local"
        assert api.test_backend("ssh:gpu-box")["ok"]
        saved = api.save_backend("box", "ssh", {"host": "gpu-box"})
        assert saved["name"] == "box"
        started = api.start_training(tiny_classifier_dict(epochs=1), on="box")
        assert started["backend"] == "box"
        assert manager.wait(started["run_id"], timeout=180)["state"] == "finished"
        with pytest.raises(api.ApiError, match="unknown backend"):
            api.start_training(tiny_classifier_dict(epochs=1), on="nowhere")
        api.delete_backend("box")
        assert all(b["name"] != "box" for b in api.list_backends()["backends"])
    finally:
        api.set_manager(None)


def test_web_routes(manager, fake_ssh, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    api.set_manager(manager)
    try:
        client = TestClient(create_app(token="", projects_dir=tmp_path / "projects"))
        assert client.get("/api/backends").json()["backends"][0]["name"] == "local"
        saved = client.post("/api/backends", json={"name": "box", "kind": "ssh",
                                                   "options": {"host": "gpu-box"}})
        assert saved.status_code == 200, saved.text
        assert client.post("/api/backends", json={"name": "bad", "kind": "ssh",
                                                  "options": {"token": "x"}}).status_code == 400
        tested = client.post("/api/backends/test", json={"on": "box"}).json()
        assert tested["ok"] and tested["kind"] == "ssh"
        started = client.post("/api/runs", json={"graph": tiny_classifier_dict(epochs=1),
                                                 "on": "box"}).json()
        assert manager.wait(started["run_id"], timeout=180)["state"] == "finished"
        assert client.get(f"/api/runs/{started['run_id']}").json()["backend"] == "box"
        assert client.post("/api/runs/import", content=b"").status_code == 400
        assert client.delete("/api/backends/box").json() == {"deleted": "box"}
    finally:
        api.set_manager(None)


@pytest.mark.skipif(not __import__("os").environ.get("AIME_SSH_TEST_HOST"),
                    reason="set AIME_SSH_TEST_HOST to a host in ~/.ssh/config (CI: localhost)")
def test_real_ssh_run_matches_local_and_survives_a_cut(manager):
    """Real OpenSSH (CI runs sshd on localhost): same metrics as a local run, and a killed
    connection is picked up again from the last event."""
    import os

    host = os.environ["AIME_SSH_TEST_HOST"]
    backends.save("real", "ssh", host=host, python=sys.executable,
                  remote_root=".aime-remote-test")
    local = _final(manager, manager.start(_design()))
    remote = _final(manager, manager.start(_design(), on="real"))
    assert remote["record"].status == "finished", remote["record"].error
    assert [e["metrics"] for e in remote["epochs"]] == [e["metrics"] for e in local["epochs"]]

    run_id = manager.start(_design(epochs=40, samples=20000), on="real")
    run = manager.get(run_id)
    deadline = time.time() + 120
    while not run.epochs and time.time() < deadline:
        time.sleep(0.2)
    assert run.epochs, run.logs[-5:]
    run.process.kill()  # the follower's ssh connection dies mid-run
    result = _final(manager, run_id, timeout=300)
    assert result["record"].status == "finished", result["record"].error
    assert [e["epoch"] for e in result["epochs"]] == list(range(1, 41))
    assert any("reconnecting" in line for line in run.logs)


@pytest.mark.skipif(not (__import__("os").environ.get("AIME_ONLINE")
                         and __import__("os").environ.get("KAGGLE_KEY")),
                    reason="needs AIME_ONLINE=1 and a Kaggle token (the nightly job's secret)")
def test_real_kaggle_kernel(manager):
    """One real Kaggle kernel (CPU, so it does not wait for a GPU slot)."""
    backends.save("kaggle-ci", "kaggle", accelerator="none", poll_seconds=20)
    run_id = manager.start(_design(epochs=2), on="kaggle-ci")
    result = _final(manager, run_id, timeout=3600)
    assert result["record"].status == "finished", result["record"].error
    assert [e["epoch"] for e in result["epochs"]] == [1, 2]
    assert result["record"].final_metrics


def test_the_gpu_check_trains_valid_designs_and_reports():
    """scripts/gpu_check.py runs on real GPUs only by hand; here its designs generate
    the training each case asks for, and its report renders."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import gpu_check

    from ai_made_easy.core.graph import Graph
    from ai_made_easy.core.training.generate import generate_training

    for name, (settings, _expect) in gpu_check.CASES.items():
        graph = Graph.from_dict(gpu_check.design(settings))
        assert not [i for i in graph.validate() if i.severity == "error"], name
        source = generate_training(graph, "pytorch")
        assert "GPUS = 2" in source and f"SHARDED = {settings['sharding']}" in source
        assert f'PRECISION = "{settings["precision"]}"' in source
    text = gpu_check.summary({"gpu": "L4", "date": "2026-10-27", "results": {
        "ddp_float16": {"ok": True, "accuracy": 0.91, "why": ""},
        "sharded_bfloat16": {"ok": None, "why": "T4 has no bfloat16"}}})
    assert "| ddp_float16 | passed | 0.910 |  |" in text and "| sharded_bfloat16 | skipped" in text
