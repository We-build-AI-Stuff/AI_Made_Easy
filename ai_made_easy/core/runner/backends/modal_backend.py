"""Modal backend: train in a Modal sandbox on a cloud GPU, with live logs.

Sign-in is Modal's own (``modal token new`` writes ``~/.modal.toml``). The bundle goes
to a Modal volume (``aime-runs``), the launcher runs in a sandbox whose image has the
run's requirements, events stream back live, and the outputs come back from the volume.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Iterator

from ai_made_easy.core.runner.backends import BackendError, RemoteBackend, device_for_gpu

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

APP_NAME = "ai-made-easy"
VOLUME = "aime-runs"
GPUS = ("none", "T4", "L4", "A10G", "A100", "A100-80GB", "H100")
_SKIP = ("bundle.tar.gz", "pid", "exit_code")


class ModalClient:
    """The Modal SDK calls the backend needs."""

    def __init__(self) -> None:
        try:
            import modal
        except ImportError as exc:
            raise BackendError("Modal needs the modal package: "
                               "pip install 'ai-made-easy[modal]'") from exc
        self.modal = modal
        self.sandbox = None

    def check(self) -> str:
        """Raises unless Modal is signed in; returns the workspace name if known."""
        try:
            self.modal.App.lookup(APP_NAME, create_if_missing=True)
        except Exception as exc:  # noqa: BLE001 — the SDK raises its own auth errors
            raise BackendError(f"Modal is not signed in (run `modal token new`): {exc}") from exc
        return "Modal"

    def _volume(self):  # noqa: ANN202
        return self.modal.Volume.from_name(VOLUME, create_if_missing=True)

    def run(self, run_id: str, bundle: bytes, launcher: str, *, gpu: str,
            requirements: list[str], timeout: int) -> Iterator[str]:
        modal = self.modal
        volume = self._volume()
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(bundle), f"/{run_id}/bundle.tar.gz")
        image = modal.Image.debian_slim(python_version="3.11")
        if requirements:
            image = image.pip_install(*requirements)
        app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.sandbox = modal.Sandbox.create(
            "python", "-c", launcher, f"/vol/{run_id}", "--bundle", f"/vol/{run_id}/bundle.tar.gz",
            image=image, gpu=None if gpu == "none" else gpu, volumes={"/vol": volume},
            app=app, timeout=timeout)
        for line in self.sandbox.stdout:
            yield line
        self.sandbox.wait()

    def returncode(self) -> int:
        code = getattr(self.sandbox, "returncode", None)
        return 1 if code is None else int(code)

    def download(self, run_id: str, dest: Path) -> None:
        volume = self._volume()
        for entry in volume.listdir(f"/{run_id}", recursive=True):
            if "FILE" not in str(getattr(entry, "type", "")).upper():
                continue
            rel = PurePosixPath(str(entry.path).lstrip("/")).relative_to(run_id)
            if rel.name in _SKIP or ".." in rel.parts:
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                for chunk in volume.read_file(str(entry.path)):
                    fh.write(chunk)

    def terminate(self) -> None:
        if self.sandbox is not None:
            self.sandbox.terminate()


MOST_GPUS = {"A10G": 4}   # GPUs per sandbox; the others take up to 8


def _count(value) -> int:  # noqa: ANN001
    """gpu_count as given: absent means 1; anything but a whole number is 0 (rejected),
    so 2.9 or 0 never quietly become another count."""
    if value is None or value == "":
        return 1
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else 0


class ModalBackend(RemoteBackend):
    kind = "modal"
    options_help = {
        "gpu": f"one of {', '.join(GPUS)} (default T4)",
        "gpu_count": "GPUs in the sandbox, 1 to 8 (A10G: 4; default 1); the Trainer's GPUs "
                     "uses them",
        "timeout_minutes": "the longest a run may take (default 240)",
    }
    client_factory = ModalClient

    def __init__(self, name: str = "", **options) -> None:  # noqa: ANN003
        super().__init__(name, **options)
        self.gpu = str(options.get("gpu") or "T4")
        if self.gpu not in GPUS:
            raise BackendError(f"Modal gpu must be one of {', '.join(GPUS)}")
        self.gpu_count = _count(options.get("gpu_count"))
        most = MOST_GPUS.get(self.gpu, 8)
        if not 1 <= self.gpu_count <= most or (self.gpu == "none" and self.gpu_count > 1):
            raise BackendError(f"Modal gpu_count must be a whole number from 1 to {most}"
                               + (" (A10G machines have 4)" if most == 4 else ", with a gpu"))
        self.timeout = int(float(options.get("timeout_minutes", 240)) * 60)

    def gpu_spec(self) -> str:
        """What Modal is asked for: ``A100`` or, for several, ``A100:2``."""
        return self.gpu if self.gpu_count == 1 else f"{self.gpu}:{self.gpu_count}"

    def accelerator(self) -> dict:
        if self.gpu == "none":
            return {"gpu": "", "device": "cpu_laptop"}
        return {"gpu": self.gpu_spec(), "count": self.gpu_count,
                "device": device_for_gpu(self.gpu, 80 if "80" in self.gpu else None)}

    def test(self) -> dict:
        try:
            self.client_factory().check()
        except BackendError as exc:
            return {"ok": False, "message": str(exc), "accelerator": self.accelerator()}
        return {"ok": True, "message": "signed in to Modal", "accelerator": self.accelerator()}

    def launch(self, run: TrainingRun, python: str) -> None:
        threading.Thread(target=self._launch, args=(run,), daemon=True).start()

    def _launch(self, run: TrainingRun) -> None:
        try:
            client = self.client_factory()
            run.client = client
            run.accelerator = self.accelerator()
            self.check_budget(run)
            if run.history is not None:
                run.history.update(run.run_id, remote=self.remote_state(run))
            self.note(run, f"starting a Modal sandbox ({self.gpu_spec()}); the first run builds "
                           "the image, which takes a few minutes")
            for line in client.run(run.run_id, run.bundle.tar_bytes(), run.bundle.launcher(),
                                   gpu=self.gpu_spec(), requirements=run.bundle.requirements,
                                   timeout=self.timeout):
                self.feed(run, line)
            code = client.returncode()
            client.download(run.run_id, run.workdir)
        except Exception as exc:  # noqa: BLE001
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
            return
        self.finish(run, code)

    def remote_state(self, run: TrainingRun) -> dict:
        return {"volume": VOLUME, "path": f"/{run.run_id}", "accelerator": run.accelerator}

    def stop(self, run: TrainingRun) -> None:
        client = getattr(run, "client", None)
        if client is not None:
            try:
                client.terminate()
            except Exception as exc:  # noqa: BLE001 — say so: the GPU may still be billing
                self.note(run, f"could not stop the Modal job ({exc}); it may still be "
                               "running: stop it with `modal app stop` or the Modal dashboard")
