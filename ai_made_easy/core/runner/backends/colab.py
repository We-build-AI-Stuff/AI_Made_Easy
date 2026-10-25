"""Colab backend: a notebook hand-off (Colab has no job API).

Two ways back, neither needs a password or token in AI Made Easy:

- **Google Drive for desktop** (found automatically, or the ``drive_folder`` option):
  the run's bundle and notebook go to ``My Drive/AI Made Easy``. Open the notebook in
  Colab and choose Run all; it writes events and checkpoints back to the same Drive
  folder, which Drive for desktop syncs here, and the app follows them (per poll).
- **Manual:** the notebook (bundle inside) is saved in the run folder. Upload it to
  Colab, run it, and it downloads a results zip; ``aime runs import <zip>`` (or the
  Import results button) completes the run.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ai_made_easy.core.runner.backends import BackendError, RemoteBackend
from ai_made_easy.core.runner.backends.bundle import LAUNCHER_NAME
from ai_made_easy.core.runner.backends.ssh import STALE
from ai_made_easy.core.runs.history import FINAL_STATES

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

APP_FOLDER = "AI Made Easy"
COLAB_DRIVE = "/content/drive/MyDrive"
GPUS = {"T4": "t4", "L4": "l4", "A100": "a100_40", "none": "cpu_laptop"}


def find_drive_folder() -> Path | None:
    """The local "My Drive" folder of Google Drive for desktop, if it is installed."""
    home = Path.home()
    candidates = sorted((home / "Library" / "CloudStorage").glob("GoogleDrive-*/My Drive"))
    candidates += [home / "Google Drive" / "My Drive", home / "My Drive",
                   Path("G:/My Drive")]
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def notebook(run: TrainingRun, gpu: str, drive_run_dir: str | None) -> dict:
    """The hand-off notebook: through Drive when ``drive_run_dir`` is set, else
    self-contained (the bundle inside) ending in a results download."""
    accel = ("Runtime → Change runtime type → " + gpu + " GPU, then " if gpu != "none" else "")
    intro = (f"# AI Made Easy run {run.run_id}\n\n"
             f"Design **{run.name}**. {accel}Runtime → Run all.\n\n")
    if drive_run_dir:
        intro += ("Events and checkpoints are written to your Google Drive "
                  f"(`{APP_FOLDER}/runs/{run.run_id}`); AI Made Easy follows them there.")
        cells = [
            _cell("markdown", intro),
            _cell("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
            _cell("code", f"RUN = {drive_run_dir!r}\n"
                          f'!python "{{RUN}}/{LAUNCHER_NAME}" "{{RUN}}" '
                          '--bundle "{RUN}/bundle.tar.gz" --install'),
            _cell("code", "drive.flush_and_unmount()  # make sure Drive has every file"),
        ]
    else:
        intro += ("When it finishes, the last cell downloads a results zip: import it with "
                  "**Import results** in AI Made Easy or `aime runs import <zip>`.")
        cells = [
            _cell("markdown", intro),
            _cell("code", f"%%writefile /content/{LAUNCHER_NAME}\n"
                          + run.bundle.launcher(embed=True)),
            _cell("code", f"!python /content/{LAUNCHER_NAME} /content/aime_run --install"),
            _cell("code", "import shutil\nfrom google.colab import files\n"
                          f"zip_path = shutil.make_archive('/content/aime_run_{run.run_id}', "
                          "'zip', '/content/aime_run')\nfiles.download(zip_path)"),
        ]
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"accelerator": "GPU" if gpu != "none" else "None",
                         "colab": {"provenance": [], "gpuType": gpu},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}}}


def drive_file_id(path: Path) -> str:
    """The Drive id Google Drive for desktop records on a synced file ("" if unknown)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["xattr", "-p", "com.google.drivefs.item-id#S", str(path)],
                                 capture_output=True, encoding="utf-8", errors="replace", timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return out.stdout.strip() if out.returncode == 0 else ""
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return ""
    try:
        return getxattr(str(path), "user.drive.id").decode().strip()
    except OSError:
        return ""


class ColabBackend(RemoteBackend):
    kind = "colab"
    live = False
    options_help = {
        "drive_folder": "your local Google Drive \"My Drive\" folder (found automatically)",
        "gpu": "the runtime to ask for: T4, L4, A100 or none (default T4)",
        "manual": "skip Drive: save the notebook and import a results zip (default false)",
        "poll_seconds": "how often to look for new events in Drive (default 10)",
    }

    def __init__(self, name: str = "", **options) -> None:  # noqa: ANN003
        super().__init__(name, **options)
        self.gpu = str(options.get("gpu") or "T4")
        if self.gpu not in GPUS:
            raise BackendError(f"Colab gpu must be one of {', '.join(GPUS)}")
        self.poll = float(options.get("poll_seconds", 10))

    def drive(self) -> Path | None:
        if self.options.get("manual"):
            return None
        folder = self.options.get("drive_folder")
        if folder:
            path = Path(str(folder)).expanduser()
            if not path.is_dir():
                raise BackendError(f"the Drive folder {path} does not exist")
            return path
        return find_drive_folder()

    def accelerator(self) -> dict:
        return {"gpu": "" if self.gpu == "none" else self.gpu, "device": GPUS[self.gpu]}

    def test(self) -> dict:
        try:
            drive = self.drive()
        except BackendError as exc:
            return {"ok": False, "message": str(exc), "accelerator": self.accelerator()}
        if drive is None:
            return {"ok": True, "accelerator": self.accelerator(),
                    "message": "no Google Drive for desktop folder: runs use a notebook "
                               "and a results zip (Import results)"}
        return {"ok": True, "accelerator": self.accelerator(),
                "message": f"runs sync through Google Drive at {drive}"}

    def launch(self, run: TrainingRun, python: str) -> None:
        try:
            drive = self.drive()
            run.accelerator = self.accelerator()
            self.check_budget(run)
            if drive is None:
                path = run.workdir / f"{run.run_id}_colab.ipynb"
                path.write_text(json.dumps(notebook(run, self.gpu, None), indent=1), encoding="utf-8")
                run.remote_url = str(path)
                self.note(run, f"notebook saved: {path}. Upload it at "
                               "https://colab.research.google.com (File → Upload notebook), "
                               "Run all, then import the results zip it downloads.")
            else:
                folder = drive / APP_FOLDER / "runs" / run.run_id
                folder.mkdir(parents=True, exist_ok=True)
                for name in STALE:   # a resumed run's last attempt would end the follow
                    (folder / name).unlink(missing_ok=True)
                (folder / "bundle.tar.gz").write_bytes(run.bundle.tar_bytes())
                (folder / LAUNCHER_NAME).write_text(run.bundle.launcher(), encoding="utf-8")
                book = drive / APP_FOLDER / f"{run.run_id}.ipynb"
                remote_dir = f"{COLAB_DRIVE}/{APP_FOLDER}/runs/{run.run_id}"
                book.write_text(json.dumps(notebook(run, self.gpu, remote_dir), indent=1), encoding="utf-8")
                run.remote_dir = str(folder)
                run.remote_url = str(book)
                self.note(run, f"notebook in Google Drive: {APP_FOLDER}/{run.run_id}.ipynb — "
                               "open it in Colab (File → Open notebook → Google Drive) and "
                               "Run all. Curves arrive as Drive syncs.")
            if run.history is not None:
                run.history.update(run.run_id, remote=self.remote_state(run))
        except Exception as exc:  # noqa: BLE001
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
            return
        if getattr(run, "remote_dir", ""):
            threading.Thread(target=self._follow, args=(run,), daemon=True).start()

    def remote_state(self, run: TrainingRun) -> dict:
        return {"dir": getattr(run, "remote_dir", ""), "notebook": run.remote_url,
                "accelerator": run.accelerator, "manual": not getattr(run, "remote_dir", "")}

    def reattach(self, run: TrainingRun) -> None:
        if getattr(run, "remote_dir", ""):
            threading.Thread(target=self._follow, args=(run,), daemon=True).start()

    def _follow(self, run: TrainingRun) -> None:
        folder = Path(run.remote_dir)
        events, exit_file = folder / "events.jsonl", folder / "exit_code"
        announced = False
        while run.state not in FINAL_STATES:
            if not announced and run.remote_url:
                file_id = drive_file_id(Path(run.remote_url))
                if file_id:
                    announced = True
                    self.note(run, f"open in Colab: https://colab.research.google.com/drive/"
                                   f"{file_id}")
            finished = exit_file.exists()
            if events.exists():
                lines = events.read_text(errors="replace", encoding="utf-8").split("\n")
                complete = lines[:-1]  # the last piece may still be syncing
                for line in complete[run.remote_offset:]:
                    self.feed(run, line)
            if finished:
                for item in folder.iterdir():
                    if item.name in ("bundle.tar.gz", LAUNCHER_NAME, "pid", "exit_code"):
                        continue
                    target = run.workdir / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
                code = exit_file.read_text(encoding="utf-8").strip()
                self.finish(run, int(code) if code.lstrip("-").isdigit() else 1)
                return
            time.sleep(self.poll)

    def stop(self, run: TrainingRun) -> None:
        self.note(run, "stopped following; stop the notebook in Colab (Runtime → Interrupt)")
        run.process_exited(130)
