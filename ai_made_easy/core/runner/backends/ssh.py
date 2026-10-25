"""SSH backend: train on any host in ``~/.ssh/config``.

Uses the system ``ssh`` client, so host aliases, keys, the agent, jump hosts and
``known_hosts`` work exactly as in a terminal (``BatchMode``: it never prompts for a
password). The same backend covers GPU clouds that rent SSH machines (RunPod, Lambda,
Vast.ai, Paperspace): add the machine to ``~/.ssh/config`` and use its alias.

The run keeps going if the connection drops: the launcher runs under ``nohup`` and
writes events to a file on the host; a small follower streams that file from the
last event received, and reconnects when the connection comes back.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from ai_made_easy.core.runner.backends import (
    BackendError,
    RemoteBackend,
    device_for_gpu,
    split_command,
)
from ai_made_easy.core.runner.backends.bundle import LAUNCHER_NAME, unpack_outputs
from ai_made_easy.core.runs.history import FINAL_STATES

if TYPE_CHECKING:
    from ai_made_easy.core.runner.manager import TrainingRun

# Runs on the host (standard library only): print events.jsonl from line OFFSET on,
# then {"type": "_exit", "code": N} once the launcher has written its exit code.
FOLLOWER = r'''
import json, os, sys, time
run_dir, offset = sys.argv[1], int(sys.argv[2])
events, exit_file = os.path.join(run_dir, "events.jsonl"), os.path.join(run_dir, "exit_code")
log, started = os.path.join(run_dir, "launch.log"), time.time()
seen, pos, buf = 0, 0, ""
while True:
    finished = os.path.exists(exit_file)
    if os.path.exists(events):
        with open(events, encoding="utf-8") as fh:
            fh.seek(pos)
            buf += fh.read()
            pos = fh.tell()
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            if seen >= offset:
                print(line, flush=True)
            seen += 1
    elif time.time() - started > 60 and os.path.exists(log) and os.path.getsize(log):
        # the launcher never started (no python, a bad path): report its output
        text = open(log, encoding="utf-8").read()[-3000:]
        print(json.dumps({"type": "error", "traceback": text}), flush=True)
        print(json.dumps({"type": "_exit", "code": 1}), flush=True)
        break
    if finished:
        code = open(exit_file, encoding="utf-8").read().strip()
        print(json.dumps({"type": "_exit", "code": int(code) if code.lstrip("-").isdigit()
                          else 1}), flush=True)
        break
    time.sleep(0.5)
'''

# a previous attempt's progress files, removed before a run (re)starts in its folder
STALE = ("events.jsonl", "exit_code", "pid")

# outputs left out of the download: the bundle and the launcher's own files
_SKIP = ("bundle.tar.gz", LAUNCHER_NAME, "pid", "launch.log", "exit_code")


def ssh_command() -> list[str]:
    """The ssh client (``$AIME_SSH`` overrides it, e.g. for tests)."""
    override = os.environ.get("AIME_SSH")
    if override:
        return split_command(override)
    found = shutil.which("ssh")
    if not found:
        raise BackendError("no ssh client found: install OpenSSH")
    return [found]


class SSHBackend(RemoteBackend):
    kind = "ssh"
    options_help = {
        "host": "a host alias from ~/.ssh/config (or user@hostname)",
        "python": "the Python on the host (default python3; a venv's python works too)",
        "remote_root": "folder for runs on the host (default .aime-remote in the home folder)",
        "install": "pip-install the run's requirements on the host first (default false)",
        "docker_image": "run inside this Docker image on the host (needs docker there)",
        "gpus": "with docker_image: pass --gpus all (default true)",
        "retries": "reconnect attempts after a dropped connection (default 30)",
    }

    def __init__(self, name: str = "", **options) -> None:  # noqa: ANN003
        super().__init__(name, **options)
        if not options.get("host"):
            raise BackendError("ssh backends need a host")
        self.host = str(options["host"])
        if self.host.startswith("-"):
            raise BackendError(f"invalid host {self.host!r}")
        self.python = str(options.get("python") or "python3")
        self.root = str(options.get("remote_root") or ".aime-remote")
        self.retries = int(options.get("retries", 30))
        self._accelerator: dict = {}

    @property
    def spec(self) -> str:
        return self.name if self.name != "ssh" else f"ssh:{self.host}"

    # ------------------------------------------------------------- plumbing
    def _argv(self, command: str) -> list[str]:
        return [*ssh_command(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                self.host, command]

    def _run(self, command: str, data: bytes | None = None,
             timeout: float = 300) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(self._argv(command), input=data, capture_output=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.host}: no answer within {timeout:.0f} s") from exc

    def _check(self, proc: subprocess.CompletedProcess, what: str) -> bytes:
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip()[-800:]
            raise BackendError(f"{self.host}: {what} failed (exit {proc.returncode}): {err}")
        return proc.stdout

    def _dir(self, run_id: str) -> str:
        return shlex.quote(f"{self.root}/{run_id}")

    # ---------------------------------------------------------------- probe
    def test(self) -> dict:
        probe = (f"echo aime-ok; {shlex.quote(self.python)} --version 2>&1; "
                 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null"
                 " | head -n 1; uname -sm")
        try:
            proc = self._run(probe, timeout=40)
        except BackendError as exc:
            return {"ok": False, "message": str(exc), "accelerator": {}}
        out = proc.stdout.decode(errors="replace").splitlines()
        if proc.returncode != 0 or not out or out[0].strip() != "aime-ok":
            err = proc.stderr.decode(errors="replace").strip()[-500:]
            return {"ok": False, "accelerator": {},
                    "message": f"could not connect to {self.host}: {err or 'no answer'}"}
        python = out[1].strip() if len(out) > 1 else ""
        if not python.startswith("Python 3"):
            return {"ok": False, "accelerator": {},
                    "message": f"{self.python} on {self.host} is not Python 3: {python}"}
        self._accelerator = self._parse_accelerator(out[2:])
        return {"ok": True, "message": f"{self.host}: {python}", "accelerator": self._accelerator}

    @staticmethod
    def _parse_accelerator(lines: list[str]) -> dict:
        for line in lines:
            if "," in line:  # nvidia-smi: "Tesla T4, 15360 MiB"
                name, memory = (p.strip() for p in line.split(",", 1))
                try:
                    gb = float(memory.split()[0]) / 1024
                except (ValueError, IndexError):
                    gb = None
                return {"gpu": name, "memory_gb": round(gb, 1) if gb else None,
                        "device": device_for_gpu(name, gb)}
        uname = " ".join(lines)
        if "Darwin" in uname and "arm64" in uname:
            return {"gpu": "Apple silicon", "device": "apple_m_series"}
        return {"gpu": "", "device": "cpu_laptop"} if uname else {}

    def accelerator(self) -> dict:
        return self._accelerator

    # ---------------------------------------------------------------- launch
    def _start_command(self, run: TrainingRun) -> str:
        d = self._dir(run.run_id)
        flags = "--quiet" + (" --install" if self.options.get("install") else "")
        image = self.options.get("docker_image")
        if image:
            gpus = "--gpus all " if self.options.get("gpus", True) else ""
            inner = (f"docker run --rm --name aime-{run.run_id} {gpus}-v \"$PWD\":/aime/run "
                     f"-w /aime/run {shlex.quote(str(image))} python {LAUNCHER_NAME} "
                     "/aime/run --bundle bundle.tar.gz --quiet --install")
        else:
            inner = (f"{shlex.quote(self.python)} {LAUNCHER_NAME} . --bundle bundle.tar.gz "
                     f"{flags}")
        # a resumed run reuses the folder: its last attempt's events and exit code go first,
        # or the follower (started right after) could replay them before the launcher runs.
        # Not "cd && nohup … &": that backgrounds a subshell that keeps ssh's output open
        return (f"cd {d} || exit 1; rm -f {' '.join(STALE)}; "
                f"nohup {inner} > launch.log 2>&1 < /dev/null & echo started")

    def launch(self, run: TrainingRun, python: str) -> None:
        bundle = run.bundle
        assert bundle is not None
        threading.Thread(target=self._launch, args=(run,), daemon=True).start()

    def _launch(self, run: TrainingRun) -> None:
        d = self._dir(run.run_id)
        try:
            probe = self.test()
            if not probe["ok"]:
                raise BackendError(probe["message"])
            run.accelerator = probe["accelerator"]
            self.note(run, f"connected: {probe['message']}"
                           + (f", {probe['accelerator'].get('gpu')}"
                              if probe["accelerator"].get("gpu") else ""))
            self.check_budget(run)
            self._check(self._run(f"mkdir -p {d} && cat > {d}/bundle.tar.gz",
                                  run.bundle.tar_bytes()), "uploading the run")
            self._check(self._run(f"cat > {d}/{LAUNCHER_NAME}",
                                  run.bundle.launcher().encode()), "uploading the launcher")
            if run.state in FINAL_STATES:
                return
            self._check(self._run(self._start_command(run), timeout=60), "starting the run")
            if run.history is not None:
                run.history.update(run.run_id, remote=self.remote_state(run))
            self.note(run, f"training on {self.host} in {self.root}/{run.run_id}")
        except Exception as exc:  # noqa: BLE001 — every failure ends the run with a reason
            run.record({"type": "error", "traceback": str(exc)})
            run.process_exited(1)
            return
        self._follow(run)

    def reattach(self, run: TrainingRun) -> None:
        threading.Thread(target=self._follow, args=(run,), daemon=True).start()

    def remote_state(self, run: TrainingRun) -> dict:
        return {"host": self.host, "dir": f"{self.root}/{run.run_id}",
                "accelerator": run.accelerator}

    # ---------------------------------------------------------------- follow
    def _follow(self, run: TrainingRun) -> None:
        attempts = 0
        code = None
        while code is None:
            command = f"{shlex.quote(self.python)} - {self._dir(run.run_id)} {run.remote_offset}"
            proc = subprocess.Popen(self._argv(command), stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
                                    bufsize=1)
            run.process = proc
            assert proc.stdin is not None and proc.stdout is not None
            try:
                proc.stdin.write(FOLLOWER)
                proc.stdin.close()
            except OSError:
                pass
            received = False
            for line in proc.stdout:
                if line.startswith('{"type": "_exit"'):
                    code = int(json.loads(line)["code"])
                    break
                received = True
                self.feed(run, line)
            proc.wait()
            if code is not None:
                break
            if received:
                attempts = 0  # it was working: start the retry budget again
            attempts += 1
            if attempts > self.retries:
                run.record({"type": "error", "traceback":
                            f"lost the connection to {self.host} and could not reconnect; "
                            "the run may still be going there. Follow it again with "
                            f"`aime runs attach {run.run_id}`."})
                run.process_exited(1)
                return
            self.note(run, f"connection lost; reconnecting ({attempts}/{self.retries})")
            time.sleep(min(2 * attempts, 30) if not os.environ.get("AIME_SSH") else 0.2)
        self._download(run)
        self.finish(run, code)

    def _download(self, run: TrainingRun) -> None:
        excludes = " ".join(f"--exclude={shlex.quote(name)}" for name in _SKIP)
        for attempt in range(3):
            try:
                data = self._check(self._run(
                    # COPYFILE_DISABLE: a macOS host's tar would add ._ metadata files
                    f"cd {self._dir(run.run_id)} && COPYFILE_DISABLE=1 tar czf - {excludes} .",
                    timeout=1800),
                    "downloading the outputs")
                unpack_outputs(data, run.workdir)
                return
            except (BackendError, OSError, ValueError) as exc:
                if attempt == 2:
                    self.note(run, f"could not download the outputs: {exc}")
                time.sleep(1)

    def stop(self, run: TrainingRun) -> None:
        d = self._dir(run.run_id)
        command = f"cd {d} 2>/dev/null && kill $(cat pid) 2>/dev/null"
        if self.options.get("docker_image"):
            command = f"docker stop aime-{run.run_id} >/dev/null 2>&1; " + command
        try:
            self._run(command + "; true", timeout=30)
        except BackendError:
            pass
