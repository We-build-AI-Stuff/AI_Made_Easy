"""DataService: dataset profiling, split and augmentation previews off the GUI thread.

Profiles are cached by data signature (``core.data.lints.ProfileCache``), so the
Problems panel can show data lints on every graph settle without touching the
disk: ``issues_for`` returns what is cached and schedules what is not; when a
profile lands, ``changed`` fires and the caller re-publishes its issues.
"""
from __future__ import annotations

import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6 import QtCore

from ai_made_easy.core.data.lints import LOCAL_BLOCKS, ProfileCache, data_issues
from ai_made_easy.core.graph import Graph, ValidationIssue


class DataService(QtCore.QObject):
    changed = QtCore.Signal()                    # a profile landed: re-lint
    profiled = QtCore.Signal(object, object)     # key, DataProfile
    rows_ready = QtCore.Signal(object, object)   # key, DataFrame | None
    split_ready = QtCore.Signal(object, str)     # SplitPreview | None, error
    preview_ready = QtCore.Signal(object, str)   # augmentation result | None, error

    def __init__(self, parent=None, python: str | None = None):
        super().__init__(parent)
        self.python = python
        self.base = None  # folder relative dataset paths resolve against
        self.cache = ProfileCache()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aime-data")
        self._lock = threading.Lock()
        self._inflight: set = set()
        self._futures: list[Future] = []
        self._closed = False
        self._emitting = threading.Lock()   # shutdown waits for an emit under way

    # ------------------------------------------------------------ profiles
    def issues_for(self, graph: Graph) -> list[ValidationIssue]:
        """Cached data lints now; uncached datasets are profiled in the background."""
        for type_id, params in self.cache.pending(graph, self.base):
            self._profile(type_id, params)
        return data_issues(graph, self.cache, self.base, compute=False)

    def request(self, type_id: str, params: dict, *, refresh: bool = False):
        """Profile a dataset block; returns the key ``profiled`` will carry.

        A cached profile is emitted right away.
        """
        if refresh:
            self.cache.clear()
        key = self.cache.key(type_id, params, self.base)
        hit = self.cache.get(type_id, params, self.base)
        if hit is not None:
            self.profiled.emit(key, hit)
        else:
            self._profile(type_id, params)
        return key

    def request_path(self, path: str, target: str | None = None):
        key = ("path", str(path), target)

        def work():
            from ai_made_easy.core.data.profile import profile_path

            self._emit("profiled", key, profile_path(path, target=target))

        self._submit(work)
        return key

    def _profile(self, type_id: str, params: dict) -> None:
        if type_id not in LOCAL_BLOCKS:
            from ai_made_easy.core.data.profile import profile_dataset

            self.profiled.emit(self.cache.key(type_id, params, self.base),
                               profile_dataset(type_id, params, self.base))
            return
        key = self.cache.key(type_id, params, self.base)
        with self._lock:
            if key in self._inflight:
                return
            self._inflight.add(key)
        base = self.base

        def work():
            try:
                profile = self.cache.profile(type_id, params, base)
            finally:
                with self._lock:
                    self._inflight.discard(key)
            self._emit("profiled", key, profile)
            self._emit("changed")

        self._submit(work)

    def load_rows(self, key, profile, limit: int = 50_000) -> None:
        """Read the first ``limit`` rows of a table profile for the row viewer."""
        def work():
            from ai_made_easy.core.data.profile import guess_format, read_table

            try:
                frame = read_table(profile.source, profile.format
                                   or guess_format(profile.source), nrows=limit)
            except Exception:  # noqa: BLE001 — the profile already reported it
                frame = None
            self._emit("rows_ready", key, frame)

        self._submit(work)

    # ------------------------------------------------------------ previews
    def split(self, graph: Graph) -> None:
        def work():
            from ai_made_easy.core import api

            try:
                from ai_made_easy.core.data.splits import SplitPreview

                data = api.split_preview(graph, base=self.base)
                preview = SplitPreview(totals=data["totals"], per_class=data["per_class"],
                                       method=data["method"], notes=data["notes"])
                self._emit("split_ready", preview, "")
            except Exception as exc:  # noqa: BLE001 — shown in the page
                self._emit("split_ready", None, str(exc))

        self._submit(work)

    def augment(self, graph: Graph, images: int = 4, variants: int = 6) -> None:
        out = tempfile.mkdtemp(prefix="aime_augment_")

        def work():
            from ai_made_easy.core.data.augment import augmentation_preview

            try:
                result = augmentation_preview(graph, out, base=self.base, python=self.python,
                                              images=images, variants=variants)
                self._emit("preview_ready", result, "")
            except Exception as exc:  # noqa: BLE001 — shown in the page
                self._emit("preview_ready", None, str(exc))

        self._submit(work)

    # ------------------------------------------------------------ plumbing
    def _submit(self, fn) -> None:  # noqa: ANN001
        if self._closed:  # late settles after the window closed
            return
        self._futures = [f for f in self._futures if not f.done()]
        self._futures.append(self._pool.submit(fn))

    def busy(self) -> bool:
        return any(not f.done() for f in self._futures)

    def wait(self, timeout: float = 60.0) -> None:
        """Block until queued work finishes and deliver its signals (tests, shutdown)."""
        import time

        deadline = time.monotonic() + timeout
        while self.busy() and time.monotonic() < deadline:
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.01)
        QtCore.QCoreApplication.processEvents()

    def _emit(self, signal: str, *args) -> None:  # noqa: ANN002
        """Emit from a worker thread, unless the service has shut down: the window may
        delete it then, and emitting on a deleted object crashes."""
        with self._emitting:
            if not self._closed:
                getattr(self, signal).emit(*args)

    def shutdown(self) -> None:
        with self._emitting:   # after this, no work still running emits
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

