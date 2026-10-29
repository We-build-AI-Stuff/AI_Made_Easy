"""Import, deploy and export.

The ``act_*`` slots of :class:`~ai_made_easy.ui.context.AppContext` for this area,
as a mixin: the methods use the stores, services and panels AppContext builds (split out of
context.py in 3.3; no behaviour change).
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

from ai_made_easy.ui.context_common import _json_or, _load_trained, api_module, tracking_label
from ai_made_easy.ui.services.export_service import default_filename
from ai_made_easy.ui.services.process_service import python_executable


class ExportActions:
    # ============================================================= import

    def act_import_model(self, *_):
        from ai_made_easy.ui.features.import_model import ImportModelDialog

        if not self._confirm_discard():
            return
        dialog = ImportModelDialog(self.window, python=python_executable(),
                                   start_dir=self._last_dir())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.result is None:
            return
        self.open_imported(dialog.result.graph)

    def open_imported(self, graph: dict) -> bool:
        """Load an imported graph as a new, unsaved project and arrange it."""
        if not self.project_service.apply_graph_dict(graph):
            return False
        self.project_store.set_path(None)
        self.canvas.auto_layout(self.project_service.snapshot(), undoable=False)
        self.canvas.center_view()
        self.project_store.mark_dirty()
        self.status_message.emit(f"Imported {graph.get('name', 'model')} — "
                                 f"{len(graph.get('nodes', []))} blocks")
        return True

    # ============================================================= deploy

    def act_deploy(self, *_):
        """Deploy the selected run, else this project's latest finished run."""
        history = self.experiment_service.history
        selected = self.experiments_page.selected_run_ids()
        candidates = [history.get(r) for r in selected] if len(selected) == 1 else \
            history.list(project=self.project_store.name)
        run = next((r for r in candidates if r.status == "finished"), None)
        if run is None:
            self.status_message.emit("Train the model first — deployment needs a finished run")
            return
        self._deploy_run(run.run_id)

    def _deploy_run(self, run_id: str) -> None:
        from ai_made_easy.ui.features.deploy import DeployDialog

        rec = self.experiment_service.history.get(run_id)
        if rec.framework == "llm":   # a fine-tune: its model goes out as GGUF
            self._export_gguf(run_id)
            return
        DeployDialog(self.window, self.experiment_service.history.path(run_id), rec.framework,
                     rec.project or rec.name, default_dir=self._last_dir(),
                     python=python_executable(),
                     subtitle=f"Source: run {run_id} ({rec.name})").exec()

    def _export_gguf(self, run_id: str) -> None:
        """A LoRA fine-tune merged into its base model, as one GGUF file (in the
        background: merging and checking take a few minutes)."""
        from ai_made_easy.core.deploy import hf_gguf
        from ai_made_easy.ui.features import tools

        choices = ["Q8_0 (8-bit, about a quarter of the size)",
                   "Q4_0 (4-bit, about an eighth)", "float32"]
        picked, ok = QtWidgets.QInputDialog.getItem(
            self.window, "Export GGUF", "The LoRA adapter merged into its base model, for "
            "llama.cpp, Ollama and LM Studio.\nWeights:", choices, 0, False)
        if not ok:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self.window, "Save the GGUF file in",
                                                            self._last_dir())
        if not folder:
            return
        quant = {0: "q8_0", 1: "q4_0"}.get(choices.index(picked), "")
        run_dir = self.experiment_service.history.path(run_id)
        worker = tools.Background(self.window)

        def finished(result, error: str) -> None:  # noqa: ANN001
            if result is None:
                QtWidgets.QMessageBox.warning(self.window, "Export GGUF", error)
                return
            self.status_message.emit(f"Saved {result['file']} — {result['note']}")

        worker.done.connect(finished)
        self.status_message.emit(f"Exporting run {run_id} to GGUF…")
        worker.start(lambda: hf_gguf.export_run(run_dir, Path(folder), quant))

    def _try_run(self, run_id: str) -> None:
        from ai_made_easy.ui.features.try_dialog import TryDialog

        TryDialog(self.window, self.experiment_service.history, run_id).exec()

    def _deploy_model(self, name: str, version: int) -> None:
        from ai_made_easy.ui.features.deploy import DeployDialog

        registry = self.experiment_service.registry
        mv = registry.get(name, version)
        DeployDialog(self.window, registry.path(name, version), mv.framework, name,
                     version=str(version), default_dir=self._last_dir(),
                     python=python_executable(),
                     subtitle=f"Source: {name} v{version} ({mv.stage})").exec()

    def _register_run(self, run_id: str) -> None:
        import re

        from ai_made_easy.core.deploy import RegistryError

        rec = self.experiment_service.history.get(run_id)
        default = re.sub(r"[^A-Za-z0-9._-]+", "-", rec.project or rec.name).strip("-") or "model"
        name, ok = QtWidgets.QInputDialog.getText(
            self.window, "Register Model", "Model name (letters, digits, . _ -):",
            text=default)
        if not ok or not name.strip():
            return
        try:
            mv = self.experiment_service.registry.register(
                self.experiment_service.history.path(run_id), name.strip())
        except (RegistryError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Register Model", str(exc))
            return
        self._refresh_experiments()
        self.experiments_page.tabs.setCurrentIndex(2)
        self.status_message.emit(f"Registered {mv.name} v{mv.version}")

    def _set_model_stage(self, name: str, version: int, stage: str) -> None:
        self.experiment_service.registry.set_stage(name, version, stage)
        self._refresh_experiments()
        self.status_message.emit(f"{name} v{version} → {stage}")

    def _share_model(self, name: str, version: int, target: str) -> None:
        """MLflow / W&B registries, a Hub model repo, or a Docker Space (in the
        background; publishing to the Hub is confirmed first)."""
        from ai_made_easy.ui.features import tools

        if target in ("mlflow", "wandb"):
            label = tracking_label(target)
            work = lambda: api_module().mirror_model(name, version, [target])  # noqa: E731
            done = lambda r: f"{name} v{version} registered in {label}: {r['mirrors'][-1]['uri']}"  # noqa: E731
        else:
            space = target == "space"
            repo = tools.ask_repo(self.window, "Hugging Face Space" if space else
                                  "Hugging Face Hub", "Space id (owner/name):" if space else
                                  "Model repo (owner/name):", f"your-name/{name}")
            if not repo or not tools.confirm_publish(
                    self.window, f"{name} v{version}" + (" as a model server" if space else ""),
                    repo):
                return
            if space:
                work = lambda: api_module().deploy_to_space(repo, model=name, version=version)  # noqa: E731
            else:
                work = lambda: api_module().push_to_hub(repo, model=name, version=version)  # noqa: E731
            done = lambda r: f"Published to {r['url']}"  # noqa: E731
        worker = tools.Background(self.window)

        def finished(result, error: str) -> None:  # noqa: ANN001
            if result is None:
                QtWidgets.QMessageBox.warning(self.window, "Share model", error)
                return
            self._refresh_experiments()
            self.status_message.emit(done(result))

        worker.done.connect(finished)
        self.status_message.emit(f"Sharing {name} v{version}…")
        worker.start(work)

    def act_tracking(self, *_):
        from ai_made_easy.core import tracking
        from ai_made_easy.ui.features.tools import TrackingDialog

        meta = self.graph_service.adapter.meta
        try:
            chosen = tracking.parse(meta.get("tracking"))
        except tracking.TrackingError:
            chosen = []
        dialog = TrackingDialog(self.window, chosen)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        picked = dialog.selected()
        if picked:
            meta["tracking"] = picked
        else:
            meta.pop("tracking", None)
        self.project_store.mark_dirty()
        self.status_message.emit("Runs of this project mirror to "
                                 + (", ".join(tracking.LABELS[p] for p in picked) or
                                    "no tracker"))

    def act_fetch_dataset(self, *_):
        from ai_made_easy.ui.features.tools import FetchDatasetDialog

        dialog = FetchDatasetDialog(self.window)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.path:
            self.act_new_from_task(initial_path=dialog.path)

    def act_export_runs(self, *_):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Run History", str(Path(self._last_dir()) / "runs.csv"),
            "CSV (*.csv);;Parquet (*.parquet)")
        if not path:
            return
        try:
            result = api_module().export_runs(path, project=None)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self.window, "Export Run History", str(exc))
            return
        self.status_message.emit(f"Wrote {result['rows']} run(s) to {result['path']}")

    def _delete_model(self, name: str, version: int) -> None:
        answer = QtWidgets.QMessageBox.question(
            self.window, "Delete model version",
            f"Delete {name} v{version} from the registry? This cannot be undone.")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.experiment_service.registry.delete(name, version)
            self._refresh_experiments()

    # ============================================================ export

    def act_export(self, target: str, *_):
        ir = self.project_service.snapshot()
        try:
            code = self.export_service.render(ir, target)
        except Exception as exc:  # noqa: BLE001
            self._export_error(target, exc)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export", str(self._last_dir() / default_filename(ir, target)),
            "Python (*.py)")
        if path:
            self._remember_dir(path)
            Path(path).write_text(code, encoding="utf-8")
            self.log_bus.info(f"exported → {path}")
            self.status_message.emit(f"Exported {Path(path).name}")

    def _export_error(self, target: str, exc: Exception) -> None:
        from ai_made_easy.ui.services.export_service import target_label

        QtWidgets.QMessageBox.warning(self.window, "Export not available",
                                      f"{target_label(target)} cannot be generated:\n\n{exc}")

    def act_export_pytorch_model(self, *_):
        self.act_export("pytorch_model")

    def act_export_pytorch_train(self, *_):
        self.act_export("pytorch_train")

    def act_export_keras_model(self, *_):
        self.act_export("keras_model")

    def act_export_keras_train(self, *_):
        self.act_export("keras_train")

    def act_export_sklearn(self, *_):
        self.act_export("sklearn_train")

    def act_export_llm(self, *_):
        self.act_export("llm")

    def act_export_onnx(self, *_):
        self.act_runtime_export("onnx")

    def act_export_jit(self, *_):
        self.act_runtime_export("jit")

    def act_runtime_export(self, kind: str, *_):
        ir = self.project_service.snapshot()
        suffix = ".onnx" if kind == "onnx" else ".pt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, f"Export {'ONNX' if kind == 'onnx' else 'TorchScript'}",
            str(self._last_dir() / f"{ir.name}{suffix}"), f"Model (*{suffix})")
        if not path:
            return
        try:
            script = self.export_service.write_runtime_script(ir, kind, Path(path))
        except Exception as exc:  # noqa: BLE001
            self._export_error("pytorch_model", exc)
            return
        self._remember_dir(path)
        self.process_service.run_script(script, script.parent, kind)

    def act_export_web(self, *_):
        from ai_made_easy.core.codegen.training_gen import collect_spec
        from ai_made_easy.core.web_export import build_web_demo, layers_from_torch

        wd = self._run_workdir()
        if wd is None or not any(wd.glob("*_best.pt")):
            self.status_message.emit("The web demo needs a trained PyTorch image model")
            return
        try:
            spec = collect_spec(self.project_service.snapshot())
            model, shape, norm = _load_trained(wd, spec.class_name)
            classes = _json_or(wd / "classes.json", None) or [str(i) for i in
                                                              range(spec.num_outputs)]
            html = build_web_demo(layers_from_torch(model), classes, list(shape), norm,
                                  title=self.project_store.name)
        except Exception as exc:  # noqa: BLE001
            self._export_error("pytorch_model", exc)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Web Demo",
            str(self._last_dir() / f"{self.project_store.name}_demo.html"), "HTML (*.html)")
        if path:
            Path(path).write_text(html, encoding="utf-8")
            self.log_bus.info(f"web demo saved → {path}")
