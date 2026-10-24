"""AppContext: the composition root.

Builds every store, service and panel, and is the only place signals are
wired. ``act_*`` methods are the intent slots the actions catalog binds to;
the Workbench only arranges what the context builds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy import __version__
from ai_made_easy.core import budget
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.summary import summarize
from ai_made_easy.ui.canvas import CanvasArea, CanvasController, block_mime_data
from ai_made_easy.ui.context_common import (  # noqa: F401 — re-exported
    _SETTINGS,
    _TRAINER_BLOCK,
    _json_or,
    _load_trained,
    api_module,
    tracking_label,
)
from ai_made_easy.ui.context_canvas import CanvasActions
from ai_made_easy.ui.context_data import DataActions
from ai_made_easy.ui.context_exports import ExportActions
from ai_made_easy.ui.context_files import FileActions
from ai_made_easy.ui.context_runs import RunActions
from ai_made_easy.ui.dialogs import (
    AboutDialog,
    ShortcutsDialog,
)
from ai_made_easy.ui.features.data_workspace import DataPage
from ai_made_easy.ui.features.experiments import ExperimentsPage
from ai_made_easy.ui.features.inspector import AssistantPage, CodePage, SummaryPage
from ai_made_easy.ui.features.library import BlockLibrary
from ai_made_easy.ui.features.problems import ProblemsPanel
from ai_made_easy.ui.features.project_field import ProjectNameField
from ai_made_easy.ui.features.properties import PropertyInspector
from ai_made_easy.ui.features.runconsole import OutputPage, TrainingPage
from ai_made_easy.ui.services.data_service import DataService
from ai_made_easy.ui.services.experiment_service import ExperimentService
from ai_made_easy.ui.services.export_service import ExportService
from ai_made_easy.ui.services.graph_service import GraphService
from ai_made_easy.ui.services.process_service import ProcessService, python_executable
from ai_made_easy.ui.services.project_service import (
    DEMO_SEED,
    ProjectService,
    project_kind,
)
from ai_made_easy.ui.stores import LogBus, ProjectStore, RunStore, ValidationStore
from ai_made_easy.ui.theme import ThemeService


class StatusChip(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusChip")

    def set_state(self, text: str, state: str = "") -> None:
        self.setText(text)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class AppContext(FileActions, ExportActions, RunActions, DataActions, CanvasActions,
                 QtCore.QObject):
    """Builds and wires everything. One instance per application."""

    status_message = QtCore.Signal(str)
    _device_found = QtCore.Signal(str, str)      # label, chip state (from a worker thread)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window: QtWidgets.QMainWindow | None = None
        # ---- services & stores
        self.theme = ThemeService()
        self.log_bus = LogBus(self)
        self.log = self.log_bus
        self.project_store = ProjectStore(self)
        self.run_store = RunStore(self)
        self.validation_store = ValidationStore(self)
        self.canvas = CanvasController()
        self.canvas.on_files_dropped(self._on_canvas_files)
        self.graph_service = GraphService(self.canvas, self.log_bus, self)
        self.export_service = ExportService(self.log_bus)
        self.process_service = ProcessService(self.log_bus, self.run_store, self)
        self.experiment_service = ExperimentService(self.process_service.history, self.log_bus,
                                                    self, python=python_executable())
        self.project_service = ProjectService(self.project_store, self.graph_service,
                                              self.log_bus)
        self.data_service = DataService(self, python=python_executable())
        self._last_ir: Graph | None = None
        self._run_kind = ""
        self._known_errors: set[tuple] = set()

        # ---- panels
        self.canvas_area = CanvasArea(self.canvas)
        self.project_field = ProjectNameField(self.project_store)
        self.library = BlockLibrary(block_mime_data)
        self.properties = PropertyInspector()
        self.summary_page = SummaryPage()
        self.code_page = CodePage()
        self.assistant_page = AssistantPage()
        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.setDocumentMode(True)
        for page, title in ((self.properties, "Properties"), (self.summary_page, "Summary"),
                            (self.code_page, "Code"), (self.assistant_page, "Assistant")):
            self.inspector_tabs.addTab(page, title)
        self.problems = ProblemsPanel(self.canvas.node_title, self._fixable)
        self.output_page = OutputPage(self.log_bus)
        self.training_page = TrainingPage(self.run_store)
        self.experiments_page = ExperimentsPage()
        self.data_page = DataPage()
        self._data_key = None
        self.validation_chip = StatusChip("Not validated")
        self.params_chip = StatusChip("")
        self.kind_chip = StatusChip("")
        self.device_chip = StatusChip("")
        self.status_chips = [self.validation_chip, self.params_chip, self.kind_chip,
                             self.device_chip, StatusChip(f"v{__version__}")]
        self._wire()

    # ================================================================ wiring

    def _wire(self) -> None:
        gs = self.graph_service
        gs.graph_settled.connect(self._on_graph_settled)
        gs.guard_message.connect(lambda msg: (self.status_message.emit(msg),
                                              self.log_bus.warning(msg)))
        graph = self.canvas.node_graph
        graph.node_selection_changed.connect(lambda *_: self._on_selection())
        graph.node_selected.connect(lambda *_: self._on_selection())
        graph.node_double_clicked.connect(self._preview_data_block)

        self.library.place_requested.connect(self.place_block)
        self.properties.param_changed.connect(self.canvas.set_param)
        self.properties.reset_requested.connect(self._reset_params)
        self.properties.table_edit_requested.connect(self._edit_table)
        self.properties.find_lr_requested.connect(self._find_lr)
        self.problems.locate_requested.connect(self._locate)
        self.problems.fix_requested.connect(self._apply_fix)
        self.summary_page.budget_changed.connect(self._on_budget_changed)
        self.code_page.target_changed.connect(lambda _t: self._refresh_preview())
        self.code_page.export_requested.connect(self.act_export)
        self.assistant_page.apply_requested.connect(
            lambda data: self.assistant_page.applied(self.project_service.apply_graph_dict(data)))

        self.training_page.train_clicked.connect(self.act_train)
        self.training_page.stop_clicked.connect(self.act_stop)
        self.training_page.run_on.manage_requested.connect(self.act_backends)
        self.training_page.framework.activated.connect(self._on_framework_chosen)
        self.training_page.errors_clicked.connect(self.act_error_analysis)
        self.training_page.saliency_clicked.connect(self.act_saliency)
        self.training_page.card_clicked.connect(self.act_model_card)
        self.training_page.folder_clicked.connect(self.act_open_run_folder)

        ep, es = self.experiments_page, self.experiment_service
        ep.scope.currentIndexChanged.connect(lambda *_: self._refresh_experiments())
        ep.compare_requested.connect(self._compare_runs)
        ep.stop_pipeline_requested.connect(lambda _pid: self.experiment_service.stop_pipeline())
        ep.resume_pipeline_requested.connect(
            lambda pid: self._run_pipeline(None, resume_from=pid))
        ep.results_requested.connect(self._load_run_results)
        ep.restore_requested.connect(self._restore_run)
        ep.diff_requested.connect(self._diff_run)
        ep.folder_requested.connect(self._open_run_folder)
        ep.delete_requested.connect(self._delete_runs)
        ep.tag_requested.connect(self._tag_run)
        ep.new_sweep_requested.connect(self.act_new_sweep)
        ep.stop_sweep_requested.connect(lambda _sid: es.stop_sweep())
        ep.apply_best_requested.connect(self._apply_best_sweep)
        ep.sweep_again_requested.connect(self._sweep_again)
        ep.register_requested.connect(self._register_run)
        ep.deploy_run_requested.connect(self._deploy_run)
        ep.try_run_requested.connect(self._try_run)
        ep.train_more_requested.connect(self._train_more)
        ep.resume_requested.connect(self._resume_run)
        ep.groups_requested.connect(self._results_by_group)
        ep.model_deploy_requested.connect(self._deploy_model)
        ep.model_stage_requested.connect(self._set_model_stage)
        ep.model_delete_requested.connect(self._delete_model)
        ep.model_share_requested.connect(self._share_model)
        es.changed.connect(self._refresh_experiments)
        es.sweep_finished.connect(self._on_sweep_finished)
        self.process_service.history_changed.connect(self._refresh_experiments)
        self.project_store.name_changed.connect(lambda *_: self._refresh_experiments())

        dp, ds = self.data_page, self.data_service
        dp.dataset_selected.connect(lambda _nid: self._profile_selected())
        dp.refresh_requested.connect(lambda: self._profile_selected(refresh=True))
        dp.open_file_requested.connect(self.act_profile_file)
        dp.rows_requested.connect(self._load_rows)
        dp.split_requested.connect(self.act_split_preview)
        dp.augment_requested.connect(self.act_augmentation_preview)
        ds.profiled.connect(self._on_profiled)
        ds.rows_ready.connect(lambda key, frame: dp.set_rows(frame)
                              if key == self._data_key else None)
        ds.split_ready.connect(dp.set_split)
        ds.preview_ready.connect(dp.set_augmentation)
        ds.changed.connect(self._republish_issues)

        self.run_store.state_changed.connect(self._on_run_state)
        ps = self.process_service
        ps.epoch_received.connect(self.training_page.on_epoch)
        ps.samples_received.connect(self.training_page.on_samples)
        ps.epoch_received.connect(self._on_epoch_progress)
        ps.log_received.connect(self.log_bus.info)
        ps.error_received.connect(self.log_bus.error)
        ps.finished.connect(self._on_run_finished)

    def attach_window(self, window: QtWidgets.QMainWindow) -> None:
        """Called by the Workbench once its chrome exists."""
        self.window = window
        # The context lives and dies with its window: its services and timers (validation,
        # autosave) drive the window's widgets, so they must not outlive them.
        self.setParent(window)
        self.status_message.connect(lambda m: window.statusBar().showMessage(m, 6000))
        self.status_message.connect(window.toasts.toast)
        theme = QtCore.QSettings(_SETTINGS).value("theme", self.theme.active())
        self._apply_theme(str(theme), announce=False)
        self._on_run_state(self.run_store.state, "")
        QtCore.QTimer.singleShot(0, self._boot)
        QtCore.QTimer.singleShot(400, self._detect_device)
        self._autosave_timer = QtCore.QTimer(self, interval=120_000, timeout=self._autosave)
        self._autosave_timer.start()

    def _boot(self) -> None:
        self.project_store.reset("mnist_cnn")
        self.graph_service.load(Graph.from_dict(json.loads(DEMO_SEED.read_text(encoding="utf-8"))))
        self.project_store.set_name(json.loads(DEMO_SEED.read_text(encoding="utf-8")).get("name", "untitled"))
        self.project_store.mark_clean()
        self.canvas.widget.setFocus()
        self.log_bus.info(f"AI Made Easy {__version__} ready — {len(get_registry().all())} "
                          "blocks available")
        QtCore.QTimer.singleShot(0, self._offer_recovery)
        QtCore.QTimer.singleShot(600, self._maybe_tour)
        minimap = str(QtCore.QSettings(_SETTINGS).value("minimap", False)).lower() in ("true", "1")
        if minimap:
            self.act_minimap(True)

    # ================================================================ settle

    def _on_graph_settled(self, ir: Graph) -> None:
        self._last_ir = ir
        self._publish_issues(ir)
        self._refresh_datasets(ir)
        kind = project_kind(ir.to_dict())
        from ai_made_easy.core.tasks import task_of

        try:
            task = task_of(ir)
        except Exception:  # noqa: BLE001 — incomplete designs have no task yet
            task = None
        self.kind_chip.set_state(f"{kind} · {task.label}" if task else kind)
        self.summary_page.set_kind(kind.split()[0] if kind != "Neural network" else "Neural net")
        try:
            summary = summarize(ir) if kind == "Neural network" else None
        except Exception:  # noqa: BLE001 — summaries need a valid model
            summary = None
        estimate, checks = None, []
        if summary is not None:
            try:
                estimate = budget.estimate(ir)
                checks = budget.check(ir, estimate)
            except Exception:  # noqa: BLE001 — an estimate never blocks the summary
                estimate = None
        flops = ([c.flops for c in estimate.layers]
                 if estimate and len(estimate.layers) == len(summary.layers) else None)
        self.summary_page.set_summary(
            summary, "" if summary else ("Classic ML pipelines have no layer summary."
                                         if kind.startswith("Classic") else ""), flops)
        self.summary_page.set_budget(budget.budget_of(ir))
        if kind == "Neural network":
            from ai_made_easy.core.api import describe_design

            described = describe_design(ir)
            self.training_page.set_frameworks(described["frameworks"],
                                              ir.meta.get("framework") or "auto")
        else:
            self.training_page.set_frameworks([], "auto")
        self.summary_page.set_estimate(estimate, checks)
        self.params_chip.set_state(f"{summary.total_params_display} parameters"
                                   if summary else "")
        self._refresh_preview(ir)
        self.assistant_page.set_graph(ir.to_dict())
        self._refresh_properties()
        if not self.graph_service.settled_from_load:
            self.project_store.mark_dirty()

    def _on_framework_chosen(self, _index: int) -> None:
        """Train with this framework: saved on the design as ``meta["framework"]``."""
        value = self.training_page.framework.currentData()
        meta = self.graph_service.adapter.meta
        if value == "auto":
            meta.pop("framework", None)
        else:
            meta["framework"] = value
        self.graph_service.schedule()

    def _on_budget_changed(self, values: dict) -> None:
        """The Summary's budget controls edited ``meta["budget"]``: re-check the design."""
        self.graph_service.adapter.meta["budget"] = values
        self.graph_service.schedule()

    def _republish_issues(self) -> None:
        """Data profiles landed in the background: refresh the problems only."""
        if self._last_ir is not None:
            self._publish_issues(self._last_ir)

    def _publish_issues(self, ir: Graph) -> None:
        from ai_made_easy.core.data.lints import project_base

        self.data_service.base = project_base(self.project_store.path)
        issues = ir.validate() + self.data_service.issues_for(ir)
        self.validation_store.update(issues)
        self.graph_service.note_shapes(ir)
        self.graph_service.apply_validation(issues)
        self.graph_service.refresh_inline(ir, self.data_service.base)
        self.problems.set_issues(issues)
        current = {(i.node_id, i.message) for i in issues if i.severity == "error"}
        fresh = [m for nid, m in current - self._known_errors]
        if fresh and not self.graph_service.settled_from_load:
            self.status_message.emit(fresh[0].split(" Fix:")[0])
        self._known_errors = current
        errors = len([i for i in issues if i.severity == "error"])
        warnings = len(issues) - errors
        if errors:
            self.validation_chip.set_state(f"{errors} error{'s' * (errors != 1)}"
                                           + (f", {warnings} warning{'s' * (warnings != 1)}"
                                              if warnings else ""), "error")
        elif warnings:
            self.validation_chip.set_state(f"{warnings} warning{'s' * (warnings != 1)}",
                                           "warning")
        else:
            self.validation_chip.set_state("No problems", "ok")

    def _refresh_preview(self, ir: Graph | None = None) -> None:
        ir = ir or self._last_ir
        if ir is None:
            return
        target = self.code_page.current_target()
        try:
            self.code_page.set_code(self.export_service.render(ir, target))
        except Exception as exc:  # noqa: BLE001 — shown in the code panel
            self.code_page.set_error(str(exc))

    # ============================================================ selection

    def _on_selection(self) -> None:
        ids = self.canvas.selected_ids()
        if len(ids) != 1:
            self.properties.clear()
            return
        node_id = ids[0]
        block = self.canvas.block_of(node_id)
        if block is None:
            self.properties.clear()
            return
        if self.properties.node_id != node_id:
            issues, shape, count = self._node_facts(node_id)
            self.properties.show_node(node_id, block, self.canvas.params_of(node_id),
                                      issues, shape, count, why=self._why(node_id))
            self.inspector_tabs.setCurrentWidget(self.properties)

    def _refresh_properties(self) -> None:
        node_id = self.properties.node_id
        if node_id is None:
            return
        if self.canvas.block_of(node_id) is None:
            self.properties.clear()
            return
        self.properties.update_status(*self._node_facts(node_id))

    def _edit_table(self, node_id: str, param: str) -> None:
        """Open the grid editor for a table parameter (a variable's probability table)."""
        from ai_made_easy.core.pgm.network import layout
        from ai_made_easy.ui.features.table_editor import TableEditorDialog

        graph = self.canvas.to_ir()
        info = layout(graph, node_id)
        if "error" in info:
            self.status_message.emit(info["error"])
            QtWidgets.QMessageBox.information(self.window, "Probability table", info["error"])
            return
        dialog = TableEditorDialog(self.window, info)
        if dialog.exec() and dialog.result_text is not None:
            self.canvas.set_param(node_id, param, dialog.result_text)
            self.properties.show_node(node_id, graph.nodes[node_id].definition(),
                                      self.canvas.params_of(node_id),
                                      *self._node_facts(node_id))

    def _why(self, node_id: str) -> str:
        """The recipe's note on why this block is in the design ("" when none)."""
        recipe = (getattr(self.canvas, "meta", None) or {}).get("recipe") or {}
        return str((recipe.get("why") or {}).get(node_id, ""))

    def _node_facts(self, node_id: str):
        issues = [i for i in self.validation_store.issues if i.node_id == node_id]
        shape = self.graph_service.last_shapes.get(node_id)
        count = None
        ir = self._last_ir
        if ir is not None and node_id in ir.nodes:
            node = ir.nodes[node_id]
            defn = node.definition()
            if defn.param_fn is not None:
                in_shapes = [self.graph_service.last_shapes.get(e.source_id)
                             for e in ir.incoming(node_id)]
                if all(s is not None for s in in_shapes) and in_shapes:
                    try:
                        count = int(defn.param_fn(in_shapes, node.resolved_params()))
                    except Exception:  # noqa: BLE001
                        count = None
        return issues, shape, count

    def _reset_params(self, node_id: str) -> None:
        block = self.canvas.block_of(node_id)
        if block is None:
            return
        for spec in block.params:
            self.canvas.set_param(node_id, spec.name, spec.default)
        issues, shape, count = self._node_facts(node_id)
        self.properties.show_node(node_id, block, self.canvas.params_of(node_id), issues,
                                  shape, count)

    def _locate(self, node_id: str) -> None:
        self.canvas.select_and_center(node_id)
        self._on_selection()

    # ============================================================== fixes

    def _fixable(self, issue) -> bool:
        from ai_made_easy.core.fixes import fix_for_issue

        if self._last_ir is None:
            return False
        try:
            return fix_for_issue(self._last_ir, issue) is not None
        except Exception:  # noqa: BLE001
            return False

    def _apply_fix(self, issue) -> None:
        from ai_made_easy.core.fixes import fix_for_issue

        ir = self.project_service.snapshot()
        result = fix_for_issue(ir, issue)
        if result is None:
            self.status_message.emit("No automatic fix is available for this problem")
            return
        label, description, fixed = result
        self.graph_service.load(fixed)
        self.project_store.mark_dirty()
        self.log_bus.info(f"Quick fix applied: {description}")
        self.status_message.emit(f"{label}: {description}")

    # ============================================================ canvas

    def place_block(self, type_id: str) -> None:
        self.graph_service.place_block(type_id)

    def _preview_data_block(self, node) -> None:  # noqa: ANN001
        definition = self.canvas.definition_of(node)
        if definition is None or not definition.type_id.startswith("data."):
            return
        self.data_page.select(node.id)
        self.act_data()

    # ================================================ compare and drop files

    def _diff_run(self, run_id: str) -> None:
        """What changed on the canvas since a run (3.3)."""
        from ai_made_easy.core import api

        try:
            d = api.diff_run(run_id, self.canvas.to_ir(self.project_store.name).to_dict())
        except Exception as exc:  # noqa: BLE001
            self.status_message.emit(f"Could not compare: {exc}")
            return
        lines = [d["summary"], ""]
        for c in d["changed"]:
            lines.append(f"{c['id']}.{c['param']}: {c['from']!r} → {c['to']!r}")
        for key, label in (("added", "Added"), ("removed", "Removed")):
            lines += [f"{label}: {x['id']} ({x['type']})" for x in d[key]]
        lines += [f"Replaced: {x['id']}: {x['from']} → {x['to']}" for x in d["retyped"]]
        lines += [f"New wire: {w['from']} → {w['to']}" for w in d["wired"]]
        lines += [f"Removed wire: {w['from']} → {w['to']}" for w in d["unwired"]]
        box = QtWidgets.QMessageBox(self.window)
        box.setWindowTitle(f"Run {run_id} vs the canvas")
        box.setText(d["summary"])
        if not d["same"]:
            box.setDetailedText("\n".join(lines[2:]))
        self.last_diff = d
        box.open()

    def open_data_file(self, path: str) -> None:
        """A data file or folder dropped on the window: start a project from it."""
        self.act_new_from_task(initial_path=path)

    def _on_canvas_files(self, paths: list[str]) -> None:
        from ai_made_easy.ui.workbench import Workbench

        data = [p for p in paths if Path(p).is_dir()
                or Path(p).suffix.lower() in Workbench.DATA_SUFFIXES]
        if data:
            self.open_data_file(data[0])
        else:
            self.status_message.emit("Drop a table (.csv, .parquet, .xlsx, .json), a text "
                                     "file or a folder of images / audio")

    # ============================================================== view

    def act_theme_dark(self, *_):
        self._apply_theme("dark")

    def act_theme_light(self, *_):
        self._apply_theme("light")

    def _apply_theme(self, name: str, announce: bool = True) -> None:
        app = QtWidgets.QApplication.instance()
        self.theme.apply(app, name)
        tokens = self.theme.tokens()
        self.canvas.apply_theme(tokens)
        self.canvas_area.set_theme(tokens)
        self.training_page.set_theme(tokens)
        self.code_page.set_theme(self.theme.active())
        if self.window is not None:
            from ai_made_easy.ui import icons
            from ai_made_easy.ui.actions_catalog import CATALOG

            for spec in CATALOG:
                if spec.icon and spec.id != "run.train":
                    self.window.actions[spec.id].setIcon(icons.icon(spec.icon))
            self.window.actions[f"view.theme_{self.theme.active()}"].setChecked(True)
        if announce:
            self.log_bus.info(f"theme: {self.theme.active()}")

    def act_tour(self, *_):
        """The three-step first-run tour (also shown once on first start)."""
        from ai_made_easy.ui.features.tour import Tour

        if self.window is None:
            return None
        if getattr(self, "tour", None) is not None:
            self.tour.finish()
        win = self.window
        train = win.findChild(QtWidgets.QToolButton, "primaryAction")
        self.tour = Tour(win, [lambda: win.docks["library"], lambda: self.canvas_area,
                               lambda: train])
        self.tour.finished.connect(self._tour_done)
        return self.tour

    def _tour_done(self) -> None:
        QtCore.QSettings(_SETTINGS).setValue("tour_done", True)
        self.tour = None

    def _maybe_tour(self) -> None:
        """First start only, and never under automation (offscreen Qt, AIME_NO_TOUR=1)."""
        settings = QtCore.QSettings(_SETTINGS)
        if str(settings.value("tour_done", False)).lower() in ("true", "1"):
            return
        if os.environ.get("AIME_NO_TOUR") == "1" or \
                QtWidgets.QApplication.platformName() in ("offscreen", "minimal"):
            return
        self.act_tour()

    # ============================================================= model

    def act_validate(self, *_):
        self.graph_service.settle_now()
        if self.window is not None:
            self.window.docks["problems"].show()
            self.window.docks["problems"].raise_()
        errors, warnings = self.problems.counts
        self.status_message.emit("No problems found" if not errors + warnings
                                 else f"{errors} error(s), {warnings} warning(s)")

    # ============================================================== help

    def act_shortcuts(self, *_):
        from ai_made_easy.ui.actions_catalog import CATALOG

        ShortcutsDialog(self.window, CATALOG).exec()

    def act_about(self, *_):
        import importlib.util

        env = [("Python", sys.version.split()[0])]
        for module, label in (("torch", "PyTorch"), ("keras", "Keras"),
                              ("sklearn", "scikit-learn"), ("PySide6", "Qt for Python")):
            if importlib.util.find_spec(module) is not None:
                try:
                    env.append((label, __import__(module).__version__))
                except Exception:  # noqa: BLE001
                    env.append((label, "installed"))
            else:
                env.append((label, "not installed"))
        env.append(("Blocks", str(len(get_registry().all()))))
        AboutDialog(self.window, __version__, env).exec()

    # ============================================================ helpers

    def confirm_close(self, window) -> bool:  # noqa: ANN001
        if self.run_store.is_running:
            self.process_service.stop()
        self.experiment_service.shutdown()
        ok = self._confirm_discard()
        if ok:
            self._clear_recovery()
        return ok

    def _confirm_discard(self) -> bool:
        if not self.project_store.dirty or self.window is None:
            return True
        answer = QtWidgets.QMessageBox.question(
            self.window, "Unsaved changes",
            f"Save changes to '{self.project_store.name}' before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self.act_save()
            return not self.project_store.dirty
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def _last_dir(self) -> Path:
        value = QtCore.QSettings(_SETTINGS).value("last_dir", "")
        path = Path(str(value)) if value else Path.home()
        return path if path.exists() else Path.home()

    def _remember_dir(self, path: str) -> None:
        QtCore.QSettings(_SETTINGS).setValue("last_dir", str(Path(path).parent))
