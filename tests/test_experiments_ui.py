"""Experiments panel: run history table, comparison, results loading, sweeps."""
from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("PySide6")

from conftest import tiny_classifier_dict  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("experiments-tests")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def ctx(app, isolated_home):
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    context = AppContext()
    win = Workbench(context)
    context._boot()
    app.processEvents()
    yield context
    context.experiment_service.shutdown()
    context.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _fake_run(history, lr, acc, project):
    rec = history.create(tiny_classifier_dict(lr=lr), project=project)
    for epoch in (1, 2):
        history.append_epoch(rec.run_id, {"type": "epoch", "epoch": epoch, "total": 2,
                                          "metrics": {"val_loss": 1.0 / epoch,
                                                      "accuracy": acc * epoch / 2}})
    (history.path(rec.run_id) / "metrics.json").write_text(json.dumps({"accuracy": acc}), encoding="utf-8")
    history.finalize(rec.run_id, "finished", 0)
    return rec.run_id


def test_runs_table_compare_and_load(ctx, app):
    from ai_made_easy.ui.features.experiment_dialogs import CompareDialog

    history = ctx.experiment_service.history
    project = ctx.project_store.name
    a = _fake_run(history, 0.05, 0.8, project)
    b = _fake_run(history, 0.1, 0.9, project)
    _fake_run(history, 0.2, 0.5, "other project")
    page = ctx.experiments_page
    ctx.act_experiments()
    assert page.runs_table.rowCount() == 2
    page.scope.setCurrentIndex(1)
    assert page.runs_table.rowCount() == 3
    headers = [page.runs_table.horizontalHeaderItem(c).text()
               for c in range(page.runs_table.columnCount())]
    assert "accuracy" in headers

    page.runs_table.selectAll()
    assert len(page.selected_run_ids()) == 3 and page.compare_btn.isEnabled()
    assert not page.restore_btn.isEnabled()

    dialog = CompareDialog(None, [history.get(a), history.get(b)],
                           {a: history.epochs(a), b: history.epochs(b)})
    names = [dialog.table.item(r, 1).text() for r in range(dialog.table.rowCount())]
    assert "opt.lr" in names and "accuracy" in names
    assert dialog.metric_combo.count() >= 2
    dialog.close()

    ctx._load_run_results(b)
    assert ctx.process_service.last_workdir == history.path(b)
    assert ctx.training_page.last_metrics()["accuracy"] == pytest.approx(0.9)

    history.delete(a)
    ctx._refresh_experiments()
    assert page.runs_table.rowCount() == 2


def test_sweep_dialog_builds_a_valid_spec(app):
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.core.sweeps import SweepSpec, sweepable_params
    from ai_made_easy.ui.features.experiment_dialogs import SweepDialog

    g = Graph.from_dict(tiny_classifier_dict())
    dialog = SweepDialog(None, sweepable_params(g), ["val_loss", "accuracy"])
    dialog.check_row("opt.lr")
    dialog.check_row("d1.units")
    dialog.check_row("opt.nesterov")
    spec = dialog.build_spec()
    assert [d["param"] for d in spec["dimensions"]] == ["units", "lr", "nesterov"] or \
        {d["param"] for d in spec["dimensions"]} == {"units", "lr", "nesterov"}
    nesterov = next(d for d in spec["dimensions"] if d["param"] == "nesterov")
    assert nesterov["values"] == [False, True]
    SweepSpec.from_dict(spec).check(g)
    dialog.close()


def test_sweep_runs_from_the_ui_service(ctx, app):
    pytest.importorskip("torch")
    from ai_made_easy.core.graph import Graph

    service = ctx.experiment_service
    sid = service.start_sweep(Graph.from_dict(tiny_classifier_dict(epochs=1)), {
        "dimensions": [{"node": "opt", "param": "lr", "kind": "choice",
                        "values": [0.05, 0.1]}],
        "metric": "accuracy", "strategy": "grid", "max_trials": 2}, project="sweep-ui")
    assert sid and service.is_running()
    deadline = time.time() + 300
    while service.runner is not None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert service.runner is None, "the sweep was still running after 300s"
    record = service.sweeps.get(sid)
    assert record.state == "finished" and len(record.trials) == 2
    ctx.experiments_page.scope.setCurrentIndex(1)
    ctx._refresh_experiments()
    assert ctx.experiments_page.sweeps_table.rowCount() == 1
    ctx.experiments_page.sweeps_table.selectRow(0)
    assert ctx.experiments_page.trials_table.rowCount() == 2
    assert ctx.experiments_page.apply_best_btn.isEnabled()


def test_multi_objective_sweeps_show_the_pareto_front(ctx, app):
    from ai_made_easy.core.sweeps import SweepRecord

    trials = [{"number": n, "values": {"recipe": "tabular_mlp"}, "run_id": f"r{n}",
               "state": "finished", "score": score, "message": "",
               "objectives": {"latency_ms": latency}}
              for n, score, latency in ((0, 0.9, 5.0), (1, 0.85, 1.0), (2, 0.8, 2.0))]
    record = SweepRecord("automl-x", "automl_binary", graph={}, trials=trials,
                         spec={"metric": "accuracy", "direction": "max", "max_trials": 3,
                               "strategy": "tpe", "objectives": ["latency_ms"]},
                         best={"number": 0, "run_id": "r0", "score": 0.9, "values": {}})
    page = ctx.experiments_page
    page.set_sweeps([record])
    page.sweeps_table.selectRow(0)
    assert "latency_ms min" in page.sweeps_table.item(0, 4).text()
    marks = {page.trials_table.item(r, 0).data(QtCore.Qt.ItemDataRole.DisplayRole)
             for r in range(3)}
    assert marks == {"0  ★  ◆", "1  ◆", "2"}               # 2 is slower and less accurate
    assert not page.pareto_plot.isHidden()
    page.set_sweeps([SweepRecord("s", "plain", graph={}, spec={"metric": "accuracy"})])
    page.sweeps_table.selectRow(0)
    assert page.pareto_plot.isHidden()


def test_register_and_deploy_from_the_ui(ctx, app, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.ui.features.deploy import DeployDialog

    service = ctx.experiment_service
    run_id = service.manager.start(Graph.from_dict(tiny_classifier_dict(epochs=1)),
                                   project=ctx.project_store.name)
    service.manager.wait(run_id, 300)
    ctx._refresh_experiments()
    page = ctx.experiments_page
    page.runs_table.selectRow(0)
    assert page.register_btn.isEnabled() and page.deploy_btn.isEnabled()

    service.registry.register(service.history.path(run_id), "ui-model")
    ctx._refresh_experiments()
    assert page.models_table.rowCount() == 1
    page.models_table.selectRow(0)
    assert page._selected_model() == ("ui-model", 1)
    ctx._set_model_stage("ui-model", 1, "production")
    assert service.registry.get("ui-model", 1).stage == "production"

    dialog = DeployDialog(None, service.registry.path("ui-model", 1), "pytorch", "ui-model")
    dialog.folder.setText(str(tmp_path / "server"))
    for check in dialog.format_checks.values():
        check.setChecked(False)
    dialog._build()
    deadline = time.time() + 300
    while dialog.result_path is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert dialog.result_path == tmp_path / "server", dialog.log.toPlainText()
    assert (tmp_path / "server" / "Dockerfile").exists()
    assert dialog.open_btn.isEnabled()
    dialog.close()
