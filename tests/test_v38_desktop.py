"""3.8 in the desktop app: Find Learning Rate on optimizer blocks, Train More and By Group
on finished runs."""
from __future__ import annotations

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
    QtCore.QCoreApplication.setApplicationName("v38-tests")
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


def _finished(history, framework: str, project: str) -> str:
    rec = history.create(tiny_classifier_dict(), framework=framework, project=project)
    history.finalize(rec.run_id, "finished", 0)
    return rec.run_id


def _buttons(widget) -> list[str]:  # noqa: ANN001
    return [b.text() for b in widget.findChildren(QtWidgets.QPushButton)]


def test_run_buttons_follow_the_selected_run(ctx, app):
    history = ctx.experiment_service.history
    project = ctx.project_store.name
    net = _finished(history, "pytorch", project)
    forest = _finished(history, "sklearn", project)
    page = ctx.experiments_page
    ctx.act_experiments()
    rows = {page.runs_table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole): r
            for r in range(page.runs_table.rowCount())}
    page.runs_table.selectRow(rows[net])
    assert page.more_btn.isEnabled() and page.groups_btn.isEnabled()
    page.runs_table.clearSelection()
    page.runs_table.selectRow(rows[forest])
    assert page.groups_btn.isEnabled() and not page.more_btn.isEnabled()
    page.runs_table.clearSelection()
    assert not page.more_btn.isEnabled() and not page.groups_btn.isEnabled()


def test_only_optimizer_blocks_offer_the_finder(ctx, app, monkeypatch):
    from ai_made_easy.core.registry import get_registry
    from ai_made_easy.ui.features import run_tools

    class Opened:                     # stands in for the modal dialog
        def __init__(self, *args) -> None:  # noqa: ANN002
            self.suggested = None

        def exec(self) -> int:
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(run_tools, "LRFinderDialog", Opened)
    reg = get_registry()
    seen = []
    ctx.properties.find_lr_requested.connect(seen.append)
    ctx.properties.show_node("opt", reg.get("train.adam"), {"lr": 0.01}, [])
    assert "Find Learning Rate…" in _buttons(ctx.properties)
    ctx.properties.find_lr_btn.click()
    assert seen == ["opt"]
    ctx.properties.show_node("d1", reg.get("core.dense"), {"units": 8}, [])
    assert "Find Learning Rate…" not in _buttons(ctx.properties)


def test_groups_dialog_shows_flagged_groups(ctx, app):
    from ai_made_easy.core import groups
    from ai_made_easy.ui.features.run_tools import GroupsDialog

    run_id = _finished(ctx.experiment_service.history, "pytorch", "p")
    dialog = GroupsDialog(None, ctx.experiment_service.manager, run_id)
    assert not dialog.score_btn.isEnabled()          # the design has no table to group by
    assert "trained on a CSV" in dialog.summary.text()
    result = groups.summarize({
        "column": "region", "how": "values", "overall": {"count": 300, "accuracy": 0.8},
        "groups": [{"value": "east", "count": 100, "accuracy": 0.55},
                   {"value": "north", "count": 200, "accuracy": 0.92}]})
    dialog.show_result(result)
    assert dialog.table.rowCount() == 3
    assert "Worse for east" in dialog.summary.text()
    east = dialog.table.item(1, 3)
    assert "worse" in east.text() and east.font().bold()
    assert not dialog.table.item(2, 0).font().bold()
    assert dialog.fairness.text() == ""                  # no classes: no fairness check
    result["fairness"] = groups.fairness({
        "classes": ["1"], "overall": {"count": 300},
        "groups": [{"value": "east", "count": 100,
                    "rates": {"selection": [0.2], "tpr": [0.6], "fpr": [0.3]}},
                   {"value": "north", "count": 200,
                    "rates": {"selection": [0.5], "tpr": [0.9], "fpr": [0.1]}}]})
    dialog.show_result(result)
    assert "under the four-fifths rule" in dialog.fairness.text()
    assert "Treated differently: east." in dialog.fairness.text()
    dialog.close()


def test_train_more_dialog_defaults(app):
    from ai_made_easy.ui.features.run_tools import TrainMoreDialog

    dialog = TrainMoreDialog(None, "r1")
    assert dialog.epochs.value() == 10 and dialog.scale.value() == 1.0
    dialog.close()


@pytest.mark.slow
def test_finder_dialog_runs_and_suggests(ctx, app):
    pytest.importorskip("torch")
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.ui.features.run_tools import LRFinderDialog

    dialog = LRFinderDialog(None, ctx.experiment_service.manager,
                            Graph.from_dict(tiny_classifier_dict()))
    dialog.show()
    deadline = time.time() + 240
    while not dialog.use_btn.isEnabled() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert dialog.use_btn.isEnabled(), dialog.status.text()
    assert dialog.suggested and dialog.use_btn.text().startswith("Use ")
    assert "Suggested" in dialog.status.text() and dialog.marker.isVisible()
    dialog.accept()
