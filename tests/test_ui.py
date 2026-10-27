"""Professional UI: panels, inspector, problems, canvas cards, analysis dialogs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ai_made_easy.core.families import all_families  # noqa: E402
from ai_made_easy.core.graph import ValidationIssue  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("ui-tests")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def ctx(app):
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    context = AppContext()
    win = Workbench(context)
    app.processEvents()
    context._boot()
    app.processEvents()
    yield context
    context.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _node_ids(ctx, type_id=None):
    return [n.id for n in ctx.canvas.node_graph.all_nodes()
            if type_id is None or (ctx.canvas.block_of(n.id)
                                   and ctx.canvas.block_of(n.id).type_id == type_id)]


# ------------------------------------------------------------------ library

def test_library_lists_every_block_and_searches(app):
    from ai_made_easy.ui.features.library import ROLE_TYPE, BlockLibrary

    lib = BlockLibrary(lambda t: QtCore.QMimeData())
    listed = set()
    it = QtWidgets.QTreeWidgetItemIterator(lib.tree)
    while it.value():
        if it.value().data(0, ROLE_TYPE):
            listed.add(it.value().data(0, ROLE_TYPE))
        it += 1
    assert listed == {b.type_id for b in get_registry().all()}
    lib.search.setText("lstm")
    visible = [i.data(0, ROLE_TYPE) for i in lib._visible_blocks()]
    assert "core.lstm" in visible and "core.conv2d" not in visible
    placed = []
    lib.place_requested.connect(placed.append)
    lib._place_first_visible()
    assert placed and placed[0] in visible


def test_library_drag_payload_creates_canvas_nodes(app):
    from ai_made_easy.ui.canvas import CanvasController, block_mime_data
    from ai_made_easy.ui.canvas.node_factory import node_type_for

    CanvasController()  # registers node classes, as the app does at start-up
    mime = block_mime_data("core.conv2d")
    assert mime.hasFormat("OdenGraphQt/nodes")
    assert node_type_for("core.conv2d") in bytes(mime.data("OdenGraphQt/nodes")).decode()


# --------------------------------------------------------------- inspector

def test_selecting_a_block_opens_typed_editors(ctx, app):
    conv = _node_ids(ctx, "core.conv2d")[0]
    ctx._locate(conv)
    app.processEvents()
    assert ctx.properties.node_id == conv
    body = ctx.properties.scroll.widget()
    spins = body.findChildren(QtWidgets.QSpinBox)
    checks = body.findChildren(QtWidgets.QCheckBox)
    assert len(spins) >= 5 and checks, "int params -> spin boxes, bool -> checkbox"
    titles = [lbl.text() for lbl in body.findChildren(QtWidgets.QLabel)]
    assert "Conv2D" in titles


def test_editing_a_parameter_updates_canvas_and_revalidates(ctx, app):
    conv = _node_ids(ctx, "core.conv2d")[0]
    ctx.canvas.set_param(conv, "out_channels", 24)
    ctx.graph_service.settle_now()
    app.processEvents()
    assert ctx.canvas.params_of(conv)["out_channels"] == 24
    assert ctx.graph_service.last_shapes[conv][0] == 24
    assert ctx.project_store.dirty


def test_invalid_edit_surfaces_in_problems_and_on_the_card(ctx, app):
    conv = _node_ids(ctx, "core.conv2d")[0]
    ctx.canvas.set_param(conv, "kernel_size", 99)
    ctx.graph_service.settle_now()
    app.processEvents()
    errors, _ = ctx.problems.counts
    assert errors >= 1
    assert any(i.node_id == conv for i in ctx.validation_store.errors)
    view = next(n.view for n in ctx.canvas.node_graph.all_nodes() if n.id == conv)
    assert view._aime_badge == "error"
    assert ctx.validation_chip.property("state") == "error"


def test_cards_show_shapes_and_settings(ctx):
    nodes = {ctx.canvas.block_of(n.id).type_id: n.view for n in ctx.canvas.node_graph.all_nodes()
             if ctx.canvas.block_of(n.id)}
    assert nodes["core.conv2d"]._aime_subtitle.startswith("[")
    assert "lr" in nodes["train.adam"]._aime_subtitle


# ---------------------------------------------------------------- problems

def test_problems_panel_counts_filters_and_locates(app):
    from ai_made_easy.ui.features.problems import ProblemsPanel

    panel = ProblemsPanel(lambda nid: f"block {nid}", lambda issue: issue.node_id == "a")
    panel.set_issues([ValidationIssue("error", "bad", "a"),
                      ValidationIssue("warning", "meh", "b")])
    assert panel.counts == (1, 1)
    assert panel.tree.topLevelItemCount() == 2
    panel.show_warnings.setChecked(False)
    assert panel.tree.topLevelItemCount() == 1
    located, fixed = [], []
    panel.locate_requested.connect(located.append)
    panel.fix_requested.connect(fixed.append)
    item = panel.tree.topLevelItem(0)
    panel.tree.setCurrentItem(item)
    assert panel.fix_button.isEnabled()
    panel._locate(item)
    panel._fix_current()
    assert located == ["a"] and fixed[0].message == "bad"


def test_quick_fix_inserts_flatten(ctx, app):
    ctx.project_service.new_project()
    app.processEvents()
    from ai_made_easy.core.graph import Edge, Graph, NodeInstance

    g = Graph(name="fixme")
    for nid, tid, params in (("i", "core.input", {"shape": "1, 28, 28"}),
                             ("c", "core.conv2d", {}), ("d", "core.dense", {"units": 10}),
                             ("o", "core.output", {})):
        g.add_node(NodeInstance(nid, tid, params))
    for a, b in (("i", "c"), ("c", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    ctx.graph_service.load(g)
    app.processEvents()
    issue = next(i for i in ctx.validation_store.issues if "Flatten" in i.message)
    assert ctx._fixable(issue)
    ctx._apply_fix(issue)
    app.processEvents()
    assert _node_ids(ctx, "core.flatten")


# ------------------------------------------------------------ project flow

def test_new_project_is_empty_input_output(ctx, app):
    ctx.project_service.new_project()
    app.processEvents()
    types = sorted(ctx.canvas.block_of(i).type_id for i in _node_ids(ctx))
    assert types == ["core.input", "core.output"]


def test_every_example_opens_valid_without_overlaps(ctx, app):
    entries = ctx.project_service.list_samples()
    assert len(entries) >= 8
    for path, name, _desc, kind in entries:
        ctx.project_service.open_sample(path)
        app.processEvents()
        errors = [str(i) for i in ctx.validation_store.errors]
        assert not errors, (name, errors)
        assert not ctx.canvas.has_overlaps(), name
        assert kind in {f.label for f in all_families()}
        assert not ctx.project_store.dirty, name


def test_delete_selection_and_select_all(ctx, app):
    before = len(_node_ids(ctx))
    ctx.canvas.node_graph.clear_selection()
    target = _node_ids(ctx, "eval.accuracy")[0]
    ctx._locate(target)
    ctx.act_delete()
    app.processEvents()
    assert len(_node_ids(ctx)) == before - 1
    ctx.act_select_all()
    assert len(ctx.canvas.selected_ids()) == before - 1


def test_switching_themes_keeps_the_style(ctx, app):
    """Setting the style again deletes the old one, which widgets still alive (another
    window waiting to be deleted) may use: that crashed opening a second window."""
    import shiboken6

    before = app.style()
    ctx.act_theme_light()
    ctx.act_theme_dark()
    assert shiboken6.isValid(before) and app.style() is before


def test_theme_switch_updates_canvas_tokens(ctx, app):
    from ai_made_easy.ui.canvas import painter

    ctx.act_theme_light()
    assert ctx.theme.active() == "light"
    assert painter._T["NODE_BODY"] == "#ffffff"
    ctx.act_theme_dark()
    assert painter._T["NODE_BODY"] == "#23262d"


def test_code_page_offers_every_target(ctx, app):
    targets = [ctx.code_page.selector.itemData(i)
               for i in range(ctx.code_page.selector.count())]
    assert {"pytorch_model", "keras_model", "pytorch_train", "keras_train",
            "sklearn_train", "llm"} <= set(targets)
    ctx.code_page.select_target("pytorch_model")
    ctx._refresh_preview()
    assert "class " in ctx.code_page.view.toPlainText()
    ctx.code_page.select_target("sklearn_train")
    ctx._refresh_preview()
    assert "not available" in ctx.code_page.view.toPlainText()


def test_train_requires_a_valid_design(ctx, app, monkeypatch):
    calls = []
    monkeypatch.setattr(ctx.process_service, "run_training", lambda g, **kw: calls.append(g))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    ctx.validation_store.update([ValidationIssue("error", "broken", None)])
    ctx.act_train()
    assert calls == []
    ctx.validation_store.update([])
    ctx.act_train()
    assert len(calls) == 1


def test_command_palette_lists_actions_and_blocks(ctx, app):
    palette = ctx.window.command_palette
    palette._refill("train")
    texts = [palette.list.item(i).text() for i in range(palette.list.count())]
    assert any(t.startswith("&Train") or t.startswith("Train") for t in texts)
    palette._refill("lstm")
    texts = [palette.list.item(i).text() for i in range(palette.list.count())]
    assert any("LSTM" in t for t in texts)


# ------------------------------------------------------------- run panels

def test_training_page_tiles_curves_and_results(app, tmp_path):
    from ai_made_easy.ui.features.runconsole import TrainingPage
    from ai_made_easy.ui.stores import RunStore

    store = RunStore()
    page = TrainingPage(store)
    store.set(RunStore.RUNNING, "train")
    assert page.stop_btn.isEnabled() and not page.start_btn.isEnabled()
    for e in (1, 2):
        page.on_epoch({"epoch": e, "total": 2, "metrics": {"train_loss": 1.0 / e,
                                                            "val_loss": 1.1 / e,
                                                            "accuracy": 0.5 * e, "lr": 0.1}})
    assert set(page.last_metrics()) == {"train_loss", "val_loss", "accuracy"}
    assert page._tiles["accuracy"].value.text() == "1"
    store.set(RunStore.FINISHED, "train")
    assert "finished" in page.status.text()
    (tmp_path / "predictions.json").write_text("[]", encoding="utf-8")
    (tmp_path / "x_best.pt").write_text("", encoding="utf-8")
    page.set_results_available(tmp_path)
    assert page.errors_btn.isEnabled() and page.saliency_btn.isEnabled()


def _artifacts(tmp_path: Path) -> Path:
    preds = [{"index": i, "true": i % 3, "probs": [0.7, 0.2, 0.1] if i % 2 else [0.1, 0.2, 0.7],
              "file": None} for i in range(12)]
    (tmp_path / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    (tmp_path / "mistakes.json").write_text(json.dumps(
        [p for p in preds if max(range(3), key=lambda c: p["probs"][c]) != p["true"]]), encoding="utf-8")
    (tmp_path / "classes.json").write_text(json.dumps(["ant", "bee", "cat"]), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"accuracy": 0.42}), encoding="utf-8")
    return tmp_path


def test_error_analysis_dialog_builds(app, tmp_path):
    from ai_made_easy.ui.features.analysis import ErrorAnalysisDialog

    dialog = ErrorAnalysisDialog(None, _artifacts(tmp_path))
    tables = dialog.findChildren(QtWidgets.QTableWidget)
    assert tables and tables[0].rowCount() == 3
    assert tables[0].item(0, 0).text() == "ant"


def test_model_card_dialog_renders_markdown(app, tmp_path):
    from ai_made_easy.ui.features.analysis import ModelCardDialog

    dialog = ModelCardDialog(None, "demo", "Synthetic data", {"epochs": 3},
                             _artifacts(tmp_path), {"framework": "PyTorch"})
    dialog.use.setPlainText("internal triage")
    markdown = dialog.markdown()
    assert "| accuracy | 0.4200 |" in markdown and "internal triage" in markdown


def test_data_profile_describes_datasets(tmp_path):
    from ai_made_easy.core.data.profile import describe, profile_path

    csv = tmp_path / "t.csv"
    csv.write_text("a,b,label\n1,2,x\n3,,y\n", encoding="utf-8")
    profile = profile_path(csv, target="label")
    assert [c.name for c in profile.columns] == ["a", "b", "label"] and profile.rows == 2
    assert "1 missing" in profile.summary
    assert "Handwritten digits" in describe("data.torchvision", {"dataset": "mnist"})
    root = tmp_path / "imgs"
    (root / "a").mkdir(parents=True)
    assert "1 class" in describe("data.image_folder", {"root": str(root)})


def test_fuzzy_score_and_toasts(app):
    from ai_made_easy.ui.features.command_palette import fuzzy_score
    from ai_made_easy.ui.features.toasts import ToastLayer

    assert fuzzy_score("Train", "tr") > fuzzy_score("Export Training Script", "tr")
    assert fuzzy_score("Train", "xyz") is None
    host = QtWidgets.QWidget()
    layer = ToastLayer(host)
    for i in range(5):
        layer.toast(f"message {i}")
    assert len(layer.findChildren(QtWidgets.QLabel)) <= 5


def test_every_block_can_be_placed_and_edited(ctx, app):
    """Regression: params named like canvas built-ins (height/width) must still work."""
    ctx.project_service.new_project()
    for block in get_registry().all():
        ctx.canvas.place_block(block.type_id)
    app.processEvents()
    ids = {ctx.canvas.block_of(i).type_id: i for i in _node_ids(ctx)}
    assert set(ids) == {b.type_id for b in get_registry().all()}
    resize = ids["prep.resize"]
    ctx.canvas.set_param(resize, "height", 48)
    assert ctx.canvas.params_of(resize)["height"] == 48
    ir = ctx.canvas.to_ir()
    assert ir.nodes[resize].params["height"] == 48


def test_autosave_recovery_and_recent_files(ctx, app, tmp_path):
    ctx.project_store.mark_dirty()
    ctx._autosave()
    recovery = ctx._recovery_file()
    assert recovery.exists()
    assert json.loads(recovery.read_text(encoding="utf-8"))["nodes"]
    ctx._clear_recovery()
    assert not recovery.exists()
    project = tmp_path / "p.json"
    ctx.project_service.save_as(project)
    ctx._add_recent(project)
    assert ctx.recent_files()[0] == str(project)
    menu = QtWidgets.QMenu()
    ctx.populate_recent(menu)
    assert menu.actions()[0].text().split("\t")[0] == "p.json"
