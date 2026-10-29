"""3.4 — the items 3.3 deferred: SHAP values, collapsing groups and SimCLR (NT-Xent on
two augmented views of each image). Tiny synthetic data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import chain, write_images  # noqa: E402

from ai_made_easy.core import canvas as cv  # noqa: E402
from ai_made_easy.core.codegen import CodegenError  # noqa: E402
from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402

TRAINER = ("train.trainer", {"epochs": 6, "batch_size": 32, "device": "cpu", "seed": 0})


def _run(graph, tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    code = generate_training(graph, "pytorch")
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return code


# ------------------------------------------------------------------ SHAP

def test_shap_is_an_optional_extra():
    defn = get_registry().get("eval.shap_values")
    assert defn.extra == "explain" and defn.requires == ("shap",)


def test_shap_values_rank_the_informative_feature(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("shap")
    pd = pytest.importorskip("pandas")
    import numpy as np

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"signal": rng.normal(size=400), "noise": rng.normal(size=400),
                          "other": rng.normal(size=400)})
    frame["label"] = (frame["signal"] > 0).astype(int)
    csv = tmp_path / "table.csv"
    frame.to_csv(csv, index=False)
    graph = chain("shap", "3", [("core.dense", {"units": 16}), ("core.relu", {}),
                                ("core.dense", {"units": 2})],
                  [("data.csv", {"path": str(csv), "target_column": "label"}),
                   ("train.adam", {"lr": 0.02}), ("train.trainer", {**TRAINER[1], "epochs": 10}),
                   ("eval.shap_values", {"background": 50, "max_samples": 80})])
    code = _run(graph, tmp_path, monkeypatch)
    assert "shap.GradientExplainer" in code and "import shap" in code
    explained = json.loads((tmp_path / "explain.json").read_text(encoding="utf-8"))
    assert explained["method"] == "shap" and explained["background"] == 50
    assert explained["features"][0]["name"] == "signal"


# ------------------------------------------------------------------ collapsing groups

def test_collapse_flags_round_trip_in_the_design():
    meta: dict = {}
    gid = cv.add_group(meta, ["a", "b"])
    assert cv.set_collapsed(meta, gid, True) and meta["groups"][0]["collapsed"] is True
    assert cv.hidden_by(meta) == {"a": gid, "b": gid}
    assert cv.set_collapsed(meta, gid, False) and "collapsed" not in meta["groups"][0]
    assert cv.hidden_by(meta) == {} and not cv.set_collapsed(meta, "nope", True)
    # code generation never sees groups, folded or not
    design = {"nodes": [], "edges": [], "meta": {"groups": [{"id": "g", "nodes": ["a"],
                                                             "collapsed": True}]}}
    assert "groups" not in cv.strip(design)["meta"]


@pytest.fixture()
def desk(isolated_home):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.core.graph import Graph
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from tests.conftest import tiny_classifier_dict

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("v34")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    ctx = AppContext()
    win = Workbench(ctx)
    app.processEvents()
    design = tiny_classifier_dict()
    for i, node in enumerate(design["nodes"]):
        node["position"] = [i * 300, 0] if node["type"].startswith("core.") else [i * 300, 400]
    ctx.canvas.load_ir(Graph.from_dict(design))
    yield ctx, win, app
    ctx.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _core_nodes(ctx) -> list:
    return [n for n in ctx.canvas.node_graph.all_nodes()
            if (b := ctx.canvas.block_of(n.id)) and b.type_id.startswith("core.")]


def test_desktop_groups_fold_into_a_card_and_open_where_they_moved(desk):
    ctx, win, app = desk
    members = _core_nodes(ctx)[:3]
    for n in ctx.canvas.node_graph.all_nodes():
        n.set_selected(n in members)
    group = ctx.canvas.group_selected("Encoder")
    before = {n.id: tuple(n.pos()) for n in members}
    for n in members:                                    # a block inside the group
        n.set_selected(n is members[0])
    win.actions["edit.collapse_group"].trigger()
    app.processEvents()
    assert group.folded and tuple(group.size()) == group.FOLDED_SIZE
    assert all(not n.view.isVisible() for n in members)
    pipes = [p for n in members for port in n.output_ports() for p in port.view.connected_pipes]
    assert pipes and all(not p.isVisible() for p in pipes)
    meta = ctx.canvas.to_ir().meta
    assert meta["groups"][0]["collapsed"] is True
    assert sorted(meta["groups"][0]["nodes"]) == sorted(before)    # members kept while folded
    group.set_pos(group.x_pos() + 100, group.y_pos() + 50)
    assert ctx.canvas.set_group_collapsed(group, False)
    app.processEvents()
    assert all(n.view.isVisible() for n in members)
    for n in members:
        assert tuple(n.pos()) == pytest.approx((before[n.id][0] + 100, before[n.id][1] + 50))
    assert "collapsed" not in ctx.canvas.to_ir().meta["groups"][0]


def test_desktop_folded_groups_load_and_give_their_blocks_back(desk):
    ctx, _win, app = desk
    graph = ctx.canvas.to_ir()
    ids = [n.id for n in _core_nodes(ctx)[:2]]
    graph.meta["groups"] = [{"id": "group1", "title": "Stem", "nodes": ids,
                             "color": "#91A7FF", "collapsed": True}]
    ctx.canvas.load_ir(graph)
    app.processEvents()
    group = ctx.canvas._annotations("group")[0]
    hidden = ctx.canvas.group_members(group)   # the loaded blocks (new canvas ids)
    assert len(hidden) == len(ids)
    assert group.folded and all(not n.view.isVisible() for n in hidden)
    group.set_selected(True)
    assert ctx.canvas.delete_selected() == 1                    # deleting the card ...
    app.processEvents()
    assert all(n.view.isVisible() for n in hidden)              # ... keeps its blocks
    assert len(_core_nodes(ctx)) >= 2


# ------------------------------------------------------------------ SimCLR

def _simclr_graph(root: Path):
    return chain("simclr", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 8})], [
        ("data.image_folder", {"root": str(root)}),
        ("prep.resize", {"height": 16, "width": 16}),
        ("prep.random_flip", {"mode": "both"}), ("prep.random_rotation", {"degrees": 15}),
        ("train.loss_nt_xent", {"classes": 2, "temperature": 0.5}),
        ("train.adam", {"lr": 0.01}), ("eval.accuracy", {}),
        ("train.trainer", {**TRAINER[1], "epochs": 3, "batch_size": 16})])


def test_simclr_trains_on_two_views_of_each_image(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    graph = _simclr_graph(write_images(tmp_path))
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    code = _run(graph, tmp_path, monkeypatch)
    assert "TwoViews(build_transforms(train=True" in code
    assert "xb, yb, loss_fn.paired = views(xb, yb)" in code
    out = capsys.readouterr().out
    assert "test:" in out and "accuracy=" in out.split("test:")[-1]
    assert Path(tmp_path, "calibration.npz").exists()          # single images, not pairs
    import numpy as np

    assert np.load(tmp_path / "calibration.npz")["x"].shape[1:] == (3, 16, 16)


def test_simclr_needs_images():
    graph = chain("simclr_table", "6", [("core.dense", {"units": 8})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 2}),
                   ("train.loss_nt_xent", {"classes": 2}), ("train.adam", {}), TRAINER])
    with pytest.raises(CodegenError, match="two augmented views"):
        generate_training(graph, "pytorch")


def test_other_scripts_have_no_views():
    code = generate_training(chain("plain_images", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 4, "kernel_size": 3, "padding": 1}),
        ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})],
        [("data.image_folder", {"root": "imgs"}), ("prep.resize", {"height": 16, "width": 16}),
         ("train.adam", {}), TRAINER]), "pytorch")
    assert "TwoViews" not in code and "views(" not in code


def test_simclr_recipe_is_valid():
    from ai_made_easy.core.recipes import build

    graph = build("image_simclr", task="multiclass")
    assert "train.loss_nt_xent" in {n.type_id for n in graph.nodes.values()}
    assert not [i for i in graph.validate() if i.severity == "error"]
    code = generate_training(graph, "pytorch")
    assert "NTXentLoss(EMBED_DIM, NUM_CLASSES" in code and "TwoViews" in code


# ------------------------------------------------------------------ pair losses

PAIR_DATA = ("data.synthetic", {"n_features": 6, "n_classes": 3, "n_samples": 300, "seed": 1})
EMBED = [("core.dense", {"units": 32}), ("core.relu", {}), ("core.dense", {"units": 8})]


@pytest.mark.parametrize("loss", ["train.loss_cosine_embedding", "train.loss_margin_ranking"])
def test_pair_losses_learn_from_pairs_made_from_the_labels(loss, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    assert get_registry().get(loss).description
    graph = chain(loss.split(".")[1], "6", EMBED,
                  [PAIR_DATA, (loss, {"classes": 3}), ("train.adamw", {"lr": 0.01}), TRAINER,
                   ("eval.accuracy", {})])
    assert not [i for i in graph.validate() if i.severity == "error"]
    code = _run(graph, tmp_path, monkeypatch)
    assert "EMBED_DIM = 8" in code and "NUM_CLASSES = 3" in code
    lines = [x for x in capsys.readouterr().out.splitlines() if "accuracy=" in x]
    assert float(lines[-1].split("accuracy=")[1].split()[0]) > 0.6     # chance is 1/3


def test_pair_losses_handle_batches_without_pairs():
    torch = pytest.importorskip("torch")
    from torch import nn

    from ai_made_easy.core.training import catalog as cat

    ns = {"torch": torch, "nn": nn}
    for name in ("CosineEmbeddingPairsLoss", "MarginRankingPairsLoss"):
        exec(cat.TRAIN_HELPERS[name], ns)
        fn = ns[name](4, 3)
        emb = torch.randn(3, 4, requires_grad=True)
        one_class = fn(emb, torch.tensor([1, 1, 1]))        # no negatives
        all_different = fn(emb, torch.tensor([0, 1, 2]))    # no positives
        for value in (one_class, all_different):
            assert torch.isfinite(value)
            value.backward()


def test_desktop_folded_groups_draw_stand_ins_for_wires_that_cross_them(desk):
    ctx, _win, app = desk
    canvas = ctx.canvas
    core = _core_nodes(ctx)
    inside = core[1:3]                        # a group in the middle of the chain
    for n in canvas.node_graph.all_nodes():
        n.set_selected(n in inside)
    group = canvas.group_selected("Middle")
    ids = {n.id for n in inside}
    crossing = {id(p) for n in inside
                for port in [*n.input_ports(), *n.output_ports()]
                for p in port.view.connected_pipes
                if p.input_port and p.output_port
                and (p.input_port.node.id in ids) != (p.output_port.node.id in ids)}
    assert crossing
    assert canvas.set_group_collapsed(group, True)
    app.processEvents()
    items = canvas.stand_ins.items
    assert {id(i.pipe) for i in items} == crossing
    assert all(i.isVisible() and i.scene() is not None for i in items)
    rect = group.view.sceneBoundingRect()
    ends = [i.path().pointAtPercent(t) for i in items for t in (0.0, 1.0)]
    assert any(abs(p.x() - rect.left()) < 1 or abs(p.x() - rect.right()) < 1 for p in ends)
    before = [i.path().boundingRect() for i in items]
    group.set_pos(group.x_pos() + 200, group.y_pos() + 120)   # stand-ins follow the card
    app.processEvents()
    app.processEvents()
    assert [i.path().boundingRect() for i in items] != before
    assert canvas.set_group_collapsed(group, False)
    assert canvas.stand_ins.items == [] and all(i.scene() is None for i in items)


# ------------------------------------------------------------------ Grad-CAM

def _cnn_graph(root: Path, extra=()):
    return chain("gradcam", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})], [
        ("data.image_folder", {"root": str(root)}),
        ("prep.resize", {"height": 16, "width": 16}),
        ("train.adam", {"lr": 0.01}), ("eval.accuracy", {}),
        ("train.trainer", {**TRAINER[1], "epochs": 2, "batch_size": 16}), *extra])


def test_grad_cam_saves_heatmaps_with_the_run(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    Image = pytest.importorskip("PIL.Image")
    graph = _cnn_graph(write_images(tmp_path), [("eval.grad_cam", {"n_samples": 3})])
    assert get_registry().get("eval.grad_cam").library == "PyTorch"
    code = _run(graph, tmp_path, monkeypatch)
    assert code.count("def grad_cam(") == 1 and "save_grad_cams(model" in code
    pictures = sorted((tmp_path / "eval_samples").glob("gradcam_*.png"))
    assert len(pictures) == 3
    with Image.open(pictures[0]) as im:
        assert im.size[0] == 2 * im.size[1] and im.size[1] >= 128   # image | heatmap
    assert "_as_" in pictures[0].stem


def test_grad_cam_is_skipped_where_it_cannot_explain():
    from ai_made_easy.core.codegen.training_gen import collect_spec

    table = chain("gradcam_table", "6", [("core.dense", {"units": 2})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 2}),
                   ("train.adam", {}), TRAINER, ("eval.grad_cam", {})])
    spec = collect_spec(table)
    assert spec.gradcam is None and any("image models" in w for w in spec.warnings)
    assert "save_grad_cams" not in generate_training(table, "pytorch")


def test_inspect_script_keeps_one_grad_cam(tmp_path):
    from ai_made_easy.core.codegen.training_gen import generate_inspect

    for extra in ((), [("eval.grad_cam", {})]):
        code = generate_inspect(_cnn_graph(tmp_path, extra))
        assert code.count("def grad_cam(") == 1
        compile(code, "inspect", "exec")


def _run_with_cams(names):
    from ai_made_easy.core import api
    from tests.conftest import tiny_classifier_dict

    h = api.manager().history
    rec = h.create(tiny_classifier_dict())
    folder = h.path(rec.run_id) / "eval_samples"
    folder.mkdir(parents=True)
    png = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                        "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")
    for name in names:
        (folder / f"{name}.png").write_bytes(png)
    return rec.run_id


def test_desktop_experiments_show_a_runs_grad_cams(desk):
    _ctx, _win, app = desk
    from ai_made_easy.ui.features.experiments import ExperimentsPage

    run_id = _run_with_cams(["gradcam_00_cat_as_dog", "gradcam_01_dog_as_dog"])
    page = ExperimentsPage()
    page._show_grad_cams(run_id)
    assert page.cam_gallery.count() == 2 and not page.cam_label.isHidden()
    assert page.cam_gallery.item(0).text() == "cat → dog"
    page._show_grad_cams(None)
    assert page.cam_gallery.count() == 0 and page.cam_label.isHidden()


@pytest.fixture()
def client(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    with TestClient(create_app(token="", projects_dir=tmp_path / "projects")) as c:
        yield c


def test_web_run_samples_carry_the_grad_cams(client):
    run_id = _run_with_cams(["gradcam_00_cat_as_dog"])
    got = client.get(f"/api/runs/{run_id}/samples").json()
    assert [s["name"] for s in got["samples"]] == ["gradcam_00_cat_as_dog"]
    assert got["samples"][0]["data_url"].startswith("data:image/png;base64,")


# ------------------------------------------------------------------ labelled pairs

def _pairs_csv(path: Path, kind: str, n: int = 400) -> Path:
    import numpy as np

    rng = np.random.default_rng(0)
    if kind == "same":
        centres = rng.normal(scale=3.0, size=(3, 4))
        ca, cb = rng.integers(0, 3, n), rng.integers(0, 3, n)
        cb = np.where(rng.random(n) < 0.5, ca, cb)               # about half are the same
        a = centres[ca] + rng.normal(scale=0.5, size=(n, 4))
        b = centres[cb] + rng.normal(scale=0.5, size=(n, 4))
        label = (ca == cb).astype(int)
    else:
        w = np.array([1.0, -2.0, 0.5, 0.0])
        a, b = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
        label = (a @ w > b @ w).astype(int)
    cols = [f"f{i}" for i in range(4)]
    rows = ["," .join([*(f"a_{c}" for c in cols), *(f"b_{c}" for c in cols), "label"])]
    rows += [",".join([*map(str, a[i]), *map(str, b[i]), str(label[i])]) for i in range(n)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("loss,units", [("train.loss_cosine_embedding", 8),
                                        ("train.loss_margin_ranking", 1)])
def test_labelled_pairs_train_an_encoder(loss, units, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    kind = "ranking" if "ranking" in loss else "same"
    csv = _pairs_csv(tmp_path / "pairs.csv", kind)
    graph = chain(f"pairs_{kind}", "4", [("core.dense", {"units": 16}), ("core.relu", {}),
                                         ("core.dense", {"units": units})],
                  [("data.pairs", {"path": str(csv)}), (loss, {"classes": 2}),
                   ("train.adam", {"lr": 0.01}), ("eval.accuracy", {}),
                   ("train.trainer", {**TRAINER[1], "epochs": 15})])
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    code = _run(graph, tmp_path, monkeypatch)
    assert "_Encoder = " in code and "x.flatten(0, 1)" in code
    out = capsys.readouterr().out
    accuracy = float(out.split("test:")[-1].split("accuracy=")[1].split()[0])
    assert accuracy > 0.8, out[-1500:]
    classes = json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))
    assert classes == (["b ranks higher", "a ranks higher"] if kind == "ranking"
                       else ["different", "same"])


def test_labelled_pairs_need_a_pair_loss_and_ranking_needs_one_score(tmp_path):
    csv = tmp_path / "pairs.csv"
    base = [("data.pairs", {"path": str(csv)}), ("train.adam", {}), TRAINER]
    wrong = chain("pairs_ce", "4", [("core.dense", {"units": 2})],
                  [*base, ("train.loss_cross_entropy", {})])
    with pytest.raises(CodegenError, match="pair loss"):
        generate_training(wrong, "pytorch")
    wide = chain("pairs_rank", "4", [("core.dense", {"units": 3})],
                 [*base, ("train.loss_margin_ranking", {"classes": 2})])
    with pytest.raises(CodegenError, match="one output unit"):
        generate_training(wide, "pytorch")


def test_other_designs_have_no_pair_wrapper():
    code = generate_training(chain("plain", "6", [("core.dense", {"units": 3})],
                                   [PAIR_DATA, ("train.adam", {}), TRAINER]), "pytorch")
    assert "_Encoder" not in code


def test_pairs_recipe_trains_on_its_demo_pairs(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from ai_made_easy.core.recipes import build

    graph = build("pairs_verification", task="binary")
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    trainer = next(n for n in graph.nodes.values() if n.type_id == "train.trainer")
    trainer.params.update(epochs=8, device="cpu")
    code = _run(graph, tmp_path, monkeypatch)
    assert "demo pairs" in code
    out = capsys.readouterr().out.split("test:")[-1]
    assert float(out.split("accuracy=")[1].split()[0]) > 0.8, out


def test_desktop_undoing_a_folded_group_gives_its_blocks_back(desk):
    ctx, _win, app = desk
    canvas = ctx.canvas
    canvas.node_graph.clear_undo_stack()
    inside = _core_nodes(ctx)[1:3]
    for n in canvas.node_graph.all_nodes():
        n.set_selected(n in inside)
    group = canvas.group_selected("Middle")
    assert canvas.set_group_collapsed(group, True)
    app.processEvents()
    stack = canvas.node_graph.undo_stack()
    while stack.canUndo():
        stack.undo()
    app.processEvents()
    assert not canvas._annotations("group")
    assert all(n.view.isVisible() for n in inside) and canvas.stand_ins.items == []
    while stack.canRedo():
        stack.redo()
    app.processEvents()
    groups = canvas._annotations("group")       # the card comes back folded ...
    assert len(groups) == 1 and groups[0].folded
    assert all(not n.view.isVisible() for n in inside)          # ... and hides them again
    assert len(canvas.stand_ins.items) == 2


# ------------------------------------------------------------------ serving metric models

def _train_ns(graph, tmp_path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "aime_training"}
    exec(compile(generate_training(graph, "pytorch"), f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return ns


def test_serving_a_metric_model_scores_against_its_classes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import numpy as np

    graph = chain("serve_arcface", "6", EMBED,
                  [PAIR_DATA, ("train.loss_arcface", {"classes": 3}),
                   ("train.adamw", {"lr": 0.01}), TRAINER])
    ns = _train_ns(graph, tmp_path, monkeypatch)
    ns["load_predictor"](str(tmp_path))
    x, y = (ns["make_arrays"]()["test"])
    out = ns["infer"](x[:40].tolist())
    assert len(out[0]["embedding"]) == 8 and len(out[0]["probabilities"]) == 3
    accuracy = np.mean([r["class_index"] == int(t) for r, t in zip(out, y[:40], strict=True)])
    assert accuracy > 0.6                     # before the fix: argmax over embedding dims


def test_serving_a_pairs_model_embeds_or_scores_single_items(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    for loss, units, key in (("train.loss_cosine_embedding", 8, "embedding"),
                             ("train.loss_margin_ranking", 1, "score")):
        folder = tmp_path / key
        folder.mkdir()
        csv = _pairs_csv(folder / "pairs.csv", "ranking" if units == 1 else "same")
        graph = chain(f"serve_{key}", "4", [("core.dense", {"units": 16}), ("core.relu", {}),
                                            ("core.dense", {"units": units})],
                      [("data.pairs", {"path": str(csv)}), (loss, {"classes": 2}),
                       ("train.adam", {"lr": 0.01}), TRAINER])
        ns = _train_ns(graph, folder, monkeypatch)
        ns["load_predictor"](str(folder))
        out = ns["infer"]([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 2.0]])
        assert len(out) == 2 and key in out[0]
        if key == "embedding":
            assert len(out[0]["embedding"]) == 8


def test_feature_explanations_skip_labelled_pairs(tmp_path):
    from ai_made_easy.core.codegen.training_gen import collect_spec

    graph = chain("pairs_explain", "4", [("core.dense", {"units": 8})],
                  [("data.pairs", {}), ("train.loss_cosine_embedding", {"classes": 2}),
                   ("train.adam", {}), TRAINER, ("eval.permutation_importance", {})])
    spec = collect_spec(graph)
    assert spec.explain is None and any("labelled pairs" in w for w in spec.warnings)


def test_desktop_a_moved_folded_card_reopens_where_it_was_moved(desk):
    ctx, _win, app = desk
    canvas = ctx.canvas
    inside = _core_nodes(ctx)[1:3]
    for n in canvas.node_graph.all_nodes():
        n.set_selected(n in inside)
    group = canvas.group_selected("Middle")
    before = {n.id: tuple(n.pos()) for n in inside}
    canvas.set_group_collapsed(group, True)
    group.set_pos(group.x_pos() + 400, group.y_pos() + 250)
    graph = canvas.to_ir()
    for nid, (x, y) in before.items():                  # saved where they will open
        assert graph.nodes[nid].position == pytest.approx((x + 400, y + 250))
    canvas.load_ir(graph)
    app.processEvents()
    card = canvas._annotations("group")[0]
    assert card.folded
    assert canvas.set_group_collapsed(card, False)
    reopened = sorted(tuple(n.pos()) for n in canvas.group_members(card))
    wanted = sorted((x + 400, y + 250) for x, y in before.values())
    assert len(reopened) == len(wanted)
    for got, want in zip(reopened, wanted, strict=True):   # pytest 9: approx isn't nested
        assert got == pytest.approx(want)
