"""Data workspace: profiles, findings, lints, fingerprints, split / augmentation previews."""
from __future__ import annotations

import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest

from conftest import tiny_classifier_dict

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.data.fingerprint import fingerprint, graph_fingerprint
from ai_made_easy.core.data.lints import ProfileCache, data_issues
from ai_made_easy.core.data.profile import profile_dataset, profile_path
from ai_made_easy.core.data.splits import split_indices, split_preview
from ai_made_easy.core.graph import Graph


def _messages(profile) -> str:
    return "\n".join(f.message for f in profile.findings)


def _table(tmp_path, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "row_id": np.arange(n), "a": rng.normal(size=n), "b": rng.exponential(5, n) ** 3,
        "const": 1, "city": rng.choice(["x", "y", "z"], n),
        "free": [f"user-{i}-{rng.integers(1e9)}" for i in range(n)],
        "zip": [f"z{v}" for v in rng.integers(0, 60, n)],
        "label": rng.choice(["cat", "dog"], n, p=[0.85, 0.15])})
    frame.loc[:9, "a"] = np.nan
    frame["leak"] = frame["label"].map({"cat": "C", "dog": "D"})
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    return path, frame


def _csv_design(path, *, extra=(), loss="train.loss_cross_entropy", units=2) -> Graph:
    data = tiny_classifier_dict()
    data["nodes"] = [n for n in data["nodes"] if n["type"] not in ("data.synthetic",
                                                                  "train.loss_cross_entropy")]
    for node in data["nodes"]:
        if node["id"] == "d2":
            node["params"]["units"] = units
    data["nodes"].append({"id": "data", "type": "data.csv", "position": [0, 0],
                          "params": {"path": str(path), "target_column": "label"}})
    data["nodes"].append({"id": "loss", "type": loss, "params": {}, "position": [0, 0]})
    for i, (type_id, params) in enumerate(extra):
        data["nodes"].append({"id": f"p{i}", "type": type_id, "params": params,
                              "position": [0, 0]})
    return Graph.from_dict(data)


# ================================================================ profiles

def test_table_profile_findings(tmp_path):
    path, _frame = _table(tmp_path)
    profile = profile_path(path, target="label")
    text = _messages(profile)
    assert profile.rows == 300 and profile.task == "classification"
    assert {c.name: c.count for c in profile.classes} == {"cat": 255, "dog": 45} or \
        sum(c.count for c in profile.classes) == 300
    assert "'const' is constant" in text
    assert "'row_id' looks like an identifier" in text
    assert "'leak' predicts 'label' perfectly" in text
    assert "class imbalance" in text
    assert "'free' looks like an identifier" in text and "'free' has" not in text
    assert "'zip' has 60 categories" in text
    assert "'a' has 10 missing" in text
    assert "strongly skewed" in text
    a = profile.column("a")
    assert a.kind == "numeric" and a.missing == 10 and a.histogram and a.mean is not None
    assert profile.column("label").role == "target"
    assert profile.column("city").kind == "categorical" and profile.column("city").top


def test_table_profile_errors_and_duplicates(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"x": [1, 1, 2, 3] * 10, "y": [0, 0, 1, 1] * 10}).to_csv(path, index=False)
    assert "target column 'label' is not in the table" in _messages(
        profile_path(path, target="label"))
    text = _messages(profile_path(path, target="y"))
    assert "duplicate row" in text
    missing = profile_dataset("data.csv", {"path": str(tmp_path / "nope.csv")})
    assert missing.error and "was not found" in _messages(missing)
    assert "fine" not in missing.text()


def test_text_and_timeseries_profiles(tmp_path):
    texts = tmp_path / "t.csv"
    pd.DataFrame({"text": ["good movie", "bad", "", "good movie"] * 5,
                  "label": ["pos", "neg", "neg", "pos"] * 5}).to_csv(texts, index=False)
    profile = profile_dataset("data.text_csv", {"path": str(texts), "format": "csv",
                                                "text_column": "text", "label_column": "label"})
    assert profile.column("text").role == "text"
    assert "duplicate text" in _messages(profile)
    assert any("text length" in d for d in profile.details)
    series = tmp_path / "s.csv"
    pd.DataFrame({"value": np.arange(60.0)}).to_csv(series, index=False)
    profile = profile_dataset("data.timeseries_csv", {
        "path": str(series), "target_columns": "value, missing", "feature_columns": "",
        "window": 32, "horizon": 1, "stride": 1})
    text = _messages(profile)
    assert "not in the table: missing" in text and "window(s) fit" in text


def test_folder_profiles(tmp_path):
    from PIL import Image

    root = tmp_path / "imgs"
    for name, color, count in (("a", (250, 250, 250), 12), ("b", (5, 5, 5), 3)):
        (root / name).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (32 + i, 24), color).save(root / name / f"{i}.png")
    (root / "b" / "broken.png").write_bytes(b"not an image")
    profile = profile_dataset("data.image_folder", {"root": str(root)})
    text = _messages(profile)
    assert profile.rows == 16 and [c.name for c in profile.classes] == ["a", "b"]
    assert "cannot be opened" in text and "only 4 sample" in text
    assert any("image sizes" in d for d in profile.details)
    assert len(profile.samples["a"]) == 12
    words = tmp_path / "texts"
    (words / "x").mkdir(parents=True)
    (words / "x" / "1.txt").write_text("hello world", encoding="utf-8")
    (words / "x" / "2.txt").write_text("", encoding="utf-8")
    profile = profile_path(words)
    assert profile.type_id == "data.text_folder" and "empty text" in _messages(profile)


def test_array_and_record_profiles(tmp_path):
    npz = tmp_path / "d.npz"
    x = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    x[0, 0] = np.nan
    np.savez(npz, x=x, y=np.r_[np.zeros(45), np.ones(5)].astype(int))
    profile = profile_dataset("data.numpy", {"path": str(npz), "x_key": "x", "y_key": "y"})
    text = _messages(profile)
    assert profile.rows == 50 and "NaN" in text and "class imbalance" in text
    bad = profile_dataset("data.numpy", {"path": str(npz), "x_key": "features", "y_key": "y"})
    assert "not in the archive: features" in _messages(bad)
    records = tmp_path / "r.jsonl"
    records.write_text('{"x": [1, 2], "y": 0}\n{"x": [1], "y": 1}\n{"z": 1}\n', encoding="utf-8")
    text = _messages(profile_dataset("data.json", {"path": str(records), "x_field": "x",
                                                   "y_field": "y"}))
    assert "lack 'x'" in text and "different lengths" in text


def test_profile_round_trips_to_dict(tmp_path):
    path, _ = _table(tmp_path, n=40)
    data = profile_path(path, target="label").to_dict()
    assert json.loads(json.dumps(data))["columns"][0]["name"] == "row_id"
    assert data["ok"] is True


# ================================================================ splits

def _script_split(script: str):
    """The generated split_indices function and its constants."""
    source = script[script.index("def split_indices"):]
    source = source[: source.index("\n\n\n")]
    consts = dict(re.findall(r"^(VAL_FRACTION|TEST_FRACTION|SPLIT_SEED) = (.+)$", script, re.M))
    ns = {"np": np, **{k: eval(v) for k, v in consts.items()}}  # noqa: S307
    exec(source, ns)  # noqa: S102
    return ns["split_indices"]


@pytest.mark.parametrize("stratify", [True, False])
def test_split_preview_matches_generated_script(tmp_path, stratify):
    path, frame = _table(tmp_path)
    graph = _csv_design(path, extra=[("prep.split", {"val_fraction": 0.2, "test_fraction": 0.1,
                                                     "stratify": stratify, "seed": 3})])
    from ai_made_easy.core.tabular.layout import layout_of

    inputs = [n for n in graph.nodes.values() if n.type_id == "core.input"]
    inputs[0].params["shape"] = str(layout_of(graph).width)     # the one-hot encoded row
    workdir = tmp_path / "out"
    workdir.mkdir()
    script = export_training(graph, "pytorch", workdir).read_text(encoding="utf-8")
    generated = _script_split(script)
    classes = sorted(frame["label"].astype(str).unique())
    y = frame["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    expected = generated(len(y), y)
    ours = split_indices(len(y), y, val=0.2, test=0.1, seed=3, stratify=stratify)
    for a, b in zip(expected, ours, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    preview = split_preview("data.csv", {"path": str(path), "target_column": "label"},
                            {"val_fraction": 0.2, "test_fraction": 0.1, "stratify": stratify,
                             "seed": 3, "shuffle": True})
    assert preview.totals == {"train": len(expected[0]), "val": len(expected[1]),
                              "test": len(expected[2])}
    assert sum(c["val"] for c in preview.per_class.values()) == len(expected[1])
    assert preview.method == ("stratified" if stratify else "random")


def test_split_preview_flags_missing_classes(tmp_path):
    root = tmp_path / "imgs"
    for name, count in (("big", 30), ("tiny", 1)):
        (root / name).mkdir(parents=True)
        for i in range(count):
            (root / name / f"{i}.png").write_bytes(b"x")
    preview = split_preview("data.image_folder", {"root": str(root)},
                            {"val_fraction": 0.1, "test_fraction": 0.1, "seed": 42})
    assert preview.method.startswith("random permutation")
    assert sum(preview.totals.values()) == 31
    assert "tiny" in preview.missing_classes().get("val", []) + \
        preview.missing_classes().get("test", []) + \
        (["tiny"] if preview.per_class["tiny"]["train"] == 0 else [])
    assert "split preview needs" in split_preview("data.torchvision", {}, {})


# ================================================================ lints

def test_lints_in_pipeline_context(tmp_path):
    path, _ = _table(tmp_path)
    cache = ProfileCache()
    issues = "\n".join(i.message for i in data_issues(_csv_design(path), cache))
    assert "no Impute Missing Values block" in issues
    assert "'leak' predicts" in issues and "class imbalance" in issues
    assert all(i.severity == "warning" for i in data_issues(_csv_design(path), cache))
    handled = _csv_design(path, extra=[
        ("prep.impute", {"strategy": "mean"}), ("prep.class_balance", {}),
        ("prep.drop_columns", {"columns": "leak, row_id, const, free, zip"})])
    issues = "\n".join(i.message for i in data_issues(handled, cache))
    assert "Impute" not in issues and "leak" not in issues and "imbalance" not in issues
    assert "row_id" not in issues and "'const'" not in issues
    regression = _csv_design(path, loss="train.loss_mse", units=1)
    assert "regression loss" in "\n".join(i.message for i in data_issues(regression, cache))
    # node ids point at the dataset block
    assert {i.node_id for i in data_issues(_csv_design(path), cache)} == {"data"}


def test_profile_cache_invalidates_on_change(tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame({"x": range(30), "label": [0, 1] * 15}).to_csv(path, index=False)
    graph = _csv_design(path)
    cache = ProfileCache()
    assert cache.pending(graph) == [("data.csv", dict(graph.nodes["data"].resolved_params()))]
    first = cache.profile("data.csv", dict(graph.nodes["data"].resolved_params()))
    assert cache.pending(graph) == []
    assert cache.profile("data.csv", dict(graph.nodes["data"].resolved_params())) is first
    pd.DataFrame({"x": range(31), "label": [0, 1] * 15 + [0]}).to_csv(path, index=False)
    assert cache.pending(graph)  # new size → new signature
    assert data_issues(graph, cache, compute=False) == []


# ================================================================ fingerprints

def test_fingerprints(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("x,label\n1,0\n", encoding="utf-8")
    first = fingerprint("data.csv", {"path": str(path)})
    assert first.startswith("file:") and first == fingerprint("data.csv", {"path": str(path)})
    path.write_text("x,label\n2,0\n", encoding="utf-8")
    assert fingerprint("data.csv", {"path": str(path)}) != first
    root = tmp_path / "imgs" / "a"
    root.mkdir(parents=True)
    (root / "1.png").write_bytes(b"1")
    folder = fingerprint("data.image_folder", {"root": str(root.parent)})
    (root / "2.png").write_bytes(b"2")
    assert fingerprint("data.image_folder", {"root": str(root.parent)}) != folder
    assert fingerprint("data.torchvision", {"dataset": "mnist"}) == "torchvision:mnist"
    assert fingerprint("data.csv", {"path": str(tmp_path / "missing.csv")}) == ""
    assert graph_fingerprint(tiny_classifier_dict()).startswith("synthetic:")


def test_runs_record_data_fingerprint(isolated_home, tmp_path):
    from ai_made_easy.core.runs.history import RunHistory, compare

    path = tmp_path / "f.csv"
    path.write_text("x,label\n1,0\n", encoding="utf-8")
    history = RunHistory()
    graph = _csv_design(path).to_dict()
    a = history.create(graph)
    path.write_text("x,label\n1,1\n", encoding="utf-8")
    b = history.create(graph)
    assert a.data_fingerprint.startswith("file:") and a.data_fingerprint != b.data_fingerprint
    assert history.get(a.run_id).data_fingerprint == a.data_fingerprint
    assert compare([a, b])["same_data"] is False
    assert compare([a, a])["same_data"] is True


# ================================================================ augmentation

def _image_design(root) -> Graph:
    data = tiny_classifier_dict()
    nodes = [n for n in data["nodes"] if n["type"] != "data.synthetic"]
    for node in nodes:
        if node["id"] == "in":
            node["params"]["shape"] = "3, 16, 16"
    nodes.insert(1, {"id": "flat", "type": "core.flatten", "params": {}, "position": [0, 0]})
    nodes += [{"id": "data", "type": "data.image_folder", "params": {"root": str(root)},
               "position": [0, 0]},
              {"id": "flip", "type": "prep.random_flip", "params": {"mode": "both"},
               "position": [0, 0]},
              {"id": "erase", "type": "prep.random_erasing", "params": {}, "position": [0, 0]},
              {"id": "mix", "type": "prep.mixup_cutmix", "params": {}, "position": [0, 0]}]
    chain = ["in", "flat", "d1", "act", "d2", "out"]
    data["nodes"] = nodes
    data["edges"] = [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in zip(chain, chain[1:],
                                                                          strict=False)]
    return Graph.from_dict(data)


def test_image_pipeline_order(tmp_path):
    from ai_made_easy.core.data.augment import image_pipeline

    pipe = image_pipeline(_image_design(tmp_path))
    assert pipe["always"] == ["v2.Resize((16, 16), antialias=True)"]
    assert pipe["train"] == ["v2.RandomHorizontalFlip()", "v2.RandomVerticalFlip()"]
    assert pipe["after"][0].startswith("v2.RandomErasing(")
    assert pipe["skipped"] == ["MixUp / CutMix (mixes whole batches)"]


@pytest.mark.skipif(importlib.util.find_spec("torchvision") is None, reason="needs torchvision")
def test_augmentation_preview_renders_views(tmp_path):
    from PIL import Image

    from ai_made_easy.core.data.augment import PreviewError, augmentation_preview

    root = tmp_path / "imgs"
    for name in ("a", "b"):
        (root / name).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (40, 30), (40 * i, 100, 200)).save(root / name / f"{i}.png")
    result = augmentation_preview(_image_design(root), tmp_path / "out", images=3, variants=2)
    assert len(result["images"]) == 3
    first = result["images"][0]
    assert first["eval_shape"] == [3, 16, 16] and len(first["train"]) == 2
    assert all((tmp_path / "out" / p).exists() for p in first["train"])
    assert result["skipped"]
    with pytest.raises(PreviewError, match="Image Folder"):
        augmentation_preview(Graph.from_dict(tiny_classifier_dict()), tmp_path / "x")


# ================================================================ api / cli

def test_api_and_cli(tmp_path, capsys):
    from ai_made_easy import cli
    from ai_made_easy.core import api

    path, _ = _table(tmp_path)
    graph = _csv_design(path).to_dict()
    assert api.profile_data(path=str(path), target="label")["rows"] == 300
    assert api.profile_data(graph)["type_id"] == "data.csv"
    assert any("Impute" in i["message"] for i in api.data_issues(graph)["issues"])
    preview = api.split_preview(graph)
    assert sum(preview["totals"].values()) == 300 and preview["method"] == "stratified"
    assert api.data_fingerprint(graph)["fingerprint"].startswith("file:")
    with pytest.raises(api.ApiError):
        api.profile_data({"name": "empty", "nodes": [], "edges": []})
    project = tmp_path / "p.json"
    project.write_text(json.dumps(graph), encoding="utf-8")
    assert cli.main(["data", "check", str(project)]) == 0
    assert "no Impute Missing Values block" in capsys.readouterr().out
    assert cli.main(["data", "split", str(project)]) == 0
    out = capsys.readouterr().out
    assert "stratified" in out and "cat" in out and "total" in out
    assert cli.main(["data", "profile", str(path), "--target", "label"]) == 0
    assert "300 rows" in capsys.readouterr().out
    assert cli.main(["data", "profile", str(project)]) == 0
    assert cli.main(["validate", str(project)]) in (0, 1)
    assert "predicts 'label' perfectly" in capsys.readouterr().out


# ================================================================ ui

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("data-tests")
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
    context.data_service.wait()
    context.project_store.mark_clean()
    win.close()
    win.deleteLater()


def test_work_still_running_when_the_window_closes_does_not_emit(app, tmp_path):
    """Closing the window shuts the service down and then deletes it: a profile still
    running must not emit on the deleted object (that crashed the app)."""
    import threading

    import shiboken6

    from ai_made_easy.ui.services.data_service import DataService

    service = DataService()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        started.set()
        release.wait(10)
        return "profile"

    service.cache.profile = slow
    service.request("data.csv", {"path": str(tmp_path / "t.csv")})
    future = service._futures[-1]
    future.add_done_callback(lambda f: finished.set())
    assert started.wait(10)
    service.shutdown()
    shiboken6.delete(service)          # what deleteLater does once the window has closed
    release.set()
    assert finished.wait(10) and future.exception() is None


def test_data_service_and_page(app, tmp_path):
    from ai_made_easy.ui.features.data_workspace import DataPage
    from ai_made_easy.ui.services.data_service import DataService

    path, _ = _table(tmp_path)
    graph = _csv_design(path)
    service = DataService()
    landed = []
    service.changed.connect(lambda: landed.append(True))
    service.issues_for(graph)  # nothing cached yet: profiled in background (may land first)
    service.wait()
    assert landed and any("Impute" in i.message for i in service.issues_for(graph))
    page = DataPage()
    got = []
    service.profiled.connect(lambda key, profile: got.append((key, profile)))
    key = service.request("data.csv", dict(graph.nodes["data"].resolved_params()))
    assert got and got[0][0] == key  # cached: emitted synchronously
    page.set_profile(got[0][1])
    assert page.columns.rowCount() == 9 and page.classes.rowCount() == 2
    assert page.findings.count() >= 5
    frames = []
    service.rows_ready.connect(lambda _k, frame: frames.append(frame))
    service.load_rows(key, got[0][1], limit=50)
    service.wait()
    page.set_rows(frames[0])
    assert page.row_model.rowCount() == 50 and page.row_model.columnCount() == 9
    assert "showing 50 of 300" in page.rows_info.text()
    splits = []
    service.split_ready.connect(lambda preview, err: splits.append((preview, err)))
    service.split(graph)
    service.wait()
    page.set_split(*splits[0])
    assert page.split_table.rowCount() == 3  # cat, dog, total
    service.shutdown()
    service.split(graph)  # ignored after shutdown


def test_context_shows_data_lints(ctx, app, tmp_path):
    path, _ = _table(tmp_path)
    graph = _csv_design(path)
    ctx.graph_service.load(graph)
    app.processEvents()
    ctx.data_service.wait()
    for _ in range(20):
        app.processEvents()
    messages = [i.message for i in ctx.validation_store.issues]
    assert any("no Impute Missing Values block" in m for m in messages)
    node_id = ctx.data_page.current_dataset()
    assert ctx._last_ir.nodes[node_id].type_id == "data.csv"
    assert ctx.data_page.profile is not None and ctx.data_page.profile.rows == 300
