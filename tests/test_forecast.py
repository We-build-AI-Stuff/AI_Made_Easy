"""Time-series forecasting: models and heads, windows, metrics, rules, profiles, training."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import api, budget
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.tasks import task_of

torch = pytest.importorskip("torch")

MODELS = ["forecast.dlinear", "forecast.nbeats", "forecast.nhits", "forecast.tcn",
          "forecast.patchtst", "forecast.tide", "forecast.rnn", "forecast.transformer",
          "forecast.nlinear", "forecast.itransformer", "forecast.tsmixer", "forecast.timesnet"]
SMALL = {"forecast.nbeats": {"width": 32, "layers": 2}, "forecast.nhits": {"width": 32},
         "forecast.tcn": {"channels": 8, "levels": 5}, "forecast.patchtst": {"d_model": 16},
         "forecast.tide": {"hidden": 16}, "forecast.rnn": {"hidden": 8},
         "forecast.transformer": {"d_model": 16}, "forecast.itransformer": {"d_model": 16},
         "forecast.tsmixer": {"hidden": 8}, "forecast.timesnet": {"d_model": 8, "hidden": 8}}


@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.forecast.runtime import namespace

    return namespace()


def design(model: str = "forecast.dlinear", params: dict | None = None, *, window: int = 48,
           horizon: int = 12, channels: int = 2, data: dict | None = None, extra: tuple = (),
           epochs: int = 1, name: str = "forecast_case") -> dict:
    data = {"n_series": 6, "length": 160, "season_length": 24, "window": window,
            "horizon": horizon, "stride": 4, **(data or {})}
    return {"name": name, "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": f"{window}, {channels}"},
         "position": [0, 0]},
        {"id": "m", "type": model, "params": {"horizon": horizon, **(params or {})},
         "position": [200, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [400, 0]},
        {"id": "data", "type": "data.synthetic_series", "params": data, "position": [0, 200]},
        {"id": "tr", "type": "train.trainer",
         "params": {"epochs": epochs, "batch_size": 32, "device": "cpu"}, "position": [0, 300]},
        *extra],
        "edges": [{"from": "in/out", "to": "m/in"}, {"from": "m/out", "to": "out/in"}]}


def issues(data: dict) -> list:
    return [i for i in Graph.from_dict(data).validate() if "optimizer" not in i.message]


# ================================================================ registry / task

def test_forecasting_blocks_and_task_are_registered():
    reg = get_registry()
    for type_id in (*MODELS, "data.forecast_csv", "data.synthetic_series", "eval.mase",
                    "eval.crps", "seq.tcn", "seq.mamba", "seq.s4d", "seq.xlstm",
                    "audio.mel_spectrogram", "audio.mfcc"):
        assert reg.get(type_id)
    graph = Graph.from_dict(design())
    task = task_of(graph)
    assert task.id == "forecasting" and task.trainer_kind == "forecasting"


def test_sample_validates_and_resolves():
    graph = Graph.from_dict(api.read_sample("demand_forecasting.json"))
    assert [i for i in graph.validate() if i.severity == "error"] == []
    assert task_of(graph).id == "forecasting"


# ================================================================ models

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("head", ["point", "quantile", "student_t", "negbin"])
def test_model_shapes_and_costs_match_torch(model, head):
    from ai_made_easy.core.codegen import class_name_for, generate

    future = 1 if model in ("forecast.tcn", "forecast.tide", "forecast.rnn") else 0
    graph = Graph.from_dict(design(model, {**SMALL.get(model, {}), "head": head,
                                           "future_covariates": future}))
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(generate(graph, "pytorch"), "<forecast>", "exec"), ns)  # noqa: S102
    net = ns[class_name_for(graph.name)]().eval()
    expected = graph.infer_shapes()["m"]
    with torch.no_grad():
        out = net(torch.randn(3, 48, 2))
        assert list(out.shape[1:]) == list(expected)
        inner = next(m for m in net.modules() if hasattr(m, "forecast"))
        fut = torch.randn(3, 12, 1) if future else None
        assert list(inner.forecast(torch.randn(3, 48, 2), fut).shape[1:]) == list(expected)
    assert budget.estimate(graph).params == sum(p.numel() for p in net.parameters())
    if head == "quantile":   # non-crossing quantiles
        assert torch.all(out[..., 1:] >= out[..., :-1])


def test_bad_quantiles_and_patch_length_are_shape_errors():
    msgs = [i.message for i in issues(design(params={"head": "quantile",
                                                     "quantiles": "0.1, 1.5"}))]
    assert any("between 0 and 1" in m for m in msgs)
    msgs = [i.message for i in issues(design("forecast.patchtst", {"patch_len": 64}))]
    assert any("patch_len 64 is longer" in m for m in msgs)


# ================================================================ data / metrics

def test_windows_split_chronologically(rt):
    series = rt["synthetic_series"](2, 100, 12, 0.0, 0.1, False, True, 0)
    train = rt["make_windows"](series, 24, 6, 2, 2, 1, "train", 12)
    val = rt["make_windows"](series, 24, 6, 2, 2, 1, "val", 12)
    test = rt["make_windows"](series, 24, 6, 2, 2, 1, "test", 12)
    assert train["x"].shape[1:] == (24, 2) and train["future"].shape[1:] == (6, 1)
    # no training target reaches into validation; val / test step by the horizon
    assert (train["origin"] + 6).max() <= 100 - 4 * 6
    assert sorted(set(val["origin"])) == [76, 82] and sorted(set(test["origin"])) == [88, 94]
    s = series[0]
    i = int(np.flatnonzero(test["series"] == 0)[0])
    np.testing.assert_allclose(test["y"][i], s["target"][88:94])
    np.testing.assert_allclose(test["x"][i, :, 0], s["target"][64:88])


def test_metrics_hand_cases(rt):
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = rt["point_metrics"](y, y + 1, np.array([2.0, 0.5]))
    assert m["mae"] == 1 and m["rmse"] == 1 and m["wape"] == pytest.approx(0.4)
    assert m["mase"] == pytest.approx((0.5 + 2) / 2)
    q = np.stack([y - 1, y, y + 1], -1)
    assert rt["pinball"](y, q, (0.1, 0.5, 0.9)) == pytest.approx((0.1 + 0 + 0.1) / 3)
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((4000, 1, 1))
    # CRPS of N(0, 1) at y = 0 is (sqrt(2) - 1) / sqrt(pi)
    assert rt["crps_from_samples"](np.zeros((1, 1)), samples) == pytest.approx(0.2337, abs=0.01)
    assert rt["coverage"](y, y - 1, y) == 1.0


def test_seasonal_scale_and_season_detection(rt):
    from ai_made_easy.core.forecast.profile import season_of

    t = np.arange(240)
    y = 10 + np.sin(2 * np.pi * t / 24)
    assert season_of(y) == 24
    assert rt["seasonal_scale"](y, 24) == pytest.approx(0, abs=1e-6) or \
        rt["seasonal_scale"](y, 24) == 1.0
    assert season_of(np.random.default_rng(0).standard_normal(300)) == 1


def test_read_forecast_table(rt, tmp_path):
    pytest.importorskip("pandas")
    path = tmp_path / "s.csv"
    path.write_text("date,store,sales,promo\n2024-01-02,b,3,1\n2024-01-01,b,2,0\n"
                    "2024-01-01,a,5,0\n", encoding="utf-8")
    out = rt["read_forecast_table"](path, "csv", "date", "sales", "store", "", "promo")
    assert [s["id"] for s in out] == ["a", "b"]
    assert out[1]["target"].tolist() == [2, 3] and out[1]["future"][:, 0].tolist() == [0, 1]
    with pytest.raises(ValueError, match="not found"):
        rt["read_forecast_table"](path, "csv", "date", "revenue")


# ================================================================ rules

def test_input_shape_rule_and_fix():
    found = issues(design(channels=3))
    issue = next(i for i in found if "Set the Input shape" in i.message)
    fixed = fix_for_issue(Graph.from_dict(design(channels=3)), issue)
    assert fixed and fixed[2].nodes["in"].params["shape"] == "48, 2"


@pytest.mark.parametrize("params, needle, fixed_param", [
    ({"horizon": 6}, "set horizon to 12", ("horizon", 12)),
])
def test_horizon_rule_and_fix(params, needle, fixed_param):
    data = design("forecast.nhits", params, horizon=12)
    data["nodes"][1]["params"].update(params)
    issue = next(i for i in issues(data) if needle in i.message)
    fixed = fix_for_issue(Graph.from_dict(data), issue)
    assert fixed[2].nodes["m"].params[fixed_param[0]] == fixed_param[1]


def test_covariate_rules():
    msgs = [i.message for i in issues(design("forecast.tide"))]
    assert any("set future_covariates to 1" in m for m in msgs)
    info = [i for i in issues(design()) if "only sees the covariates' past" in i.message]
    assert info and info[0].severity == "info"
    assert not any("covariate" in i.message for i in issues(design(
        channels=1, data={"promotions": False})))


def test_tcn_receptive_field_rule():
    msgs = [i.message for i in issues(design("forecast.tcn", {"levels": 2,
                                                              "future_covariates": 1}))]
    assert any("receptive field" in m and "set levels to 4" in m for m in msgs)


def test_head_rules():
    msgs = [i.message for i in issues(design(params={"head": "negbin"}))]
    assert any("continuous" in m for m in msgs)
    msgs = [i.message for i in issues(design(data={"counts": True}))]
    assert any("negbin head" in m for m in msgs)
    loss = ({"id": "l", "type": "train.loss_mse", "params": {}, "position": [0, 0]},)
    msgs = [i.message for i in issues(design(params={"head": "quantile"}, extra=loss))]
    assert any("is ignored" in m for m in msgs)


def test_generic_layers_must_output_the_horizon():
    data = design()
    data["nodes"][1] = {"id": "m", "type": "seq.tcn", "params": {"levels": 5},
                        "position": [0, 0]}
    msgs = [i.message for i in issues(data)]
    assert any("one value per future step" in m for m in msgs)


def test_window_checks_on_the_dataset():
    msgs = [i.message for i in issues(design(window=8, horizon=12, data={"window": 8}))]
    assert any("shorter than the horizon" in m for m in msgs)


# ================================================================ profile

def test_profile_forecast_datasets(tmp_path):
    pytest.importorskip("pandas")
    from ai_made_easy.core.data.profile import profile_dataset

    params = {p.name: p.default for p in get_registry().get("data.synthetic_series").params}
    p = profile_dataset("data.synthetic_series", params)
    assert p.task == "forecasting" and "30 series" in p.summary
    assert any("Input [48, 2]" in d for d in p.details)
    rows = ["date,store,sales"]
    for store in "ab":
        for day in range(1, 29):
            if store == "b" and day == 5:
                continue
            rows.append(f"2024-02-{day:02d},{store},{3 + 4 * (day % 7 == 0)}")
    (tmp_path / "s.csv").write_text("\n".join(rows), encoding="utf-8")
    csv = {p.name: p.default for p in get_registry().get("data.forecast_csv").params}
    csv.update(path="s.csv", target_column="sales", id_column="store", window=7, horizon=3,
               season_length=1)
    p = profile_dataset("data.forecast_csv", csv, tmp_path)
    found = " ".join(f.message for f in p.findings)
    assert "non-negative counts" in found and "break the regular spacing" in found
    assert "every 7 steps" in found
    csv["window"] = 40
    assert "shorter than" in " ".join(f.message for f in
                                       profile_dataset("data.forecast_csv", csv,
                                                       tmp_path).findings)
    assert "does not exist" in profile_dataset("data.forecast_csv", {**csv, "path": "x.csv"},
                                               tmp_path).error


# ================================================================ training / serving

def _train(data: dict, tmp_path: Path) -> tuple[Path, str]:
    from ai_made_easy.core.training.generate import generate_training

    script = tmp_path / "train.py"
    script.write_text(generate_training(Graph.from_dict(data), "pytorch"), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert "resources: peak_memory_mb" in proc.stdout
    return tmp_path, proc.stdout


def _infer(folder: Path, items: list) -> list:
    code = ("import sys, json; sys.path.insert(0, '.'); import train; "
            "train.load_predictor('.'); "
            f"print('OUT ' + json.dumps(train.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@pytest.mark.parametrize("model, params, counts", [
    ("forecast.dlinear", {"head": "point"}, False),
    ("forecast.tide", {"head": "quantile", "future_covariates": 1, "hidden": 16}, False),
    ("forecast.rnn", {"head": "negbin", "future_covariates": 1, "hidden": 8}, True),
    ("forecast.nhits", {"head": "student_t", "width": 32}, False),
], ids=["point", "quantile", "negbin", "student_t"])
def test_heads_train_and_serve(model, params, counts, tmp_path):
    run, stdout = _train(design(model, params, data={"counts": counts}), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert {"mae", "mase", "smape", "wape", "crps", "naive_mase", "mae_by_step"} <= set(metrics)
    assert len(metrics["mae_by_step"]) == 12
    if params["head"] != "point":
        assert "coverage_80" in metrics
    assert list((run / "eval_samples").glob("*.png"))
    history = [float(v) for v in np.linspace(10, 30, 60)]
    out = _infer(run, [{"history": history, "future_covariates": [[0.0]] * 72}, history])
    assert len(out) == 2 and len(out[0]["forecast"]) == 12
    if params["head"] == "quantile":
        assert set(out[0]["quantiles"]) == {"0.1", "0.5", "0.9"}
    elif params["head"] != "point":
        assert all(lo <= hi for lo, hi in zip(out[0]["lower"], out[0]["upper"], strict=True))
    if counts:
        assert min(out[0]["forecast"]) >= 0


def test_generic_layers_train(tmp_path):
    data = design(channels=1, data={"promotions": False})
    data["nodes"][1] = {"id": "m", "type": "seq.xlstm", "params": {"hidden": 8},
                        "position": [0, 0]}
    data["nodes"] += [{"id": "f", "type": "core.flatten", "params": {}, "position": [0, 0]},
                      {"id": "lin", "type": "core.dense", "params": {"units": 12},
                       "position": [0, 0]}]
    data["edges"] = [{"from": "in/out", "to": "m/in"}, {"from": "m/out", "to": "f/in"},
                     {"from": "f/out", "to": "lin/in"}, {"from": "lin/out", "to": "out/in"}]
    assert not [i for i in issues(data) if i.severity == "error"]
    run, _ = _train(data, tmp_path)
    assert "mase" in json.loads((run / "metrics.json").read_text(encoding="utf-8"))


def test_forecast_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(design(name="fc_run", params={"head": "quantile"})))
        status = mgr.wait(run_id, 600)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        record = mgr.history.get(run_id)
        assert "mase" in record.final_metrics
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text(encoding="utf-8"))
        assert meta["task"] == "forecasting" and meta["input_kind"] == "series"
        assert api.run_samples(run_id)["samples"]
    finally:
        api.set_manager(None)


def test_timesnet_predicts_with_the_periods_it_learned(rt):
    """In training TimesNet picks each batch's periods and keeps a running spectrum; in
    prediction it uses the spectrum's periods, measured with cosine / sine bases: a
    forecast no longer depends on the rest of the batch, and matches the FFT."""
    import math

    from ai_made_easy.core.forecast.helpers import FORECAST_HELPERS

    ns = {"torch": torch, "nn": torch.nn, "math": math}
    for name in ("ForecastHead", "TimesNet"):
        exec(FORECAST_HELPERS[name], ns)  # noqa: S102
    torch.manual_seed(0)
    net = ns["TimesNet"](48, 2, 12, d_model=8, hidden=8, top_k=2)
    t = torch.arange(48.0)
    series = torch.stack([torch.sin(2 * math.pi * t / 12), torch.zeros(48)], -1)
    batch = series + 0.05 * torch.randn(16, 48, 2)
    net.train()
    for _ in range(5):
        net(batch)
    assert net.spectrum.abs().sum() > 0
    net.eval()
    with torch.no_grad():
        alone = net(batch[:1])
        mixed = net(torch.cat([batch[:1], torch.randn(7, 48, 2) * 5]))
        assert torch.allclose(alone, mixed[:1], atol=1e-5)
        z = net.embed(batch) + net.position
        periods, weight = net._periods(z, 0)
        freqs = net.spectrum[0, 1:].topk(2).indices + 1
        fft = torch.fft.rfft(z, dim=1).abs().mean(-1)[:, freqs]
        assert torch.allclose(weight, fft, atol=1e-3)
        assert periods[0] == 48 // int(freqs[0])


@pytest.mark.slow
def test_timesnet_exports_to_executorch_float_and_int8(tmp_path):
    """Eval mode fixes TimesNet's periods as plain ints and gives each period its own copy
    of the convolutions: export traces constants (float ExecuTorch failed on a
    data-dependent shape) and the int8 XNNPACK program loads (a convolution used at two
    call sites gave one that didn't)."""
    pytest.importorskip("executorch")
    from test_export import _build

    from ai_made_easy.core.deploy import exporters as ex
    from ai_made_easy.core.deploy.exportability import blockers

    graph = Graph.from_dict(design("forecast.timesnet", {"d_model": 8, "hidden": 8}))
    assert set(blockers(graph)) == {"gguf"}          # GGUF holds Llama-style decoders only
    model, samples = _build(graph)
    report = ex.export_all(model, "pytorch", ["executorch", "executorch_int8"], tmp_path,
                           samples, log=lambda _m: None)
    assert report["executorch"]["ok"], report["executorch"].get("error")
    assert report["executorch_int8"]["ok"], report["executorch_int8"].get("error")
