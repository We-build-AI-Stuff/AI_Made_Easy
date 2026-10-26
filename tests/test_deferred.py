"""3.2 Phase 5 — the last deferred items: Glow, OGB datasets, Atari, custom environments,
and quantization-aware training for embedding models."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ai_made_easy.core.graph import Graph

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "ai_made_easy" / "samples"


def _sample(name: str) -> dict:
    data = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    return data.get("graph", data)


def _errors(graph: Graph) -> list[str]:
    return [i.message for i in graph.validate() if i.severity == "error"]


def _train(graph: Graph, tmp_path: Path, env: dict | None = None,
           timeout: float = 1800) -> dict:
    from ai_made_easy.core.codegen import export_training

    script = export_training(graph, "pytorch", tmp_path)
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=timeout,
                          env={**os.environ, "KERAS_BACKEND": "torch", **(env or {})})
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-4000:]
    return json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


# ============================================================================ Glow

def _chain(layers: list) -> tuple[list, list]:
    nodes, edges, prev = [], [], None
    for nid, type_id, params in layers:
        nodes.append({"id": nid, "type": type_id, "params": params, "position": [0, 0]})
        if prev:
            edges.append({"from": f"{prev}/out", "to": f"{nid}/in"})
        prev = nid
    return nodes, edges


def _image_flow(layers: list, size: int, n: int, epochs: int) -> Graph:
    nodes, edges = _chain([*layers, ("output", "core.output", {})])
    nodes += [
        {"id": "data", "type": "data.density_images", "position": [0, 0],
         "params": {"kind": "shapes", "size": str(size), "n_samples": n, "seed": 0,
                    "val_fraction": 0.1, "test_fraction": 0.1}},
        {"id": "opt", "type": "train.adam", "position": [0, 0], "params": {"lr": 1e-3}},
        {"id": "trainer", "type": "train.trainer", "position": [0, 0],
         "params": {"epochs": epochs, "batch_size": 64, "seed": 0,
                    "early_stopping_patience": 0}}]
    return Graph.from_dict({"name": "flow", "nodes": nodes, "edges": edges})


def glow(size: int = 8, hidden: int = 32, steps: int = 4, n: int = 2000,
         epochs: int = 8) -> Graph:
    layers = [("input", "core.input", {"shape": f"1, {size}, {size}"}),
              ("squeeze", "flow.squeeze", {})]
    for k in range(steps):
        layers += [(f"actnorm{k}", "flow.actnorm", {}),
                   (f"invconv{k}", "flow.invconv1x1", {"seed": k}),
                   (f"coupling{k}", "flow.conv_coupling",
                    {"hidden": hidden, "parity": "even" if k % 2 == 0 else "odd"})]
    return _image_flow(layers, size, n, epochs)


def realnvp(size: int = 8, hidden: int = 16, steps: int = 4, n: int = 2000,
            epochs: int = 8) -> Graph:
    layers = [("input", "core.input", {"shape": str(size * size)})]
    for k in range(steps):
        layers += [(f"actnorm{k}", "flow.actnorm", {}),
                   (f"coupling{k}", "flow.affine_coupling",
                    {"hidden": hidden, "layers": 2, "parity": "even" if k % 2 == 0 else "odd"})]
    return _image_flow(layers, size, n, epochs)


def test_glow_layers_invert_and_their_logdet_is_exact():
    torch = pytest.importorskip("torch")
    from ai_made_easy.core.codegen import generate

    graph = Graph.from_dict(_sample("glow_shapes"))
    assert not _errors(graph)
    assert graph.infer_shapes()["output"] == [8, 4, 4]       # squeezed, split, squeezed
    scope: dict = {}
    exec(generate(graph, "pytorch"), scope)                    # noqa: S102 — our own code
    model = scope["build_model"]()
    layers = [m for m in model.modules() if getattr(m, "is_flow", False)]
    for layer in layers:              # random, non-identity parameters
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.05)
    model.eval()
    x = torch.randn(3, 1, 16, 16)
    z = model(x)
    # without the split, the flow inverts exactly
    split = next(m for m in layers if type(m).__name__ == "FlowSplit")
    before = layers[:layers.index(split)]
    h = x
    for layer in before:
        h = layer(h)
    back = h
    for layer in reversed(before):
        back = layer.inverse(back)
    assert torch.allclose(back, x, atol=1e-4)
    # log-determinants match autograd's Jacobian on a tiny input
    small = [m for m in before]
    xs = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    for layer in small:
        layer.double()

    def run(v):  # noqa: ANN001, ANN202
        out = v.view(1, 1, 4, 4)
        for layer in small:
            out = layer(out)
        return out.reshape(-1)

    jac = torch.autograd.functional.jacobian(run, xs.reshape(-1))
    run(xs.reshape(-1))
    analytic = sum(float(m.logdet[0]) for m in small)
    assert analytic == pytest.approx(float(torch.slogdet(jac)[1]), abs=1e-6)
    assert z.shape == (3, 8, 4, 4)


def test_glow_rules():
    graph = Graph.from_dict(_sample("glow_shapes"))
    data = graph.nodes["data"]
    data.params["size"] = "8"
    assert any("1×8×8" in m for m in _errors(graph))
    odd = Graph.from_dict(_sample("glow_shapes"))
    odd.nodes["input"].params["shape"] = "1, 15, 15"
    assert any("even height and width" in m for m in _errors(odd))


def test_glow_beats_a_realnvp_of_similar_size(tmp_path):
    pytest.importorskip("torch")
    from ai_made_easy.core import budget

    g, r = glow(), realnvp()
    g_params, r_params = budget.estimate(g).params, budget.estimate(r).params
    assert 0.5 < r_params / g_params < 2.0, (g_params, r_params)
    (tmp_path / "glow").mkdir()
    (tmp_path / "realnvp").mkdir()
    glow_bpd = _train(g, tmp_path / "glow")["bits_per_dim"]
    realnvp_bpd = _train(r, tmp_path / "realnvp")["bits_per_dim"]
    assert glow_bpd < realnvp_bpd, (glow_bpd, realnvp_bpd)
    assert (tmp_path / "glow" / "eval_samples" / "density.png").exists()


def test_glow_recipe_and_serving(tmp_path):
    pytest.importorskip("torch")
    from ai_made_easy.core import recipes
    from ai_made_easy.core.recipes import DataFacts

    ranked = recipes.recommend("density_estimation", DataFacts(), None, modality="image")
    assert ranked[0].recipe.id == "glow" and ranked[0].ready
    graph = Graph.from_dict(_sample("glow_shapes"))
    graph.nodes["trainer"].params["epochs"] = 1
    graph.nodes["data"].params["n_samples"] = 300
    metrics = _train(graph, tmp_path)
    assert metrics["bits_per_dim"] < metrics["baseline_nll"] / 256 / 0.6931 + 0.5
    script = next(tmp_path.glob("*_train_pytorch.py"))
    probe = textwrap.dedent(f"""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("m", {str(script)!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.load_predictor({str(tmp_path)!r})
        out = m.infer([{{"sample": 2, "seed": 1}}, [0] * 256])
        print(json.dumps([len(out[0]["samples"]), out[1]["bits_per_dim"] > 0]))
    """)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path)
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [2, True], proc.stderr[-2000:]


# ============================================================================= OGB

FAKE_OGB = '''
import numpy as np, torch

class _Data:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.num_nodes = len(kw["x"])

def _graph(rng, n, f, classes=None):
    x = torch.tensor(rng.normal(0, 1, (n, f)), dtype=torch.float32)
    src = rng.integers(0, n, 3 * n); dst = rng.integers(0, n, 3 * n)
    ei = torch.tensor(np.stack([np.r_[src, dst], np.r_[dst, src]]), dtype=torch.long)
    return x, ei

class _Split(dict):
    pass

def _acc(d):
    return {"acc": float((np.asarray(d["y_true"]) == np.asarray(d["y_pred"])).mean())}

def _rocauc(d):
    yt, yp = np.asarray(d["y_true"]).ravel(), np.asarray(d["y_pred"]).ravel()
    pos, neg = yp[yt == 1], yp[yt == 0]
    return {"rocauc": float((pos[:, None] > neg[None, :]).mean()) if len(pos) and len(neg) else 0.5}

def _hits(d):
    pos, neg = np.asarray(d["y_pred_pos"]), np.asarray(d["y_pred_neg"])
    k = np.sort(neg)[-min(50, len(neg))]
    return {"hits@50": float((pos > k).mean())}
'''

FAKE_NODES = '''
from ogb._fake import _Data, _graph, _acc, np, torch
class PygNodePropPredDataset:
    def __init__(self, name, root=""):
        rng = np.random.default_rng(0)
        x, ei = _graph(rng, 300, 128)
        y = torch.tensor(rng.integers(0, 40, (300, 1)))
        x[:, :40] += 3 * torch.nn.functional.one_hot(y[:, 0], 40)
        self.data, self.num_classes = _Data(x=x, edge_index=ei, y=y), 40
    def __getitem__(self, i):
        return self.data
    def get_idx_split(self):
        idx = torch.arange(300)
        return {"train": idx[:200], "valid": idx[200:250], "test": idx[250:]}
class Evaluator:
    def __init__(self, name):
        self.name = name
    def eval(self, d):
        return _acc(d)
'''

FAKE_GRAPHS = '''
from ogb._fake import _Data, _graph, _rocauc, np, torch
class PygGraphPropPredDataset:
    def __init__(self, name, root=""):
        rng = np.random.default_rng(0)
        self.items = []
        for i in range(120):
            label = i % 2
            x, ei = _graph(rng, int(rng.integers(6, 12)), 9)
            x[:, 0] += 2.0 * label
            self.items.append(_Data(x=x, edge_index=ei, y=torch.tensor([[label]])))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]
    def get_idx_split(self):
        idx = torch.arange(120)
        return {"train": idx[:80], "valid": idx[80:100], "test": idx[100:]}
class Evaluator:
    def __init__(self, name):
        self.name = name
    def eval(self, d):
        return _rocauc(d)
'''

FAKE_LINKS = '''
from ogb._fake import _Data, _graph, _hits, np, torch
class PygLinkPropPredDataset:
    def __init__(self, name, root=""):
        rng = np.random.default_rng(0)
        x, ei = _graph(rng, 200, 128)
        self.data, self.rng = _Data(x=x, edge_index=ei), rng
    def __getitem__(self, i):
        return self.data
    def get_edge_split(self):
        e = self.data.edge_index.T[: self.data.edge_index.shape[1] // 2]
        neg = torch.tensor(self.rng.integers(0, 200, (60, 2)))
        return {"train": {"edge": e[:400]}, "valid": {"edge": e[400:500], "edge_neg": neg},
                "test": {"edge": e[500:600], "edge_neg": neg}}
class Evaluator:
    def __init__(self, name):
        self.name = name
    def eval(self, d):
        return _hits(d)
'''


@pytest.fixture()
def fake_ogb(tmp_path):
    pkg = tmp_path / "fakes" / "ogb"
    for sub, code in (("nodeproppred", FAKE_NODES), ("graphproppred", FAKE_GRAPHS),
                      ("linkproppred", FAKE_LINKS)):
        (pkg / sub).mkdir(parents=True)
        (pkg / sub / "__init__.py").write_text(code, encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "_fake.py").write_text(FAKE_OGB, encoding="utf-8")
    return {"PYTHONPATH": os.pathsep.join([str(pkg.parent), str(ROOT),
                                           os.environ.get("PYTHONPATH", "")])}


def _ogb_design(sample: str, block: str, name: str, features: int, classes: int) -> Graph:
    graph = Graph.from_dict(_sample(sample))
    data = next(n for n in graph.nodes.values() if n.type_id.startswith("data."))
    data.type_id = block
    data.params = {"name": name, "root": "/tmp/unused", "val_fraction": 0.1,
                   "test_fraction": 0.2}
    graph.nodes["input"].params["shape"] = str(features)
    if classes:
        head = graph.model_nodes()[-2]
        key = "units" if head.type_id == "core.dense" else "out_channels"
        head.params[key] = classes
    for node in graph.nodes.values():
        if node.type_id == "train.trainer":
            node.params["epochs"] = 3
    return graph


@pytest.mark.parametrize("sample,block,name,features,classes,metric", [
    ("graph_node_classification", "data.ogb_nodes", "ogbn-arxiv", 128, 40, "ogb_acc"),
    ("graph_classification_gin", "data.ogb_graphs", "ogbg-molhiv", 9, 2, "ogb_rocauc"),
    ("graph_link_prediction", "data.ogb_links", "ogbl-collab", 128, 0, "ogb_hits@50"),
], ids=["nodes", "graphs", "links"])
def test_ogb_datasets_use_the_official_split_and_evaluator(tmp_path, fake_ogb, sample, block,
                                                           name, features, classes, metric):
    pytest.importorskip("torch_geometric")
    graph = _ogb_design(sample, block, name, features, classes)
    assert not _errors(graph), _errors(graph)
    run = tmp_path / "run"
    run.mkdir()
    metrics = _train(graph, run, env=fake_ogb)
    assert 0.0 <= metrics[metric] <= 1.0, metrics
    log = next(run.glob("*_train_pytorch.py")).read_text(encoding="utf-8")
    assert "ogb" in log


def test_ogb_rules():
    graph = _ogb_design("graph_node_classification", "data.ogb_links", "ogbl-collab", 128, 4)
    assert any("no node labels" in m for m in _errors(graph))
    wrong = _ogb_design("graph_classification_gin", "data.ogb_graphs", "ogbg-molhiv", 9, 3)
    assert any("2 classes" in m for m in _errors(wrong))


@pytest.mark.skipif(not os.environ.get("AIME_ONLINE"), reason="downloads (nightly job only)")
def test_real_ogb_molbace(tmp_path):
    pytest.importorskip("ogb")
    graph = _ogb_design("graph_classification_gin", "data.ogb_graphs", "ogbg-molbace", 9, 2)
    graph.nodes["data"].params["root"] = str(Path.home() / ".cache" / "aime" / "ogb")
    metrics = _train(graph, tmp_path)
    assert metrics["ogb_rocauc"] > 0.5


# ======================================================================= RL: Atari

def _rl(sample: str) -> Graph:
    return Graph.from_dict(_sample(sample))


def _messages(graph: Graph) -> list[str]:
    return [i.message for i in graph.validate()]


def test_atari_rules_and_shapes():
    pytest.importorskip("gymnasium")
    graph = _rl("rl_atari_pong_ppo")
    assert not _errors(graph), _errors(graph)
    graph.nodes["preprocess"].params["screen_size"] = 64
    assert any("Set the Input shape to '4, 64, 64'" in m for m in _errors(graph))
    single = _rl("rl_atari_pong_ppo")
    single.nodes["stack"].params["frames"] = 1
    single.nodes["input"].params["shape"] = "1, 84, 84"
    assert any("one frame shows no motion" in m for m in _messages(single))
    raw = _sample("rl_atari_pong_ppo")
    raw["nodes"] = [n for n in raw["nodes"] if n["id"] not in ("preprocess", "stack")]
    bare = Graph.from_dict(raw)
    assert any("standard preprocessing" in m for m in _messages(bare))
    assert not _errors(bare)                         # the defaults give [4, 84, 84]


def test_atari_rom_and_package_rules(monkeypatch):
    pytest.importorskip("gymnasium")
    import importlib.util as util

    from ai_made_easy.core.rl import rules

    real = util.find_spec
    monkeypatch.setattr(rules.importlib.util, "find_spec",
                        lambda name, *a: None if name == "ale_py" else real(name, *a))
    assert any("pip install 'ai-made-easy[atari]'" in m
               for m in _messages(_rl("rl_atari_pong_ppo")))
    # ale-py there (whether or not this machine has it), but without the game's ROM
    monkeypatch.setattr(rules.importlib.util, "find_spec",
                        lambda name, *a: object() if name == "ale_py" else real(name, *a))
    monkeypatch.setattr(rules, "_rom_missing", lambda rom: True)
    assert any("the pong ROM is missing" in m and "never ships ROMs" in m
               for m in _errors(_rl("rl_atari_pong_ppo")))


def test_atari_environment_in_the_generated_script(tmp_path):
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")
    pytest.importorskip("stable_baselines3")
    from ai_made_easy.core.codegen import export_training

    script = export_training(_rl("rl_atari_pong_ppo"), "pytorch", tmp_path)
    probe = textwrap.dedent(f"""
        import importlib.util, json
        spec = importlib.util.spec_from_file_location("m", {str(script)!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        env = m.make_env()
        obs, _ = env.reset(seed=0)
        print(json.dumps([list(obs.shape), str(obs.dtype), int(env.action_space.n)]))
    """)
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          timeout=300)
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [[4, 84, 84], "uint8", 6], \
        proc.stderr[-2000:]


# ============================================================ RL: custom environments

CORRIDOR = '''
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Corridor(gym.Env):
    """Walk right to the end of a corridor: position, distance left, steps used."""

    def __init__(self, length: int = 6):
        self.length = length
        self.observation_space = spaces.Box(0.0, 1.0, (3,), np.float32)
        self.action_space = spaces.Discrete(2)

    def _obs(self):
        return np.array([self.pos / self.length, 1 - self.pos / self.length,
                         self.steps / 20], np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.steps = 0, 0
        return self._obs(), {}

    def step(self, action):
        self.pos = max(0, self.pos + (1 if int(action) == 1 else -1))
        self.steps += 1
        done = self.pos >= self.length
        return self._obs(), 1.0 if done else -0.05, done, self.steps >= 20, {}


gym.register("Corridor-v0", entry_point=Corridor)
'''


def test_custom_environment_spaces_are_read_at_design_time(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("stable_baselines3")
    module = tmp_path / "corridor_env.py"
    module.write_text(CORRIDOR, encoding="utf-8")
    graph = _rl("rl_cartpole_ppo")
    env = next(n for n in graph.nodes.values() if n.type_id == "rl.env")
    env.params.update(env_id="custom", custom_id="Corridor-v0", module=str(module))
    assert any("Corridor-v0 observations have 3 values: Set the Input shape to '3'" in m
               for m in _errors(graph))
    graph.nodes["input"].params["shape"] = "3"
    assert not _errors(graph), _errors(graph)
    algo = next(n for n in graph.nodes.values() if n.type_id.startswith("rl."))
    sac = Graph.from_dict(graph.to_dict())
    node = next(n for n in sac.nodes.values() if n.type_id == "rl.ppo")
    node.type_id = "rl.sac"
    node.params = {}
    assert any("Corridor-v0 has 2 discrete actions" in m for m in _errors(sac))
    broken = Graph.from_dict(graph.to_dict())
    next(n for n in broken.nodes.values() if n.type_id == "rl.env").params["custom_id"] = \
        "Nowhere-v0"
    assert any("could not create Nowhere-v0" in m for m in _errors(broken))
    for node in graph.nodes.values():
        if node.type_id == "rl.ppo":
            node.params.update(total_timesteps=1024, n_steps=128, batch_size=64,
                               evaluations=2, eval_episodes=2)
    assert algo is not None
    (tmp_path / "run").mkdir()
    metrics = _train(graph, tmp_path / "run")
    assert "mean_reward" in metrics


# =============================================================== QAT for embeddings

def test_qat_text_classifier_stays_within_a_point(tmp_path, isolated_home):
    pytest.importorskip("torch")
    from ai_made_easy.core import recipes
    from ai_made_easy.core.pipelines.runner import PipelineRunner, PipelineStore
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    import random

    rng = random.Random(0)
    topics = {"sport": "goal match team score league coach", "food": "pasta bread cheese "
              "spice bake recipe", "space": "planet orbit rocket star galaxy launch"}
    filler = "the a of and to in it was very really quite".split()
    rows = ["text,label"]
    for i in range(900):
        label = list(topics)[i % 3]
        words = rng.sample(topics[label].split(), 3) + rng.sample(filler, 5)
        rng.shuffle(words)
        rows.append(f"{' '.join(words)},{label}")
    table = tmp_path / "texts.csv"
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    facts = recipes.detect(table, target="label")
    assert facts.kind != "demo", facts
    built = recipes.build("text_bag_of_embeddings", "multiclass", facts, {"epochs": 8}, None)
    design = built.to_dict() if hasattr(built, "to_dict") else built
    head = next(n for n in design["nodes"] if n["type"] == "core.input")
    assert str(head["params"].get("dtype", "")).startswith("int")       # token ids
    stage = lambda sid, t, **p: {"id": sid, "type": t, "params": p, "position": [0, 0]}  # noqa: E731
    data = {"name": "qat_text", "meta": {"designs": {"text": design}},
            "nodes": [stage("t", "pipeline.train", design="embedded:text"),
                      stage("q", "pipeline.qat", epochs=2)],
            "edges": [{"from": "t/out", "to": "q/in"}]}
    graph = Graph.from_dict(data)
    issues = graph.validate()
    assert not [i for i in issues if i.severity == "error"], issues
    assert any("float scales" in i.message for i in issues)
    runner = PipelineRunner(RunManager(RunHistory(tmp_path / "runs")),
                            PipelineStore(tmp_path / "pipelines"), graph)
    runner.start()
    record = runner.wait(1800)
    assert record.state == "finished", {k: v["message"][-800:] for k, v in record.stages.items()}
    q = record.stages["q"]["metrics"]
    assert q["accuracy"] >= q["accuracy_float"] - 0.01, q
    assert q["size_mb"] < q["size_mb_float"]
