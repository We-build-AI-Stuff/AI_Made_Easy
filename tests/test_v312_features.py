"""3.12 features. Phase 1: GGUF for more language models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_made_easy.core.graph import Graph

SAMPLES = Path(__file__).resolve().parents[1] / "ai_made_easy" / "samples"


def _sample(name: str, **gpt) -> dict:
    design = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    for node in design["nodes"]:
        if node["type"] == "gen.gpt":
            node["params"].update(gpt)
    return design


# ------------------------------------------------------------ Phase 1: GGUF GPTs
def _gpt(vocab: int = 256):  # noqa: ANN202 — a torch module
    """A Tiny GPT as the generated script builds it, for any vocabulary."""
    import math

    import torch
    from torch import nn

    from ai_made_easy.core.generative import helpers

    ns = {"torch": torch, "nn": nn, "math": math}
    exec(helpers.CAUSAL + helpers.GPT, ns)  # noqa: S102 — the generated code's own classes

    class TinyGpt(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt = ns["GPT"](vocab, 64, 128, 3, 4, 0.1, True)

        def forward(self, x):  # noqa: ANN001, ANN202
            return self.gpt(x)

    torch.manual_seed(0)
    return TinyGpt().eval()


def test_gpts_export_to_gguf_and_other_models_say_why_not():
    from ai_made_easy.core.deploy.exportability import blockers

    assert "gguf" not in blockers(Graph.from_dict(_sample("tiny_gpt")))
    assert "gguf" not in blockers(Graph.from_dict(_sample("tiny_llama")))
    assert "gguf" not in blockers(Graph.from_dict(_sample("tiny_rwkv")))
    from conftest import tiny_classifier_dict

    reason, _node = blockers(Graph.from_dict(tiny_classifier_dict()))["gguf"]
    assert "GPTs (one GPT block" in reason and "RWKV-7" in reason


@pytest.mark.parametrize("bpe", [False, True], ids=["bytes", "bpe"])
def test_a_gpt_in_gguf_matches_pytorch_and_the_runs_tokens(tmp_path, bpe):
    """llama.cpp's gpt2 logits are PyTorch's within its float16 GELU, token by token, and
    it splits text into the run's tokens."""
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    llama_cpp = pytest.importorskip("llama_cpp")
    from test_v311_features import _text_tools

    from ai_made_easy.core.deploy import exporters

    tools = _text_tools()
    text = tools["synthetic_text"](400, 0)
    merges = tools["train_bpe"](text, 300) if bpe else None
    vocab = 256 + len(merges or [])
    model = _gpt(vocab)
    samples = exporters.random_samples("tensor", (64,), integer=True, vocab=vocab)
    report = exporters.export_all(model, "pytorch", ["gguf"], tmp_path, samples,
                                  merges=merges)["gguf"]
    assert report["ok"], report
    assert report["note"].startswith("matches PyTorch")
    assert "llama.cpp's GELU rounds to float16" in report["note"]
    assert ("BPE merges" in report["note"]) == bpe
    llm = llama_cpp.Llama(model_path=str(tmp_path / "model.gguf"), vocab_only=True,
                          verbose=False)
    sample = "The robot's map isn't here: 1234 apples, naïve café!\n"
    expected = tools["BPE"](merges).encode(sample) if bpe else tools["encode_bytes"](sample)
    assert llm.tokenize(sample.encode(), add_bos=False, special=False) == list(expected)


def test_gguf_reads_a_gpts_layout():
    pytest.importorskip("torch")
    import torch
    from test_export import _build

    from ai_made_easy.core.deploy import exporters

    model, _ = _build(Graph.from_dict(_sample("tiny_gpt")))
    layout = exporters.gpt_layout(model)
    assert (layout["vocab"], layout["dim"], len(layout["blocks"]), layout["heads"],
            layout["hidden"], layout["context"]) == (256, 128, 3, 4, 512, 64)
    assert exporters.gpt_layout(torch.nn.Sequential(torch.nn.Linear(4, 2))) is None


@pytest.mark.slow
def test_a_trained_gpt_with_bpe_exports_to_gguf_from_its_package(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    design = _sample("tiny_gpt", vocab_size=300, layers=2)
    for node in design["nodes"]:
        if node["type"] == "train.trainer":
            node["params"].update(epochs=2, device="cpu")
        if node["type"] == "data.synthetic_text":
            node["params"]["n_sentences"] = 600
    design["nodes"].append({"id": "tok", "type": "gen.bpe_tokenizer", "position": [0, 0],
                            "params": {"vocab_size": 300}})
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    run_id = mgr.start(Graph.from_dict(design))
    assert mgr.wait(run_id, 600)["state"] == "finished", mgr.history.get(run_id).error[-2000:]
    result = build_package(mgr.history.path(run_id), tmp_path / "pkg", formats=("gguf",))
    entry = result.formats["gguf"]
    assert entry["ok"], entry
    assert "BPE merges" in entry["note"] and entry["note"].startswith("matches PyTorch")
    assert (tmp_path / "pkg" / entry["file"]).is_file()


def test_a_language_models_layer_norm_must_not_see_later_tokens():
    """A LayerNorm over every position mixes later tokens into earlier ones' outputs:
    Tiny RWKV's did (its file predates per-token LayerNorm being the default)."""
    from ai_made_easy.core.fixes import fix_for_issue

    design = _sample("tiny_rwkv")
    assert not [i for i in Graph.from_dict(design).validate() if i.severity == "error"]
    for node in design["nodes"]:
        if node["type"] == "core.layer_norm":
            node["params"]["normalized_dims"] = "all"
    graph = Graph.from_dict(design)
    [issue] = [i for i in graph.validate() if "normalizes over every position" in i.message]
    assert issue.severity == "error" and issue.node_id == "norm"
    _label, _desc, fixed = fix_for_issue(graph, issue)
    assert fixed.nodes["norm"].resolved_params()["normalized_dims"] == "last"


# ------------------------------------------------------------ Phase 1: Hugging Face → GGUF
_TEXT = ("The robot's map isn't here: 1234 apples, naïve café! Hello world.\n\n"
         "Two  spaces, tabs\tand émojis 🙂 too. ") * 40


def _bpe_tokenizer(folder, regex: str) -> int:  # noqa: ANN001
    """A small byte-level BPE tokenizer (Llama 3's or Qwen2's split), saved to ``folder``."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([_TEXT], trainers.BpeTrainer(
        vocab_size=320, special_tokens=["<|begin|>", "<|end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin|>",
                                   eos_token="<|end|>")
    fast.save_pretrained(folder)
    return len(fast)


def _spm_tokenizer(folder, cls: str) -> int:  # noqa: ANN001
    """A small SentencePiece tokenizer (byte fallback), saved as ``cls`` to ``folder``."""
    import sentencepiece as spm
    import transformers

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "corpus.txt").write_text(_TEXT, encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(folder / "corpus.txt"), model_prefix=str(folder / "sp"), vocab_size=400,
        model_type="bpe", byte_fallback=True, character_coverage=1.0,
        bos_id=1, eos_id=2, unk_id=0, pad_id=-1, minloglevel=2)
    (folder / "sp.model").replace(folder / "tokenizer.model")   # read as transformers 5 does
    tok = getattr(transformers, cls).from_pretrained(folder)
    tok.save_pretrained(folder)
    return len(tok)


def _tiny_hf(family: str, folder):  # noqa: ANN001
    """A tiny random Hugging Face model of ``family`` with its kind of tokenizer."""
    import torch
    import transformers

    from ai_made_easy.core.deploy import hf_gguf

    shared = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "max_position_embeddings": 256, "tie_word_embeddings": False}
    if family == "llama":
        vocab = _bpe_tokenizer(folder, next(r for r, n in hf_gguf._PRE.items()
                                            if n == "llama-bpe"))
        config = transformers.LlamaConfig(vocab_size=vocab, rms_norm_eps=1e-5, **shared)
        model = transformers.LlamaForCausalLM(config)
    elif family == "qwen2":
        vocab = _bpe_tokenizer(folder, next(r for r, n in hf_gguf._PRE.items() if n == "qwen2"))
        config = transformers.Qwen2Config(vocab_size=vocab, **shared)
        model = transformers.Qwen2ForCausalLM(config)
    elif family == "phi3":
        vocab = _spm_tokenizer(folder, "LlamaTokenizer")
        config = transformers.Phi3Config(vocab_size=vocab, pad_token_id=0, **shared)
        model = transformers.Phi3ForCausalLM(config)
    else:
        vocab = _spm_tokenizer(folder, "GemmaTokenizer")
        config = transformers.Gemma3TextConfig(vocab_size=vocab, head_dim=16,
                                               query_pre_attn_scalar=16, sliding_window=64,
                                               **shared)
        model = transformers.Gemma3ForCausalLM(config)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    model.save_pretrained(folder, safe_serialization=True)
    return model


FAMILIES = ["llama", "qwen2", "phi3", "gemma3"]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_hugging_face_model_in_gguf_matches_transformers(tmp_path, family):
    """llama.cpp splits text into the tokenizer's tokens and scores the next token as
    transformers does, for each family llama.cpp reads."""
    for module in ("torch", "transformers", "gguf", "llama_cpp", "sentencepiece", "tokenizers"):
        pytest.importorskip(module)
    from ai_made_easy.core.deploy import hf_gguf

    _tiny_hf(family, tmp_path / "model")
    report = hf_gguf.export(tmp_path / "model", tmp_path / "model.gguf")
    assert report["ok"] and report["family"] == family and report["checked"], report
    assert report["worst"] <= hf_gguf.TOLERANCE
    assert ("SentencePiece" in report["note"]) == (family in ("phi3", "gemma3"))


def test_a_sentencepiece_tokenizer_saved_as_tokenizer_json_exports(tmp_path):
    """transformers 5 saves SentencePiece tokenizers as tokenizer.json alone."""
    for module in ("torch", "transformers", "gguf", "llama_cpp", "sentencepiece"):
        pytest.importorskip(module)
    from ai_made_easy.core.deploy import hf_gguf

    _tiny_hf("gemma3", tmp_path / "model")
    (tmp_path / "model" / "tokenizer.model").unlink()
    report = hf_gguf.export(tmp_path / "model", tmp_path / "model.gguf")
    assert report["ok"] and "SentencePiece" in report["note"], report


def test_a_file_llama_cpp_cant_read_is_an_error_not_a_crash(tmp_path):
    """llama.cpp stops its process on a broken file: the check runs in another one."""
    for module in ("torch", "transformers", "gguf", "llama_cpp", "tokenizers"):
        pytest.importorskip(module)
    from ai_made_easy.core.deploy import hf_gguf

    _tiny_hf("llama", tmp_path / "model")
    (tmp_path / "broken.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    with pytest.raises(hf_gguf.HfGgufError, match="llama.cpp could not use the file"):
        hf_gguf.check(tmp_path / "model", tmp_path / "broken.gguf")


def test_a_failed_check_leaves_no_file_and_a_slow_one_is_an_error(tmp_path, monkeypatch):
    import subprocess

    from ai_made_easy.core.deploy import hf_gguf

    def slow(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise subprocess.TimeoutExpired("check", hf_gguf.CHECK_SECONDS)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(hf_gguf.HfGgufError, match="took over 30 minutes"):
        hf_gguf.check(tmp_path, tmp_path / "model.gguf")

    def written(model_dir, path, quant=""):  # noqa: ANN001, ANN202
        path.write_bytes(b"GGUF")
        return {"family": "llama", "tokenizer": "BPE", "size": 1.0}

    monkeypatch.setattr(hf_gguf, "write_gguf", written)
    with pytest.raises(hf_gguf.HfGgufError):
        hf_gguf.export(tmp_path, tmp_path / "model.gguf")
    assert not (tmp_path / "model.gguf").exists()


def test_a_run_export_names_missing_packages_before_it_starts(tmp_path, monkeypatch):
    import importlib.util

    from ai_made_easy.core.deploy import hf_gguf

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "peft" else real(name, *a))
    with pytest.raises(hf_gguf.HfGgufError, match=r"needs peft: pip install"):
        hf_gguf.export_run(tmp_path, tmp_path / "out")


@pytest.mark.parametrize("quant", ["Q8_0", "Q4_0"])
def test_a_quantized_hugging_face_gguf_keeps_its_next_tokens(tmp_path, quant):
    for module in ("torch", "transformers", "gguf", "llama_cpp", "tokenizers"):
        pytest.importorskip(module)
    from ai_made_easy.core.deploy import hf_gguf

    _tiny_hf("llama", tmp_path / "model")
    float_size = None
    report = hf_gguf.export(tmp_path / "model", tmp_path / "q.gguf", quant)
    hf_gguf.write_gguf(tmp_path / "model", tmp_path / "f.gguf")
    float_size = (tmp_path / "f.gguf").stat().st_size
    assert (tmp_path / "q.gguf").stat().st_size < float_size * (0.4 if quant == "Q8_0" else 0.3)
    assert report["top1"] >= (0.9 if quant == "Q8_0" else 0.6), report
    assert "of the float size" in report["note"]


def test_other_hugging_face_models_say_why_they_cant_export(tmp_path):
    import json as _json

    from ai_made_easy.core.deploy import hf_gguf

    (tmp_path / "config.json").write_text(_json.dumps({"architectures": ["GPT2LMHeadModel"]}),
                                          encoding="utf-8")
    with pytest.raises(hf_gguf.HfGgufError, match="Llama, Qwen2, Phi-3 and Gemma 3"):
        hf_gguf.write_gguf(tmp_path, tmp_path / "x.gguf")
    with pytest.raises(hf_gguf.HfGgufError, match="pre-tokenizer isn't one llama.cpp knows"):
        hf_gguf._pre_tokenizer({"pre_tokenizer": {"type": "Split",
                                                  "pattern": {"Regex": "\\w+"}}})


@pytest.mark.slow
def test_a_lora_fine_tune_run_exports_to_gguf(tmp_path):
    """The run's adapter, merged into its base model, goes out as GGUF (here the base is a
    tiny local model, so nothing is downloaded)."""
    for module in ("torch", "transformers", "peft", "gguf", "llama_cpp", "tokenizers"):
        pytest.importorskip(module)
    import json as _json

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from ai_made_easy.core.deploy import hf_gguf

    base = _tiny_hf("qwen2", tmp_path / "base")
    run = tmp_path / "run"
    adapter = run / "chat_lora_adapter"
    torch.manual_seed(1)
    tuned = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                            init_lora_weights=False))
    tuned.save_pretrained(adapter)
    AutoTokenizer.from_pretrained(tmp_path / "base").save_pretrained(adapter)
    (run / "run.json").write_text(_json.dumps({"run_id": "r1", "graph": {"nodes": [
        {"id": "m", "type": "llm.model", "params": {"repo_id": str(tmp_path / "base")}},
        {"id": "l", "type": "llm.lora", "params": {}}]}}), encoding="utf-8")
    report = hf_gguf.export_run(run, tmp_path / "out", "q8_0")
    assert report["ok"] and report["file"].endswith("chat-q8_0.gguf"), report
    assert report["base"] == str(tmp_path / "base") and report["top1"] >= 0.9
    with pytest.raises(hf_gguf.HfGgufError, match="quant must be one of"):
        hf_gguf.export_run(run, tmp_path / "out", "q2_k")


# ------------------------------------------------------------ Phase 1: RWKV-7
def _rwkv7(dim: int = 16, layers: int = 2, head: int = 8, hidden: int = 24):  # noqa: ANN202
    import math

    import torch

    import ai_made_easy.core.blocks  # noqa: F401  (registers the helpers)
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    ns = {"torch": torch, "nn": torch.nn, "math": math}
    exec(PYTORCH_HELPERS["RWKV7"], ns)  # noqa: S102 — our own helper source
    torch.manual_seed(0)
    stack = ns["RWKV7"](dim, layers, head, hidden, (32, 32, 32, 32)).eval()
    with torch.no_grad():   # away from the start, where outputs and the gate are zero
        for p in stack.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return stack


def test_rwkv7_in_keras_computes_what_pytorch_does():
    pytest.importorskip("torch")
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("compared on the torch backend")
    import numpy as np
    import torch
    from keras import layers, ops

    from ai_made_easy.core.codegen.helpers import KERAS_HELPERS

    reference = _rwkv7()
    ns = {"keras": keras, "layers": layers, "ops": ops}
    for helper in ("TorchLayout", "RWKV7"):
        exec(KERAS_HELPERS[helper], ns)  # noqa: S102
    layer = ns["RWKV7"](2, 8, 24, (32, 32, 32, 32))
    x = np.random.default_rng(0).standard_normal((2, 11, 16)).astype("float32")
    layer(ops.convert_to_tensor(x))
    weights = {w.name: w for w in layer.weights}
    params = {n.replace(".", "_"): p for n, p in reference.named_parameters()}
    assert set(weights) == set(params)
    for n, p in params.items():
        weights[n].assign(p.detach().numpy())
    with torch.no_grad():
        want = reference(torch.from_numpy(x)).numpy()
    got = ops.convert_to_numpy(layer(ops.convert_to_tensor(x)))
    assert np.abs(got - want).max() <= 1e-4 * np.abs(want).max()


def test_rwkv7_counts_its_parameters_and_sees_only_the_past():
    pytest.importorskip("torch")
    import torch

    from ai_made_easy.core.blocks.library312 import rwkv7_params

    stack = _rwkv7(dim=64, layers=3, head=16, hidden=96)
    assert sum(p.numel() for p in stack.parameters()) == rwkv7_params(64, 3, 96)
    x = torch.randn(1, 12, 64)
    later = x.clone()
    later[:, 8:] = torch.randn(1, 4, 64)
    with torch.no_grad():
        assert torch.allclose(stack(x)[:, :8], stack(later)[:, :8], atol=1e-6)


def test_tiny_rwkv_is_rwkv7_and_its_gguf_matches_pytorch(tmp_path):
    """Tiny RWKV now uses RWKV-7, which llama.cpp reads: its GGUF scores tokens as PyTorch
    does."""
    pytest.importorskip("torch")
    pytest.importorskip("gguf")
    pytest.importorskip("llama_cpp")
    import torch
    from test_export import _build

    from ai_made_easy.core.deploy import exporters

    graph = Graph.from_dict(_sample("tiny_rwkv"))
    assert [n.type_id for n in graph.nodes.values() if n.type_id.startswith("seq.")] == \
        ["seq.rwkv7"]
    torch.manual_seed(0)
    model, samples = _build(graph)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    report = exporters.export_all(model, "pytorch", ["gguf", "gguf_q8_0"], tmp_path,
                                  samples)
    assert report["gguf"]["ok"] and report["gguf"]["note"].startswith("matches PyTorch"), report
    assert report["gguf_q8_0"]["ok"] and "of the float size" in report["gguf_q8_0"]["note"]
    assert report["gguf_q8_0"]["metrics"]["top1_agreement"] >= 0.9


def test_gguf_from_a_run_says_why_when_the_run_is_not_a_fine_tune(isolated_home, capsys):
    """REST, the CLI and the API answer a network trained here with how to export it."""
    from conftest import tiny_classifier_dict

    from ai_made_easy import cli
    from ai_made_easy.core import api
    from ai_made_easy.core.runner.manager import RunManager

    mgr = RunManager()
    api.set_manager(mgr)
    try:
        record = mgr.history.create(tiny_classifier_dict(), framework="pytorch", project="p")
        with pytest.raises(api.ApiError, match="takes LoRA fine-tunes"):
            api.gguf_run(record.run_id, str(isolated_home / "out"))
        assert cli.main(["gguf", record.run_id, "-o", str(isolated_home / "out")]) != 0
        assert "takes LoRA fine-tunes" in capsys.readouterr().err
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ai_made_easy.server.app import create_app

        got = TestClient(create_app()).get(f"/api/runs/{record.run_id}/gguf?quant=q8_0")
        assert got.status_code == 400 and "use the gguf export format" in got.text
    finally:
        api.set_manager(None)


# ------------------------------------------------------------ Phase 2: thresholds in use
@pytest.mark.slow
def test_adopted_thresholds_decide_try_it_and_the_package(isolated_home, tmp_path):
    """Use a Results-by-group run's thresholds: the scored run's Try it, and a package
    built from it, decide each row's class with its group's threshold; stop using them
    and 0.5 decides again."""
    pytest.importorskip("torch")
    from test_v38_features import _network
    from test_v311_features import _shifted_table

    from ai_made_easy.core import api
    from ai_made_easy.core.deploy import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = api.start_training(_network(_shifted_table(tmp_path / "t.csv")))["run_id"]
        assert mgr.wait(run_id, 600)["state"] == "finished"
        scoring = api.results_by_group(run_id, "region")["run_id"]
        assert mgr.wait(scoring, 600)["state"] == "finished"
        with pytest.raises(api.ApiError, match="not a Results by group run"):
            api.use_group_thresholds(run_id)
        cuts = api.use_group_thresholds(scoring)
        assert cuts["run_id"] == run_id and cuts["column"] == "region"
        assert api.group_thresholds(run_id)["thresholds"]["thresholds"] == cuts["thresholds"]
        east = cuts["thresholds"]["east"]
        assert api.try_describe(run_id)["group_thresholds"]["column"] == "region"
        row = {"a": 0.3, "b": 0.3}
        got = api.try_run(run_id, [{**row, "region": "east"}, row])["predictions"]
        p = got[0]["probabilities"]["1"]
        assert got[0]["threshold"] == east and got[0]["group"] == "east"
        assert got[0]["label"] == ("1" if p >= east else "0")
        assert got[1]["threshold"] == 0.5 and got[1]["group"] is None   # no region: default
        package = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        meta = json.loads((tmp_path / "pkg" / "metadata.json").read_text(encoding="utf-8"))
        assert meta["group_thresholds"]["thresholds"] == cuts["thresholds"]
        assert "'region'" in meta["input_help"] and package.path.is_dir()
        assert (tmp_path / "pkg" / "model" / "group_thresholds.json").is_file()
        assert api.stop_group_thresholds(run_id)["removed"]
        again = api.try_run(run_id, [{**row, "region": "east"}])["predictions"][0]
        assert "threshold" not in again and again["label"] == ("1" if p >= 0.5 else "0")
    finally:
        api.try_service().close()
        api.set_manager(None)


@pytest.mark.slow
def test_a_classic_classifier_uses_adopted_thresholds_too(isolated_home, tmp_path):
    pytest.importorskip("sklearn")
    from test_v311_features import _shifted_table

    from ai_made_easy.core import api, recipes
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        facts = recipes.detect(_shifted_table(tmp_path / "t.csv"))
        run_id = mgr.start(recipes.build("random_forest", facts.task, facts,
                                         {"n_estimators": 50}))
        assert mgr.wait(run_id, 600)["state"] == "finished"
        scoring = api.results_by_group(run_id, "region")["run_id"]
        assert mgr.wait(scoring, 600)["state"] == "finished"
        cuts = api.use_group_thresholds(scoring)["thresholds"]
        got = api.try_run(run_id, [{"a": 0.4, "b": 0.4, "region": "east"}])["predictions"][0]
        assert got["group"] == "east" and got["threshold"] == cuts.get("east", 0.5)
        p = list(got["probabilities"].values())[1]
        assert got["label"] == list(got["probabilities"])[int(p >= got["threshold"])]
    finally:
        api.try_service().close()
        api.set_manager(None)


@pytest.mark.slow
def test_the_desktop_uses_thresholds_and_asks_for_the_group(isolated_home, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("PySide6")
    import time

    from PySide6 import QtCore, QtWidgets
    from test_v38_features import _network
    from test_v311_features import _shifted_table

    from ai_made_easy.core import api
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from ai_made_easy.ui.features.run_tools import GroupsDialog
    from ai_made_easy.ui.features.try_dialog import TryDialog, describe_prediction

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    _ensure_qt_plugin_path()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        run_id = api.start_training(_network(_shifted_table(tmp_path / "t.csv")))["run_id"]
        assert mgr.wait(run_id, 600)["state"] == "finished"
        scoring = api.results_by_group(run_id, "region")["run_id"]
        assert mgr.wait(scoring, 600)["state"] == "finished"
        groups = GroupsDialog(None, mgr, run_id)
        groups.scoring = scoring
        groups.show_result(api.group_result(scoring))
        assert not groups.use_btn.isHidden() and groups.stop_btn.isHidden()
        groups.use_btn.click()
        assert groups.in_use.text() == "In use, by region." and not groups.stop_btn.isHidden()
        dialog = TryDialog(None, mgr.history, run_id)
        deadline = time.time() + 120
        while dialog.form_info is None and time.time() < deadline:
            app.processEvents()
            QtCore.QThread.msleep(20)
        assert dialog._group is not None and dialog._group[0] == "region"
        assert list(dialog._fields) == ["a", "b"]           # the model's own columns
        dialog._fields["a"].setText("0.3")
        dialog._fields["b"].setText("0.3")
        dialog._group[1].setText("east")
        assert dialog.inputs() == [{"a": 0.3, "b": 0.3, "region": "east"}]
        text = describe_prediction({"label": "1", "confidence": 0.6, "threshold": 0.31,
                                    "group": "east", "probabilities": {"0": 0.4, "1": 0.6}})
        assert "decided at 0.31 (east's threshold)" in text
        groups.stop_btn.click()
        assert groups.in_use.text() == "" and not groups.use_btn.isHidden()
        dialog.deleteLater()
        groups.deleteLater()
    finally:
        api.try_service().close()
        api.set_manager(None)


# ------------------------------------------------------------ Phase 2: a whole file's changes
@pytest.mark.slow
def test_a_files_predictions_can_say_what_would_change_each(isolated_home, tmp_path):
    pytest.importorskip("torch")
    import csv
    import io

    from test_v38_features import _network
    from test_v311_features import _shifted_table

    from ai_made_easy.core import api
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = api.start_training(_network(_shifted_table(tmp_path / "t.csv")))["run_id"]
        assert mgr.wait(run_id, 600)["state"] == "finished"
        data = b"a,b\n0.2,0.3\n0.9,0.8\n0.5,0.45\n"
        plain = api.try_file(run_id, "rows.csv", data)
        assert "smallest_change" not in plain["csv"]
        got = api.try_file(run_id, "rows.csv", data, changes=True)
        rows = list(csv.DictReader(io.StringIO(got["csv"])))
        assert len(rows) == 3 and {"smallest_change", "change_predicts"} <= set(rows[0])
        for row in rows:   # each predicts the other class after its change
            assert row["smallest_change"] and row["change_predicts"] != row["label"], row
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from ai_made_easy.server.app import create_app

        sent = TestClient(create_app()).post(f"/api/runs/{run_id}/try/file?changes=true",
                                             files={"file": ("rows.csv", data, "text/csv")})
        assert sent.status_code == 200 and "smallest_change" in sent.text
    finally:
        api.try_service().close()
        api.set_manager(None)
