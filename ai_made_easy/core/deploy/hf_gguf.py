"""Fine-tuned Hugging Face language models as GGUF (3.12), for llama.cpp, Ollama and LM
Studio.

A fine-tune run saves its LoRA adapter. Exporting merges it into the base model
(transformers + PEFT), then writes one GGUF file the way llama.cpp's own converter does
for that family:

- **Llama** (``llama``): query and key rows reordered for llama.cpp's rotary layout;
  Llama 3's frequency scaling as a ``rope_freqs`` tensor.
- **Qwen2** (``qwen2``): as is, with the attention biases.
- **Phi-3** (``phi3``): fused QKV and gate/up projections; LongRoPE's factors.
- **Gemma 3** (``gemma3``): norm weights stored with their +1, sliding-window and global
  rope bases.

The tokenizer goes in too: SentencePiece (``tokenizer.model``) or byte-level BPE
(``tokenizer.json``, with its pre-tokenizer recognised from its regex). The file is then
checked: llama.cpp must split sample texts into the tokenizer's tokens, and its
next-token scores must match transformers'. Other architectures say why they can't
export.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FAMILIES = {"LlamaForCausalLM": "llama", "Qwen2ForCausalLM": "qwen2",
            "Phi3ForCausalLM": "phi3", "Gemma3ForCausalLM": "gemma3"}
TOLERANCE = 1e-3          # of the logits' scale: llama.cpp's float32 kernels sum in their own order
SAMPLE_TEXTS = ("The robot's map isn't here: 1234 apples, naïve café!",
                "Hello world.\n\n  Two  spaces, tabs\tand émojis 🙂 too.")
# byte-level BPE pre-tokenizers llama.cpp knows, by the regex in tokenizer.json
_PRE = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "llama-bpe",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen2",
}


class HfGgufError(ValueError):
    """A model or run this export can't take; the message says why."""


# ------------------------------------------------------------------ merge

def merge_adapter(base: str, adapter: Path, out: Path) -> Path:
    """The base model with the LoRA adapter merged in, saved (float32) to ``out``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    source = adapter if (adapter / "tokenizer_config.json").is_file() else base
    AutoTokenizer.from_pretrained(source).save_pretrained(out)
    if not (out / "tokenizer.model").is_file():   # SentencePiece's own file, if the base has it
        spm = _base_file(base, "tokenizer.model")
        if spm is not None:
            (out / "tokenizer.model").write_bytes(spm.read_bytes())
    return out


def _base_file(base: str, name: str) -> Path | None:
    """A file of the base model: in its folder, or from the Hugging Face Hub (cached)."""
    local = Path(base) / name
    if local.is_file():
        return local
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(base, name))
    except Exception:  # noqa: BLE001 — no such file (BPE tokenizers have none) or offline
        return None


# ------------------------------------------------------------------ convert

def _rope(config: dict) -> tuple[dict, float]:
    rope = dict(config.get("rope_parameters") or config.get("rope_scaling") or {})
    rope = rope.get("full_attention", rope)
    theta = config.get("rope_theta") or rope.get("rope_theta") or 10000.0
    return rope, float(theta)


def _permute(w, heads: int):  # noqa: ANN001, ANN202 — torch tensors
    """Llama's query / key rows, from Hugging Face's rotary layout to llama.cpp's."""
    return (w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:])
            .swapaxes(1, 2).reshape(w.shape))


def _llama3_factors(rope: dict, theta: float, dim: int) -> list[float]:
    """Llama 3's per-frequency rope scaling, as llama.cpp's converter computes it."""
    factor = rope.get("factor", 8.0)
    low, high = rope.get("low_freq_factor", 1.0), rope.get("high_freq_factor", 4.0)
    old = rope.get("original_max_position_embeddings", 8192)
    out = []
    for i in range(0, dim, 2):
        wavelen = 2 * math.pi * theta ** (i / dim)
        if wavelen < old / high:
            out.append(1.0)
        elif wavelen > old / low:
            out.append(factor)
        else:
            smooth = (old / wavelen - low) / (high - low)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return out


def _parameters(writer, family: str, config: dict) -> list[tuple[str, np.ndarray]]:  # noqa: ANN001
    """The family's hyperparameters; returns extra tensors (rope factors) to write."""
    import gguf

    heads = config["num_attention_heads"]
    hidden = config["hidden_size"]
    head_dim = config.get("head_dim") or hidden // heads
    rope, theta = _rope(config)
    kind = (rope.get("rope_type") or rope.get("type") or "default").lower()
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_context_length(config.get("max_position_embeddings", 2048))
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(heads)
    writer.add_head_count_kv(config.get("num_key_value_heads") or heads)
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
    writer.add_rope_freq_base(theta)
    writer.add_vocab_size(config["vocab_size"])
    if config.get("head_dim"):
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
    extra = []
    if kind == "linear":
        writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        writer.add_rope_scaling_factor(rope["factor"])
    elif kind == "yarn":
        writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        writer.add_rope_scaling_factor(rope["factor"])
        writer.add_rope_scaling_orig_ctx_len(rope["original_max_position_embeddings"])
    elif kind == "llama3" and family == "llama":
        extra.append(("rope_freqs.weight",
                      np.asarray(_llama3_factors(rope, theta, head_dim), np.float32)))
    elif kind in ("longrope", "su") and family == "phi3":
        writer.add_rope_scaling_type(gguf.RopeScalingType.LONGROPE)
    elif kind != "default":
        raise HfGgufError(f"its rope scaling ({kind}) isn't one this export writes for "
                          f"{family}")
    if family == "llama":
        writer.add_rope_dimension_count(head_dim)
    elif family == "phi3":
        rot = int(config.get("partial_rotary_factor", 1.0) * hidden) // heads
        orig = rope.get("original_max_position_embeddings",
                        config.get("original_max_position_embeddings",
                                   config.get("max_position_embeddings", 2048)))
        writer.add_rope_scaling_orig_ctx_len(orig)
        writer.add_rope_dimension_count(rot)
        writer.add_sliding_window(config.get("sliding_window") or 0)
        if rope.get("long_factor"):
            scale = config.get("max_position_embeddings", orig) / orig
            writer.add_rope_scaling_attn_factors(
                math.sqrt(1 + math.log(scale) / math.log(orig)) if scale > 1 else 1.0)
            extra += [("rope_factors_long.weight", np.asarray(rope["long_factor"], np.float32)),
                      ("rope_factors_short.weight", np.asarray(rope["short_factor"], np.float32))]
    elif family == "gemma3":
        writer.add_key_length(head_dim)
        writer.add_value_length(head_dim)
        if config.get("final_logit_softcapping"):
            writer.add_final_logit_softcapping(config["final_logit_softcapping"])
        if config.get("attn_logit_softcapping"):
            raise HfGgufError("llama.cpp's gemma3 has no attention soft-capping")
        if config.get("sliding_window"):
            writer.add_sliding_window(config["sliding_window"])
        if config.get("rope_local_base_freq"):
            writer.add_rope_freq_base_swa(config["rope_local_base_freq"])
    return extra


def _tensors(model_dir: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        out.update(load_file(str(f)))
    if not out:
        raise HfGgufError(f"no safetensors weights in {model_dir}")
    return out


def _vocabulary(writer, family: str, model_dir: Path, vocab_size: int) -> str:  # noqa: ANN001
    """The tokenizer, as llama.cpp reads it; returns its kind for the report."""
    import gguf

    if (model_dir / "tokenizer.model").is_file():
        from sentencepiece import SentencePieceProcessor

        sp = SentencePieceProcessor()
        sp.LoadFromFile(str(model_dir / "tokenizer.model"))
        tokens = [f"[PAD{i}]".encode() for i in range(vocab_size)]
        scores = [-10000.0] * vocab_size
        types = [gguf.TokenType.UNUSED] * vocab_size
        for i in range(min(sp.vocab_size(), vocab_size)):
            tokens[i], scores[i] = sp.IdToPiece(i).encode("utf-8"), sp.GetScore(i)
            types[i] = (gguf.TokenType.UNKNOWN if sp.IsUnknown(i) else
                        gguf.TokenType.CONTROL if sp.IsControl(i) else
                        gguf.TokenType.UNUSED if sp.IsUnused(i) else
                        gguf.TokenType.BYTE if sp.IsByte(i) else gguf.TokenType.NORMAL)
        config = model_dir / "tokenizer_config.json"
        added = json.loads(config.read_text(encoding="utf-8")).get(
            "added_tokens_decoder", {}) if config.is_file() else {}
        for key, token in added.items():
            i = int(key)
            if i < vocab_size and types[i] == gguf.TokenType.UNUSED:
                tokens[i], scores[i] = token["content"].encode("utf-8"), -1000.0
                types[i] = (gguf.TokenType.CONTROL if token.get("special")
                            else gguf.TokenType.USER_DEFINED)
        writer.add_tokenizer_model("llama")
        writer.add_tokenizer_pre("default")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(types)
        if family == "gemma3":
            writer.add_add_space_prefix(False)
        gguf.SpecialVocab(model_dir, n_vocab=len(tokens)).add_to_gguf(writer)
        return "SentencePiece"
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise HfGgufError("the model has neither tokenizer.model nor tokenizer.json")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if (spec.get("model") or {}).get("byte_fallback"):   # SentencePiece, saved as tokenizer.json
        tokens, scores, types = _sentencepiece_from_json(model_dir, spec, vocab_size)
        writer.add_tokenizer_model("llama")
        writer.add_tokenizer_pre("default")
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(types)
        writer.add_add_space_prefix(_space_prefix(spec))
        gguf.SpecialVocab(model_dir, n_vocab=len(tokens)).add_to_gguf(writer)
        return "SentencePiece"
    if (spec.get("model") or {}).get("type") != "BPE":
        raise HfGgufError(f"its tokenizer is {spec.get('model', {}).get('type')}, and this "
                          "export writes SentencePiece and byte-level BPE")
    pre = _pre_tokenizer(spec)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    reverse = {i: t for t, i in tok.get_vocab().items()}
    special = {i for i, t in tok.added_tokens_decoder.items() if t.special}
    added = set(tok.get_added_vocab().values())
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in reverse:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
            continue
        tokens.append(reverse[i])
        types.append(gguf.TokenType.CONTROL if i in special else
                     gguf.TokenType.USER_DEFINED if i in added else gguf.TokenType.NORMAL)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(pre)
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    gguf.SpecialVocab(model_dir, load_merges=True).add_to_gguf(writer)
    return f"byte-level BPE ({pre})"


def _sentencepiece_from_json(model_dir: Path, spec: dict, vocab_size: int) -> tuple:
    """A SentencePiece vocabulary from tokenizer.json, by token id. llama.cpp merges the
    pair whose result scores highest and Hugging Face the earliest merge, so a merged
    piece scores minus its merge's rank: both then split text the same way."""
    import re

    import gguf
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    reverse = {i: t for t, i in tok.get_vocab().items()}
    special = {i for i, t in tok.added_tokens_decoder.items() if t.special}
    added = set(tok.get_added_vocab().values())
    rank = {}
    for r, merge in enumerate((spec.get("model") or {}).get("merges") or []):
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        rank.setdefault(left + right, r)
    tokens = [f"[PAD{i}]".encode() for i in range(vocab_size)]
    scores = [-10000.0] * vocab_size
    types = [gguf.TokenType.UNUSED] * vocab_size
    for i in range(vocab_size):
        if i not in reverse:
            continue
        text = reverse[i]
        tokens[i], scores[i] = text.encode("utf-8"), -float(rank.get(text, 0))
        types[i] = (gguf.TokenType.UNKNOWN if text == tok.unk_token else
                    gguf.TokenType.CONTROL if i in special else
                    gguf.TokenType.USER_DEFINED if i in added else
                    gguf.TokenType.BYTE if re.fullmatch(r"<0x[0-9A-F]{2}>", text) else
                    gguf.TokenType.NORMAL)
    return tokens, scores, types


def _space_prefix(spec: dict) -> bool:
    """Whether a SentencePiece tokenizer saved as tokenizer.json puts a space before the
    text (its "dummy prefix"), as its Metaspace or Prepend step says."""
    pre = spec.get("pre_tokenizer") or {}
    for step in pre.get("pretokenizers", [pre]):
        if step.get("type") == "Metaspace":
            return step.get("prepend_scheme", "always") != "never" and \
                step.get("add_prefix_space", True)
    norm = spec.get("normalizer") or {}
    return any(step.get("type") == "Prepend" for step in norm.get("normalizers", [norm]))


def _pre_tokenizer(spec: dict) -> str:
    """llama.cpp's name for a byte-level BPE tokenizer's pre-tokenizer."""
    pre = spec.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", [pre])
    split = next((s for s in steps if s.get("type") == "Split"), None)
    if split is None:
        if any(s.get("type") == "ByteLevel" and s.get("use_regex", True) for s in steps):
            return "gpt-2"
        raise HfGgufError("its tokenizer splits text in a way this export doesn't recognise")
    regex = (split.get("pattern") or {}).get("Regex", "")
    if regex not in _PRE:
        raise HfGgufError("its tokenizer's pre-tokenizer isn't one llama.cpp knows "
                          "(Llama 3's, Qwen2's or GPT-2's)")
    return _PRE[regex]


def write_gguf(model_dir: Path, path: Path, quant: str = "") -> dict:
    """``model_dir`` (a Hugging Face model saved with its tokenizer) as one GGUF file."""
    import gguf

    from ai_made_easy.core.deploy.exporters import add_gguf_tensor

    model_dir, path = Path(model_dir), Path(path)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    arch = (config.get("architectures") or ["?"])[0]
    family = FAMILIES.get(arch)
    if family is None:
        raise HfGgufError(f"GGUF export takes Llama, Qwen2, Phi-3 and Gemma 3 language "
                          f"models; this one is {arch}")
    writer = gguf.GGUFWriter(str(path), family)
    writer.add_name(config.get("_name_or_path") or "AI Made Easy fine-tune")
    extra = _parameters(writer, family, config)
    qtype = getattr(gguf.GGMLQuantizationType, quant) if quant else None
    writer.add_file_type(getattr(gguf.LlamaFileType, f"MOSTLY_{quant}") if quant
                         else gguf.LlamaFileType.ALL_F32)
    tokenizer = _vocabulary(writer, family, model_dir, config["vocab_size"])
    names = gguf.get_tensor_name_map(getattr(gguf.MODEL_ARCH, family.upper()),
                                     config["num_hidden_layers"])
    heads = config["num_attention_heads"]
    kv = config.get("num_key_value_heads") or heads
    float_bytes = 0
    for name, tensor in _tensors(model_dir).items():
        if name.endswith("rotary_emb.inv_freq"):
            continue
        tensor = tensor.float()
        if family == "llama" and name.endswith(("q_proj.weight", "q_proj.bias")):
            tensor = _permute(tensor, heads)
        elif family == "llama" and name.endswith(("k_proj.weight", "k_proj.bias")):
            tensor = _permute(tensor, kv)
        elif family == "gemma3" and name.endswith("norm.weight"):
            tensor = tensor + 1.0
        new = names.get_name(name, try_suffixes=(".weight", ".bias"))
        if new is None:
            raise HfGgufError(f"llama.cpp's {family} has no tensor for {name}")
        data = tensor.numpy()
        float_bytes += data.nbytes
        add_gguf_tensor(writer, new, data, qtype)
    for name, data in extra:
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"family": family, "tokenizer": tokenizer,
            "size": path.stat().st_size / max(float_bytes, 1)}


# ------------------------------------------------------------------ check

def check(model_dir: Path, path: Path, quant: str = "") -> dict:
    """llama.cpp against transformers, in another process: llama.cpp stops the whole
    process when a file is wrong, and this may run inside the app's server."""
    import subprocess

    from ai_made_easy.core.runner.manager import python_executable

    done = subprocess.run([python_executable(), "-m", "ai_made_easy.core.deploy.hf_gguf",
                           str(model_dir), str(path), quant], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=1800, check=False)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        why = (done.stderr.strip().splitlines() or ["no message"])[-1]
        raise HfGgufError(f"llama.cpp could not use the file (it stopped with code "
                          f"{done.returncode}: {why})")
    result = json.loads(lines[-1])
    if "error" in result:
        raise HfGgufError(result["error"])
    return result


def _check(model_dir: Path, path: Path, quant: str = "") -> dict:
    """llama.cpp against transformers: the same tokens for sample texts, and next-token
    scores within TOLERANCE (or, quantized, how often the top token agrees)."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return {"checked": False, "note": "not checked here (pip install llama-cpp-python "
                                          "to check it)"}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = [tok.encode(t, add_special_tokens=False) for t in SAMPLE_TEXTS]
    vocab = Llama(model_path=str(path), vocab_only=True, verbose=False)
    for text, want in zip(SAMPLE_TEXTS, ids, strict=True):
        got = vocab.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        if got != want:
            raise HfGgufError(f"llama.cpp splits {text!r} into {got[:12]}…, the tokenizer "
                              f"into {want[:12]}…")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    longest = max(len(i) for i in ids)
    llm = Llama(model_path=str(path), n_ctx=max(longest, 64), logits_all=True, verbose=False,
                n_gpu_layers=0, type_k=0, type_v=0, flash_attn=False)
    worst, agree, total = 0.0, 0, 0
    for row in ids:
        with torch.no_grad():
            ref = model(torch.tensor([row])).logits[0].numpy().astype(np.float64)
        llm.reset()
        llm.eval(row)
        got = np.asarray(llm.scores[:len(row)], np.float64)
        worst = max(worst, float(np.abs(got - ref).max()) / max(float(np.abs(ref).max()), 1e-9))
        agree += int((got.argmax(-1) == ref.argmax(-1)).sum())
        total += len(row)
    if quant:
        return {"checked": True, "top1": agree / total,
                "note": f"the top next token agrees with the float model on {agree / total:.1%}"}
    if worst > TOLERANCE:
        raise HfGgufError(f"llama.cpp's next-token scores differ from transformers' by "
                          f"{worst:.2e} of their scale")
    return {"checked": True, "worst": worst,
            "note": f"matches transformers ({worst:.1e} of the scores' scale)"}


def export(model_dir: Path, out: Path, quant: str = "") -> dict:
    """Write and check ``out``; the report says what went in and how it was checked."""
    info = write_gguf(model_dir, out, quant)
    result = check(model_dir, out, quant)
    size = f", {info['size']:.0%} of the float size" if quant else ""
    return {"ok": True, "file": str(out), "family": info["family"], **result,
            "note": f"{info['family']}, {info['tokenizer']} tokenizer{size}; {result['note']}"}


# ------------------------------------------------------------------ runs

QUANTS = {"": "", "f32": "", "q8_0": "Q8_0", "q4_0": "Q4_0"}


def export_run(run_dir: Path, out_dir: Path, quant: str = "") -> dict:
    """A finished fine-tune run as GGUF: its LoRA adapter merged into the base model."""
    import tempfile

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if quant.lower() not in QUANTS:
        raise HfGgufError(f"quant must be one of {', '.join(q for q in QUANTS if q)}")
    quant = QUANTS[quant.lower()]
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    nodes = (record.get("graph") or {}).get("nodes", [])
    model = next((n for n in nodes if n.get("type") == "llm.model"), None)
    if model is None or not any(n.get("type") == "llm.lora" for n in nodes):
        raise HfGgufError("GGUF from a run takes LoRA fine-tunes of a Hugging Face model; "
                          "for a network trained here, use the gguf export format")
    adapter = next(iter(sorted(run_dir.glob("*_lora_adapter"))), None)
    if adapter is None:
        raise HfGgufError("the run saved no LoRA adapter: did it finish?")
    base = (model.get("params") or {}).get("repo_id") or "sshleifer/tiny-gpt2"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = adapter.name.removesuffix("_lora_adapter")
    path = out_dir / (f"{name}-{quant.lower()}.gguf" if quant else f"{name}.gguf")
    with tempfile.TemporaryDirectory(prefix="aime_gguf_") as work:
        merged = merge_adapter(base, adapter, Path(work) / "merged")
        return {"run_id": record.get("run_id", run_dir.name), "base": base,
                **export(merged, path, quant)}


if __name__ == "__main__":   # the check, in its own process (see ``check``)
    import sys

    try:
        print(json.dumps(_check(Path(sys.argv[1]), Path(sys.argv[2]),
                                sys.argv[3] if len(sys.argv) > 3 else "")))
    except HfGgufError as exc:
        print(json.dumps({"error": str(exc)}))
