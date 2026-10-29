"""Execute generated model code for every NN block and compare real output
shapes against the designer's inferred shapes (PyTorch + Keras 3).

Usage: KERAS_BACKEND=torch python scripts/verify_codegen.py [--keras] [-v] [--coverage]
Also importable: ``block_cases()`` / ``run_case()`` power the test suite and
``coverage()`` reports, per block family, which layers the harness runs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402
from ai_made_easy.core.spec import missing_requirements  # noqa: E402

VARIANTS: dict[str, list[dict]] = {
    "vision.swin_block": [{"shift": True}],
    "core.gqa_attention": [{"heads": 4, "kv_heads": 1}, {"causal": False, "rope": False}],
    "core.mixture_of_experts": [{"experts": 2, "top_k": 1}],
    "core.gated_ffn": [{"activation": "gelu", "hidden": 24}],
    "core.conv2d": [{"padding": 2, "stride": 2}, {"dilation": 2, "padding": 2},
                    {"groups": 2, "out_channels": 8, "bias": False},
                    {"kernel_size": 5, "padding": 0, "stride": 3}],
    "core.conv1d": [{"padding": 0, "stride": 2}, {"padding": 3, "kernel_size": 3}],
    "core.conv3d": [{"stride": 2}],
    "core.conv_transpose2d": [{"stride": 2, "padding": 1, "output_padding": 1},
                              {"stride": 2, "kernel_size": 2, "padding": 0},
                              {"padding": 1, "dilation": 2}],
    "core.conv_transpose1d": [{"stride": 3, "padding": 2, "output_padding": 2}],
    "core.depthwise_conv2d": [{"depth_multiplier": 2, "stride": 2}],
    "core.separable_conv2d": [{"padding": 0, "bias": False}],
    "core.maxpool2d": [{"padding": 1, "kernel_size": 3, "stride": 2}],
    "core.avgpool1d": [{"padding": 1, "kernel_size": 3, "stride": 1}],
    "core.adaptive_avgpool2d": [{"output_size": 4}, {"output_size": 3}],
    "core.adaptive_maxpool1d": [{"output_size": 8}],
    "core.lstm": [{"bidirectional": True}, {"num_layers": 2, "dropout": 0.2},
                  {"return_sequences": False}, {"bidirectional": True,
                                                "return_sequences": False},
                  {"proj_size": 16, "num_layers": 2},
                  {"proj_size": 16, "bidirectional": True, "return_sequences": False}],
    "core.gru": [{"bidirectional": True, "num_layers": 2, "return_sequences": False}],
    "core.rnn": [{"nonlinearity": "relu", "bias": False}],
    "core.layer_norm": [{"normalized_dims": "all"}, {"affine": False}],
    "core.batch_norm1d": [{"affine": False}],
    "core.gelu": [{"approximate": "tanh"}],
    "core.softplus": [{"beta": 2.0}],
    "core.softmax": [{"dim": 0}],
    "core.upsample2d": [{"mode": "bilinear"}, {"mode": "bicubic", "scale_factor": 3}],
    "core.upsample1d": [{"mode": "linear"}],
    "core.reduce": [{"op": "sum", "dim": 0, "keepdim": True}, {"op": "std"},
                    {"op": "max", "dim": 0}],
    "core.slice": [{"dim": -1, "start": 2, "end": -2}],
    "core.transpose": [{"dim0": -1, "dim1": 0}],
    "core.unsqueeze": [{"dim": -1}],
    "core.transformer_encoder": [{"activation": "gelu", "norm_first": True}],
    "core.dot": [{"normalize": True}],
    "core.embedding": [{"padding_idx": 0}],
    "core.math": [{"op": "square"}, {"op": "sin"}],
    "core.reshape": [{"target": "4, 8"}, {"target": "8, 2, 16"}],
    "core.pretrained_backbone": [{"architecture": a, "weights": "none"} for a in (
        "mobilenet_v3_small", "efficientnet_b0", "densenet121", "convnext_tiny", "vgg16")],
    "core.squeeze": [{"dim": 0}],
    "seq.tcn": [{"levels": 2, "kernel_size": 5, "channels": 8}],
    "seq.xlstm": [{"return_sequences": False, "num_layers": 2}],
    "audio.mfcc": [{"n_fft": 512, "hop_length": 256, "n_mels": 40, "n_mfcc": 20}],
    "audio.hf_encoder": [{"model_id": m, "weights": "none", "pooling": "none"} for m in (
        "facebook/wav2vec2-base", "openai/whisper-tiny")] + [
        {"model_id": "MIT/ast-finetuned-audioset-10-10-0.4593", "weights": "none"}],
    **{f"forecast.{m}": [{"head": h} for h in ("quantile", "student_t", "negbin")]
       for m in ("dlinear", "nbeats", "nhits", "tcn", "patchtst", "tide", "rnn",
                 "transformer", "tft", "nlinear", "itransformer", "tsmixer", "timesnet")},
    "audio.mel_spectrogram": [{"layout": "sequence [frames, bins]"}],
    "gen.diffusion_unet": [{"num_classes": 4, "attention": False, "depth": 2}],
    "gen.gpt": [{"tie_weights": False, "layers": 2}],
    "gen.seq2seq_transformer": [{"encoder_layers": 1, "decoder_layers": 2, "max_target_len": 5}],
    "gen.dcgan_generator": [{"channels": 3, "image_size": 16, "width": 16}],
    "vision.unet": [{"variant": v, "depth": 3} for v in (
        "unet_plus_plus", "attention_unet", "resunet")] + [{"upsample": "bilinear",
                                                             "norm": "group"}],
    "vision.segmenter": [{"arch": "lraspp_mobilenet_v3_large", "weights": "none"}],
    "vision.detector": [{"arch": a, "weights": "none"} for a in (
        "ssdlite320_mobilenet_v3_large", "fcos_resnet50_fpn")],
}

# defaults that would download weights are replaced for offline verification
DEFAULT_OVERRIDES = {"core.pretrained_backbone": {"weights": "none"},
                     "core.linformer_attention": {"k": 4},  # k must fit the test sequence
                     "core.hf_text_encoder": {"weights": "none"},
                     **{t: {"weights": "none"} for t in (
                         "vision.detector", "vision.instance_segmenter",
                         "vision.keypoint_detector", "vision.hf_detector", "vision.segmenter",
                         "vision.hf_segmenter", "vision.hf_mask2former", "vision.timm_backbone",
                         "vision.hf_image_encoder", "audio.hf_encoder")}}

# blocks that only make sense inside a larger design: (input shape, dtype, layers before,
# layers after) — e.g. a VAE bottleneck must be decoded back to the input size
CONTEXTS: dict[str, tuple] = {
    "gen.reparameterize": ([32], "float32", [], [("core.dense", {"units": 32})]),
    "gen.causal_transformer": ([16], "int64",
                               [("core.embedding", {"num_embeddings": 256,
                                                    "embedding_dim": 32})],
                               [("core.dense", {"units": 256})]),
    # flows need at least one learnable transform next to ActNorm / permutations
    "flow.actnorm": ([8], "float32", [], [("flow.affine_coupling", {})]),
    "flow.permute": ([8], "float32", [], [("flow.affine_coupling", {})]),
    # Glow's image layers, next to a convolutional coupling
    "flow.invconv1x1": ([4, 8, 8], "float32", [], [("flow.conv_coupling", {})]),
    "flow.squeeze": ([4, 8, 8], "float32", [], [("flow.conv_coupling", {})]),
    "flow.split": ([4, 8, 8], "float32", [("flow.conv_coupling", {})],
                   [("flow.conv_coupling", {})]),
    # Fold reads the columns an Unfold makes (3.5)
    "core.fold": ([4, 8, 8], "float32", [("core.unfold", {"kernel_size": 3})], []),
    # a small RWKV-7 stack: its width must divide into heads (3.12)
    "seq.rwkv7": ([12, 32], "float32", [], []),
    # the YOLO-style head reads features from a backbone
    "vision.yolo_head": ([3, 64, 64], "float32",
                         [("core.conv2d", {"out_channels": 8, "kernel_size": 3, "stride": 2,
                                           "padding": 1})], []),
}

CANDIDATE_SHAPES = ([32], [8, 16], [4, 16, 16], [4, 8, 8, 8], [8, 16, 16], [1, 16],
                    [3, 64, 64], [3, 224, 224], [1, 16000], [48, 1])


def _graph_for(type_id: str, shape: list[int], params: dict | None = None,
               context: tuple | None = None) -> Graph:
    defn = get_registry().get(type_id)
    g = Graph(name=f"case_{type_id.replace('.', '_')}")
    dtype = context[1] if context else ("int64" if defn.input_dtype == "int" else "float32")
    g.add_node(NodeInstance("inp", "core.input",
                            {"shape": ",".join(map(str, shape)), "dtype": dtype}))
    before, after = (context[2], context[3]) if context else ([], [])
    prev = "inp"
    for i, (t, p) in enumerate(before):
        g.add_node(NodeInstance(f"pre{i}", t, dict(p)))
        g.add_edge(Edge(prev, "out", f"pre{i}", "in"))
        prev = f"pre{i}"
    g.add_node(NodeInstance("blk", type_id, dict(params or {})))
    for port in defn.inputs:
        g.add_edge(Edge(prev, "out", "blk", port.name))
    prev, port_out = "blk", defn.outputs[0].name
    for i, (t, p) in enumerate(after):
        g.add_node(NodeInstance(f"post{i}", t, dict(p)))
        g.add_edge(Edge(prev, port_out, f"post{i}", "in"))
        prev, port_out = f"post{i}", "out"
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_edge(Edge(prev, port_out, "out", "in"))
    return g


def block_cases(extra_params: dict | None = None):
    """Yield (type_id, input_shape, params, graph) for every NN block that
    accepts at least one candidate shape."""
    reg = get_registry()
    for defn in reg.all():
        if defn.shape_fn is None or defn.builder is not None:
            continue
        if defn.type_id in ("core.input", "core.output"):
            continue
        if DEFAULT_OVERRIDES.get(defn.type_id, {}) is None:
            continue  # needs network downloads (covered by tests that opt in)
        if missing_requirements(defn):
            continue  # optional package not installed (the extra's CI job covers it)
        variants = [DEFAULT_OVERRIDES.get(defn.type_id, {})] + list(
            (extra_params or {}).get(defn.type_id, []))
        context = CONTEXTS.get(defn.type_id)
        for params in variants:
            for shape in ([context[0]] if context else CANDIDATE_SHAPES):
                g = _graph_for(defn.type_id, shape, params, context)
                if not [i for i in g.validate() if i.severity == "error"]:
                    yield defn.type_id, shape, params, g
                    break


def coverage(extra_params: dict | None = None) -> dict[str, dict[str, list[str]]]:
    """family (type-id prefix) -> status -> block ids. Statuses: ``run`` (the harness
    executes a case), ``needs_extra`` (an optional package is missing here; its CI job
    runs it), ``download`` (needs pretrained weights), ``no_case`` (no candidate input
    shape is valid: add a CONTEXTS entry) and ``config`` (data / training / evaluation /
    graph-model blocks without generated layer code: their family tests cover them)."""
    cases = {type_id for type_id, *_rest in block_cases(
        VARIANTS if extra_params is None else extra_params)}
    out: dict[str, dict[str, list[str]]] = {}
    for defn in get_registry().all():
        if defn.type_id in ("core.input", "core.output"):
            continue
        family = defn.type_id.split(".")[0]
        if defn.shape_fn is None or defn.builder is not None:
            status = "config"
        elif defn.type_id in cases:
            status = "run"
        elif DEFAULT_OVERRIDES.get(defn.type_id, {}) is None:
            status = "download"
        elif missing_requirements(defn):
            status = "needs_extra"
        else:
            status = "no_case"
        out.setdefault(family, {}).setdefault(status, []).append(defn.type_id)
    return out


def print_coverage(report: dict) -> None:
    columns = ("run", "needs_extra", "download", "no_case", "config")
    print(f"{'family':<12}" + "".join(f"{c:>13}" for c in columns))
    totals = dict.fromkeys(columns, 0)
    for family in sorted(report):
        row = report[family]
        print(f"{family:<12}" + "".join(f"{len(row.get(c, [])):>13}" for c in columns))
        for c in columns:
            totals[c] += len(row.get(c, []))
    print(f"{'total':<12}" + "".join(f"{totals[c]:>13}" for c in columns))
    layers = totals["run"] + totals["needs_extra"] + totals["download"] + totals["no_case"]
    print(f"layer blocks run by the harness here: {totals['run']}/{layers}")
    for family in sorted(report):
        for type_id in report[family].get("no_case", []):
            print(f"  no case: {type_id}")


def _sample(shape, int_input: bool, vocab: int = 10):
    import torch

    if int_input:
        return torch.randint(0, vocab, (2, *shape))
    return torch.randn(2, *shape)


def run_torch(graph: Graph) -> tuple[int, ...]:
    import torch

    from ai_made_easy.core.codegen import generate

    code = generate(graph, "pytorch")
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(code, f"<{graph.name}_pytorch>", "exec"), ns)
    from ai_made_easy.core.codegen import class_name_for

    model = ns[class_name_for(graph.name)]()
    model.eval()
    shapes = graph.infer_shapes()
    in_shape = shapes["inp"]
    int_input = graph.nodes["inp"].params.get("dtype") == "int64"
    x = _sample(in_shape, int_input)
    with torch.no_grad():
        if "graph_context(x, edge_index, batch)" in code:   # graph models: a real edge
            out = model(x, torch.tensor([[0, 1], [1, 0]]))
        else:
            out = model(x)
    return tuple(out.shape[1:])


def run_keras(graph: Graph) -> tuple[int, ...]:
    import numpy as np

    from ai_made_easy.core.codegen import generate, keras_output_shape

    code = generate(graph, "keras")
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(code, f"<{graph.name}_keras>", "exec"), ns)
    model = ns["build_model"]()
    in_shape = tuple(int(d) for d in model.inputs[0].shape[1:])
    if graph.nodes["inp"].params.get("dtype") == "int64":
        x = np.random.randint(0, 10, size=(2, *in_shape))
    else:
        x = np.random.randn(2, *in_shape).astype("float32")
    out = model(x)
    got = tuple(int(d) for d in out.shape[1:])
    return got, tuple(keras_output_shape(graph))


def main() -> int:
    verbose = "-v" in sys.argv
    do_keras = "--keras" in sys.argv
    if "--coverage" in sys.argv:
        report = coverage()
        print_coverage(report)
        return 1 if any(r.get("no_case") for r in report.values()) else 0
    fails = 0
    total = 0
    for type_id, shape, params, g in block_cases(VARIANTS):
        total += 1
        want = tuple(g.infer_shapes()["out"])
        try:
            got = run_torch(g)
            ok = got == want
            msg = "" if ok else f"shape {got} != designer {want}"
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {str(exc)[:160]}"
        line = f"torch  {type_id:32} in={shape} {params or ''} {msg}"
        if not ok:
            fails += 1
            print("FAIL", line)
        elif verbose:
            print("ok  ", line)
        if do_keras and get_registry().get(type_id).supports("keras"):
            try:
                got_k, want_k = run_keras(g)
                ok = got_k == want_k
                msg = "" if ok else f"shape {got_k} != expected {want_k}"
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{type(exc).__name__}: {str(exc)[:160]}"
            if not ok and "no Keras equivalent" in msg:
                ok = True  # a configuration Keras cannot express, reported cleanly
            line = f"keras  {type_id:32} in={shape} {params or ''} {msg}"
            if not ok:
                fails += 1
                print("FAIL", line)
            elif verbose:
                print("ok  ", line)
    print(f"{total} block cases, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
