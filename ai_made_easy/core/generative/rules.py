"""Design rules for generative models (registered as lints; Quick Fixes in core.fixes).

"set <param> to <value>" and "Set the Input shape to '…'" phrasing becomes a
one-click fix on the flagged block.
"""
from __future__ import annotations

from ai_made_easy.core.generative.blocks import (
    BYTE_VOCAB,
    CONFIG_BLOCKS,
    GENERATIVE_DATA,
    SEQ2SEQ_VOCAB,
)
from ai_made_easy.core.generative.tasks import dataset_of, generative_kind
from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule

SHAPES_CLASSES = {"shapes": 3, "blobs": 1}
# layers that let a position see later positions (break next-token prediction)
NON_CAUSAL = {"core.transformer_encoder", "core.transformer_decoder",
              "core.multihead_attention"}


def image_shape(node) -> list[int] | None:  # noqa: ANN001
    """[C, H, W] a generative image dataset produces when fixed by the dataset (generated
    images); folders and benchmarks are resized to the design."""
    if node is None or node.type_id != "data.synthetic_images":
        return None
    p = node.resolved_params()
    return [int(p["channels"]), int(p["image_size"]), int(p["image_size"])]


def pair_limits(node) -> tuple[int, int] | None:  # noqa: ANN001
    """(longest source, longest target) bytes of generated pairs."""
    if node is None or node.type_id != "data.synthetic_pairs":
        return None
    p = node.resolved_params()
    n = int(p["max_length"])
    if p["task"] == "digits":
        k = max(1, n // 5)
        return k, 6 * k - 1   # "five" + space per digit, at most 5 letters
    return n, n


def generative_rules(ctx: LintContext) -> list:
    kind = generative_kind(ctx.graph)
    data = dataset_of(ctx.graph)
    if kind is None:
        if data is not None and data.type_id == "data.synthetic_images":
            return [_issue("warning", "Synthetic Images trains generative models: add a "
                                      "Reparameterize block (VAE), a Discriminator (GAN) or a "
                                      "Noise Scheduler with a Diffusion U-Net",
                           data.instance_id)]
        return []
    conflicts = _conflicts(ctx, kind)
    if conflicts:
        return conflicts   # the other checks assume one consistent kind of model
    rules = {"vae": _vae, "adversarial": _gan, "diffusion": _diffusion,
             "text_to_image": _text_to_image,
             "language_model": _language_model, "seq2seq": _seq2seq}
    return rules[kind](ctx, data)


def _conflicts(ctx: LintContext, kind: str) -> list:
    types = {n.type_id for n in ctx.graph.nodes.values()}
    marks = [t for t in ("gen.reparameterize", *CONFIG_BLOCKS) if t in types]
    if len(marks) > 1:
        node = next(n for n in ctx.graph.nodes.values() if n.type_id == marks[1])
        return [_issue("error", "a design trains one kind of generative model: keep only one "
                                "of Reparameterize (VAE), Discriminator (GAN) and Noise "
                                "Scheduler (diffusion)", node.instance_id)]
    data = dataset_of(ctx.graph)
    want = {"vae": "images", "adversarial": "images", "diffusion": "images",
            "language_model": "text", "seq2seq": "pairs", "text_to_image": "images"}[kind]
    if data is not None and GENERATIVE_DATA.get(data.type_id, "images") != want:
        return [_issue("error", f"{_name(data)} does not fit this design ({want} needed)",
                       data.instance_id)]
    return []


def _text_to_image(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    sd = ctx.nodes_of("gen.sd_lora")[0]
    layers = [n for n in ctx.graph.nodes.values()
              if n.type_id in ("core.input", "core.output")
              or n.definition().shape_fn is not None]
    if layers:
        out.append(_issue("error", "Stable Diffusion LoRA is a complete pretrained model: "
                                   "remove the Input, Output and layer blocks",
                          layers[0].instance_id))
    for node in ctx.nodes_of("eval.fid", "eval.kid"):
        if node.resolved_params()["features"] != "pixels":
            out.append(_issue("error", f"{_name(node)} uses Inception features, which Stable "
                                       "Diffusion runs do not load: set features to pixels",
                              node.instance_id))
    for node in ctx.nodes_of("eval.inception_score"):
        out.append(_issue("warning", "the Inception Score is not computed for Stable "
                                     "Diffusion LoRA runs (use FID on pixels)",
                          node.instance_id))
    p = sd.resolved_params()
    if data is not None and data.type_id == "data.image_folder" and \
            "{class}" not in str(p["prompt"]):
        out.append(_issue("info", "every image gets the same caption: put {class} in the "
                                  "prompt to name each class folder", sd.instance_id))
    if p["weights"] == "pretrained":
        from ai_made_easy.core.zoo import download

        info = download("gen.sd_lora", p) or {"download_mb": 4500, "license": "see the model card"}
        out.append(_issue("info", f"downloads {p['base_model']} (about "
                                  f"{info['download_mb'] / 1000:.1f} GB, {info['license']} "
                                  "license) on the first run; a GPU with 8 GB or more is "
                                  "recommended", sd.instance_id))
    return out


def _input(ctx: LintContext):
    inputs = ctx.nodes_of("core.input")
    return (inputs[0], ctx.shapes.get(inputs[0].instance_id)) if inputs else (None, None)


def _shape_issue(node, have, want, source) -> list:  # noqa: ANN001
    if node is None or have is None or list(have) == list(want):
        return []
    text = ", ".join(map(str, want))
    return [_issue("error", f"{source} gives {list(want)} images but the Input is "
                            f"{list(have)}. Set the Input shape to '{text}'",
                   node.instance_id)]


def _vae(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    reps = ctx.nodes_of("gen.reparameterize")
    if len(reps) > 1:
        out.append(_issue("error", "a VAE has one bottleneck: keep one Reparameterize block",
                          reps[1].instance_id))
    inp, have = _input(ctx)
    want = image_shape(data)
    if want:
        out += _shape_issue(inp, have, want, _name(data))
    last = ctx.last_compute()
    if last is not None and have is not None:
        made = ctx.shapes.get(last.instance_id)
        if made is not None and list(made) != list(have):
            out.append(_issue("error", f"the decoder outputs {list(made)} but a VAE must "
                                       f"reconstruct its {list(have)} input: adjust the decoder "
                                       "layers", last.instance_id))
        if ctx.nodes_of("train.loss_bce") and last.type_id != "core.sigmoid":
            out.append(_issue("warning", "binary cross-entropy reconstruction needs outputs in "
                                         "[0, 1]: end the decoder with Sigmoid",
                              last.instance_id))
    return out


def _gan(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    inp, have = _input(ctx)
    if have is not None and len(have) != 1:
        out.append(_issue("error", f"the generator reads a latent noise vector but the Input "
                                   f"is {list(have)}. Set the Input shape to '128'",
                          inp.instance_id))
    last = ctx.last_compute()
    made = ctx.shapes.get(last.instance_id) if last is not None else None
    want = image_shape(data)
    if made is not None and len(made) != 3:
        out.append(_issue("error", f"the generator must output images [C, H, W], got "
                                   f"{list(made)}", last.instance_id))
    elif made is not None and want and list(made) != want:
        fix = ""
        if last.type_id == "gen.dcgan_generator":
            fix = (f": set image_size to {want[1]}" if made[1:] != want[1:]
                   else f": set channels to {want[0]}")
        out.append(_issue("error", f"the generator makes {list(made)} images but "
                                   f"{_name(data)} has {want}{fix}", last.instance_id))
    if last is not None and last.type_id not in ("core.tanh", "gen.dcgan_generator"):
        out.append(_issue("warning", "GAN images are scaled to [-1, 1]: end the generator "
                                     "with Tanh", last.instance_id))
    return out


def _diffusion(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    unets = ctx.nodes_of("gen.diffusion_unet")
    if not unets:
        sched = ctx.nodes_of("gen.noise_scheduler")
        return [_issue("error", "diffusion needs a denoiser: put a Diffusion U-Net between "
                                "the Input and the Output", sched[0].instance_id)]
    unet = unets[0]
    prods, cons = ctx.producers(unet.instance_id), ctx.consumers(unet.instance_id)
    if (prods and prods[0].type_id != "core.input") or \
            (cons and cons[0].type_id != "core.output"):
        out.append(_issue("error", f"{_name(unet)} takes the noisy image straight from the "
                                   "Input and feeds the Output: remove the blocks in between",
                          unet.instance_id))
    inp, have = _input(ctx)
    want = image_shape(data)
    if want:
        out += _shape_issue(inp, have, want, _name(data))
    k = int(unet.resolved_params()["num_classes"])
    if k and data is not None and data.type_id == "data.synthetic_images":
        classes = SHAPES_CLASSES[data.resolved_params()["kind"]]
        if k != classes:
            out.append(_issue("error", f"the dataset has {classes} class(es) but the U-Net is "
                                       f"conditioned on {k}: set num_classes to {classes}",
                              unet.instance_id))
    for sched in ctx.nodes_of("gen.noise_scheduler"):
        if not k and float(sched.resolved_params()["guidance"]) != 1.0:
            out.append(_issue("info", "guidance only applies to class-conditional U-Nets "
                                      "(num_classes > 0); sampling is unguided",
                              sched.instance_id))
    return out


def _language_model(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    inp, have = _input(ctx)
    if inp is not None:
        if have is not None and len(have) != 1:
            out.append(_issue("error", f"a language model reads a token sequence [L] but the "
                                       f"Input is {list(have)}. Set the Input shape to '128'",
                              inp.instance_id))
        if not str(inp.resolved_params().get("dtype", "")).startswith("int"):
            out.append(_issue("error", "tokens are integers: set the Input dtype to int64",
                              inp.instance_id))
    for node in ctx.chain:
        p = node.resolved_params()
        if node.type_id in NON_CAUSAL or (node.type_id in ("core.lstm", "core.gru")
                                          and p.get("bidirectional")) or \
                (node.type_id == "core.conv1d" and int(p.get("kernel_size", 1)) > 1):
            out.append(_issue("error", f"{_name(node)} lets each position see later tokens, so "
                                       "the model would copy the answer instead of predicting "
                                       "it: use Causal Transformer / GPT, or a one-directional "
                                       "RNN", node.instance_id))
        shape = ctx.shapes.get(node.instance_id)
        if node.type_id == "core.layer_norm" and p.get("normalized_dims") == "all" \
                and shape is not None and len(shape) >= 2:
            out.append(_issue("error", f"{_name(node)} normalizes over every position, so each "
                                       "one's output depends on later tokens: set "
                                       "normalized_dims to last", node.instance_id))
    bpe = next(iter(ctx.nodes_of("gen.bpe_tokenizer")), None)
    vocab = int(bpe.resolved_params()["vocab_size"]) if bpe is not None else BYTE_VOCAB
    what = (f"the BPE tokenizer makes {vocab} tokens" if bpe is not None
            else f"text is byte-level ({BYTE_VOCAB} tokens)")
    last = ctx.last_compute()
    made = ctx.shapes.get(last.instance_id) if last is not None else None
    if made is not None and have is not None:
        if len(made) != 2 or made[0] != have[0]:
            out.append(_issue("error", f"a language model outputs next-token scores for every "
                                       f"position [{have[0]}, {vocab}] but this one "
                                       f"outputs {list(made)}", last.instance_id))
        elif made[1] != vocab:
            param = "vocab_size" if last.type_id == "gen.gpt" else "units"
            out.append(_issue("error", f"{what} but the model scores {made[1]}: set {param} "
                                       f"to {vocab}", last.instance_id))
    for node in ctx.chain:   # the first lookup table must hold every token id
        if node.type_id == "core.embedding":
            n = int(node.resolved_params()["num_embeddings"])
            if n < vocab:
                out.append(_issue("error", f"{what} but {_name(node)} has {n} rows: set "
                                           f"num_embeddings to {vocab}", node.instance_id))
            break
    return out


def _seq2seq(ctx: LintContext, data) -> list:  # noqa: ANN001
    out = []
    models = ctx.nodes_of("gen.seq2seq_transformer")
    for bpe in ctx.nodes_of("gen.bpe_tokenizer"):
        out.append(_issue("warning", "the BPE Tokenizer applies to language models: "
                                     "sequence-to-sequence models read bytes, so it is ignored",
                          bpe.instance_id))
    if not models:
        node = data or ctx.last_compute()
        return [_issue("error", "sequence-to-sequence training needs a Seq2Seq Transformer "
                                "between the Input and the Output",
                       node.instance_id if node is not None else None)]
    model = models[0]
    p = model.resolved_params()
    if int(p["vocab_size"]) != SEQ2SEQ_VOCAB:
        out.append(_issue("error", f"byte-level text with PAD / BOS / EOS needs "
                                   f"{SEQ2SEQ_VOCAB} tokens: set vocab_size to {SEQ2SEQ_VOCAB}",
                          model.instance_id))
    inp, have = _input(ctx)
    if inp is not None and not str(inp.resolved_params().get("dtype", "")).startswith("int"):
        out.append(_issue("error", "tokens are integers: set the Input dtype to int64",
                          inp.instance_id))
    limits = pair_limits(data)
    if limits and have is not None:
        src, tgt = limits
        if have[0] < src:
            out.append(_issue("warning", f"sources have up to {src} bytes but the Input holds "
                                         f"{have[0]}: longer ones are cut. Set the Input shape "
                                         f"to '{src}'", inp.instance_id))
        if int(p["max_target_len"]) < tgt + 1:
            out.append(_issue("warning", f"targets have up to {tgt} bytes plus EOS: set "
                                         f"max_target_len to {tgt + 1}", model.instance_id))
    return out


register_rule(generative_rules)


def text_to_image_issues(graph) -> list:  # noqa: ANN001
    """Validator of the Stable Diffusion LoRA family (no layer graph to lint)."""
    return generative_rules(LintContext(graph, [], {}))
