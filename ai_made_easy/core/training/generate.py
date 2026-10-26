"""Render complete, runnable training scripts from a TrainingSpec."""
from __future__ import annotations

import re

from typing import Callable

from jinja2 import Environment, StrictUndefined, Undefined

from ai_made_easy.core.codegen import (
    CodegenError,
    emit_graph,
    require_framework,
    template_context,
)
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.training import catalog as cat
from ai_made_easy.core.training import data_catalog as dcat
from ai_made_easy.core.training.data_template import DATA_TEMPLATE
from ai_made_easy.core.training.inference_template import (
    INFERENCE_COMMON,
    INFERENCE_KERAS,
    INFERENCE_TORCH,
    INFERENCE_TORCH_IMAGES,
)
from ai_made_easy.core.training.keras_template import KERAS_HEADER, KERAS_TRAINING
from ai_made_easy.core.training.metrics_code import metrics_code
from ai_made_easy.core.training.spec import TrainingSpec, collect_spec, dataset_comment
from ai_made_easy.core.training.torch_template import (
    TORCH_ARRAY_LOADERS,
    TORCH_HEADER,
    TORCH_IMAGE_DATA,
    TORCH_MAIN,
    TORCH_TRAINING,
)

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                   undefined=Undefined)
_env.filters["repr"] = repr
_strict = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                      undefined=StrictUndefined)



def _csv_list(value) -> list[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def _task_label(spec: TrainingSpec) -> str:
    from ai_made_easy.core.tasks import get_task

    label = get_task(spec.task).label.lower()
    if spec.task in ("multiclass", "multilabel"):
        return f"{label} ({spec.num_outputs} classes)"
    if spec.task == "regression":
        return f"{label} ({spec.num_outputs} output{'s' if spec.num_outputs > 1 else ''})"
    return label


def _requirements(spec: TrainingSpec, framework: str, torchvision: bool) -> str:
    reqs = ["torch", "numpy"] if framework == "pytorch" else ["keras>=3", "numpy"]
    block = spec.dataset["block"]
    if torchvision:   # JPEG and Gaussian noise augmentation (3.6) arrived in torchvision 0.21
        newer = {"prep.random_jpeg", "prep.gaussian_noise"} & {t for t, _ in spec.image_train}
        reqs.append("torchvision>=0.21" if newer else "torchvision")
    if block in ("data.csv", "data.text_csv", "data.timeseries_csv"):
        reqs.append("pandas")
    if block == "data.image_folder" or (spec.is_classification and torchvision) \
            or ((spec.gradcam or spec.attention or spec.saliency) and framework == "pytorch"):
        reqs.append("pillow")
    if block == "data.sklearn":
        reqs.append("scikit-learn")
    if block == "data.huggingface":
        reqs.append("datasets")
    tok = spec.steps.get("prep.tokenize")
    if tok and tok.get("method") == "huggingface":
        reqs.append("transformers")
    if spec.explain and spec.explain["method"] == "shap" and framework == "pytorch":
        reqs.append("shap")
    return ", ".join(dict.fromkeys(reqs))


def _default_tokenizer(spec: TrainingSpec, graph: Graph) -> dict:
    vocab = 20000
    for node in graph.nodes.values():
        if node.type_id == "core.embedding":
            vocab = int(node.resolved_params()["num_embeddings"])
            break
    return {"method": "word", "max_length": spec.input_shape[0], "vocab_size": vocab,
            "min_freq": 1, "hf_tokenizer": "bert-base-uncased"}


def _image_channel_fix(spec: TrainingSpec) -> str | None:
    want = spec.input_shape[0]
    explicit = any(t == "prep.grayscale" for t, _ in spec.image_always)
    if explicit:
        return None
    block = spec.dataset["block"]
    native = (dcat.TORCHVISION[spec.dataset["dataset"]][1][0] if block == "data.torchvision"
              else (1 if spec.dataset.get("grayscale") else 3))
    if native != want:
        return f"v2.Grayscale(num_output_channels={want})"
    if block == "data.image_folder" and want == 1:
        return "v2.Grayscale(num_output_channels=1)"
    return None


def _synthetic_table_code() -> str:
    from ai_made_easy.core.tabular.synthetic import SYNTHETIC_TABLE_CODE

    return SYNTHETIC_TABLE_CODE


def _raw_columns(graph: Graph) -> list[int]:
    """Input positions holding categorical codes (tabular models embed them, so scaling
    must leave them as integers)."""
    from ai_made_easy.core.tabular.blocks import CATEGORICAL_BLOCKS, ints

    return sorted({i for n in graph.nodes.values() if n.type_id in CATEGORICAL_BLOCKS
                   for i in ints(n.resolved_params()["categorical"])})


def _data_ctx(spec: TrainingSpec, graph: Graph, framework: str) -> dict:
    d = dict(spec.dataset)
    steps = dict(spec.steps)
    text = spec.modality == "text"
    tok = steps.get("prep.tokenize") or (_default_tokenizer(spec, graph) if text else None)
    feat = steps.get("prep.audio_features") or (
        {"kind": "waveform", "n_fft": 512, "hop_length": 160, "n_mels": 64, "n_mfcc": 20,
         "log_scale": True} if spec.modality == "audio" else None)
    image_keras = framework == "keras" and spec.modality == "image"
    if image_keras:
        steps.pop("prep.normalize", None)  # normalization lives inside the Keras model
    norm = steps.get("prep.normalize")
    ctx = {
        "d": d, "steps": steps, "comment": dataset_comment(spec), "task": spec.task,
        "classification": spec.is_classification, "n_outputs": spec.num_outputs,
        "n_targets": spec.num_outputs,
        "feature_columns": _csv_list(d.get("feature_columns")),
        "ts_targets": _csv_list(d.get("target_columns")),
        "drop_columns": _csv_list((steps.get("prep.drop_columns") or {}).get("columns")),
        "one_hot_columns": _csv_list((steps.get("prep.one_hot") or {}).get("columns")),
        "ordinal_columns": _csv_list((steps.get("prep.ordinal_encode") or {}).get("columns")),
        "max_categories": int((steps.get("prep.one_hot") or {}).get("max_categories", 50)),
        "table_steps": d["block"] in ("data.csv", "data.synthetic_table"),
        "synthetic_table_code": _synthetic_table_code(),
        "raw_columns": _raw_columns(graph),
        "text": text, "tok": tok,
        "audio": spec.modality == "audio", "feat": feat,
        "needs_wav": spec.modality == "audio",
        "sample_rate": d.get("sample_rate", 16000),
        "timeseries": spec.modality == "timeseries",
        "ts_channels_first": str(d.get("layout", "")).startswith("channels"),
        "target_scaling": bool(norm or steps.get("prep.minmax")),
        "chronological": spec.modality == "timeseries",
        "stratify": bool(spec.split.get("stratify")) and spec.task in ("multiclass", "binary"),
        "shuffle": bool(spec.split.get("shuffle", True)),
        "per_channel": len(spec.input_shape) >= 3 and spec.modality in ("image", "array"),
        "norm_mean": [float(v) for v in _csv_list(norm.get("mean"))] if norm else None,
        "norm_std": [float(v) for v in _csv_list(norm.get("std"))] if norm else None,
        "y_dtype": "np.int64" if spec.task in ("multiclass", "binary") else "np.float32",
        "sk_loader": dcat.SKLEARN[d["dataset"]][0] if d["block"] == "data.sklearn" else "",
        "keras_dataset": dcat.KERAS_DATASETS.get(d.get("dataset", ""), ""),
    }
    return ctx


def _common_ctx(graph: Graph, spec: TrainingSpec, framework: str) -> dict:
    plan = emit_graph(graph)
    require_framework(plan, framework)
    model_ctx = template_context(graph, plan)
    meta = cat.COMPONENTS[spec.loss["kind"]].meta
    metrics = [(m, p, cat.COMPONENTS[m].meta["key"]) for m, p in spec.metrics]
    balance = (spec.steps.get("prep.class_balance") or {}).get("strategy")
    if balance == "class weights" and spec.loss["kind"] not in (
            "train.loss_cross_entropy", "train.loss_focal", "train.loss_bce_logits"):
        raise CodegenError("class weights need Cross-Entropy, Focal or BCE (logits) loss; "
                           "use the 'oversample' strategy instead")
    torchvision = (framework == "pytorch" and spec.modality == "image")
    ctx = {
        **model_ctx,
        "spec": spec, "plan": plan,
        "task_label": _task_label(spec),
        "filename": f"{spec.name}_train_{framework}.py",
        "requirements": _requirements(spec, framework, torchvision),
        "input_shape_literal": tuple(spec.input_shape),
        "keras_transpose": plan.keras_input_transpose,
        "loss_input": meta["input"],
        "loss_helper": meta.get("helper"),
        "balance": balance,
        "metrics_code": metrics_code(spec.task, metrics),
        "torchvision_pipeline": torchvision,
        "image_inputs": spec.modality == "image",
        "image_explicit_size": False,
        **_data_ctx(spec, graph, framework),
    }
    norm = spec.steps.get("prep.normalize")
    ctx["norm_fixed"] = bool(norm and norm["mode"] == "fixed" and spec.modality == "image")
    ctx["norm_fit"] = bool(norm and norm["mode"] == "fit" and spec.modality == "image")
    return ctx


# ------------------------------------------------------------------ PyTorch

def _torch_ctx(graph: Graph, spec: TrainingSpec) -> dict:
    ctx = _common_ctx(graph, spec, "pytorch")
    loss_expr = cat.render(spec.loss["kind"], spec.loss, "pytorch")
    if loss_expr is None:
        raise CodegenError(f"{cat.COMPONENTS[spec.loss['kind']].name} has no PyTorch form")
    sched_step = None
    sched_expr = None
    if spec.scheduler:
        sched_expr = cat.render(spec.scheduler["kind"], spec.scheduler, "pytorch")
        sched_step = cat.COMPONENTS[spec.scheduler["kind"]].meta["step"]
    from ai_made_easy.core.bayes.codegen import probabilistic_context

    ctx.update(
        prob=probabilistic_context(graph),
        aux_loss=any((n.definition().meta or {}).get("aux_loss")
                     for n in graph.nodes.values()),
        loss_expr=loss_expr,
        optimizer_expr=cat.render(spec.optimizer["kind"], spec.optimizer, "pytorch"),
        ema=bool(float(spec.trainer.get("ema_decay") or 0)),
        swa=int(spec.trainer.get("swa_start") or 0),
        metric_loss=spec.metric_loss,
        two_views=cat.COMPONENTS[spec.loss["kind"]].meta.get("views") == 2,
        sam=bool(cat.COMPONENTS[spec.optimizer["kind"]].meta.get("sam")),
        closure=bool(cat.COMPONENTS[spec.optimizer["kind"]].meta.get("closure")),
        schedule_free=bool(cat.COMPONENTS[spec.optimizer["kind"]].meta.get("schedule_free")),
        scheduler_expr=sched_expr, scheduler_step=sched_step,
        uses_functional=ctx["uses_functional"] or ctx["loss_helper"] == "FocalLoss",
        grad_cam_fn=GRAD_CAM_FN,
        pairs=spec.pairs,
    )
    if spec.pairs == "ranking" and spec.embed_dim != 1:
        raise CodegenError("Margin Ranking on labelled pairs scores each item: end the "
                           f"model with one output unit (it has {spec.embed_dim})")
    if ctx["two_views"] and not ctx["torchvision_pipeline"]:
        raise CodegenError(f"{cat.COMPONENTS[spec.loss['kind']].name} learns from two "
                           "augmented views of each image: use an image dataset")
    if ctx["ema"] and ctx["swa"]:
        raise CodegenError("use one kind of weight averaging on the Trainer: EMA or SWA")
    ctx["gpus"] = int(spec.trainer.get("gpus") or 1)
    for kind in (spec.optimizer["kind"], spec.loss["kind"]):
        helper = cat.COMPONENTS[kind].meta.get("torch_helper")
        if helper:
            ctx["torch_helpers"] = [*ctx["torch_helpers"], cat.TRAIN_HELPERS[helper]]
    image_train, after_tensor = [], []
    for tid, params in spec.image_train:
        blk = dcat.BLOCKS[tid]
        (after_tensor if blk.meta.get("after_tensor") else image_train).append(blk.torch(params))
    ctx["image_always"] = [dcat.BLOCKS[t].torch(p) for t, p in spec.image_always]
    ctx["image_train"] = image_train
    ctx["image_after_tensor"] = after_tensor
    mix = spec.steps.get("prep.mixup_cutmix")
    ctx["mixup"] = None
    if mix:
        a = mix["alpha"]
        ctx["mixup"] = {
            "mixup": f"v2.MixUp(num_classes=NUM_CLASSES, alpha={a})",
            "cutmix": f"v2.CutMix(num_classes=NUM_CLASSES, alpha={a})",
            "both": (f"v2.RandomChoice([v2.MixUp(num_classes=NUM_CLASSES, alpha={a}), "
                     f"v2.CutMix(num_classes=NUM_CLASSES, alpha={a})])"),
        }[mix["mode"]]
    if spec.modality == "image":
        ctx["image_channels_fix"] = _image_channel_fix(spec)
        # only always-applied transforms fix the evaluation size; random crops are
        # training-only, so evaluation / serving images still need resizing
        explicit_size = any(t in ("prep.resize", "prep.center_crop")
                            for t, _ in spec.image_always)
        ctx["image_explicit_size"] = explicit_size
        ctx["image_resize_fix"] = (
            f"v2.Resize(({spec.input_shape[1]}, {spec.input_shape[2]}), antialias=True)"
            if spec.dataset["block"] == "data.image_folder" and not explicit_size else None)
        if spec.dataset["block"] == "data.torchvision":
            tv = dcat.TORCHVISION[spec.dataset["dataset"]]
            ctx["tv_class"], ctx["tv_split_kw"] = tv[0], tv[3]
    return ctx


def _render_torch(graph: Graph, spec: TrainingSpec, ctx: dict, main: str,
                  inference: bool = True) -> str:
    parts = [_env.from_string(TORCH_HEADER).render(**ctx)]
    if ctx["torchvision_pipeline"]:
        parts.append(_env.from_string(TORCH_IMAGE_DATA).render(**ctx))
    else:
        parts.append(_env.from_string(DATA_TEMPLATE).render(**ctx))
        parts.append(_env.from_string(TORCH_ARRAY_LOADERS).render(**ctx))
    parts.append(_env.from_string(TORCH_TRAINING).render(**ctx))
    if ctx.get("prob"):
        from ai_made_easy.core.bayes.template import PROBABILISTIC

        parts.append(_env.from_string(PROBABILISTIC).render(**ctx))
    if inference:
        parts.append(_env.from_string(INFERENCE_COMMON).render(**ctx))
        if ctx["torchvision_pipeline"]:
            parts.append(_env.from_string(INFERENCE_TORCH_IMAGES).render(**ctx))
        parts.append(_env.from_string(INFERENCE_TORCH).render(**ctx))
    parts.append(_env.from_string(main).render(**ctx))
    return "".join(parts)


# ------------------------------------------------------------------- Keras

def _with_kwargs(expr: str, **kwargs) -> str:
    extra = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    return f"{expr[:-1]}, {extra})" if extra else expr


def _keras_ctx(graph: Graph, spec: TrainingSpec) -> dict:
    ctx = _common_ctx(graph, spec, "keras")
    unsupported = []
    loss_expr = cat.render(spec.loss["kind"], spec.loss, "keras")
    if loss_expr is None:
        unsupported.append(cat.COMPONENTS[spec.loss["kind"]].name)
    opt_expr = cat.render(spec.optimizer["kind"], spec.optimizer, "keras")
    if opt_expr is None:
        unsupported.append(cat.COMPONENTS[spec.optimizer["kind"]].name)
    if int(spec.trainer.get("swa_start") or 0):
        unsupported.append("SWA weight averaging (use EMA in Keras)")
    torch_only = [label for label, used in (
        ("gradient checkpointing", bool(spec.trainer.get("gradient_checkpointing"))),
        ("several GPUs", int(spec.trainer.get("gpus") or 1) > 1)) if used]
    if torch_only:
        spec.warnings.append(f"the Trainer's {', '.join(torch_only)} apply to PyTorch: the "
                             "Keras script trains without")
    if opt_expr is None:
        pass
    elif float(spec.trainer.get("ema_decay") or 0):  # Keras keeps the average itself
        opt_expr = (f"{opt_expr[:-1]}, use_ema=True, "
                    f"ema_momentum={float(spec.trainer['ema_decay'])})")
    sched = None
    if spec.scheduler:
        comp = cat.COMPONENTS[spec.scheduler["kind"]]
        rendered = cat.render(spec.scheduler["kind"], spec.scheduler, "keras")
        if rendered is None:
            unsupported.append(comp.name)
        elif comp.meta["step"] == "metric":
            sched = rendered
        else:
            sched = f"keras.callbacks.LearningRateScheduler({rendered})"
    if spec.kfold:
        unsupported.append("K-Fold cross-validation")
    if spec.steps.get("prep.mixup_cutmix"):
        unsupported.append("MixUp / CutMix")
    if spec.steps.get("prep.spec_augment"):
        unsupported.append("SpecAugment")
    if spec.modality == "image" and spec.dataset["block"] == "data.torchvision" \
            and spec.dataset["dataset"] not in dcat.KERAS_DATASETS:
        unsupported.append(f"the {spec.dataset['dataset']} dataset (Keras provides "
                           + ", ".join(dcat.KERAS_DATASETS) + ")")
    augment = []
    preprocess = []
    for bucket, items in ((preprocess, spec.image_always), (augment, spec.image_train)):
        for tid, params in items:
            blk = dcat.BLOCKS[tid]
            layer = blk.keras(params) if blk.keras else None
            if layer is None:
                unsupported.append(blk.name)
            else:
                bucket.append(layer)
    if unsupported:
        raise CodegenError("not available in Keras 3: " + ", ".join(dict.fromkeys(unsupported)))
    meta = cat.COMPONENTS[spec.loss["kind"]].meta
    one_hot = bool(meta.get("keras_one_hot") and meta["keras_one_hot"](spec.loss))
    tr = spec.trainer
    opt_expr = _with_kwargs(
        opt_expr,
        clipnorm=tr["grad_clip_norm"] if float(tr["grad_clip_norm"]) > 0 else None,
        gradient_accumulation_steps=(tr["accumulation_steps"]
                                     if int(tr["accumulation_steps"]) > 1 else None))
    norm = spec.steps.get("prep.normalize")
    image_norm = None
    if norm and spec.modality == "image":
        if norm["mode"] == "fixed":
            mean = [float(v) for v in _csv_list(norm["mean"])]
            var = [float(v) ** 2 for v in _csv_list(norm["std"])]
            image_norm = f"layers.Normalization(axis=-1, mean={mean}, variance={var})"
        else:
            image_norm = ("layers.Normalization(axis=-1, mean=flat.mean(axis=0), "
                          "variance=flat.var(axis=0))")
    ctx.update(loss_expr=loss_expr, optimizer_expr=opt_expr, scheduler_callback=sched,
               one_hot=one_hot, augment=preprocess + augment, image_norm=image_norm,
               uses_ops=ctx["uses_ops"] or "ops." in (loss_expr or ""))
    return ctx


def _render_keras(ctx: dict) -> str:
    training = KERAS_TRAINING.replace(
        "    normalize = layers.Normalization(axis=-1, mean=flat.mean(axis=0), "
        "variance=flat.var(axis=0))",
        "    normalize = {{ image_norm }}")
    body = _env.from_string(training).render(**ctx)
    tail = '\n\nif __name__ == "__main__":'
    head, _, rest = body.rpartition(tail)
    return "".join([
        _env.from_string(KERAS_HEADER).render(**ctx),
        _env.from_string(DATA_TEMPLATE).render(**ctx),
        head,
        _env.from_string(INFERENCE_COMMON).render(**ctx),
        _env.from_string(INFERENCE_KERAS).render(**ctx),
        tail, rest,
    ])


# ---------------------------------------------------------------- public

def _validate(graph: Graph) -> None:
    if errors := [i for i in graph.validate() if i.severity == "error"]:
        raise CodegenError("graph has validation errors:\n"
                           + "\n".join(f"  - {e}" for e in errors))


# trainer kind -> renderer(graph, spec, framework); the training loop a task needs.
# Renderers registered with needs_spec=False get spec=None and read the graph
# themselves (detection / segmentation pipelines have their own data formats).
TrainerRenderer = Callable[[Graph, "TrainingSpec | None", str], str]
TRAINERS: dict[str, TrainerRenderer] = {}
_GRAPH_TRAINERS: set[str] = set()


def own_loop_requirements(model_code: str, extra: tuple[str, ...] = ()) -> str:
    """The "Needs:" line of a task-specific training script (deploy packages read it)."""
    reqs = ["torch", "numpy", "pillow"]
    for module in ("torchvision", "transformers", "timm", "torch_geometric"):
        if re.search(rf"^\s*(import|from) {module}\b", model_code, re.M):
            reqs.append(module)
    return ", ".join(dict.fromkeys([*reqs, *extra]))


def register_trainer(kind: str, renderer: TrainerRenderer, needs_spec: bool = True) -> None:
    """Add a training loop for a trainer kind (adversarial, diffusion, rl, ...)."""
    from ai_made_easy.core.tasks import TRAINER_KINDS

    if kind not in TRAINER_KINDS:
        raise ValueError(f"unknown trainer kind {kind!r}; one of {TRAINER_KINDS}")
    TRAINERS[kind] = renderer
    if needs_spec:
        _GRAPH_TRAINERS.discard(kind)
    else:
        _GRAPH_TRAINERS.add(kind)


def generate_training(graph: Graph, framework: str) -> str:
    from ai_made_easy.core.classic.generate import generate_classic, is_classic
    from ai_made_easy.core.tasks import get_task

    from ai_made_easy.core.families import generator_for

    own = generator_for(graph, framework)
    if own is not None:
        _plain_optimizer(graph)
        return own(graph)
    if is_classic(graph) or framework == "sklearn":
        if not is_classic(graph):
            raise CodegenError("scikit-learn export needs a classic ML estimator block")
        return generate_classic(graph)
    _validate(graph)
    from ai_made_easy.core.tasks import task_of

    task = task_of(graph)
    if task is not None and task.trainer_kind in _GRAPH_TRAINERS:
        _plain_optimizer(graph)
        return TRAINERS[task.trainer_kind](graph, None, framework)
    spec = collect_spec(graph)
    kind = get_task(spec.task).trainer_kind
    if kind not in TRAINERS:
        raise CodegenError(f"{get_task(spec.task).label} needs the {kind} trainer, "
                           "which is not available")
    return TRAINERS[kind](graph, spec, framework)


def _plain_optimizer(graph: Graph) -> None:
    """SAM and L-BFGS step differently (two passes, or a closure): only the standard
    supervised loop knows how, so the family trainers refuse them with a clear message."""
    for node in graph.nodes.values():
        meta = cat.COMPONENTS[node.type_id].meta if node.type_id in cat.OPTIMIZER_IDS else {}
        if meta.get("sam") or meta.get("closure"):
            raise CodegenError(f"{cat.COMPONENTS[node.type_id].name} works with the standard "
                               "training loop (tables, images, text, audio, arrays); this "
                               "design's trainer steps the usual way: pick another optimizer")


def _supervised(graph: Graph, spec: TrainingSpec, framework: str) -> str:
    if framework == "pytorch":
        return _render_torch(graph, spec, {**_torch_ctx(graph, spec), "inspect": False},
                             TORCH_MAIN)
    if framework == "keras":
        return _render_keras(_keras_ctx(graph, spec))
    raise ValueError(f"unknown framework {framework!r}")


register_trainer("supervised", _supervised)


GRAD_CAM_FN = r'''
def grad_cam(model: nn.Module, x: torch.Tensor, layer: nn.Module, class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap for one input on the given convolution layer."""
    stored = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        stored["acts"] = out

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad()
        with torch.enable_grad():
            out = model(x.clone())
            out[0, class_idx].backward()
        acts = stored["acts"]
        weights = acts.grad.mean(dim=tuple(range(2, acts.ndim)), keepdim=True)
        cam = torch.relu((weights * acts).sum(1, keepdim=True))
        if cam.ndim == 4:
            cam = torch.nn.functional.interpolate(cam, size=x.shape[2:], mode="bilinear",
                                                  align_corners=False)
        cam = cam[0, 0]
        return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-9)).detach().cpu().numpy()
    finally:
        handle.remove()


'''


INSPECT_MAIN = r'''

{% if not spec.gradcam %}
{{ grad_cam_fn }}{% endif %}
def feature_grid(maps: torch.Tensor, count: int = 16) -> np.ndarray:
    maps = maps[0][:count]
    maps = torch.nn.functional.interpolate(maps.unsqueeze(1), size=(40, 40), mode="bilinear",
                                           align_corners=False)[:, 0]
    lo = maps.amin(dim=(1, 2), keepdim=True)
    hi = maps.amax(dim=(1, 2), keepdim=True)
    maps = (maps - lo) / (hi - lo + 1e-9)
    side = max(int(len(maps) ** 0.5), 1)
    rows = []
    for r in range(0, len(maps), side):
        chunk = list(maps[r:r + side])
        while len(chunk) < side:
            chunk.append(chunk[-1])
        rows.append(torch.cat(chunk, dim=1))
    return torch.cat(rows, dim=0).numpy()


def main() -> None:
    """Load the trained checkpoint and explain one test prediction."""
    _, _, test_loader = make_loaders()
    model = {{ spec.class_name }}()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    args = sys.argv[1:]
    image_path = None
    if args and args[0] == "--image":
        from PIL import Image

        image_path = args[1]
        mode = "L" if INPUT_SHAPE[0] == 1 else "RGB"
        img = Image.open(image_path).convert(mode).resize((INPUT_SHAPE[2], INPUT_SHAPE[1]))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr[None] if arr.ndim == 2 else arr.transpose(2, 0, 1))
{% if norm_fixed %}
        x = (x - torch.tensor(NORM_MEAN).view(-1, 1, 1)) / torch.tensor(NORM_STD).view(-1, 1, 1)
{% endif %}
    else:
        idx = int(args[0]) if args else 0
        x, _ = test_loader.dataset[idx % len(test_loader.dataset)]
    x = x.unsqueeze(0).float()
    convs = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d))]
    feats = {}
    hooks = []
    if convs:
        hooks.append(convs[0].register_forward_hook(
            lambda _m, _i, o: feats.__setitem__("first", o.detach())))
    with torch.no_grad():
        out = model(x)
    for h in hooks:
        h.remove()
    probs = to_scores(out.numpy())[0]
    single = probs.shape[0] == 1
    top = np.argsort(probs)[::-1][:3]
    result = {"image": image_path, "single": single,
              "top": [{"class": int(c), "prob": round(float(probs[c]), 4)} for c in top]}
    np.save("input.npy", x[0].numpy())
    if feats.get("first") is not None and feats["first"].ndim == 4:
        np.save("feats.npy", feature_grid(feats["first"]))
    if convs and not single and isinstance(convs[-1], nn.Conv2d):
        np.save("cam.npy", grad_cam(model, x, convs[-1], int(top[0])))
    Path("inspect.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("inspect:", result)


if __name__ == "__main__":
    main()
'''


def generate_inspect(graph: Graph) -> str:
    """Inspect script: Grad-CAM + first-layer features for one test example."""
    _validate(graph)
    spec = collect_spec(graph)
    if not spec.is_classification:
        raise CodegenError("inspection needs a classification model")
    return _render_torch(graph, spec, {**_torch_ctx(graph, spec), "inspect": True},
                         INSPECT_MAIN, inference=False)
