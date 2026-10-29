"""Inference section appended to every generated training script.

After training, the script saves its fitted preprocessing (``INFERENCE_STATE``)
next to the model. ``load_predictor(folder)`` + ``infer(raw)`` then turn raw
inputs into predictions with exactly the training-time preprocessing — the
deployment package's server imports the training script for this.
"""

INFERENCE_COMMON = r'''

# ============================================================= inference
INFERENCE_FILE = "inference_state.pkl"
GROUP_THRESHOLDS: dict = {}   # thresholds per group the run adopted (group_thresholds.json)


def save_inference_state() -> None:
    """Persist the fitted preprocessing and class names for serving."""
    import pickle

    INFERENCE_STATE["classes"] = CLASS_NAMES
    with open(INFERENCE_FILE, "wb") as fh:
        pickle.dump(INFERENCE_STATE, fh)


def load_inference_state(folder: str | Path = ".") -> None:
    """Restore the state saved by a training run (trusted files only: pickle)."""
    import pickle

    global CLASS_NAMES, GROUP_THRESHOLDS
    with open(Path(folder) / INFERENCE_FILE, "rb") as fh:
        INFERENCE_STATE.update(pickle.load(fh))
    CLASS_NAMES = INFERENCE_STATE.get("classes")
    adopted = Path(folder) / "group_thresholds.json"
    GROUP_THRESHOLDS = json.loads(adopted.read_text(encoding="utf-8")) if adopted.is_file() else {}


def group_cut(item, column: str, cuts: dict, default: float) -> tuple[float, str | None]:
    """A row's group threshold, and the group it was found under. Groups are keyed as
    training wrote them: str() of the value, "(missing)" for an empty cell; a whole number
    read as 1.0 finds "1", and 1 finds "1.0". A row without the column gets the default."""
    if not isinstance(item, dict) or column not in item:
        return default, None
    group = item[column]
    if group is None or (isinstance(group, float) and group != group):
        keys = ["(missing)"]
    elif isinstance(group, float) and group.is_integer():
        keys = [str(group), str(int(group))]
    elif isinstance(group, int) and not isinstance(group, bool):
        keys = [str(group), str(float(group))]
    else:
        keys = [str(group)]
    found = next((k for k in keys if k in cuts), None)
    return (float(cuts[found]), found) if found is not None else (default, keys[0])


def with_group_thresholds(results: list[dict], raw: list) -> list[dict]:
    """A two-class prediction decided by its row's group threshold instead of 0.5, when
    the run adopted thresholds per group (Results by group). The row carries the group
    column even when the model doesn't read it; without it the default applies."""
    if not GROUP_THRESHOLDS:
        return results
    column, cuts = GROUP_THRESHOLDS["column"], GROUP_THRESHOLDS["thresholds"]
    default = float(GROUP_THRESHOLDS.get("default", 0.5))
    for result, item in zip(results, raw):
        probs = result.get("probabilities")
        if not isinstance(probs, dict) or len(probs) != 2:
            continue
        cut, group = group_cut(item, column, cuts, default)
        names = list(probs)
        p = float(probs[names[1]])
        best = int(p >= cut)
        result.update(label=names[best], class_index=best,
                      confidence=round(p if best else 1 - p, 6), threshold=cut,
                      group=group)
    return results
{% if image_inputs %}


def open_image(item):
    """Path, raw bytes, base64 / data-URL string or PIL image -> PIL image."""
    import base64
    import io

    from PIL import Image

    if isinstance(item, Image.Image):
        return item
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item))
    if isinstance(item, str):
        if item.startswith("data:"):
            item = item.split(",", 1)[1]
        elif len(item) < 1024 and _is_file(item):
            return Image.open(item)
        return Image.open(io.BytesIO(base64.b64decode(item)))
    arr = np.asarray(item)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    return Image.fromarray(arr)


def _is_file(text: str) -> bool:
    try:
        return Path(text).is_file()
    except (OSError, ValueError):  # e.g. base64 strings longer than a file name may be
        return False
{% endif %}


def class_name(index: int) -> str:
    return str(CLASS_NAMES[index]) if CLASS_NAMES and index < len(CLASS_NAMES) else str(index)


def describe(outputs: np.ndarray) -> list[dict]:
    """Model outputs -> JSON-friendly predictions."""
    scores = to_scores(np.asarray(outputs, dtype=np.float32))
    results = []
{% if spec.task == "multiclass" %}
    for row in scores:
        best = int(np.argmax(row))
        results.append({"label": class_name(best), "class_index": best,
                        "confidence": round(float(row[best]), 6),
                        "probabilities": {class_name(i): round(float(p), 6)
                                          for i, p in enumerate(row)}})
{% elif spec.task == "binary" %}
    for p in scores.reshape(len(scores), -1)[:, 0]:
        best = int(p >= 0.5)
        results.append({"label": class_name(best), "class_index": best,
                        "confidence": round(float(p if best else 1 - p), 6),
                        "probabilities": {class_name(0): round(float(1 - p), 6),
                                          class_name(1): round(float(p), 6)}})
{% elif spec.task == "multilabel" %}
    for row in scores:
        results.append({"labels": [class_name(i) for i, p in enumerate(row) if p >= 0.5],
                        "probabilities": {class_name(i): round(float(p), 6)
                                          for i, p in enumerate(row)}})
{% else %}
    values = scores.reshape(len(scores), -1)
{% if timeseries %}
    values = inverse_targets(values)
{% endif %}
    for row in values:
        results.append({"prediction": float(row[0]) if len(row) == 1
                        else [float(v) for v in row]})
{% endif %}
    return results
'''

INFERENCE_TORCH = r'''

_PREDICTOR = None


def quantized_engine() -> str:
    """A quantized-kernel backend this machine supports (fbgemm / x86 / qnnpack)."""
    engines = torch.backends.quantized.supported_engines
    if torch.backends.quantized.engine == "none" or torch.backends.quantized.engine \
            not in engines:
        torch.backends.quantized.engine = next(e for e in ("fbgemm", "x86", "qnnpack")
                                               if e in engines)
    return torch.backends.quantized.engine


def qat_prepare(model: nn.Module) -> nn.Module:
    """Fake-quantized copy of the model for quantization-aware training (FX graph mode).
    Embedding tables keep float quantization parameters (8-bit rows, each with a float
    scale and offset, weight-only); every other layer is quantized to int8."""
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.qconfig import default_embedding_qat_qconfig
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    mapping = get_default_qat_qconfig_mapping(quantized_engine())
    if any(isinstance(m, nn.Embedding) for m in model.modules()):
        mapping = mapping.set_object_type(nn.Embedding, default_embedding_qat_qconfig)
    example = (torch.zeros(1, *INPUT_SHAPE, dtype={{ "torch.long" if spec.input_dtype.startswith("int") else "torch.float32" }}),)
    return prepare_qat_fx(model.cpu().train(), mapping, example)


def qat_convert(model: nn.Module) -> nn.Module:
    """The int8 model from a fake-quantized one (CPU)."""
    from torch.ao.quantization.quantize_fx import convert_fx

    return convert_fx(model.cpu().eval())


def model_device(model: nn.Module) -> torch.device:
    """Where the model lives (quantized models may have no float parameters)."""
    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


def load_predictor(folder: str | Path = ".", device: str = "cpu") -> nn.Module:
    """Load the trained weights and preprocessing state from a run folder."""
    global _PREDICTOR
    load_inference_state(folder)
    model = {{ spec.class_name }}()
    state = torch.load(Path(folder) / CHECKPOINT, map_location="cpu")
    if QUANTIZE == "qat_int8":           # set by a pipeline's QAT stage: int8 on the CPU
        model = qat_prepare(model)
        model.load_state_dict(state)
        _PREDICTOR = qat_convert(model)
        return _PREDICTOR
    model.load_state_dict(state)
    model.eval()
    if QUANTIZE == "dynamic_int8":       # set by a pipeline's Quantize stage (CPU serving)
        quantized_engine()
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        _PREDICTOR = model
        return _PREDICTOR
    _PREDICTOR = model.to(device)
    return _PREDICTOR


@torch.no_grad()
def infer(raw: list) -> list[dict]:
    """Predictions for a list of raw inputs (see ``prepare_inputs``)."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor(folder) first")
    device = model_device(_PREDICTOR)
    x = torch.from_numpy(np.ascontiguousarray(prepare_inputs(raw))).to(device)
{% if prob %}
    return serve(_PREDICTOR, x)
{% elif pairs == "ranking" %}
    # labelled pairs: each item gets its score; the higher one ranks first
    return [{"score": round(float(v), 6)} for v in _PREDICTOR(x).float().cpu().numpy()[:, 0]]
{% elif pairs %}
    # labelled pairs or triplets: each item gets its embedding; compare two by cosine similarity
    return [{"embedding": [round(float(v), 6) for v in row]}
            for row in _PREDICTOR(x).float().cpu().numpy()]
{% elif metric_loss %}
    # the model outputs embeddings: the trained loss scores them against its classes
    emb = _PREDICTOR(x).float().cpu()
    results = describe(_scorer().scores(emb).numpy())
    for result, row in zip(results, emb.numpy(), strict=True):
        result["embedding"] = [round(float(v), 6) for v in row]
    return with_group_thresholds(results, raw)
{% else %}
    return with_group_thresholds(describe(_PREDICTOR(x).float().cpu().numpy()), raw)
{% endif %}
{% if metric_loss and not pairs %}


_SCORER_FN = None


def _scorer():
    """The metric-learning loss with its trained class centres / averages."""
    global _SCORER_FN
    if _SCORER_FN is None:
        if "scorer" not in INFERENCE_STATE:
            raise RuntimeError("this run saved no class scorer: retrain it to serve labels")
        _SCORER_FN = make_loss()
        _SCORER_FN.load_state_dict(INFERENCE_STATE["scorer"])
        _SCORER_FN.eval()
    return _SCORER_FN
{% endif %}
'''

INFERENCE_KERAS = r'''

_PREDICTOR = None


def load_predictor(folder: str | Path = ".", device: str = "cpu"):
    """Load the trained model and preprocessing state from a run folder."""
    global _PREDICTOR
    load_inference_state(folder)
    _PREDICTOR = keras.models.load_model(Path(folder) / MODEL_FILE)
    return _PREDICTOR


def infer(raw: list) -> list[dict]:
    """Predictions for a list of raw inputs (see ``prepare_inputs``)."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor(folder) first")
    x = to_model_layout(prepare_inputs(raw))
    return with_group_thresholds(describe(np.asarray(_PREDICTOR.predict(x, verbose=0))), raw)
'''

# Torch image datasets use torchvision transforms instead of DATA_TEMPLATE's
# prepare_inputs; this is their equivalent.
INFERENCE_TORCH_IMAGES = r'''


def prepare_inputs(raw: list) -> np.ndarray:
    """Images (paths, bytes, base64 strings or PIL images) -> a normalised batch."""
    mean, std = INFERENCE_STATE.get("norm") or (None, None)
    transform = build_transforms(train=False, mean=mean, std=std)
    mode = "L" if INPUT_SHAPE[0] == 1 else "RGB"
    batch = []
    for item in raw:
        img = open_image(item).convert(mode)
{% if not image_explicit_size %}
        img = img.resize((INPUT_SHAPE[2], INPUT_SHAPE[1]))
{% endif %}
        batch.append(transform(img))
    x = torch.stack(batch).numpy()
    if tuple(x.shape[1:]) != tuple(INPUT_SHAPE):
        raise ValueError(f"images become {tuple(x.shape[1:])}; the model expects "
                         f"{tuple(INPUT_SHAPE)}")
    return x
'''
