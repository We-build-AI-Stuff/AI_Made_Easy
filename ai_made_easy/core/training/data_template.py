"""Framework-neutral data layer emitted into every generated training script.

Produces ``make_arrays() -> {"train": (x, y), "val": ..., "test": ...}`` with all
preprocessing fitted on the training split only. Image datasets in PyTorch
scripts use torchvision pipelines instead (see ``torch_script``); everything
else — tables, arrays, text, audio, time series, and images in Keras scripts —
flows through this numpy code.
"""

DATA_TEMPLATE = r'''
# ================================================================== data
INFERENCE_STATE: dict = {}  # fitted preprocessing, saved with the model for serving
{% if balance == "group weights" %}
GROUP_VALUES = TRAIN_GROUPS = None   # the group column, for every row and the training rows


{{ group_weights_code }}
{% endif %}


{% if d.block in ("data.csv", "data.text_csv", "data.timeseries_csv", "data.pairs", "data.triplets") %}
def read_table(path: str, fmt: str) -> "pd.DataFrame":
    import pandas as pd

    readers = {"csv": pd.read_csv, "tsv": lambda p: pd.read_csv(p, sep="\t"),
               "parquet": pd.read_parquet, "excel": pd.read_excel,
               "json": lambda p: pd.read_json(p, lines=str(p).endswith(".jsonl"))}
    return readers[fmt](path)


{% endif %}
{% if d.block == "data.synthetic_table" %}
{{ synthetic_table_code }}


{% endif %}
{% if needs_wav %}
def read_wav(path: str, sample_rate: int, n_samples: int) -> np.ndarray:
    """Mono float32 waveform, resampled and padded/trimmed to n_samples."""
    import wave

    with wave.open(path, "rb") as fh:
        rate, width, channels = fh.getframerate(), fh.getsampwidth(), fh.getnchannels()
        raw = fh.readframes(fh.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        audio = (audio - 128.0) / 128.0
    else:
        audio /= float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and len(audio) > 1:
        target_len = max(int(round(len(audio) * sample_rate / rate)), 1)
        audio = np.interp(np.linspace(0, len(audio) - 1, target_len),
                          np.arange(len(audio)), audio).astype(np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    return audio[:n_samples]


{% endif %}
def load_raw():
    r"""Load {{ comment }}. Returns (features, targets, class names or None)."""
{% if d.block == "data.synthetic" %}
    rng = np.random.default_rng({{ d.seed }})
    n = {{ d.n_samples }}
{% if d.kind == "moons" %}
    t = rng.uniform(0, np.pi, size=(n // 2, 1))
    a = np.hstack([np.cos(t), np.sin(t)])
    b = np.hstack([1 - np.cos(t), 0.5 - np.sin(t)])
    x = np.vstack([a, b]) + rng.normal(0, {{ d.noise }}, (2 * (n // 2), 2))
    y = np.repeat([0, 1], n // 2)
    return x.astype(np.float32), y, ["0", "1"]
{% elif d.kind == "circles" %}
    angles = rng.uniform(0, 2 * np.pi, (2 * (n // 2), 1))
    radius = np.repeat([0.5, 1.0], n // 2)[:, None]
    x = np.hstack([radius * np.cos(angles), radius * np.sin(angles)])
    x += rng.normal(0, {{ d.noise }}, x.shape)
    y = np.repeat([0, 1], n // 2)
    return x.astype(np.float32), y, ["0", "1"]
{% elif d.kind == "regression" %}
    d = {{ d.n_features }}
    w = rng.normal(size=(d, {{ n_targets }}))
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x @ w + rng.normal(0, {{ d.noise }}, (n, {{ n_targets }}))).astype(np.float32)
    return x, (y[:, 0] if {{ n_targets }} == 1 else y), None
{% else %}
    d, k = {{ d.n_features }}, {{ d.n_classes }}
    centers = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(0, k, size=n)
    x = centers[y] + rng.normal(scale=0.5 + {{ d.noise }}, size=(n, d))
{% if task == "multilabel" %}
    y = (rng.random((n, {{ n_outputs }})) < 0.3).astype(np.float32)
    y[np.arange(n), rng.integers(0, {{ n_outputs }}, size=n)] = 1.0
    x = y @ rng.normal(size=({{ n_outputs }}, d)) + rng.normal(scale=0.5, size=(n, d))
    return x.astype(np.float32), y, [str(i) for i in range({{ n_outputs }})]
{% elif task == "distribution" %}
    # soft labels (e.g. annotators' vote shares): a softmax of the features
    logits = x @ rng.normal(size=(d, {{ n_outputs }})) / np.sqrt(d)
    y = np.exp(logits - logits.max(axis=1, keepdims=True))
    y /= y.sum(axis=1, keepdims=True)
    return x.astype(np.float32), y.astype(np.float32), [str(i) for i in range({{ n_outputs }})]
{% endif %}
    return x.astype(np.float32), y, [str(i) for i in range(k)]
{% endif %}
{% elif d.block == "data.sklearn" %}
    from sklearn import datasets as skds

    bunch = skds.{{ sk_loader }}()
    names = [str(n) for n in getattr(bunch, "target_names", [])] or None
    return bunch.data.astype(np.float32), bunch.target, names
{% elif d.block == "data.csv" %}
    frame = read_table({{ d.path | repr }}, {{ d.format | repr }})
    target = {{ d.target_column | repr }}
    if target not in frame.columns:
        raise SystemExit(f"target column {target!r} not found; columns: {list(frame.columns)}")
    y = frame.pop(target)
{% if balance == "group weights" %}
    global GROUP_VALUES
    GROUP_VALUES = group_column(frame, {{ group_column | repr }})
{% endif %}
{% if feature_columns %}
    frame = frame[{{ feature_columns | repr }}]
{% endif %}
{% if classification %}
    classes = sorted(y.astype(str).unique())
    return frame, y.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy(), classes
{% else %}
    return frame, y.to_numpy(dtype=np.float32), None
{% endif %}
{% elif d.block == "data.synthetic_table" %}
    frame, y = synthetic_table({{ d.n_samples }}, {{ d.noise }}, {{ d.seed }}, {{ d.task | repr }})
{% if classification %}
    classes = sorted(set(y.tolist()))
    return frame, np.asarray([classes.index(v) for v in y]), [str(c) for c in classes]
{% else %}
    return frame, y.astype(np.float32), None
{% endif %}
{% elif d.block == "data.pairs" and not d.path %}
    # demo pairs: 8 features per item, 2000 pairs
    rng = np.random.default_rng(0)
    n = 2000
{% if pairs == "ranking" %}
    w = rng.normal(size=8)
    a, b = rng.normal(size=(n, 8)), rng.normal(size=(n, 8))
    y = (a @ w > b @ w).astype(np.int64)            # 1: a ranks above b
    names = ["b ranks higher", "a ranks higher"]
{% else %}
    centres = rng.normal(scale=2.0, size=(10, 8))   # 10 kinds of item
    ka = rng.integers(0, 10, n)
    kb = np.where(rng.random(n) < 0.5, ka, rng.integers(0, 10, n))
    a = centres[ka] + rng.normal(scale=0.7, size=(n, 8))
    b = centres[kb] + rng.normal(scale=0.7, size=(n, 8))
    y = (ka == kb).astype(np.int64)                 # 1: the same kind
    names = ["different", "same"]
{% endif %}
    return np.stack([a, b], axis=1).astype(np.float32), y, names
{% elif d.block == "data.pairs" %}
    frame = read_table({{ d.path | repr }}, {{ d.format | repr }})
    label, lp, rp = {{ d.label_column | repr }}, {{ d.left_prefix | repr }}, {{ d.right_prefix | repr }}
    if label not in frame.columns:
        raise SystemExit(f"label column {label!r} not found; columns: {list(frame.columns)}")
    left = [c for c in frame.columns if str(c).startswith(lp)]
    right = [rp + str(c)[len(lp):] for c in left]
    missing = [c for c in right if c not in frame.columns]
    if not left or missing:
        raise SystemExit(f"pairs need matching {lp}* and {rp}* columns; missing: {missing or lp + '*'}")
    x = np.stack([frame[left].to_numpy(np.float32), frame[right].to_numpy(np.float32)], axis=1)
    y = frame[label].to_numpy()
    if not set(np.unique(y).tolist()) <= {-1, 0, 1}:
        raise SystemExit(f"{label} must be 1 or 0 (or -1) for each pair")
    return x, (y > 0).astype(np.int64), {{ ["b ranks higher", "a ranks higher"] if pairs == "ranking" else ["different", "same"] }}
{% elif d.block == "data.triplets" and not d.path %}
    # demo triplets: 8 features per item, 2000 triplets from 10 kinds of item
    rng = np.random.default_rng(0)
    n = 2000
    centres = rng.normal(scale=2.0, size=(10, 8))
    ka = rng.integers(0, 10, n)
    kn = (ka + rng.integers(1, 10, n)) % 10          # a different kind
    a, p, q = (centres[k] + rng.normal(scale=0.7, size=(n, 8)) for k in (ka, ka, kn))
    return np.stack([a, p, q], axis=1).astype(np.float32), np.ones(n, np.int64), \
        ["negative closer", "positive closer"]
{% elif d.block == "data.triplets" %}
    frame = read_table({{ d.path | repr }}, {{ d.format | repr }})
    ap, pp, np_ = {{ d.anchor_prefix | repr }}, {{ d.positive_prefix | repr }}, {{ d.negative_prefix | repr }}
    anchor = [c for c in frame.columns if str(c).startswith(ap)]
    others = [[pre + str(c)[len(ap):] for c in anchor] for pre in (pp, np_)]
    missing = [c for cols in others for c in cols if c not in frame.columns]
    if not anchor or missing:
        raise SystemExit(f"triplets need matching {ap}*, {pp}* and {np_}* columns; "
                         f"missing: {missing or ap + '*'}")
    x = np.stack([frame[cols].to_numpy(np.float32) for cols in (anchor, *others)], axis=1)
    return x, np.ones(len(x), np.int64), ["negative closer", "positive closer"]
{% elif d.block == "data.numpy" %}
    archive = np.load({{ d.path | repr }})
    x, y = archive[{{ d.x_key | repr }}], archive[{{ d.y_key | repr }}]
    return x.astype(np.float32), y, None
{% elif d.block == "data.json" %}
    import json

    text = Path({{ d.path | repr }}).read_text(encoding="utf-8").strip()
    records = json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    x = np.asarray([r[{{ d.x_field | repr }}] for r in records], dtype=np.float32)
    y = np.asarray([r[{{ d.y_field | repr }}] for r in records])
    return x, y, None
{% elif d.block == "data.huggingface" %}
    from datasets import load_dataset

    ds = load_dataset({{ d.repo_id | repr }}, split={{ d.split | repr }})
    x = np.asarray([np.asarray(v, dtype=np.float32) for v in ds[{{ d.x_field | repr }}]])
    if x.ndim == 3:  # grayscale images (N, H, W) -> (N, 1, H, W)
        x = x[:, None] / 255.0
    elif x.ndim == 4 and x.shape[-1] in (1, 3):  # (N, H, W, C) -> channels-first
        x = x.transpose(0, 3, 1, 2) / 255.0
    return x.astype(np.float32), np.asarray(ds[{{ d.y_field | repr }}]), None
{% elif d.block == "data.text_csv" %}
    frame = read_table({{ d.path | repr }}, {{ d.format | repr }})
    texts = frame[{{ d.text_column | repr }}].fillna("").astype(str).tolist()
    labels = frame[{{ d.label_column | repr }}]
{% if classification %}
    classes = sorted(labels.astype(str).unique())
    index = {c: i for i, c in enumerate(classes)}
    return texts, labels.astype(str).map(index).to_numpy(), classes
{% else %}
    return texts, labels.to_numpy(dtype=np.float32), None
{% endif %}
{% elif d.block == "data.text_folder" %}
    root = Path({{ d.root | repr }})
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class subfolders under {root}")
    texts, labels = [], []
    for ci, cls in enumerate(classes):
        for file in sorted((root / cls).glob("*.txt")):
            texts.append(file.read_text(errors="ignore", encoding="utf-8"))
            labels.append(ci)
    return texts, np.asarray(labels), classes
{% elif d.block == "data.audio_folder" %}
    root = Path({{ d.root | repr }})
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class subfolders under {root}")
    n_samples = int(round({{ d.duration }} * {{ d.sample_rate }}))
    waves, labels = [], []
    for ci, cls in enumerate(classes):
        for file in sorted((root / cls).glob("*.wav")):
            waves.append(read_wav(str(file), {{ d.sample_rate }}, n_samples))
            labels.append(ci)
    return np.stack(waves), np.asarray(labels), classes
{% elif d.block == "data.timeseries_csv" %}
    frame = read_table({{ d.path | repr }}, "csv")
    targets = {{ ts_targets | repr }}
    missing = [c for c in targets if c not in frame.columns]
    if missing:
        raise SystemExit(f"target columns {missing} not found; columns: {list(frame.columns)}")
{% if feature_columns %}
    features = {{ feature_columns | repr }}
{% else %}
    features = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
{% endif %}
    return frame[features].to_numpy(np.float32), frame[targets].to_numpy(np.float32), None
{% elif d.block in ("data.image_folder", "data.torchvision") %}
{% if d.block == "data.image_folder" %}
    from PIL import Image

    root = Path({{ d.root | repr }})
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class subfolders under {root}")
    mode = "L" if INPUT_SHAPE[0] == 1 else "RGB"
    images, labels = [], []
    for ci, cls in enumerate(classes):
        for file in sorted((root / cls).iterdir()):
            if file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
                continue
            img = Image.open(file).convert(mode).resize((INPUT_SHAPE[2], INPUT_SHAPE[1]))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(arr[None] if arr.ndim == 2 else arr.transpose(2, 0, 1))
            labels.append(ci)
    return np.stack(images), np.asarray(labels), classes
{% else %}
    (x_train, y_train), (x_test, y_test) = keras.datasets.{{ keras_dataset }}.load_data()
    x = np.concatenate([x_train, x_test]).astype(np.float32) / 255.0
    x = x[:, None] if x.ndim == 3 else x.transpose(0, 3, 1, 2)
    return x, np.concatenate([y_train, y_test]).reshape(-1), None
{% endif %}
{% endif %}


def split_indices(n: int, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test indices{% if chronological %} (chronological){% endif %}."""
    n_val, n_test = int(n * VAL_FRACTION), int(n * TEST_FRACTION)
{% if chronological %}
    idx = np.arange(n)
    return idx[: n - n_val - n_test], idx[n - n_val - n_test: n - n_test], idx[n - n_test:]
{% else %}
    rng = np.random.default_rng(SPLIT_SEED)
{% if stratify %}
    if y is not None and np.ndim(y) == 1 and len(np.unique(y)) < max(n // 5, 2):
        train, val, test = [], [], []
        for c in np.unique(y):
            members = np.flatnonzero(y == c)
            rng.shuffle(members)
            cv, ct = int(round(len(members) * VAL_FRACTION)), int(round(len(members) * TEST_FRACTION))
            val += list(members[:cv])
            test += list(members[cv:cv + ct])
            train += list(members[cv + ct:])
        return (rng.permutation(train), rng.permutation(val).astype(int),
                rng.permutation(test).astype(int))
{% endif %}
    idx = {{ "rng.permutation(n)" if shuffle else "np.arange(n)" }}
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]
{% endif %}
{% if table_steps %}


def preprocess_tables(train: "pd.DataFrame", frames: list) -> list[np.ndarray]:
    """Column-level preprocessing fitted on the training frame only."""
    import pandas as pd

{% if steps["prep.drop_columns"] %}
    drop = {{ drop_columns | repr }}
    train = train.drop(columns=[c for c in drop if c in train.columns])
{% endif %}
    state = {"columns": list(train.columns)}  # the inputs a deployed model needs
    categorical = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]
    numeric = [c for c in train.columns if c not in categorical]
{% if steps["prep.impute"] %}
    fill = {}
    for c in numeric:
{% if steps["prep.impute"].strategy == "mean" %}
        fill[c] = train[c].mean()
{% elif steps["prep.impute"].strategy == "median" %}
        fill[c] = train[c].median()
{% elif steps["prep.impute"].strategy == "mode" %}
        fill[c] = train[c].mode().iloc[0] if train[c].notna().any() else 0.0
{% else %}
        fill[c] = {{ steps["prep.impute"].constant }}
{% endif %}
    state["fill"] = fill
{% endif %}
{% if steps["prep.ordinal_encode"] %}
    ordinal = {{ ordinal_columns | repr }} or categorical
    state["codes"] = {c: {v: i for i, v in enumerate(sorted(train[c].dropna().astype(str).unique()))}
                      for c in ordinal}
    categorical = [c for c in categorical if c not in ordinal]
{% endif %}
    one_hot = {{ one_hot_columns | repr }} or categorical
    state["categories"] = {c: sorted(train[c].dropna().astype(str).unique())[:{{ max_categories }}]
                           for c in one_hot}
    INFERENCE_STATE["table"] = state
    return [table_transform(f, state) for f in frames]


def table_transform(frame: "pd.DataFrame", state: dict) -> np.ndarray:
    """Apply the fitted column preprocessing to any frame with the training columns."""
    import pandas as pd

{% if steps["prep.drop_columns"] %}
    frame = frame.drop(columns=[c for c in {{ drop_columns | repr }} if c in frame.columns])
{% endif %}
{% if steps["prep.impute"] and steps["prep.impute"].strategy != "drop rows" %}
    frame = frame.fillna(value=state["fill"])
{% endif %}
    for c, codes in state.get("codes", {}).items():
        frame = frame.assign(**{c: frame[c].astype(str).map(codes).fillna(-1)})
    cats = state["categories"]
    parts = [frame.drop(columns=list(cats)).astype(np.float32)]
    for c, values in cats.items():
        col = frame[c].astype(str)
        parts.append(pd.DataFrame({f"{c}={v}": (col == v).astype(np.float32)
                                   for v in values}, index=frame.index))
    return pd.concat(parts, axis=1).to_numpy(np.float32)
{% endif %}
{% if text %}


def clean_text(text: str) -> str:
    import re

{% if steps["prep.text_clean"] %}
{% if steps["prep.text_clean"].strip_html %}
    text = re.sub(r"<[^>]+>", " ", text)
{% endif %}
{% if steps["prep.text_clean"].lowercase %}
    text = text.lower()
{% endif %}
{% if steps["prep.text_clean"].strip_punctuation %}
    text = re.sub(r"[^\w\s]", " ", text)
{% endif %}
{% if steps["prep.text_clean"].strip_digits %}
    text = re.sub(r"\d+", " ", text)
{% endif %}
{% endif %}
    return re.sub(r"\s+", " ", text).strip()


def encode_texts(train_texts: list[str], groups: list[list[str]]) -> list[np.ndarray]:
    r"""{{ tok.method }} tokenization to [{{ tok.max_length }}] int64 ids (0 = pad, 1 = unknown)."""
{% if tok.method == "huggingface" %}
    return [encode_batch(texts) for texts in groups]


def encode_batch(texts: list[str]) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained({{ tok.hf_tokenizer | repr }})
    return np.asarray(tokenizer([clean_text(t) for t in texts], padding="max_length",
                                truncation=True, max_length={{ tok.max_length }})["input_ids"],
                      dtype=np.int64)
{% else %}
    from collections import Counter

    counts = Counter(t for text in train_texts for t in text_tokens(text))
    vocab = [t for t, c in counts.most_common({{ tok.vocab_size }} - 2) if c >= {{ tok.min_freq }}]
    INFERENCE_STATE["vocab"] = {t: i + 2 for i, t in enumerate(vocab)}
    return [encode_batch(texts) for texts in groups]


def text_tokens(text: str) -> list[str]:
    text = clean_text(text)
    return {{ "list(text)" if tok.method == "char" else "text.split()" }}


def encode_batch(texts: list[str]) -> np.ndarray:
    index = INFERENCE_STATE["vocab"]

    def encode(text: str) -> list[int]:
        ids = [index.get(t, 1) for t in text_tokens(text)][: {{ tok.max_length }}]
        return ids + [0] * ({{ tok.max_length }} - len(ids))

    return np.asarray([encode(t) for t in texts], dtype=np.int64)
{% endif %}
{% endif %}
{% if audio %}


def audio_features(waves: np.ndarray) -> np.ndarray:
    r"""{{ feat.kind }} features: (N, samples) -> (N, *INPUT_SHAPE)."""
{% if feat.kind == "waveform" %}
    return waves[:, None, :].astype(np.float32)
{% else %}
    n_fft, hop = {{ feat.n_fft }}, {{ feat.hop_length }}
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(waves, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    frames = 1 + (padded.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(padded[:, idx] * window, axis=-1)) ** 2  # (N, T, F)
{% if feat.kind in ("mel_spectrogram", "mfcc") %}
    sr, n_mels = {{ sample_rate }}, {{ feat.n_mels }}
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    points = hz(np.linspace(mel(0.0), mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fbank = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(n_mels):
        lo, mid, hi = points[m], points[m + 1], points[m + 2]
        fbank[m] = np.clip(np.minimum((bins - lo) / (mid - lo + 1e-9),
                                      (hi - bins) / (hi - mid + 1e-9)), 0, None)
    power = power @ fbank.T
{% endif %}
{% if feat.log_scale or feat.kind == "mfcc" %}
    power = np.log(power + 1e-6)
{% endif %}
{% if feat.kind == "mfcc" %}
    k = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange({{ feat.n_mfcc }})[:, None])
    power = power @ dct.T.astype(np.float32)
{% endif %}
    return power.transpose(0, 2, 1)[:, None].astype(np.float32)  # (N, 1, F, T)
{% endif %}
{% endif %}
{% if timeseries %}


def make_windows(features: np.ndarray, targets: np.ndarray):
    """Sliding windows: x (N, window, C), y (N, horizon * targets)."""
    window, horizon, stride = {{ d.window }}, {{ d.horizon }}, {{ d.stride }}
    starts = range(0, len(features) - window - horizon + 1, stride)
    x = np.stack([features[s:s + window] for s in starts])
    y = np.stack([targets[s + window:s + window + horizon].reshape(-1) for s in starts])
{% if ts_channels_first %}
    x = x.transpose(0, 2, 1)  # channels-first [C, L] for Conv1D
{% endif %}
    return x.astype(np.float32), y.astype(np.float32)
{% endif %}


class NumericPipeline:
    """Feature transforms fitted on training data only."""

    def fit(self, x: np.ndarray) -> "NumericPipeline":
        flat = x.reshape(len(x), -1)
{% if steps["prep.impute"] and not table_steps %}
        self.fill = np.nan{{ {"mean": "mean", "median": "median"}.get(steps["prep.impute"].strategy, "mean") }}(flat, axis=0)
{% if steps["prep.impute"].strategy == "constant" %}
        self.fill = np.full(flat.shape[1], {{ steps["prep.impute"].constant }}, dtype=np.float32)
{% endif %}
        flat = np.where(np.isnan(flat), self.fill, flat)
{% endif %}
{% if steps["prep.log_transform"] %}
        self.log_cols = np.all(flat >= 0, axis=0)
        flat = np.where(self.log_cols, np.log1p(np.abs(flat)), flat)
{% endif %}
{% if steps["prep.clip_outliers"] %}
{% set c = steps["prep.clip_outliers"] %}
{% if c.method == "iqr" %}
        q1, q3 = np.percentile(flat, [25, 75], axis=0)
        self.lo, self.hi = q1 - {{ c.threshold }} * (q3 - q1), q3 + {{ c.threshold }} * (q3 - q1)
{% elif c.method == "zscore" %}
        mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-12
        self.lo, self.hi = mu - {{ c.threshold }} * sd, mu + {{ c.threshold }} * sd
{% else %}
        self.lo, self.hi = np.quantile(flat, [{{ c.threshold }}, 1 - {{ c.threshold }}], axis=0)
{% endif %}
        flat = np.clip(flat, self.lo, self.hi)
{% endif %}
{% if steps["prep.variance_filter"] %}
        self.keep = flat.var(axis=0) > {{ steps["prep.variance_filter"].threshold }}
        flat = flat[:, self.keep]
{% endif %}
{% if per_channel %}
        axes = (0,) + tuple(range(2, x.ndim))
{% endif %}
{% if steps["prep.normalize"] %}
{% if steps["prep.normalize"].mode == "fixed" %}
        self.mean = np.asarray({{ norm_mean }}, dtype=np.float32)
        self.std = np.asarray({{ norm_std }}, dtype=np.float32)
{% elif per_channel %}
        self.mean = x.mean(axis=axes)
        self.std = x.std(axis=axes)
{% else %}
        self.mean, self.std = flat.mean(axis=0), flat.std(axis=0)
{% endif %}
        # constant in training: leave unscaled (like scikit-learn), never divide by ~0
        self.std = np.where(self.std < 1e-8, 1.0, self.std).astype(np.float32)
{% endif %}
{% if steps["prep.robust_scale"] %}
        self.median = np.median(flat, axis=0)
        q1, q3 = np.percentile(flat, [25, 75], axis=0)
        self.iqr = np.where(q3 - q1 > 0, q3 - q1, 1.0)
{% endif %}
{% if steps["prep.minmax"] %}
        scaled = self._scale(x)
        flat_s = scaled.reshape(len(scaled), -1)
        self.min, self.max = flat_s.min(axis=0), flat_s.max(axis=0)
{% endif %}
        return self

    def _scale(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        flat = x.reshape(len(x), -1).astype(np.float32)
{% if raw_columns %}
        raw = flat[:, {{ raw_columns | repr }}].copy()   # categorical codes stay integers
{% endif %}
{% if steps["prep.impute"] and not table_steps %}
        flat = np.where(np.isnan(flat), self.fill, flat)
{% endif %}
{% if steps["prep.log_transform"] %}
        flat = np.where(self.log_cols, np.log1p(np.abs(flat)), flat)
{% endif %}
{% if steps["prep.clip_outliers"] %}
        flat = np.clip(flat, self.lo, self.hi)
{% endif %}
{% if steps["prep.variance_filter"] %}
        flat = flat[:, self.keep]
        shape = (len(x), flat.shape[1])
{% endif %}
{% if steps["prep.normalize"] %}
{% if per_channel %}
        x = flat.reshape(shape)
        view = (1, -1) + (1,) * (x.ndim - 2)
        flat = ((x - self.mean.reshape(view)) / self.std.reshape(view)).reshape(len(x), -1)
{% else %}
        flat = (flat - self.mean) / self.std
{% endif %}
{% endif %}
{% if steps["prep.robust_scale"] %}
        flat = (flat - self.median) / self.iqr
{% endif %}
{% if raw_columns %}
        flat[:, {{ raw_columns | repr }}] = raw
{% endif %}
        return flat.reshape(shape)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._scale(x)
{% if steps["prep.minmax"] %}
        shape = x.shape
        flat = x.reshape(len(x), -1)
        span = np.where(self.max - self.min > 0, self.max - self.min, 1.0)
        flat = ((flat - self.min) / span * ({{ steps["prep.minmax"].range_max }} - {{ steps["prep.minmax"].range_min }})
                + {{ steps["prep.minmax"].range_min }})
        x = flat.reshape(shape)
{% endif %}
        return x.astype(np.float32)


def make_arrays() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load, split and preprocess the data. Statistics come from the training split."""
    x, y, classes = load_raw()
    global CLASS_NAMES
    CLASS_NAMES = classes
{% if timeseries %}
    n = len(x)
    n_val, n_test = int(n * VAL_FRACTION), int(n * TEST_FRACTION)
    bounds = {"train": (0, n - n_val - n_test), "val": (n - n_val - n_test, n - n_test),
              "test": (n - n_test, n)}
    pipeline = NumericPipeline().fit(x[: bounds["train"][1]])
    target_pipeline = NumericPipeline().fit(y[: bounds["train"][1]]) if TARGET_SCALING else None
    INFERENCE_STATE["numeric"] = dict(vars(pipeline))
    INFERENCE_STATE["target"] = dict(vars(target_pipeline)) if target_pipeline else None
    splits = {}
    for name, (lo, hi) in bounds.items():
        feats = pipeline.transform(x[lo:hi])
        targs = target_pipeline.transform(y[lo:hi]) if target_pipeline else y[lo:hi]
        if hi - lo > {{ d.window }} + {{ d.horizon }}:
            splits[name] = make_windows(feats, targs)
        else:
            splits[name] = (np.zeros((0, *INPUT_SHAPE), np.float32),
                            np.zeros((0, {{ n_outputs }}), np.float32))
    return splits
{% else %}
{% if table_steps and steps["prep.impute"] and steps["prep.impute"].strategy == "drop rows" %}
    keep = x.notna().all(axis=1).to_numpy()
    x, y = x[keep].reset_index(drop=True), np.asarray(y)[keep]
{% endif %}
    train_i, val_i, test_i = split_indices(len(y), y)
{% if balance == "group weights" %}
    global TRAIN_GROUPS
{% if table_steps and steps["prep.impute"] and steps["prep.impute"].strategy == "drop rows" %}
    TRAIN_GROUPS = GROUP_VALUES[keep][train_i]
{% else %}
    TRAIN_GROUPS = GROUP_VALUES[train_i]
{% endif %}
{% endif %}
    parts = {"train": train_i, "val": val_i, "test": test_i}
{% if table_steps %}
    train_x, val_x, test_x = preprocess_tables(
        x.iloc[train_i], [x.iloc[train_i], x.iloc[val_i], x.iloc[test_i]])
    feats = {"train": train_x, "val": val_x, "test": test_x}
{% elif text %}
    texts = {k: [x[i] for i in v] for k, v in parts.items()}
    train_x, val_x, test_x = encode_texts(texts["train"],
                                          [texts["train"], texts["val"], texts["test"]])
    feats = {"train": train_x, "val": val_x, "test": test_x}
{% elif audio %}
    x = audio_features(x)
    feats = {k: x[v] for k, v in parts.items()}
{% else %}
    feats = {k: x[v] for k, v in parts.items()}
{% endif %}
    targets = {k: np.asarray(y)[v] for k, v in parts.items()}
{% if not text %}
    pipeline = NumericPipeline().fit(feats["train"])
    INFERENCE_STATE["numeric"] = dict(vars(pipeline))
    feats = {k: pipeline.transform(v) for k, v in feats.items()}
{% endif %}
    out = {}
    for k in parts:
        xs = feats[k]
        if xs.ndim == 2 and len(INPUT_SHAPE) > 1 and xs.shape[1] == int(np.prod(INPUT_SHAPE)):
            xs = xs.reshape(len(xs), *INPUT_SHAPE)
{% if pairs %}
        if xs.shape[2:] != tuple(INPUT_SHAPE):   # each sample is a pair (or triplet) of items
            raise SystemExit(f"each {{ 'triplet' if pairs == 'triplet' else 'pair' }}'s items have shape {xs.shape[2:]} but the model "
                             f"expects {tuple(INPUT_SHAPE)}")
{% else %}
        if xs.shape[1:] != tuple(INPUT_SHAPE):
            raise SystemExit(f"data samples have shape {xs.shape[1:]} but the model expects "
                             f"{tuple(INPUT_SHAPE)}")
{% endif %}
        out[k] = (xs, targets[k].astype({{ y_dtype }}))
    return out
{% endif %}


def prepare_inputs(raw: list) -> np.ndarray:
    """Raw inputs -> a model-ready batch, preprocessed exactly as during training.

{% if table_steps %}    raw: records, e.g. [{"column": value, ...}, ...] with the training columns.
{% elif text %}    raw: texts, e.g. ["first document", "second document"].
{% elif audio %}    raw: mono waveforms ({{ sample_rate }} Hz float lists) or paths to .wav files.
{% elif timeseries %}    raw: windows of {{ d.window }} time steps x feature columns.
{% elif d.block in ("data.image_folder", "data.torchvision") %}    raw: images as file paths, bytes, base64 strings or PIL images.
{% else %}    raw: samples shaped like INPUT_SHAPE (or flattened).
{% endif %}    """
{% if table_steps %}
    import pandas as pd

    frame = pd.DataFrame.from_records(raw)
    columns = INFERENCE_STATE["table"]["columns"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    x = table_transform(frame[columns], INFERENCE_STATE["table"])
{% elif text %}
    x = encode_batch([str(t) for t in raw])
{% elif audio %}
    n = int(round({{ d.duration }} * {{ sample_rate }}))
    waves = []
    for item in raw:
        if isinstance(item, str):
            waves.append(read_wav(item, {{ sample_rate }}, n))
        else:
            wave = np.asarray(item, dtype=np.float32).reshape(-1)[:n]
            waves.append(np.pad(wave, (0, n - len(wave))))
    x = audio_features(np.stack(waves))
{% elif timeseries %}
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.shape[1] != {{ d.window }}:
        raise ValueError(f"each window needs {{ d.window }} time steps, got {x.shape[1]}")
{% elif d.block in ("data.image_folder", "data.torchvision") %}
    x = np.stack([image_array(item) for item in raw])
{% else %}
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
{% endif %}
{% if not text %}
    pipeline = NumericPipeline.__new__(NumericPipeline)
    vars(pipeline).update(INFERENCE_STATE.get("numeric") or {})
{% if timeseries %}
    n, steps, channels = x.shape
    x = pipeline.transform(x.reshape(-1, channels)).reshape(n, steps, -1)
{% if ts_channels_first %}
    x = x.transpose(0, 2, 1)
{% endif %}
{% else %}
    x = pipeline.transform(x)
{% endif %}
{% endif %}
    if x.ndim == 2 and len(INPUT_SHAPE) > 1 and x.shape[1] == int(np.prod(INPUT_SHAPE)):
        x = x.reshape(len(x), *INPUT_SHAPE)
    if tuple(x.shape[1:]) != tuple(INPUT_SHAPE):
        raise ValueError(f"inputs have shape {tuple(x.shape[1:])}; the model expects "
                         f"{tuple(INPUT_SHAPE)}")
    return x
{% if d.block in ("data.image_folder", "data.torchvision") %}


def image_array(item) -> np.ndarray:
    """An image (path, bytes, base64 string or PIL image) as a [C, H, W] array in [0, 1]."""
    img = open_image(item).convert("L" if INPUT_SHAPE[0] == 1 else "RGB")
    arr = np.asarray(img.resize((INPUT_SHAPE[2], INPUT_SHAPE[1])), dtype=np.float32) / 255.0
    return arr[None] if arr.ndim == 2 else arr.transpose(2, 0, 1)
{% endif %}
{% if timeseries %}


def inverse_targets(values: np.ndarray) -> np.ndarray:
    """Undo target scaling so predictions are in the original units."""
    state = INFERENCE_STATE.get("target")
    if not state:
        return values
    flat = values.reshape(len(values), -1).astype(np.float64)
{% if steps["prep.minmax"] %}
    span = np.where(state["max"] - state["min"] > 0, state["max"] - state["min"], 1.0)
    lo, hi = {{ steps["prep.minmax"].range_min }}, {{ steps["prep.minmax"].range_max }}
    reps = flat.shape[1] // len(span)
    flat = (flat - lo) / (hi - lo) * np.tile(span, reps) + np.tile(state["min"], reps)
{% endif %}
{% if steps["prep.normalize"] %}
    reps = flat.shape[1] // len(np.atleast_1d(state["std"]))
    flat = flat * np.tile(state["std"], reps) + np.tile(state["mean"], reps)
{% endif %}
    return flat.reshape(values.shape)
{% endif %}
'''


GROUP_WEIGHTS = r'''def group_column(frame, name: str) -> np.ndarray:
    """The group of each row, as text (missing values are a group of their own)."""
    if name not in frame.columns:
        raise SystemExit(f"group column {name!r} not found; columns: {list(frame.columns)}")
    return frame[name].astype(str).where(frame[name].notna(), "(missing)").to_numpy()


def group_weights(groups, labels) -> np.ndarray:
    """Reweighing (Kamiran & Calders): a row of group g and class c weighs
    P(g) P(c) / P(g, c), so in the weighted rows every group has each class as often as
    the data overall, and the group says nothing about the class."""
    groups, labels = np.asarray(groups), np.asarray(labels).reshape(-1)
    weights = np.ones(len(labels))
    for g in np.unique(groups):
        in_group = groups == g
        for c in np.unique(labels):
            both = in_group & (labels == c)
            if both.any():
                weights[both] = in_group.mean() * (labels == c).mean() / both.mean()
    return weights'''
