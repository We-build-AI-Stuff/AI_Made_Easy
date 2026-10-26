"""Datasets and preprocessing: the data side of the training pipeline.

Every dataset declares its *modality*; every preprocessing block declares the
modalities it applies to and *when* it runs (``stage``):

* ``fit``     — statistics are fitted on the training split only, then applied
                to all splits (no leakage): normalize, min-max, impute, ...
* ``always``  — deterministic transforms applied to every split: resize, crop
* ``train``   — random augmentation applied to training batches only
* ``config``  — pipeline settings: split, loader, class balancing, tokenizer

Image transforms render to torchvision ``transforms.v2`` (PyTorch) and Keras
preprocessing layers (Keras 3); ``None`` means no faithful equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, P

IMAGE, TABULAR, TEXT, AUDIO, TIMESERIES, ARRAY = (
    "image", "tabular", "text", "audio", "timeseries", "array")
NUMERIC = {TABULAR, ARRAY, TIMESERIES}
ALL = {IMAGE, TABULAR, TEXT, AUDIO, TIMESERIES, ARRAY}

TORCHVISION = {  # name: (class, sample shape [C, H, W], classes, split keyword)
    "mnist": ("MNIST", [1, 28, 28], 10, "train"),
    "fashion_mnist": ("FashionMNIST", [1, 28, 28], 10, "train"),
    "kmnist": ("KMNIST", [1, 28, 28], 10, "train"),
    "cifar10": ("CIFAR10", [3, 32, 32], 10, "train"),
    "cifar100": ("CIFAR100", [3, 32, 32], 100, "train"),
    "svhn": ("SVHN", [3, 32, 32], 10, "split"),
    "stl10": ("STL10", [3, 96, 96], 10, "split"),
    "usps": ("USPS", [1, 16, 16], 10, "train"),
}
KERAS_DATASETS = {"mnist": "mnist", "fashion_mnist": "fashion_mnist",
                  "cifar10": "cifar10", "cifar100": "cifar100"}
SKLEARN = {  # name: (loader, features, classes or 0 for regression)
    "iris": ("load_iris", 4, 3), "wine": ("load_wine", 13, 3),
    "breast_cancer": ("load_breast_cancer", 30, 2), "digits": ("load_digits", 64, 10),
    "diabetes": ("load_diabetes", 10, 0),
    "california_housing": ("fetch_california_housing", 8, 0),
}


@dataclass(frozen=True)
class DataBlock:
    type_id: str
    name: str
    category: str
    params: tuple = ()
    desc: str = ""
    modality: str | None = None            # datasets
    applies_to: frozenset = frozenset()    # preprocessing
    stage: str = "config"
    torch: Callable | None = None          # image transforms: v2 expression
    keras: Callable | None = None          # image transforms: keras layer
    checks: Callable | None = None
    meta: dict = field(default_factory=dict)


def _range_check(lo_key: str, hi_key: str):
    def fn(p):
        if float(p[lo_key]) > float(p[hi_key]):
            return [("error", f"{lo_key} must not exceed {hi_key}")]
        return []
    return fn


def _split_check(p):
    total = float(p["val_fraction"]) + float(p["test_fraction"])
    if total >= 1.0:
        return [("error", f"val_fraction + test_fraction = {total:g} leaves no training data")]
    if total > 0.5:
        return [("warning", f"{total:.0%} of the data is held out; training uses only "
                            f"{1 - total:.0%}")]
    return []


def _ts_check(p):
    out = []
    if int(p["stride"]) > int(p["window"]):
        out.append(("warning", "stride larger than window skips time steps between windows"))
    if not str(p["target_columns"]).strip():
        out.append(("error", "target_columns is empty — name the column(s) to forecast"))
    return out


# =================================================================== datasets

DATASETS: list[DataBlock] = [
    DataBlock(
        "data.torchvision", "Torchvision Dataset", "Data",
        (P("dataset", "enum", "mnist", options=tuple(TORCHVISION)),
         P("download", "bool", True), P("data_dir", "str", "~/.aime/data")),
        desc="Standard image benchmarks (MNIST, CIFAR, SVHN, STL-10, ...).", modality=IMAGE),
    DataBlock(
        "data.image_folder", "Image Folder", "Data",
        (P("root", "str", "images/", help="One subfolder per class"),
         P("grayscale", "bool", False)),
        desc="Images organised as root/<class>/<file>.", modality=IMAGE),
    DataBlock(
        "data.csv", "Table File", "Data",
        (P("path", "str", "data.csv"),
         P("format", "enum", "csv", options=("csv", "tsv", "parquet", "excel", "json")),
         P("target_column", "str", "label"),
         P("feature_columns", "str", "", help="Comma-separated; empty = all other columns")),
        desc="Tabular data (CSV/TSV/Parquet/Excel/JSON) loaded with pandas.", modality=TABULAR),
    DataBlock(
        "data.sklearn", "Built-in Tabular Dataset", "Data",
        (P("dataset", "enum", "iris", options=tuple(SKLEARN)),),
        desc="scikit-learn reference datasets (Iris, Wine, Breast Cancer, Digits, ...).",
        modality=TABULAR),
    DataBlock(
        "data.synthetic", "Synthetic Data", "Data",
        (P("kind", "enum", "classification",
           options=("classification", "moons", "circles", "regression")),
         P("n_samples", "int", 1000, lo=10), P("n_features", "int", 20, lo=1),
         P("n_classes", "int", 2, lo=2), P("noise", "float", 0.1, lo=0.0),
         P("seed", "int", 42, lo=0)),
        desc="Generated data for quick experiments.", modality=TABULAR,
        checks=lambda p: ([("error", "moons and circles are 2-D: set n_features to 2")]
                          if p["kind"] in ("moons", "circles") and int(p["n_features"]) != 2
                          else [])),
    DataBlock(
        "data.synthetic_table", "Synthetic Table", "Data",
        (P("task", "enum", "classification", options=("classification", "regression")),
         P("n_samples", "int", 3000, lo=50), P("noise", "float", 0.5, lo=0.0),
         P("seed", "int", 0, lo=0)),
        desc="A generated customer table with numbers and categories (age, income, tenure, "
             "city, plan, channel); churn (classification) or monthly spend (regression) "
             "depends on interactions between categories — for trying tabular models.",
        modality=TABULAR),
    DataBlock(
        "data.numpy", "NumPy Archive", "Data",
        (P("path", "str", "data.npz"), P("x_key", "str", "x"), P("y_key", "str", "y")),
        desc="Arrays stored in an .npz file (x: samples, y: targets).", modality=ARRAY),
    DataBlock(
        "data.pairs", "Labelled Pairs", "Data",
        (P("path", "str", "", help="Your table of pairs; empty = generated demo pairs "
                                   "(8 features per item)"),
         P("format", "enum", "csv", options=("csv", "tsv", "parquet")),
         P("label_column", "str", "label",
           help="1 / 0: same / different (Cosine Embedding) or a ranks above b (Margin "
                "Ranking)"),
         P("left_prefix", "str", "a_", help="Columns of the first item, e.g. a_height"),
         P("right_prefix", "str", "b_", help="The same columns of the second item")),
        desc="A table of pairs: each row holds two items (the a_ and b_ columns) and a "
             "label. The model is the encoder; it embeds both items. Train it with Cosine "
             "Embedding (same or different) or Margin Ranking (which ranks higher).",
        modality=ARRAY),
    DataBlock(
        "data.triplets", "Labelled Triplets", "Data",
        (P("path", "str", "", help="Your table of triplets; empty = generated demo triplets "
                                   "(8 features per item)"),
         P("format", "enum", "csv", options=("csv", "tsv", "parquet")),
         P("anchor_prefix", "str", "a_", help="Columns of the anchor item, e.g. a_height"),
         P("positive_prefix", "str", "p_", help="The same columns of an item like the anchor"),
         P("negative_prefix", "str", "n_", help="The same columns of an item unlike it")),
        desc="A table of triplets: each row holds an anchor, an item like it (positive) and "
             "one unlike it (negative), in a_, p_ and n_ columns. The model is the encoder; "
             "Triplet loss trains it so the positive ends up closer to the anchor than the "
             "negative. Accuracy: how often it does on the test triplets.",
        modality=ARRAY),
    DataBlock(
        "data.json", "JSON Records", "Data",
        (P("path", "str", "data.jsonl"), P("x_field", "str", "x"), P("y_field", "str", "y")),
        desc="JSON array or JSON Lines with feature and target fields.", modality=ARRAY),
    DataBlock(
        "data.huggingface", "Hugging Face Dataset", "Data",
        (P("repo_id", "str", "mnist"), P("split", "str", "train"),
         P("x_field", "str", "image"), P("y_field", "str", "label")),
        desc="Dataset from the Hugging Face Hub (numeric or image fields).", modality=ARRAY),
    DataBlock(
        "data.text_csv", "Text Table", "Data",
        (P("path", "str", "texts.csv"),
         P("format", "enum", "csv", options=("csv", "tsv", "json", "parquet")),
         P("text_column", "str", "text"), P("label_column", "str", "label")),
        desc="Text classification data: one text column, one label column.", modality=TEXT),
    DataBlock(
        "data.text_folder", "Text Folder", "Data",
        (P("root", "str", "texts/", help="One subfolder per class with .txt files"),),
        desc="Plain-text documents organised as root/<class>/<file>.txt.", modality=TEXT),
    DataBlock(
        "data.audio_folder", "Audio Folder", "Data",
        (P("root", "str", "audio/", help="One subfolder per class with .wav files"),
         P("sample_rate", "int", 16000, lo=1000), P("duration", "float", 1.0, lo=0.05,
                                                     help="Seconds; clips are padded/trimmed")),
        desc="WAV clips organised as root/<class>/<file>.wav.", modality=AUDIO),
    DataBlock(
        "data.timeseries_csv", "Time Series Table", "Data",
        (P("path", "str", "series.csv"),
         P("target_columns", "str", "value", help="Column(s) to forecast"),
         P("feature_columns", "str", "", help="Input columns; empty = all numeric"),
         P("window", "int", 32, lo=1, help="Past time steps per sample"),
         P("horizon", "int", 1, lo=1, help="Future steps to predict"),
         P("stride", "int", 1, lo=1),
         P("layout", "enum", "sequence [L, C]",
           options=("sequence [L, C]", "channels [C, L]"),
           help="[L, C] for RNN/Transformer, [C, L] for Conv1D")),
        desc="Sliding-window forecasting from a time-ordered table.", modality=TIMESERIES,
        checks=_ts_check),
]


# ============================================================ preprocessing

def _p_tv(expr: str) -> Callable:
    return lambda p: expr.format(**p)


PREPROCESSING: list[DataBlock] = [
    # ---------------------------------------------------------- pipeline
    DataBlock(
        "prep.split", "Train / Val / Test Split", "Preprocessing",
        (P("val_fraction", "float", 0.1, lo=0.0, hi=0.9),
         P("test_fraction", "float", 0.1, lo=0.0, hi=0.9),
         P("shuffle", "bool", True, help="Ignored for time series (kept chronological)"),
         P("stratify", "bool", True, help="Preserve class proportions in every split"),
         P("seed", "int", 42, lo=0)),
        desc="Hold out validation and test data.", applies_to=frozenset(ALL),
        checks=_split_check),
    DataBlock(
        "prep.dataloader", "Data Loader", "Preprocessing",
        (P("batch_size", "int", 0, lo=0, help="0 = use the Trainer batch size"),
         P("num_workers", "int", 0, lo=0), P("pin_memory", "bool", True),
         P("drop_last", "bool", False)),
        desc="Batching and worker settings (PyTorch).", applies_to=frozenset(ALL)),
    DataBlock(
        "prep.class_balance", "Class Balancing", "Preprocessing",
        (P("strategy", "enum", "class weights",
           options=("class weights", "oversample", "group weights")),
         P("group_column", "str", "",
           help="group weights: the column (e.g. region) whose groups are weighted so "
                "that each gets every class as often as the data overall")),
        desc="Counter class imbalance with loss weights or a weighted sampler, or weight "
             "rows so a group column says nothing about the class (group weights).",
        applies_to=frozenset(ALL)),
    # ---------------------------------------------------------- tabular
    DataBlock(
        "prep.drop_columns", "Drop Columns", "Preprocessing",
        (P("columns", "str", "", help="Comma-separated column names"),),
        desc="Remove columns before training.", applies_to=frozenset({TABULAR}), stage="fit"),
    DataBlock(
        "prep.impute", "Impute Missing Values", "Preprocessing",
        (P("strategy", "enum", "mean", options=("mean", "median", "mode", "constant",
                                                 "drop rows")),
         P("constant", "float", 0.0, lo=-1e12)),
        desc="Fill missing values with statistics fitted on the training split.",
        applies_to=frozenset(NUMERIC), stage="fit"),
    DataBlock(
        "prep.one_hot", "One-Hot Encode", "Preprocessing",
        (P("columns", "str", "", help="Empty = every non-numeric column"),
         P("max_categories", "int", 50, lo=2)),
        desc="Expand categorical columns into indicator features.",
        applies_to=frozenset({TABULAR}), stage="fit"),
    DataBlock(
        "prep.ordinal_encode", "Ordinal Encode", "Preprocessing",
        (P("columns", "str", "", help="Empty = every non-numeric column"),),
        desc="Map categories to integer codes (unknown → -1).",
        applies_to=frozenset({TABULAR}), stage="fit"),
    DataBlock(
        "prep.log_transform", "Log Transform", "Preprocessing",
        (P("columns", "str", "", help="Empty = all non-negative numeric columns"),),
        desc="log(1 + x) for skewed, non-negative features.",
        applies_to=frozenset(NUMERIC), stage="fit"),
    DataBlock(
        "prep.clip_outliers", "Clip Outliers", "Preprocessing",
        (P("method", "enum", "iqr", options=("iqr", "zscore", "quantile")),
         P("threshold", "float", 1.5, lo=0.0,
           help="IQR multiplier, z-score limit, or quantile (e.g. 0.01)")),
        desc="Clip extreme values using bounds fitted on the training split.",
        applies_to=frozenset(NUMERIC), stage="fit"),
    DataBlock(
        "prep.normalize", "Standardize (z-score)", "Preprocessing",
        (P("mode", "enum", "fit", options=("fit", "fixed"),
           help="fit: compute from training data; fixed: use the values below"),
         P("mean", "str", "0.0", help="Fixed mean(s), comma-separated (per channel)"),
         P("std", "str", "1.0", help="Fixed std(s), comma-separated (per channel)")),
        desc="(x − mean) / std per feature or per image channel.",
        applies_to=frozenset(ALL - {TEXT}), stage="fit",
        checks=lambda p: ([("error", "std values must be positive")]
                          if p["mode"] == "fixed" and any(
                              float(v) <= 0 for v in str(p["std"]).split(",") if v.strip())
                          else [])),
    DataBlock(
        "prep.minmax", "Min-Max Scale", "Preprocessing",
        (P("range_min", "float", 0.0, lo=-1e9), P("range_max", "float", 1.0, lo=-1e9)),
        desc="Rescale each feature to [range_min, range_max].",
        applies_to=frozenset(NUMERIC | {AUDIO}), stage="fit",
        checks=lambda p: ([("error", "range_min must be below range_max")]
                          if float(p["range_min"]) >= float(p["range_max"]) else [])),
    DataBlock(
        "prep.robust_scale", "Robust Scale", "Preprocessing", (),
        desc="(x − median) / IQR per feature (outlier-resistant).",
        applies_to=frozenset(NUMERIC), stage="fit"),
    DataBlock(
        "prep.variance_filter", "Variance Filter", "Preprocessing",
        (P("threshold", "float", 0.0, lo=0.0, help="Drop features with variance ≤ threshold"),),
        desc="Remove constant / near-constant features.",
        applies_to=frozenset({TABULAR, ARRAY}), stage="fit"),
    # ---------------------------------------------------------- text
    DataBlock(
        "prep.text_clean", "Clean Text", "Text Processing",
        (P("lowercase", "bool", True), P("strip_punctuation", "bool", True),
         P("strip_digits", "bool", False), P("strip_html", "bool", True)),
        desc="Normalise raw text before tokenization.", applies_to=frozenset({TEXT}),
        stage="always"),
    DataBlock(
        "prep.tokenize", "Tokenize", "Text Processing",
        (P("method", "enum", "word", options=("word", "char", "huggingface")),
         P("max_length", "int", 128, lo=1, help="Tokens per sample (pad / truncate)"),
         P("vocab_size", "int", 20000, lo=10, help="Word/char vocabulary incl. <pad>, <unk>"),
         P("min_freq", "int", 1, lo=1),
         P("hf_tokenizer", "str", "bert-base-uncased",
           help="Hugging Face tokenizer when method = huggingface")),
        desc="Turn text into fixed-length integer id sequences [max_length].",
        applies_to=frozenset({TEXT}), stage="fit"),
    # ---------------------------------------------------------- audio
    DataBlock(
        "prep.audio_features", "Audio Features", "Audio Processing",
        (P("kind", "enum", "mel_spectrogram",
           options=("waveform", "spectrogram", "mel_spectrogram", "mfcc")),
         P("n_fft", "int", 512, lo=16), P("hop_length", "int", 160, lo=1),
         P("n_mels", "int", 64, lo=4), P("n_mfcc", "int", 20, lo=2),
         P("log_scale", "bool", True)),
        desc="Waveform → [1, samples], spectrogram/mel → [1, F, T], MFCC → [1, n_mfcc, T].",
        applies_to=frozenset({AUDIO}), stage="always",
        checks=lambda p: ([("error", "n_mfcc cannot exceed n_mels")]
                          if p["kind"] == "mfcc" and int(p["n_mfcc"]) > int(p["n_mels"]) else [])),
    DataBlock(
        "prep.spec_augment", "SpecAugment", "Audio Processing",
        (P("freq_mask", "int", 8, lo=0, help="Max masked frequency bins"),
         P("time_mask", "int", 16, lo=0, help="Max masked time frames"),
         P("p", "float", 0.5, lo=0.0, hi=1.0)),
        desc="Random frequency/time masking of spectrograms (training only).",
        applies_to=frozenset({AUDIO}), stage="train"),
    # ---------------------------------------------------------- image (always)
    DataBlock(
        "prep.resize", "Resize", "Image Transforms",
        (P("height", "int", 32, lo=1), P("width", "int", 32, lo=1)),
        desc="Resize every image.", applies_to=frozenset({IMAGE}), stage="always",
        torch=_p_tv("v2.Resize(({height}, {width}), antialias=True)"),
        keras=_p_tv("layers.Resizing({height}, {width})")),
    DataBlock(
        "prep.center_crop", "Center Crop", "Image Transforms",
        (P("size", "int", 28, lo=1),),
        desc="Crop the central square.", applies_to=frozenset({IMAGE}), stage="always",
        torch=_p_tv("v2.CenterCrop({size})"), keras=_p_tv("layers.CenterCrop({size}, {size})")),
    DataBlock(
        "prep.grayscale", "Grayscale", "Image Transforms",
        (P("channels", "enum", "1", options=("1", "3")),),
        desc="Convert images to grayscale (1 or 3 identical channels).",
        applies_to=frozenset({IMAGE}), stage="always",
        torch=_p_tv("v2.Grayscale(num_output_channels={channels})")),
    # ---------------------------------------------------------- image (train)
    DataBlock(
        "prep.random_resized_crop", "Random Resized Crop", "Augmentation",
        (P("size", "int", 224, lo=1), P("scale_min", "float", 0.08, lo=0.01, hi=1.0),
         P("scale_max", "float", 1.0, lo=0.01, hi=1.0)),
        desc="Crop a random area and aspect ratio, then resize (Inception-style).",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomResizedCrop({size}, scale=({scale_min}, {scale_max}), antialias=True)"),
        checks=_range_check("scale_min", "scale_max")),
    DataBlock(
        "prep.random_crop", "Random Crop", "Augmentation",
        (P("size", "int", 32, lo=1), P("padding", "int", 4, lo=0)),
        desc="Pad, then crop a random window of the given size.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomCrop({size}, padding={padding})"),
        keras=lambda p: (f"layers.RandomCrop({p['size']}, {p['size']})" if not int(p["padding"])
                         else None)),
    DataBlock(
        "prep.random_flip", "Random Flip", "Augmentation",
        (P("mode", "enum", "horizontal", options=("horizontal", "vertical", "both")),),
        desc="Random horizontal and/or vertical flips.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=lambda p: {"horizontal": "v2.RandomHorizontalFlip()",
                         "vertical": "v2.RandomVerticalFlip()",
                         "both": "v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()"}[p["mode"]],
        keras=lambda p: 'layers.RandomFlip("{}")'.format(
            {"horizontal": "horizontal", "vertical": "vertical",
             "both": "horizontal_and_vertical"}[p["mode"]])),
    DataBlock(
        "prep.random_rotation", "Random Rotation", "Augmentation",
        (P("degrees", "float", 15.0, lo=0.0, hi=180.0),),
        desc="Rotate by a random angle in [−degrees, degrees].",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomRotation({degrees})"),
        keras=lambda p: f"layers.RandomRotation({float(p['degrees']) / 360.0:.6g})"),
    DataBlock(
        "prep.random_affine", "Random Affine", "Augmentation",
        (P("degrees", "float", 0.0, lo=0.0, hi=180.0),
         P("translate", "float", 0.1, lo=0.0, hi=1.0, help="Max shift as image fraction"),
         P("scale_min", "float", 0.9, lo=0.01), P("scale_max", "float", 1.1, lo=0.01),
         P("shear", "float", 0.0, lo=0.0, hi=90.0)),
        desc="Random rotation, translation, scaling and shear.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomAffine(degrees={degrees}, translate=({translate}, {translate}), "
                    "scale=({scale_min}, {scale_max}), shear={shear})"),
        checks=_range_check("scale_min", "scale_max")),
    DataBlock(
        "prep.random_perspective", "Random Perspective", "Augmentation",
        (P("distortion", "float", 0.5, lo=0.0, hi=1.0), P("p", "float", 0.5, lo=0.0, hi=1.0)),
        desc="Random perspective warp.", applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomPerspective(distortion_scale={distortion}, p={p})")),
    DataBlock(
        "prep.color_jitter", "Color Jitter", "Augmentation",
        (P("brightness", "float", 0.2, lo=0.0), P("contrast", "float", 0.2, lo=0.0),
         P("saturation", "float", 0.2, lo=0.0), P("hue", "float", 0.05, lo=0.0, hi=0.5)),
        desc="Random brightness, contrast, saturation and hue.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.ColorJitter(brightness={brightness}, contrast={contrast}, "
                    "saturation={saturation}, hue={hue})"),
        keras=_p_tv("layers.RandomColorJitter(value_range=(0, 1), brightness_factor={brightness}, "
                    "contrast_factor={contrast}, saturation_factor=(1 - {saturation}, "
                    "1 + {saturation}), hue_factor={hue})")),
    DataBlock(
        "prep.random_grayscale", "Random Grayscale", "Augmentation",
        (P("p", "float", 0.1, lo=0.0, hi=1.0),),
        desc="Convert to grayscale with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomGrayscale(p={p})"),
        keras=_p_tv("layers.RandomGrayscale(factor={p})")),
    DataBlock(
        "prep.gaussian_blur", "Gaussian Blur", "Augmentation",
        (P("kernel_size", "int", 3, lo=1), P("sigma_min", "float", 0.1, lo=0.0),
         P("sigma_max", "float", 2.0, lo=0.0)),
        desc="Blur with a random Gaussian sigma.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.GaussianBlur({kernel_size}, sigma=({sigma_min}, {sigma_max}))"),
        checks=lambda p: (([("error", "kernel_size must be odd")]
                           if int(p["kernel_size"]) % 2 == 0 else [])
                          + _range_check("sigma_min", "sigma_max")(p))),
    DataBlock(
        "prep.random_erasing", "Random Erasing", "Augmentation",
        (P("p", "float", 0.25, lo=0.0, hi=1.0), P("scale_min", "float", 0.02, lo=0.0, hi=1.0),
         P("scale_max", "float", 0.33, lo=0.0, hi=1.0)),
        desc="Erase a random rectangle (Cutout).", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomErasing(p={p}, scale=({scale_min}, {scale_max}))"),
        meta={"after_tensor": True}, checks=_range_check("scale_min", "scale_max")),
    DataBlock(
        "prep.random_autocontrast", "Random Autocontrast", "Augmentation",
        (P("p", "float", 0.5, lo=0.0, hi=1.0),),
        desc="Maximise contrast with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomAutocontrast(p={p})")),
    DataBlock(
        "prep.random_equalize", "Random Equalize", "Augmentation",
        (P("p", "float", 0.5, lo=0.0, hi=1.0),),
        desc="Histogram-equalise with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomEqualize(p={p})")),
    DataBlock(
        "prep.random_invert", "Random Invert", "Augmentation",
        (P("p", "float", 0.5, lo=0.0, hi=1.0),),
        desc="Invert colours with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomInvert(p={p})"),
        keras=_p_tv("layers.RandomInvert(factor={p}, value_range=(0, 1))")),
    DataBlock(
        "prep.random_posterize", "Random Posterize", "Augmentation",
        (P("bits", "int", 4, lo=1, hi=8), P("p", "float", 0.5, lo=0.0, hi=1.0)),
        desc="Reduce colour bits with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomPosterize(bits={bits}, p={p})")),
    DataBlock(
        "prep.random_solarize", "Random Solarize", "Augmentation",
        (P("threshold", "float", 0.5, lo=0.0, hi=1.0), P("p", "float", 0.5, lo=0.0, hi=1.0)),
        desc="Invert pixels above a threshold with probability p.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomSolarize(threshold={threshold}, p={p})")),
    DataBlock(
        "prep.random_sharpness", "Random Sharpness", "Augmentation",
        (P("sharpness_factor", "float", 2.0, lo=0.0), P("p", "float", 0.5, lo=0.0, hi=1.0)),
        desc="Adjust sharpness with probability p.", applies_to=frozenset({IMAGE}),
        stage="train", torch=_p_tv("v2.RandomAdjustSharpness({sharpness_factor}, p={p})")),
    DataBlock(
        "prep.auto_augment", "AutoAugment", "Augmentation",
        (P("policy", "enum", "imagenet", options=("imagenet", "cifar10", "svhn")),),
        desc="Learned augmentation policy (Cubuk et al., 2019).",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=lambda p: (f"v2.AutoAugment(v2.AutoAugmentPolicy."
                         f"{p['policy'].upper()})")),
    DataBlock(
        "prep.rand_augment", "RandAugment", "Augmentation",
        (P("num_ops", "int", 2, lo=1), P("magnitude", "int", 9, lo=0, hi=30)),
        desc="Random sequence of augmentations at a fixed magnitude.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandAugment(num_ops={num_ops}, magnitude={magnitude})"),
        keras=lambda p: (f"layers.RandAugment(value_range=(0, 1), num_ops={p['num_ops']}, "
                         f"factor={int(p['magnitude']) / 30:.3g})")),
    DataBlock(
        "prep.trivial_augment", "TrivialAugment", "Augmentation", (),
        desc="One random augmentation of random strength per image.",
        applies_to=frozenset({IMAGE}), stage="train", torch=lambda p: "v2.TrivialAugmentWide()"),
    DataBlock(
        "prep.augmix", "AugMix", "Augmentation",
        (P("severity", "int", 3, lo=1, hi=10),),
        desc="Mix several augmentation chains (robustness).",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.AugMix(severity={severity})"),
        keras=lambda p: "layers.AugMix(value_range=(0, 1))"),
    # 3.6
    DataBlock(
        "prep.random_jpeg", "Random JPEG Quality", "Augmentation",
        (P("quality_min", "int", 50, lo=1, hi=100), P("quality_max", "int", 95, lo=1, hi=100)),
        desc="Re-encode each image as a JPEG of random quality: robust to compression "
             "artifacts in photos from the web and phones.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.JPEG(({quality_min}, {quality_max}))"),
        checks=_range_check("quality_min", "quality_max")),
    DataBlock(
        "prep.elastic_transform", "Elastic Transform", "Augmentation",
        (P("alpha", "float", 50.0, lo=0.0, hi=1000.0, help="How far pixels move"),
         P("sigma", "float", 5.0, lo=0.1, hi=100.0, help="How smooth the displacement is (images must be over 8 × sigma wide)")),
        desc="Warps each image as if drawn on rubber (Simard et al., 2003): handwriting, "
             "cells and other shapes that bend.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.ElasticTransform(alpha={alpha}, sigma={sigma})")),
    DataBlock(
        "prep.random_channel_permutation", "Random Channel Permutation", "Augmentation", (),
        desc="Shuffle the colour channels of each image, so the model cannot rely on "
             "exact colours.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=lambda p: "v2.RandomChannelPermutation()"),
    DataBlock(
        "prep.gaussian_noise", "Gaussian Noise (images)", "Augmentation",
        (P("sigma", "float", 0.05, lo=0.0, hi=1.0, help="Noise level on 0-1 pixels"),),
        desc="Adds random noise to every pixel (after scaling to 0-1): robust to sensor "
             "noise and low light.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.GaussianNoise(sigma={sigma}, clip=True)"),
        keras=_p_tv("layers.GaussianNoise({sigma})"), meta={"after_tensor": True}),
    # 3.7
    DataBlock(
        "prep.photometric_distort", "Photometric Distort", "Augmentation",
        (P("p", "float", 0.5, lo=0.0, hi=1.0, help="Chance of each change"),),
        desc="SSD's colour augmentation (Liu et al., 2016): random brightness, contrast, "
             "saturation and hue, each with probability p, in a random order. A strong "
             "default for detection and photos in varied lighting.",
        applies_to=frozenset({IMAGE}), stage="train",
        torch=_p_tv("v2.RandomPhotometricDistort(p={p})")),
    DataBlock(
        "prep.mixup_cutmix", "MixUp / CutMix", "Augmentation",
        (P("mode", "enum", "mixup", options=("mixup", "cutmix", "both")),
         P("alpha", "float", 1.0, lo=0.0)),
        desc="Blend pairs of images and their labels per batch (classification).",
        applies_to=frozenset({IMAGE}), stage="batch"),
]

DATASET_IDS = tuple(d.type_id for d in DATASETS)
PREP_IDS = tuple(d.type_id for d in PREPROCESSING)
BLOCKS: dict[str, DataBlock] = {d.type_id: d for d in (*DATASETS, *PREPROCESSING)}

_FAMILY = {"Data": "data"}


def register_all() -> None:
    reg = get_registry()
    for blk in (*DATASETS, *PREPROCESSING):
        family = "data" if blk.category == "Data" else "preprocess"
        reg.register(BlockDefinition(
            type_id=blk.type_id, display_name=blk.name, category=blk.category,
            color=family_color(family), params=tuple(blk.params), checks_fn=blk.checks,
            description=blk.desc,
            library=("PyTorch" + (" · Keras" if blk.keras else "")) if blk.torch else "PyTorch · Keras",
            meta={"modality": blk.modality,
                  "applies_to": sorted(blk.applies_to), "stage": blk.stage},
        ))
