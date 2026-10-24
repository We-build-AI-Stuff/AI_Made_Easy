"""Training components: losses, optimizers, LR schedulers, metrics, trainer.

One table per component kind is the single source of truth: it registers the
canvas blocks *and* drives code generation for PyTorch and Keras 3. Each entry
renders a framework expression from the block's resolved params; ``keras`` is
``None`` when Keras has no faithful equivalent (reported as a clear error).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, P
from ai_made_easy.core.training.train_helpers import TRAIN_HELPERS  # noqa: F401  (re-exported)

Render = Callable[[dict], str]


@dataclass(frozen=True)
class Component:
    type_id: str
    name: str
    params: tuple = ()
    torch: Render | None = None
    keras: Render | None = None
    desc: str = ""
    meta: dict = field(default_factory=dict)
    checks: Callable | None = None


def _floats(value) -> list[float]:
    return [float(v) for v in str(value).replace(";", ",").split(",") if v.strip()]


def _ints(value) -> list[int]:
    return [int(float(v)) for v in str(value).replace(";", ",").split(",") if v.strip()]


# ===================================================================== losses
# meta: task = multiclass | binary | multilabel | regression | distribution
#       target = int | float ; input = logits | probs | log_probs | raw

def _class_weight_arg(p: dict, torch_kw: str = "weight") -> str:
    weights = _floats(p.get("class_weights", ""))
    return f"{torch_kw}=torch.tensor({weights}), " if weights else ""


def _weights_check(p):
    try:
        _floats(p.get("class_weights", ""))
    except ValueError:
        return [("error", "class_weights must be comma-separated numbers")]
    return []


LOSSES: list[Component] = [
    Component(
        "train.loss_cross_entropy", "Cross-Entropy",
        (P("label_smoothing", "float", 0.0, lo=0.0, hi=1.0),
         P("class_weights", "str", "", help="Optional per-class weights, comma-separated")),
        torch=lambda p: (f"nn.CrossEntropyLoss({_class_weight_arg(p)}"
                         f"label_smoothing={p['label_smoothing']})").replace(", )", ")"),
        keras=lambda p: (f"keras.losses.CategoricalCrossentropy(from_logits=True, "
                         f"label_smoothing={p['label_smoothing']})"
                         if float(p["label_smoothing"]) > 0 else
                         "keras.losses.SparseCategoricalCrossentropy(from_logits=True)"),
        desc="Multi-class classification on logits (softmax applied internally).",
        meta={"task": "multiclass", "target": "int", "input": "logits",
              "keras_one_hot": lambda p: float(p["label_smoothing"]) > 0},
        checks=_weights_check),
    Component(
        "train.loss_nll", "Negative Log-Likelihood",
        (P("class_weights", "str", "", help="Optional per-class weights"),),
        torch=lambda p: f"nn.NLLLoss({_class_weight_arg(p).rstrip(', ')})",
        keras=lambda p: ("lambda y, p: keras.losses.sparse_categorical_crossentropy("
                         "y, ops.exp(p))"),
        desc="Multi-class loss on log-probabilities (pair with a final LogSoftmax).",
        meta={"task": "multiclass", "target": "int", "input": "log_probs"},
        checks=_weights_check),
    Component(
        "train.loss_focal", "Focal Loss",
        (P("gamma", "float", 2.0, lo=0.0, help="Focusing parameter"),
         P("alpha", "float", 0.25, lo=0.0, hi=1.0, help="Balance factor (0 = off)")),
        torch=lambda p: f"FocalLoss(gamma={p['gamma']}, alpha={p['alpha']})",
        keras=lambda p: (f"keras.losses.CategoricalFocalCrossentropy(alpha={p['alpha']}, "
                         f"gamma={p['gamma']}, from_logits=True)"),
        desc="Cross-entropy that down-weights easy examples (class imbalance).",
        meta={"task": "multiclass", "target": "int", "input": "logits",
              "keras_one_hot": lambda p: True, "helper": "FocalLoss"}),
    Component(
        "train.loss_multi_margin", "Multi-Class Hinge",
        (P("margin", "float", 1.0, lo=0.0), P("p", "enum", "1", options=("1", "2"))),
        torch=lambda p: f"nn.MultiMarginLoss(p={p['p']}, margin={p['margin']})",
        desc="Multi-class hinge (SVM-style) loss on scores.",
        meta={"task": "multiclass", "target": "int", "input": "logits"}),
    Component(
        "train.loss_bce_logits", "Binary Cross-Entropy (logits)",
        (P("pos_weight", "float", 1.0, lo=0.0, help="Weight of positive examples"),),
        torch=lambda p: ("nn.BCEWithLogitsLoss()" if float(p["pos_weight"]) == 1.0 else
                         f"nn.BCEWithLogitsLoss(pos_weight=torch.tensor([{p['pos_weight']}]))"),
        keras=lambda p: "keras.losses.BinaryCrossentropy(from_logits=True)",
        desc="Binary (1 unit) or multi-label (K units) classification on logits.",
        meta={"task": "binary", "target": "float", "input": "logits"}),
    Component(
        "train.loss_bce", "Binary Cross-Entropy (probabilities)", (),
        torch=lambda p: "nn.BCELoss()",
        keras=lambda p: "keras.losses.BinaryCrossentropy()",
        desc="Binary / multi-label loss on probabilities (pair with a final Sigmoid).",
        meta={"task": "binary", "target": "float", "input": "probs"}),
    Component(
        "train.loss_multilabel_soft_margin", "Multi-Label Soft Margin", (),
        torch=lambda p: "nn.MultiLabelSoftMarginLoss()",
        keras=lambda p: "keras.losses.BinaryCrossentropy(from_logits=True)",
        desc="Multi-label classification on logits with multi-hot targets.",
        meta={"task": "multilabel", "target": "float", "input": "logits"}),
    Component(
        "train.loss_multilabel_margin", "Multi-Label Hinge",
        (), torch=lambda p: ("lambda pred, target: (torch.relu(1 - pred[:, :, None] + "
                             "pred[:, None, :]) * (target[:, :, None] * (1 - target[:, None, "
                             ":]))).sum((1, 2)).div(pred.shape[1]).mean()"),
        keras=lambda p: ("lambda y, p: ops.mean(ops.sum(ops.relu(1 - p[:, :, None] + "
                         "p[:, None, :]) * (ops.cast(y, p.dtype)[:, :, None] * "
                         "(1 - ops.cast(y, p.dtype)[:, None, :])), axis=(1, 2)) / p.shape[1])"),
        desc="torch's MultiLabelMarginLoss on multi-hot targets: every label an example has "
             "must score at least 1 above every label it lacks. Multi-label, on scores.",
        meta={"task": "multilabel", "target": "float", "input": "logits"}),
    Component(
        "train.loss_mse", "Mean Squared Error", (),
        torch=lambda p: "nn.MSELoss()", keras=lambda p: "keras.losses.MeanSquaredError()",
        desc="Regression: mean of squared errors.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_l1", "Mean Absolute Error", (),
        torch=lambda p: "nn.L1Loss()", keras=lambda p: "keras.losses.MeanAbsoluteError()",
        desc="Regression: mean of absolute errors (robust to outliers).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_smooth_l1", "Smooth L1",
        (P("beta", "float", 1.0, lo=1e-6, help="Transition point between L2 and L1"),),
        torch=lambda p: f"nn.SmoothL1Loss(beta={p['beta']})",
        keras=lambda p: (f"lambda y, p: keras.losses.huber(y, p, delta={p['beta']}) "
                         f"/ {p['beta']}"),
        desc="Quadratic near zero, linear beyond beta.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_huber", "Huber",
        (P("delta", "float", 1.0, lo=1e-6),),
        torch=lambda p: f"nn.HuberLoss(delta={p['delta']})",
        keras=lambda p: f"keras.losses.Huber(delta={p['delta']})",
        desc="Huber loss: L2 for small errors, L1 for large ones.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_log_cosh", "Log-Cosh", (),
        torch=lambda p: "lambda pred, target: torch.mean(torch.log(torch.cosh(pred - target)))",
        keras=lambda p: "keras.losses.LogCosh()",
        desc="Smooth regression loss: log(cosh(error)).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_poisson", "Poisson NLL", (),
        torch=lambda p: "nn.PoissonNLLLoss(log_input=True)",
        keras=lambda p: "lambda y, p: keras.losses.poisson(y, ops.exp(p))",
        desc="Count regression; the model outputs log-rates.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_kl_div", "KL Divergence", (),
        torch=lambda p: 'nn.KLDivLoss(reduction="batchmean")',
        keras=lambda p: "lambda y, p: keras.losses.kl_divergence(y, ops.exp(p))",
        desc="Distribution matching; model outputs log-probabilities, targets probabilities.",
        meta={"task": "distribution", "target": "float", "input": "log_probs"}),
]


# Metric learning (3.3): the model outputs an embedding (its last layer's width is the
# embedding size) and the loss learns how classes sit in that space. Each one also scores
# embeddings against its class centres, so accuracy and the other metrics still work.
_CLASSES = P("classes", "int", 10, lo=2, help="Number of classes in the data")
_METRIC_META = {"task": "multiclass", "target": "int", "input": "embedding", "metric": True}


def _margin(kind: str) -> Render:
    return lambda p: (f"MarginSoftmaxLoss(EMBED_DIM, NUM_CLASSES, scale={p['scale']}, "
                      f"margin={p['margin']}, kind=\"{kind}\")")


LOSSES += [
    Component(
        "train.loss_arcface", "ArcFace",
        (_CLASSES, P("scale", "float", 30.0, lo=1.0, help="Logit scale s"),
         P("margin", "float", 0.5, lo=0.0, hi=1.5, help="Additive angular margin (radians)")),
        torch=_margin("arcface"),
        desc="Additive angular margin loss (Deng et al., 2019): embeddings of a class "
             "gather around a learned centre with a margin between classes. Face and "
             "product recognition, verification. The trained network outputs embeddings.",
        meta={**_METRIC_META, "torch_helper": "MarginSoftmaxLoss"}),
    Component(
        "train.loss_cosface", "CosFace",
        (_CLASSES, P("scale", "float", 30.0, lo=1.0, help="Logit scale s"),
         P("margin", "float", 0.35, lo=0.0, hi=1.0, help="Additive cosine margin")),
        torch=_margin("cosface"),
        desc="Large margin cosine loss (Wang et al., 2018): like ArcFace with the margin "
             "on the cosine; a little easier to train. The network outputs embeddings.",
        meta={**_METRIC_META, "torch_helper": "MarginSoftmaxLoss"}),
    Component(
        "train.loss_triplet", "Triplet (batch-hard)",
        (_CLASSES, P("margin", "float", 0.3, lo=0.0, help="Distance margin")),
        torch=lambda p: f"TripletLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="For each example, the farthest same-class and the nearest other-class "
             "example in the batch (Hermans et al., 2017): same-class embeddings end up "
             "closer by a margin. Use batches with several examples per class.",
        meta={**_METRIC_META, "torch_helper": "TripletLoss"}),
    Component(
        "train.loss_supcon", "Supervised Contrastive",
        (_CLASSES, P("temperature", "float", 0.1, lo=1e-3, hi=1.0)),
        torch=lambda p: f"SupConLoss(EMBED_DIM, NUM_CLASSES, temperature={p['temperature']})",
        desc="InfoNCE / NT-Xent with labels (Khosla et al., 2020): every same-class pair "
             "in the batch is a positive, everything else a negative.",
        meta={**_METRIC_META, "torch_helper": "SupConLoss"}),
]
LOSSES += [
    Component(
        "train.loss_nt_xent", "NT-Xent (SimCLR)",
        (_CLASSES, P("temperature", "float", 0.5, lo=1e-3, hi=2.0)),
        torch=lambda p: f"NTXentLoss(EMBED_DIM, NUM_CLASSES, temperature={p['temperature']})",
        desc="Self-supervised contrastive learning (Chen et al., 2020): every training image "
             "is augmented twice and the network learns to match the two views among all "
             "the others in the batch. Labels are never used to learn, only to measure the "
             "embedding (accuracy of the nearest class average). Images, with augmentation.",
        meta={**_METRIC_META, "torch_helper": "NTXentLoss", "views": 2}),
]
# Pair losses (3.4): pairs come from the labels in each batch, so they work with any dataset
# that has classes, with no separate pairs file.
LOSSES += [
    Component(
        "train.loss_cosine_embedding", "Cosine Embedding (pairs)",
        (_CLASSES, P("margin", "float", 0.2, lo=-1.0, hi=1.0,
                     help="Different-class pairs are pushed below this cosine")),
        torch=lambda p: f"CosineEmbeddingPairsLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="Every pair of examples in the batch: same class pulls the cosine towards 1, "
             "different classes push it below the margin (torch's CosineEmbeddingLoss on "
             "pairs made from the labels). Verification and duplicate detection.",
        meta={**_METRIC_META, "torch_helper": "CosineEmbeddingPairsLoss"}),
    Component(
        "train.loss_margin_ranking", "Margin Ranking (pairs)",
        (_CLASSES, P("margin", "float", 0.2, lo=0.0, hi=2.0,
                     help="How much more similar a same-class example must be")),
        torch=lambda p: f"MarginRankingPairsLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="Ranks a same-class example above an other-class one, by a margin of cosine "
             "similarity, for every such pair around each example in the batch (torch's "
             "MarginRankingLoss on pairs made from the labels). Retrieval and search.",
        meta={**_METRIC_META, "torch_helper": "MarginRankingPairsLoss"}),
]
# Keras's other losses (3.5), with the same formula in PyTorch
_PM1 = "(2 * target - 1)"   # {0, 1} labels as {-1, +1}, as Keras converts them
LOSSES += [
    Component(
        "train.loss_hinge", "Hinge",
        (), torch=lambda p: f"lambda pred, target: torch.clamp(1 - {_PM1} * pred, min=0).mean()",
        keras=lambda p: "keras.losses.Hinge()",
        desc="The SVM loss on a raw score: zero once an example is on the right side by a "
             "margin of 1 (binary, labels 0 / 1 read as −1 / +1).",
        meta={"task": "binary", "target": "float", "input": "logits"}),
    Component(
        "train.loss_squared_hinge", "Squared Hinge",
        (), torch=lambda p: (f"lambda pred, target: torch.clamp(1 - {_PM1} * pred, "
                             "min=0).pow(2).mean()"),
        keras=lambda p: "keras.losses.SquaredHinge()",
        desc="Hinge squared: smooth at the margin and harder on large mistakes (binary).",
        meta={"task": "binary", "target": "float", "input": "logits"}),
    Component(
        "train.loss_soft_margin", "Soft Margin",
        (), torch=lambda p: f"lambda pred, target: nn.functional.soft_margin_loss(pred, {_PM1})",
        keras=lambda p: "keras.losses.BinaryCrossentropy(from_logits=True)",
        desc="torch's SoftMarginLoss, log(1 + exp(−y·x)): the logistic loss written for "
             "−1 / +1 labels (equal to binary cross-entropy on logits).",
        meta={"task": "binary", "target": "float", "input": "logits"}),
    Component(
        "train.loss_msle", "Mean Squared Log Error",
        (), torch=lambda p: ("lambda pred, target: (torch.log1p(pred.clamp(min=0)) - "
                             "torch.log1p(target.clamp(min=0))).pow(2).mean()"),
        keras=lambda p: "keras.losses.MeanSquaredLogarithmicError()",
        desc="Regression on a log scale: relative errors count, so a miss of 10 on 1000 "
             "costs about as much as 0.1 on 10. Prices, counts, durations (targets ≥ 0).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_mape", "Mean Absolute Percentage Error",
        (), torch=lambda p: ("lambda pred, target: 100 * ((target - pred).abs() / "
                             "target.abs().clamp(min=1e-7)).mean()"),
        keras=lambda p: "keras.losses.MeanAbsolutePercentageError()",
        desc="Regression error as a percentage of the target (avoid targets near zero).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_generalized_ce", "Generalized Cross-Entropy",
        (P("q", "float", 0.7, lo=0.01, hi=0.99,
           help="Towards 0: cross-entropy; towards 1: robust to wrong labels"),),
        torch=lambda p: (f"lambda pred, target: ((1 - pred.softmax(-1).gather(1, "
                         f"target.long().reshape(-1, 1)).clamp(min=1e-7) ** {float(p['q'])}) "
                         f"/ {float(p['q'])}).mean()"),
        desc="Noise-robust classification loss (Zhang & Sabuncu, 2018), (1 − p_y^q) / q: "
             "between cross-entropy and mean absolute error, it learns well when some "
             "training labels are wrong. Multi-class, on logits, PyTorch.",
        meta={"task": "multiclass", "target": "int", "input": "logits"}),
]
# 3.6: regression losses for quantiles and skewed, zero-heavy targets
LOSSES += [
    Component(
        "train.loss_quantile", "Quantile (Pinball)",
        (P("quantile", "float", 0.5, lo=0.01, hi=0.99,
           help="0.5: the median; 0.9: a value 90% of targets fall below"),),
        torch=lambda p: (f"lambda pred, target: torch.maximum({float(p['quantile'])} * "
                         f"(target - pred), ({float(p['quantile'])} - 1) * (target - pred))"
                         ".mean()"),
        keras=lambda p: (f"lambda y, p: ops.mean(ops.maximum({float(p['quantile'])} * (y - p), "
                         f"({float(p['quantile'])} - 1) * (y - p)))"),
        desc="Pinball loss: the model learns one quantile of the target instead of its "
             "mean. 0.5 is the median (robust to outliers); 0.9 an upper estimate for "
             "planning stock or capacity. For several quantiles at once use the Quantile "
             "Head.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_tweedie", "Tweedie Deviance",
        (P("power", "float", 1.5, lo=1.01, hi=1.99,
           help="Between 1 (Poisson counts) and 2 (Gamma amounts)"),),
        torch=lambda p: (f"lambda pred, target: (-target * torch.exp(pred * (1 - {float(p['power'])}))"
                         f" / (1 - {float(p['power'])}) + torch.exp(pred * (2 - "
                         f"{float(p['power'])})) / (2 - {float(p['power'])})).mean()"),
        keras=lambda p: (f"lambda y, p: ops.mean(-y * ops.exp(p * (1 - {float(p['power'])})) / "
                         f"(1 - {float(p['power'])}) + ops.exp(p * (2 - {float(p['power'])})) / "
                         f"(2 - {float(p['power'])}))"),
        desc="Tweedie negative log-likelihood (as in LightGBM and XGBoost) for targets that "
             "are often exactly zero and otherwise positive and skewed: insurance claims, "
             "sales, rainfall. The model outputs the log of the mean.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
]
# 3.7: more classification and regression losses, and two self-supervised ones
LOSSES += [
    Component(
        "train.loss_poly", "PolyLoss (Poly-1)",
        (P("epsilon", "float", 1.0, lo=-1.0, hi=10.0,
           help="Weight of the extra (1 − p) term; 0 = plain cross-entropy"),),
        torch=lambda p: (f"lambda pred, target: nn.functional.cross_entropy(pred, target) + "
                         f"{float(p['epsilon'])} * (1 - pred.softmax(-1).gather(1, "
                         "target.long().reshape(-1, 1))).mean()"),
        keras=lambda p: ("lambda y, p: keras.losses.sparse_categorical_crossentropy(y, p, "
                         f"from_logits=True) + {float(p['epsilon'])} * (1 - ops.sum(ops.one_hot("
                         "ops.cast(ops.reshape(y, (-1,)), 'int32'), ops.shape(p)[-1]) * "
                         "ops.softmax(p), axis=-1))"),
        desc="Cross-entropy plus ε × (1 − probability of the true class) (Leng et al., "
             "2022): one knob that often beats cross-entropy and focal loss. ε = 1 is the "
             "paper's default; try −1 for large, clean datasets.",
        meta={"task": "multiclass", "target": "int", "input": "logits"}),
    Component(
        "train.loss_asymmetric", "Asymmetric Loss (multi-label)",
        (P("gamma_neg", "float", 4.0, lo=0.0, hi=10.0, help="Focusing on negative labels"),
         P("gamma_pos", "float", 0.0, lo=0.0, hi=10.0, help="Focusing on positive labels"),
         P("clip", "float", 0.05, lo=0.0, hi=0.5,
           help="Negatives below this probability count as solved")),
        torch=lambda p: (f"AsymmetricLoss(gamma_neg={float(p['gamma_neg'])}, "
                         f"gamma_pos={float(p['gamma_pos'])}, clip={float(p['clip'])})"),
        keras=lambda p: (
            "lambda y, p: (lambda s, y: ops.mean(-(y * ops.power(1 - s, "
            f"{float(p['gamma_pos'])}) * ops.log(ops.maximum(s, 1e-8)) + (1 - y) * ops.power("
            f"ops.maximum(s - {float(p['clip'])}, 0), {float(p['gamma_neg'])}) * ops.log("
            f"ops.maximum(1 - ops.maximum(s - {float(p['clip'])}, 0), 1e-8)))))"
            "(ops.sigmoid(p), ops.cast(y, p.dtype))"),
        desc="Multi-label loss that treats the many negative labels differently from the "
             "few positive ones (Ridnik et al., 2021): easy negatives are discounted and "
             "very easy ones ignored. Tagging with many possible labels.",
        meta={"task": "multilabel", "target": "float", "input": "logits",
              "torch_helper": "AsymmetricLoss"}),
    Component(
        "train.loss_charbonnier", "Charbonnier",
        (P("epsilon", "float", 1e-3, lo=1e-9, hi=1.0, help="Smoothing near zero error"),),
        torch=lambda p: (f"lambda pred, target: torch.sqrt((pred - target) ** 2 + "
                         f"{float(p['epsilon'])} ** 2).mean()"),
        keras=lambda p: (f"lambda y, p: ops.mean(ops.sqrt(ops.square(p - y) + "
                         f"{float(p['epsilon'])} ** 2))"),
        desc="A smooth absolute error, √(error² + ε²): robust like L1 and differentiable at "
             "zero. The usual loss for image restoration (super-resolution, deblurring).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_ssim", "SSIM (+ L1)",
        (P("alpha", "float", 0.84, lo=0.0, hi=1.0,
           help="Share of 1 − SSIM; the rest is mean absolute error"),
         P("window", "int", 7, lo=3, hi=15, help="Window size (odd)")),
        torch=lambda p: f"SSIMLoss(alpha={float(p['alpha'])}, window={int(p['window'])})",
        desc="1 − structural similarity, mixed with L1 (Zhao et al., 2017): compares local "
             "brightness, contrast and structure, so restored images look sharper than with "
             "MSE. Images out, images as targets [C, H, W]; other shapes compare whole "
             "signals.",
        meta={"task": "regression", "target": "float", "input": "raw",
              "torch_helper": "SSIMLoss"},
        checks=lambda p: ([] if int(p["window"]) % 2 else
                          [("error", "SSIM window must be odd (3, 5, 7, …)")])),
    Component(
        "train.loss_vicreg", "VICReg (self-supervised)",
        (_CLASSES, P("invariance", "float", 25.0, lo=0.0, help="Weight: the two views agree"),
         P("variance", "float", 25.0, lo=0.0, help="Weight: every feature keeps varying"),
         P("covariance", "float", 1.0, lo=0.0, help="Weight: features stay decorrelated")),
        torch=lambda p: (f"VICRegLoss(EMBED_DIM, NUM_CLASSES, invariance={p['invariance']}, "
                         f"variance={p['variance']}, covariance={p['covariance']})"),
        desc="Variance-Invariance-Covariance regularization (Bardes et al., 2022): two "
             "augmentations of an image should embed alike, while each feature keeps "
             "varying across the batch and features stay uncorrelated, so nothing collapses. "
             "No negatives, so small batches work. Images, with augmentation.",
        meta={**_METRIC_META, "torch_helper": "VICRegLoss", "views": 2}),
    Component(
        "train.loss_barlow_twins", "Barlow Twins (self-supervised)",
        (_CLASSES, P("off_diagonal", "float", 5e-3, lo=0.0, hi=1.0,
                     help="Weight of decorrelating different features")),
        torch=lambda p: (f"BarlowTwinsLoss(EMBED_DIM, NUM_CLASSES, "
                         f"off_diagonal={p['off_diagonal']})"),
        desc="Barlow Twins (Zbontar et al., 2021): the cross-correlation between the "
             "embeddings of two augmentations of the same images should be the identity: "
             "each feature agrees across views and differs from the others. Images, with "
             "augmentation.",
        meta={**_METRIC_META, "torch_helper": "BarlowTwinsLoss", "views": 2}),
]
METRIC_LOSS_IDS = tuple(c.type_id for c in LOSSES if c.meta.get("metric"))
PAIR_LOSS_IDS = ("train.loss_cosine_embedding", "train.loss_margin_ranking")


# ================================================================= optimizers

def _wd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


def _kwd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


_LR = lambda default: P("lr", "float", default, lo=0.0, help="Learning rate")  # noqa: E731
_B1 = P("beta1", "float", 0.9, lo=0.0, hi=0.999999)
_B2 = P("beta2", "float", 0.999, lo=0.0, hi=0.999999)
_EPS = lambda d: P("eps", "float", d, lo=0.0)  # noqa: E731
_WD = lambda d=0.0: P("weight_decay", "float", d, lo=0.0)  # noqa: E731


def _sgd_checks(p):
    if p.get("nesterov") and float(p.get("momentum", 0)) <= 0:
        return [("error", "Nesterov momentum requires momentum > 0")]
    return []


OPTIMIZERS: list[Component] = [
    Component(
        "train.sgd", "SGD",
        (_LR(1e-2), P("momentum", "float", 0.9, lo=0.0, hi=0.999999),
         P("nesterov", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.SGD(model.parameters(), lr={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.SGD(learning_rate={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_kwd(p)})"),
        desc="Stochastic gradient descent with optional (Nesterov) momentum.",
        checks=_sgd_checks),
    Component(
        "train.adam", "Adam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.Adam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.Adam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + f"{_kwd(p)})"),
        desc="Adaptive moment estimation."),
    Component(
        "train.adamw", "AdamW",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(1e-2), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.AdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.AdamW(learning_rate={p['lr']}, "
                         f"weight_decay={p['weight_decay']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        desc="Adam with decoupled weight decay."),
    Component(
        "train.nadam", "NAdam",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.NAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Nadam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with Nesterov momentum."),
    Component(
        "train.radam", "RAdam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.RAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        desc="Rectified Adam (variance-adaptive warmup)."),
    Component(
        "train.adamax", "Adamax",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.Adamax(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adamax(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam variant based on the infinity norm."),
    Component(
        "train.adagrad", "Adagrad",
        (_LR(1e-2), _EPS(1e-10), _WD()),
        torch=lambda p: (f"torch.optim.Adagrad(model.parameters(), lr={p['lr']}, "
                         f"eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adagrad(learning_rate={p['lr']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Per-parameter learning rates from accumulated squared gradients."),
    Component(
        "train.adadelta", "Adadelta",
        (_LR(1.0), P("rho", "float", 0.9, lo=0.0, hi=1.0), _EPS(1e-6), _WD()),
        torch=lambda p: (f"torch.optim.Adadelta(model.parameters(), lr={p['lr']}, "
                         f"rho={p['rho']}, eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adadelta(learning_rate={p['lr']}, rho={p['rho']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Adagrad extension with a moving window of gradient updates."),
    Component(
        "train.rmsprop", "RMSprop",
        (_LR(1e-2), P("alpha", "float", 0.99, lo=0.0, hi=1.0), _EPS(1e-8),
         P("momentum", "float", 0.0, lo=0.0, hi=0.999999), P("centered", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.RMSprop(model.parameters(), lr={p['lr']}, "
                         f"alpha={p['alpha']}, eps={p['eps']}, momentum={p['momentum']}, "
                         f"centered={p['centered']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.RMSprop(learning_rate={p['lr']}, rho={p['alpha']}, "
                         f"momentum={p['momentum']}, epsilon={p['eps']}, "
                         f"centered={p['centered']}{_kwd(p)})"),
        desc="Divides the gradient by a running RMS of recent magnitudes."),
    Component(
        "train.adafactor", "Adafactor",
        (_LR(1e-2), _WD()),
        torch=lambda p: f"torch.optim.Adafactor(model.parameters(), lr={p['lr']}{_wd(p)})",
        keras=lambda p: f"keras.optimizers.Adafactor(learning_rate={p['lr']}{_kwd(p)})",
        desc="Memory-efficient adaptive optimizer with factored second moments."),
    Component(
        "train.asgd", "ASGD",
        (_LR(1e-2), P("lambd", "float", 1e-4, lo=0.0), P("alpha", "float", 0.75, lo=0.0),
         P("t0", "float", 1e6, lo=0.0), _WD()),
        torch=lambda p: (f"torch.optim.ASGD(model.parameters(), lr={p['lr']}, "
                         f"lambd={p['lambd']}, alpha={p['alpha']}, t0={p['t0']}{_wd(p)})"),
        desc="Averaged stochastic gradient descent."),
    Component(
        "train.rprop", "Rprop",
        (_LR(1e-2), P("eta_minus", "float", 0.5, lo=0.0, hi=1.0),
         P("eta_plus", "float", 1.2, lo=1.0)),
        torch=lambda p: (f"torch.optim.Rprop(model.parameters(), lr={p['lr']}, "
                         f"etas=({p['eta_minus']}, {p['eta_plus']}))"),
        desc="Resilient backpropagation (sign-based, full-batch friendly)."),
    Component(
        "train.lion", "Lion",
        (_LR(1e-4), _B1, P("beta2", "float", 0.99, lo=0.0, hi=0.999999), _WD(1e-2)),
        torch=lambda p: (f"Lion(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lion(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}{_kwd(p)})"),
        desc="Evolved sign momentum (Chen et al., 2023): one state per weight, half Adam's "
             "memory. Use a learning rate about 10x smaller than AdamW's.",
        meta={"torch_helper": "Lion"}),
    Component(
        "train.muon", "Muon",
        (_LR(0.02), P("momentum", "float", 0.95, lo=0.0, hi=0.999),
         P("nesterov", "bool", True), P("ns_steps", "int", 5, lo=1, hi=10,
                                        help="Newton–Schulz steps per update"),
         P("adamw_lr", "float", 3e-4, lo=0.0, help="Learning rate of the AdamW part "
                                                    "(biases, norms, other 1-D weights)"),
         _WD(0.0)),
        torch=lambda p: (f"Muon(model.parameters(), lr={p['lr']}, momentum={p['momentum']}, "
                         f"nesterov={bool(p['nesterov'])}, ns_steps={int(p['ns_steps'])}, "
                         f"adamw_lr={p['adamw_lr']}, weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Muon(learning_rate={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={bool(p['nesterov'])}, "
                         f"ns_steps={int(p['ns_steps'])}, "
                         f"adam_lr_ratio={float(p['adamw_lr']) / max(float(p['lr']), 1e-12)}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Muon (Jordan et al., 2024): momentum whose update is orthogonalized by a few "
             "Newton–Schulz steps for every weight matrix (and conv kernel), AdamW for the "
             "rest. Fast on transformers and CNNs; Keras 3.10 or later.",
        meta={"torch_helper": "Muon"}),
    Component(
        "train.lbfgs", "L-BFGS",
        (_LR(1.0), P("max_iter", "int", 20, lo=1, hi=200, help="Iterations per batch"),
         P("history_size", "int", 100, lo=1, hi=1000, help="Past updates kept to estimate "
                                                           "the curvature"),
         P("line_search", "bool", True, help="Strong-Wolfe line search (steadier)")),
        torch=lambda p: (f"torch.optim.LBFGS(model.parameters(), lr={p['lr']}, "
                         f"max_iter={int(p['max_iter'])}, "
                         f"history_size={int(p['history_size'])}, line_search_fn="
                         + ("\"strong_wolfe\")" if p.get("line_search") else "None)")),
        desc="Limited-memory BFGS: a quasi-Newton method that estimates the curvature from "
             "recent gradients. Converges in few steps on small, smooth problems (tabular "
             "regression, physics fits); use large batches or the whole dataset as one "
             "batch. Re-evaluates the loss several times per step; full precision.",
        meta={"closure": True}),
    Component(
        "train.lamb", "LAMB",
        (_LR(1e-3), _B1, _B2, _EPS(1e-6), _WD(1e-2)),
        torch=lambda p: (f"Lamb(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lamb(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with a per-layer trust ratio (You et al., 2020): stays stable at very "
             "large batch sizes.",
        meta={"torch_helper": "Lamb"}),
    Component(
        "train.lookahead_adamw", "Lookahead (AdamW)",
        (_LR(1e-3), _B1, _B2, _WD(1e-2), P("k", "int", 5, lo=1, help="Fast steps per sync"),
         P("alpha", "float", 0.5, lo=0.0, hi=1.0, help="How far the slow weights move")),
        torch=lambda p: (f"LookaheadAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']}, k={int(p['k'])}, "
                         f"alpha={p['alpha']})"),
        desc="AdamW whose weights are pulled toward a slow moving copy every k steps "
             "(Zhang et al., 2019): less sensitive to the learning rate.",
        meta={"torch_helper": "LookaheadAdamW"}),
    Component(
        "train.sam", "SAM (Sharpness-Aware)",
        (P("base", "enum", "sgd", options=("sgd", "adamw"), help="The optimizer SAM wraps"),
         _LR(0.05), P("rho", "float", 0.05, lo=0.0, help="Neighbourhood size"),
         P("momentum", "float", 0.9, lo=0.0, hi=1.0, help="SGD momentum"), _WD(5e-4),
         P("adaptive", "bool", False, help="ASAM: scale the step by each weight's size")),
        torch=lambda p: (f"SAM(model.parameters(), base=\"{p['base']}\", rho={p['rho']}, "
                         f"adaptive={bool(p['adaptive'])}, lr={p['lr']}, "
                         f"momentum={p['momentum']}, weight_decay={p['weight_decay']})"),
        desc="Sharpness-Aware Minimization (Foret et al., 2021): each step first moves to "
             "the worst weights nearby, then descends from there, which finds flatter minima "
             "that generalize better. Two forward/backward passes per step; runs in full "
             "precision.",
        meta={"torch_helper": "SAM", "sam": True}),
    Component(
        "train.schedule_free_adamw", "Schedule-Free AdamW",
        (_LR(2.5e-3), _B1, _B2, _EPS(1e-8), _WD(0.0),
         P("warmup_steps", "int", 0, lo=0, help="Linear warm-up steps")),
        torch=lambda p: (f"ScheduleFreeAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}, "
                         f"warmup_steps={int(p['warmup_steps'])})"),
        desc="AdamW without a learning-rate schedule (Defazio et al., 2024): it averages "
             "its iterates instead, and evaluates the average. Leave out the scheduler.",
        meta={"torch_helper": "ScheduleFreeAdamW", "schedule_free": True}),
]

# 3.6: newer adaptive optimizers
OPTIMIZERS += [
    Component(
        "train.adabelief", "AdaBelief",
        (_LR(1e-3), _B1, _B2, _EPS(1e-16), _WD(0.0)),
        torch=lambda p: (f"AdaBelief(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adam that scales each step by how far the gradient is from its running "
             "average, its \"belief\" (Zhuang et al., 2020): big steps when gradients agree, "
             "small ones when they are noisy. Adam's speed, closer to SGD's generalization.",
        meta={"torch_helper": "AdaBelief"}),
    Component(
        "train.adan", "Adan",
        (_LR(1e-3), P("beta1", "float", 0.02, lo=0.0, hi=1.0, help="Momentum weight of the "
                                                                    "new gradient"),
         P("beta2", "float", 0.08, lo=0.0, hi=1.0, help="Weight of the gradient difference"),
         P("beta3", "float", 0.01, lo=0.0, hi=1.0, help="Weight of the new squared term"),
         _EPS(1e-8), _WD(0.02)),
        torch=lambda p: (f"Adan(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}, {p['beta3']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adaptive Nesterov momentum (Xie et al., 2022): estimates where the gradient is "
             "heading from the change between steps. Strong on vision transformers, CNNs and "
             "language models; tolerates large learning rates and batches.",
        meta={"torch_helper": "Adan"}),
    Component(
        "train.ademamix", "AdEMAMix",
        (_LR(1e-3), _B1, _B2, P("beta3", "float", 0.9999, lo=0.0, hi=0.999999,
                                help="Decay of the slow momentum (remembers ~10k steps)"),
         P("alpha", "float", 5.0, lo=0.0, hi=20.0, help="Weight of the slow momentum"),
         P("warmup_steps", "int", 0, lo=0, help="Steps over which alpha and beta3 ramp up "
                                                "(0 = off; the paper uses the whole run)"),
         _EPS(1e-8), _WD(0.0)),
        torch=lambda p: (f"AdEMAMix(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}, {p['beta3']}), "
                         f"alpha={p['alpha']}, warmup_steps={int(p['warmup_steps'])}, "
                         f"eps={p['eps']}, weight_decay={p['weight_decay']})"),
        desc="AdamW with a second, very slow momentum mixed in (Pagliardini et al., 2024): "
             "old gradients keep helping for thousands of steps, so long runs reach a lower "
             "loss. Use a warm-up for alpha and beta3 when training from scratch.",
        meta={"torch_helper": "AdEMAMix"}),
]
# 3.7
OPTIMIZERS += [
    Component(
        "train.prodigy", "Prodigy (learning-rate free)",
        (_LR(1.0), _B1, _B2, P("d_coef", "float", 1.0, lo=0.01, hi=100.0,
                               help="Scales the learning rate Prodigy finds"),
         _EPS(1e-8), _WD(0.0)),
        torch=lambda p: (f"Prodigy(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), d_coef={p['d_coef']}, "
                         f"eps={p['eps']}, weight_decay={p['weight_decay']})"),
        desc="Adam that estimates its own step size while it trains (Mishchenko & Defazio, "
             "2023): leave lr at 1 and it finds the right scale, so there is no learning "
             "rate to tune. Pair it with a cosine schedule.",
        meta={"torch_helper": "Prodigy"}),
    Component(
        "train.cautious_adamw", "Cautious AdamW",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(0.01)),
        torch=lambda p: (f"CautiousAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="AdamW that skips the parts of a step pointing against the current gradient "
             "(Liang et al., 2024): a one-line change that speeds up training of language "
             "and vision models.",
        meta={"torch_helper": "CautiousAdamW"}),
    Component(
        "train.yogi", "Yogi",
        (_LR(1e-2), _B1, _B2, _EPS(1e-3), _WD(0.0)),
        torch=lambda p: (f"Yogi(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adam whose second moment changes additively, not multiplicatively (Zaheer et "
             "al., 2018), so a burst of large gradients does not shrink the step for long. "
             "Steadier on sparse or noisy gradients.",
        meta={"torch_helper": "Yogi"}),
    Component(
        "train.adopt", "ADOPT",
        (_LR(1e-3), _B1, P("beta2", "float", 0.9999, lo=0.0, hi=0.99999), _EPS(1e-6),
         _WD(0.0)),
        torch=lambda p: (f"ADOPT(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="ADOPT (Taniguchi et al., 2024): Adam that normalizes each gradient by the "
             "second moment from before it and applies momentum after that, which makes it "
             "converge for any beta2. A drop-in for Adam and AdamW, often steadier.",
        meta={"torch_helper": "ADOPT"}),
]

# PyTorch optimizers and losses that are not in torch: see train_helpers.py


# ================================================================= schedulers
# meta: step = epoch | batch | metric

def _milestones_check(p):
    try:
        ms = _ints(p.get("milestones", ""))
    except ValueError:
        return [("error", "milestones must be comma-separated integers")]
    if ms != sorted(ms) or len(set(ms)) != len(ms):
        return [("error", "milestones must be strictly increasing")]
    return []


def _cyclic_check(p):
    if float(p["base_lr"]) >= float(p["max_lr"]):
        return [("error", "base_lr must be smaller than max_lr")]
    return []


def _keras_closed_form(expr: str) -> Render:
    return lambda p: f"lambda epoch, lr: {expr.format(**p)}"


SCHEDULERS: list[Component] = [
    Component(
        "train.step_lr", "Step LR",
        (P("step_size", "int", 10, lo=1, help="Epochs between decays"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.StepLR(optimizer, step_size={p['step_size']}, "
                         f"gamma={p['gamma']})"),
        keras=_keras_closed_form("LR0 * {gamma} ** (epoch // {step_size})"),
        desc="Multiply the LR by gamma every step_size epochs.", meta={"step": "epoch"}),
    Component(
        "train.multistep_lr", "Multi-Step LR",
        (P("milestones", "str", "30, 60", help="Epochs at which to decay"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.MultiStepLR(optimizer, "
                         f"milestones={_ints(p['milestones'])}, gamma={p['gamma']})"),
        keras=lambda p: (f"lambda epoch, lr: LR0 * {p['gamma']} ** "
                         f"sum(epoch >= m for m in {_ints(p['milestones'])})"),
        desc="Decay by gamma at the given epochs.", meta={"step": "epoch"},
        checks=_milestones_check),
    Component(
        "train.exponential_lr", "Exponential LR",
        (P("gamma", "float", 0.95, lo=0.0, hi=1.0),),
        torch=lambda p: f"torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma={p['gamma']})",
        keras=_keras_closed_form("LR0 * {gamma} ** epoch"),
        desc="Multiply the LR by gamma every epoch.", meta={"step": "epoch"}),
    Component(
        "train.cosine_annealing_lr", "Cosine Annealing",
        (P("T_max", "int", 50, lo=1, help="Epochs per half cosine"),
         P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
                         f"T_max={p['T_max']}, eta_min={p['eta_min']})"),
        keras=_keras_closed_form(
            "{eta_min} + (LR0 - {eta_min}) * (1 + math.cos(math.pi * min(epoch, {T_max}) "
            "/ {T_max})) / 2"),
        desc="Cosine decay from the initial LR to eta_min.", meta={"step": "epoch"}),
    Component(
        "train.warm_restarts_lr", "Cosine Warm Restarts",
        (P("T_0", "int", 10, lo=1), P("T_mult", "int", 1, lo=1),
         P("eta_min", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, "
                         f"T_0={p['T_0']}, T_mult={p['T_mult']}, eta_min={p['eta_min']})"),
        desc="SGDR: cosine cycles that restart, optionally growing.", meta={"step": "epoch"}),
    Component(
        "train.warmup_cosine_lr", "Warmup + Cosine",
        (P("warmup_epochs", "int", 5, lo=0), P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (
            "torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=["
            f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, "
            f"total_iters={max(int(p['warmup_epochs']), 1)}), "
            f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
            f"T_max=max(EPOCHS - {p['warmup_epochs']}, 1), eta_min={p['eta_min']})], "
            f"milestones=[{max(int(p['warmup_epochs']), 1)}])"),
        keras=lambda p: (
            f"lambda epoch, lr: LR0 * (epoch + 1) / {max(int(p['warmup_epochs']), 1)} "
            f"if epoch < {p['warmup_epochs']} else {p['eta_min']} + (LR0 - {p['eta_min']}) * "
            f"(1 + math.cos(math.pi * (epoch - {p['warmup_epochs']}) / "
            f"max(EPOCHS - {p['warmup_epochs']}, 1))) / 2"),
        desc="Linear warmup, then cosine decay over the remaining epochs.",
        meta={"step": "epoch"}),
    Component(
        "train.linear_lr", "Linear LR",
        (P("start_factor", "float", 0.1, lo=1e-8, hi=1.0),
         P("end_factor", "float", 1.0, lo=0.0, hi=1.0),
         P("total_iters", "int", 5, lo=1, help="Epochs of the linear ramp")),
        torch=lambda p: (f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor="
                         f"{p['start_factor']}, end_factor={p['end_factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form(
            "LR0 * ({start_factor} + ({end_factor} - {start_factor}) * "
            "min(epoch, {total_iters}) / {total_iters})"),
        desc="Linearly ramp the LR factor (warmup or decay).", meta={"step": "epoch"}),
    Component(
        "train.constant_lr", "Constant LR (warm phase)",
        (P("factor", "float", 0.333, lo=0.0, hi=1.0), P("total_iters", "int", 5, lo=1)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ConstantLR(optimizer, factor={p['factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form("LR0 * ({factor} if epoch < {total_iters} else 1.0)"),
        desc="Scale the LR by a constant factor for the first epochs.", meta={"step": "epoch"}),
    Component(
        "train.polynomial_lr", "Polynomial LR",
        (P("total_iters", "int", 50, lo=1), P("power", "float", 1.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.PolynomialLR(optimizer, "
                         f"total_iters={p['total_iters']}, power={p['power']})"),
        keras=_keras_closed_form(
            "LR0 * (1 - min(epoch, {total_iters}) / {total_iters}) ** {power}"),
        desc="Polynomial decay to zero over total_iters epochs.", meta={"step": "epoch"}),
    Component(
        "train.plateau_lr", "Reduce on Plateau",
        (P("factor", "float", 0.5, lo=0.0, hi=0.999), P("patience", "int", 5, lo=0),
         P("min_lr", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\"min\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        keras=lambda p: (f"keras.callbacks.ReduceLROnPlateau(monitor=\"val_loss\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        desc="Reduce the LR when validation loss stops improving.", meta={"step": "metric"}),
    Component(
        "train.one_cycle_lr", "One-Cycle",
        (P("max_lr", "float", 1e-2, lo=1e-8), P("pct_start", "float", 0.3, lo=0.0, hi=1.0),
         P("anneal_strategy", "enum", "cos", options=("cos", "linear"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={p['max_lr']}, "
                         f"total_steps=EPOCHS * len(train_loader), pct_start={p['pct_start']}, "
                         f"anneal_strategy=\"{p['anneal_strategy']}\")"),
        desc="Super-convergence schedule stepped every batch.", meta={"step": "batch"}),
    Component(
        "train.cyclic_lr", "Cyclic LR",
        (P("base_lr", "float", 1e-4, lo=0.0), P("max_lr", "float", 1e-2, lo=0.0),
         P("step_size_up", "int", 2000, lo=1, help="Batches per half cycle"),
         P("mode", "enum", "triangular", options=("triangular", "triangular2", "exp_range"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr={p['base_lr']}, "
                         f"max_lr={p['max_lr']}, step_size_up={p['step_size_up']}, "
                         f"mode=\"{p['mode']}\")"),
        desc="Cycle the LR between two bounds every batch.", meta={"step": "batch"},
        checks=_cyclic_check),
    # 3.6
    Component(
        "train.inverse_sqrt_lr", "Inverse Square Root",
        (P("warmup_epochs", "int", 4, lo=1, help="Linear warm-up, then decay as 1/sqrt"),),
        torch=lambda p: (f"torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: "
                         f"min((e + 1) / {int(p['warmup_epochs'])}, "
                         f"({int(p['warmup_epochs'])} / (e + 1)) ** 0.5))"),
        keras=lambda p: (f"lambda epoch, lr: LR0 * min((epoch + 1) / {int(p['warmup_epochs'])}, "
                         f"({int(p['warmup_epochs'])} / (epoch + 1)) ** 0.5)"),
        desc="The original Transformer's schedule (Vaswani et al., 2017): a linear warm-up, "
             "then the learning rate falls as 1/√epoch. Never reaches zero, so training can "
             "go on.", meta={"step": "epoch"}),
    Component(
        "train.wsd_lr", "Warmup-Stable-Decay",
        (P("warmup_epochs", "int", 2, lo=0), P("decay_fraction", "float", 0.2, lo=0.01, hi=1.0,
                                                help="Share of the run spent decaying"),
         P("final_factor", "float", 0.0, lo=0.0, hi=1.0, help="LR at the end, as a fraction")),
        torch=lambda p: (
            "torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e, w="
            f"{int(p['warmup_epochs'])}, d=max(1, round(EPOCHS * {float(p['decay_fraction'])})), "
            f"f={float(p['final_factor'])}: (e + 1) / w if e < w else 1.0 if e < EPOCHS - d "
            "else f + (1 - f) * (EPOCHS - e - 1) / d)"),
        keras=lambda p: (
            f"lambda epoch, lr, w={int(p['warmup_epochs'])}, "
            f"d=max(1, round(EPOCHS * {float(p['decay_fraction'])})), "
            f"f={float(p['final_factor'])}: LR0 * ((epoch + 1) / w if epoch < w else 1.0 "
            "if epoch < EPOCHS - d else f + (1 - f) * (EPOCHS - epoch - 1) / d)"),
        desc="Warm up, hold the learning rate, and decay only at the end (used to train "
             "MiniCPM, DeepSeek and other LLMs): you can stop or extend a run without "
             "planning its length in advance.", meta={"step": "epoch"}),
]


# ===================================================================== metrics
# meta: tasks = set of task kinds the metric applies to; key = printed name

METRICS: list[Component] = [
    Component("eval.accuracy", "Accuracy", desc="Fraction of correct predictions.",
              meta={"tasks": {"multiclass", "binary", "multilabel"}, "key": "accuracy"}),
    Component("eval.balanced_accuracy", "Balanced Accuracy",
              desc="Mean per-class recall (robust to imbalance).",
              meta={"tasks": {"multiclass", "binary"}, "key": "balanced_accuracy"}),
    Component("eval.top_k_accuracy", "Top-k Accuracy", (P("k", "int", 5, lo=1),),
              desc="Target among the k highest scores.",
              meta={"tasks": {"multiclass"}, "key": "top_k_accuracy"}),
    Component("eval.precision", "Precision",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="TP / (TP + FP).", meta={"tasks": {"multiclass", "binary"}, "key": "precision"}),
    Component("eval.recall", "Recall",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="TP / (TP + FN).", meta={"tasks": {"multiclass", "binary"}, "key": "recall"}),
    Component("eval.f1", "F1 Score",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="Harmonic mean of precision and recall.",
              meta={"tasks": {"multiclass", "binary"}, "key": "f1"}),
    Component("eval.mcc", "Matthews Correlation",
              desc="Correlation between predictions and targets (−1..1).",
              meta={"tasks": {"multiclass", "binary"}, "key": "mcc"}),
    Component("eval.cohen_kappa", "Cohen's Kappa", desc="Agreement beyond chance.",
              meta={"tasks": {"multiclass", "binary"}, "key": "kappa"}),
    Component("eval.roc_auc", "ROC-AUC", desc="Area under the ROC curve (one-vs-rest).",
              meta={"tasks": {"multiclass", "binary", "multilabel"}, "key": "roc_auc"}),
    Component("eval.average_precision", "Average Precision (PR-AUC)",
              desc="Area under the precision-recall curve.",
              meta={"tasks": {"binary", "multilabel", "multiclass"}, "key": "avg_precision"}),
    Component("eval.log_loss", "Log Loss", desc="Cross-entropy of predicted probabilities.",
              meta={"tasks": {"multiclass", "binary"}, "key": "log_loss"}),
    Component("eval.confusion_matrix", "Confusion Matrix",
              (P("labels", "str", "", help="Optional class names, comma-separated"),),
              desc="Counts of actual vs predicted classes (saved to confusion.json).",
              meta={"tasks": {"multiclass", "binary"}, "key": "confusion"}),
    Component("eval.mse", "MSE", desc="Mean squared error.",
              meta={"tasks": {"regression"}, "key": "mse"}),
    Component("eval.rmse", "RMSE", desc="Root mean squared error.",
              meta={"tasks": {"regression"}, "key": "rmse"}),
    Component("eval.mae", "MAE", desc="Mean absolute error.",
              meta={"tasks": {"regression"}, "key": "mae"}),
    Component("eval.median_ae", "Median Absolute Error", desc="Median of absolute errors.",
              meta={"tasks": {"regression"}, "key": "median_ae"}),
    Component("eval.mape", "MAPE", desc="Mean absolute percentage error.",
              meta={"tasks": {"regression"}, "key": "mape"}),
    Component("eval.r2", "R² Score", desc="Coefficient of determination.",
              meta={"tasks": {"regression"}, "key": "r2"}),
    Component("eval.explained_variance", "Explained Variance", desc="1 − Var(err) / Var(y).",
              meta={"tasks": {"regression"}, "key": "explained_variance"}),
    # 3.6
    Component("eval.brier", "Brier Score",
              desc="Mean squared error of the predicted probabilities (0 is perfect): rewards "
                   "being right and being honest about confidence.",
              meta={"tasks": {"multiclass", "binary"}, "key": "brier"}),
    Component("eval.hamming", "Hamming Loss",
              desc="Fraction of labels predicted wrongly, over every example and label "
                   "(multi-label; lower is better).",
              meta={"tasks": {"multilabel"}, "key": "hamming"}),
    Component("eval.jaccard", "Jaccard Index (IoU)",
              desc="Overlap of predicted and true labels: per example for multi-label, per "
                   "class (macro average) otherwise.",
              meta={"tasks": {"multiclass", "binary", "multilabel"}, "key": "jaccard"}),
    Component("eval.pearson", "Pearson Correlation",
              desc="Linear correlation between predictions and targets (−1..1).",
              meta={"tasks": {"regression"}, "key": "pearson"}),
    Component("eval.spearman", "Spearman Correlation",
              desc="Rank correlation: do predictions order the examples like the targets do? "
                   "(−1..1; ignores the scale).",
              meta={"tasks": {"regression"}, "key": "spearman"}),
    Component("eval.max_error", "Max Error", desc="The largest absolute error (worst case).",
              meta={"tasks": {"regression"}, "key": "max_error"}),
    Component("eval.msle", "Mean Squared Log Error",
              desc="Squared error of log(1 + value): relative errors for positive targets.",
              meta={"tasks": {"regression"}, "key": "msle"}),
    # 3.7
    Component("eval.concordance", "Concordance (Lin's CCC)",
              desc="Agreement between predictions and targets (−1..1): like Pearson "
                   "correlation, but also penalizes a shift or a different scale.",
              meta={"tasks": {"regression"}, "key": "ccc"}),
    Component("eval.psnr", "PSNR",
              desc="Peak signal-to-noise ratio in dB, from the MSE and the targets' range: "
                   "higher is better (image restoration, ~30 dB looks good).",
              meta={"tasks": {"regression"}, "key": "psnr"}),
    Component("eval.ssim", "SSIM",
              desc="Structural similarity of predicted and target images (−1..1, 1 = "
                   "identical), from local brightness, contrast and structure in 7×7 "
                   "windows. Image outputs; for other shapes it compares whole signals.",
              meta={"tasks": {"regression"}, "key": "ssim"}),
]


# ===================================================================== others

TRAINER = Component(
    "train.trainer", "Trainer",
    (P("epochs", "int", 10, lo=1), P("batch_size", "int", 32, lo=1),
     P("device", "enum", "auto", options=("auto", "cpu", "cuda", "mps")),
     P("seed", "int", 42, lo=0),
     P("early_stopping_patience", "int", 0, lo=0, help="Epochs without improvement; 0 = off"),
     P("grad_clip_norm", "float", 0.0, lo=0.0, help="Max gradient norm; 0 = off"),
     P("accumulation_steps", "int", 1, lo=1, help="Batches per optimizer step"),
     P("mixed_precision", "bool", False, help="Automatic mixed precision on CUDA/MPS"),
     P("deterministic", "bool", False,
       help="Deterministic kernels for bit-reproducible runs (slower)"),
     P("ema_decay", "float", 0.0, lo=0.0, hi=0.99999,
       help="Evaluate and keep an exponential moving average of the weights "
            "(e.g. 0.999); 0 = off"),
     P("swa_start", "int", 0, lo=0,
       help="Stochastic weight averaging: average the weights of every epoch from this "
            "one on (PyTorch); 0 = off"),
     P("compile", "bool", False,
       help="Compile the model with torch.compile: faster steps after a slower first one. "
            "If compiling fails the run says why and trains without it (PyTorch)"),
     P("gradient_checkpointing", "bool", False,
       help="Recompute each layer's activations in the backward pass instead of keeping "
            "them: less memory for about a third more compute (PyTorch)"),
     P("gpus", "int", 1, lo=1, hi=64,
       help="GPUs of one machine that train together (DistributedDataParallel); the "
            "batch size is per GPU (PyTorch)")),
    desc="Training loop: epochs, batching, device, early stopping, clipping, AMP, "
         "weight averaging (EMA or SWA), compiling, gradient checkpointing, several GPUs.",
    checks=lambda p: ([("error", "use one kind of weight averaging: EMA or SWA")]
                      if float(p.get("ema_decay") or 0) and int(p.get("swa_start") or 0)
                      else []) + ([("warning", "SWA starts after the last epoch, so it never "
                                               "averages anything")]
                                  if int(p.get("swa_start") or 0) > int(p.get("epochs") or 1)
                                  else []))

KFOLD = Component(
    "train.kfold", "K-Fold Cross-Validation",
    (P("k", "int", 5, lo=2, hi=20), P("stratified", "bool", True), P("seed", "int", 42, lo=0),
     P("refit", "bool", False, help="Then train the final model on the training split as "
                                    "usual (stacking ensembles use its out-of-fold "
                                    "predictions)")),
    desc="Retrain on k folds and report mean ± std (tabular/array datasets); the "
         "out-of-fold predictions are saved for stacking.")

PREDICT = Component(
    "eval.predict", "Prediction Preview",
    (P("n_samples", "int", 5, lo=1, hi=50),
     P("show_probabilities", "bool", True, help="Print the top-3 class probabilities")),
    desc="Print predictions for test examples after training.")


EXPLAIN = Component(
    "eval.permutation_importance", "Feature Importance",
    (P("n_repeats", "int", 5, lo=1, hi=50, help="Shuffles per feature (more = steadier)"),
     P("max_samples", "int", 2000, lo=50, help="Test rows used (a sample keeps it fast)")),
    desc="After training, shuffles each input feature of the test set in turn and "
         "measures how much the loss gets worse: the features the model relies on most. "
         "Tables and feature vectors, PyTorch.")


EXPLAIN_IG = Component(
    "eval.integrated_gradients", "Integrated Gradients",
    (P("steps", "int", 32, lo=4, hi=256, help="Points on the path from the baseline"),
     P("baseline", "enum", "mean", options=("mean", "zeros"),
       help="Where the path starts: the average row, or all zeros"),
     P("max_samples", "int", 500, lo=10, help="Test rows explained")),
    desc="After training, integrates the gradient of the predicted class (or the output) "
         "along a path from a baseline to each test row (Sundararajan et al., 2017): how "
         "much each feature moved the prediction. Tables and feature vectors, PyTorch.")


EXPLAIN_SHAP = Component(
    "eval.shap_values", "SHAP Values",
    (P("background", "int", 100, lo=10, help="Training rows the explanation compares with"),
     P("max_samples", "int", 300, lo=10, help="Test rows explained")),
    desc="After training, SHAP values (expected gradients, Lundberg & Lee, 2017): each "
         "feature's share of the gap between a prediction and the average prediction over "
         "background rows. Tables and feature vectors, PyTorch. "
         "pip install 'ai-made-easy[explain]'.",
    meta={"requires": ("shap",), "extra": "explain"})


EXPLAIN_GRADCAM = Component(
    "eval.grad_cam", "Grad-CAM Heatmaps",
    (P("n_samples", "int", 8, lo=1, hi=12, help="Test images explained (up to 12, "
                                                       "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the heatmap is drawn")),
    desc="After training, Grad-CAM (Selvaraju et al., 2017) on the last convolution: a "
         "heatmap over each of a few test images showing where the model looked for its "
         "prediction, saved with the run and shown in both apps. Image classifiers with "
         "a Conv2d layer, PyTorch.")


EXPLAIN_ATTENTION = Component(
    "eval.attention_maps", "Attention Maps",
    (P("n_samples", "int", 6, lo=1, hi=12, help="Test samples explained (up to 12, "
                                                      "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the map is drawn over "
                                                      "an image")),
    desc="After training, the attention weights of the model's last attention layer on a "
         "few test samples, saved with the run and shown in both apps: for a vision "
         "transformer, how much each image patch is attended to, drawn over the image; for "
         "text and sequences, the attention matrix (row: a position, columns: where it "
         "looked). Models with attention or transformer blocks, PyTorch.")


EXPLAIN_SALIENCY = Component(
    "eval.audio_saliency", "Audio Saliency",
    (P("n_samples", "int", 6, lo=1, hi=12, help="Test clips explained (up to 12, "
                                                      "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the map is drawn")),
    desc="After training, gradient saliency (Simonyan et al., 2014) for a few test clips: "
         "how much each time-frequency point of the spectrogram (or each moment of a raw "
         "waveform) moves the predicted class, drawn over the input and saved with the run. "
         "Audio classifiers, PyTorch.")

PICTURE_EXPLAIN_IDS = ("eval.grad_cam", "eval.attention_maps", "eval.audio_saliency")


def _losses_by_id() -> dict[str, Component]:
    return {c.type_id: c for c in LOSSES}


LOSS_IDS = tuple(c.type_id for c in LOSSES)
OPTIMIZER_IDS = tuple(c.type_id for c in OPTIMIZERS)
SCHEDULER_IDS = tuple(c.type_id for c in SCHEDULERS)
METRIC_IDS = tuple(c.type_id for c in METRICS)
COMPONENTS: dict[str, Component] = {
    c.type_id: c for c in (*LOSSES, *OPTIMIZERS, *SCHEDULERS, *METRICS, TRAINER, KFOLD, PREDICT,
                           EXPLAIN, EXPLAIN_IG, EXPLAIN_SHAP, EXPLAIN_GRADCAM,
                           EXPLAIN_ATTENTION, EXPLAIN_SALIENCY)
}

_CATEGORY = {
    **{t: ("Loss", "training") for t in LOSS_IDS},
    **{t: ("Optimizer", "training") for t in OPTIMIZER_IDS},
    **{t: ("Scheduler", "training") for t in SCHEDULER_IDS},
    **{t: ("Metrics", "evaluation") for t in METRIC_IDS},
    "train.trainer": ("Training", "training"),
    "train.kfold": ("Training", "training"),
    "eval.predict": ("Metrics", "evaluation"),
    "eval.permutation_importance": ("Metrics", "evaluation"),
    "eval.integrated_gradients": ("Metrics", "evaluation"),
    "eval.shap_values": ("Metrics", "evaluation"),
    "eval.grad_cam": ("Metrics", "evaluation"),
    "eval.attention_maps": ("Metrics", "evaluation"),
    "eval.audio_saliency": ("Metrics", "evaluation"),
}
EXPLAIN_IDS = ("eval.permutation_importance", "eval.integrated_gradients", "eval.shap_values")
EXPLAIN_METHODS = {"eval.permutation_importance": "permutation",
                   "eval.integrated_gradients": "integrated_gradients",
                   "eval.shap_values": "shap"}


def _libraries(c: Component) -> str:
    if c.type_id in (*EXPLAIN_IDS, *PICTURE_EXPLAIN_IDS):
        return "PyTorch"
    if c.type_id in METRIC_IDS or c.type_id in ("train.trainer", "train.kfold", "eval.predict"):
        return "PyTorch · Keras"
    return " · ".join(lib for lib, ok in (("PyTorch", c.torch), ("Keras", c.keras)) if ok)


def register_all() -> None:
    reg = get_registry()
    for type_id, comp in COMPONENTS.items():
        category, family = _CATEGORY[type_id]
        reg.register(BlockDefinition(
            type_id=type_id, display_name=comp.name, category=category,
            color=family_color(family), params=tuple(comp.params),
            checks_fn=comp.checks, description=comp.desc, library=_libraries(comp),
            meta={"kind": category.lower(), **{k: v for k, v in comp.meta.items()
                                               if not callable(v)}},
            requires=tuple(comp.meta.get("requires", ())), extra=comp.meta.get("extra", ""),
        ))


def render(comp_id: str, params: dict, framework: str) -> str | None:
    comp = COMPONENTS[comp_id]
    fn = comp.torch if framework == "pytorch" else comp.keras
    return fn(params) if fn else None


def resolved(comp_id: str, params: dict | None) -> dict[str, Any]:
    comp = COMPONENTS[comp_id]
    out = {p.name: p.default for p in comp.params}
    out.update(params or {})
    return out
