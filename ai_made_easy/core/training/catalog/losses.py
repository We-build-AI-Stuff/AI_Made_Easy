"""Losses: one Component per loss block (PyTorch and Keras)."""
from __future__ import annotations

from ai_made_easy.core.spec import P
from ai_made_easy.core.training.catalog import Component, Render, _floats


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
