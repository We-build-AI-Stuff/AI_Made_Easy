"""Metrics."""
from __future__ import annotations

from ai_made_easy.core.spec import P
from ai_made_easy.core.training.catalog import Component


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
