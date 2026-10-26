"""The Trainer, K-Fold, Predict and the explanation blocks."""
from __future__ import annotations

from ai_made_easy.core.spec import P
from ai_made_easy.core.training.catalog import Component


def _trainer_checks(p: dict) -> list[tuple[str, str]]:
    ema, swa = float(p.get("ema_decay") or 0), int(p.get("swa_start") or 0)
    out = []
    if ema and swa:
        out.append(("error", "use one kind of weight averaging: EMA or SWA"))
    if swa > int(p.get("epochs") or 1):
        out.append(("warning", "SWA starts after the last epoch, so it never averages "
                               "anything"))
    if p.get("sharding") and int(p.get("gpus") or 1) < 2:
        out.append(("warning", "sharding splits a model between several GPUs: set GPUs "
                               "to 2 or more"))
    if p.get("sharding") and (ema or swa):
        out.append(("error", "sharded training keeps no averaged weights: turn off EMA / "
                             "SWA, or sharding"))
    return out


TRAINER = Component(
    "train.trainer", "Trainer",
    (P("epochs", "int", 10, lo=1), P("batch_size", "int", 32, lo=1),
     P("device", "enum", "auto", options=("auto", "cpu", "cuda", "mps")),
     P("seed", "int", 42, lo=0),
     P("early_stopping_patience", "int", 0, lo=0, help="Epochs without improvement; 0 = off"),
     P("grad_clip_norm", "float", 0.0, lo=0.0, help="Max gradient norm; 0 = off"),
     P("accumulation_steps", "int", 1, lo=1, help="Batches per optimizer step"),
     P("mixed_precision", "bool", False, help="Automatic mixed precision on CUDA/MPS"),
     P("precision", "enum", "auto", options=("auto", "float16", "bfloat16"),
       help="Mixed precision's number format: auto = float16 on CUDA, bfloat16 on MPS. "
            "bfloat16 needs no loss scaling (CUDA GPUs from Ampere on) (PyTorch)"),
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
            "batch size is per GPU (PyTorch)"),
     P("sharding", "bool", False,
       help="With several GPUs: split the weights, gradients and optimizer state between "
            "them (FSDP), so a model too big for one GPU trains. Sharded runs don't "
            "resume (PyTorch)"),
     P("qat_epochs", "int", 0, lo=0,
       help="Quantization-aware training: after training, fine-tune this many epochs "
            "with int8 rounding simulated, for the ExecuTorch int8 export; 0 = off "
            "(PyTorch, needs the export extra)")),
    desc="Training loop: epochs, batching, device, early stopping, clipping, AMP, "
         "weight averaging (EMA or SWA), compiling, gradient checkpointing, several GPUs "
         "(optionally sharded), quantization-aware training.",
    checks=lambda p: _trainer_checks(p))

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
