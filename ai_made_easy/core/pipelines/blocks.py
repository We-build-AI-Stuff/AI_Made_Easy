"""Pipeline stage blocks: each stage trains, transforms, evaluates or ships the model it
receives on its ``in`` wire and passes the result (a run folder) on its ``out`` wire."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Pipeline Stages"
IN = PortSpec("in", dtype="config", role="artifact")
MEMBERS = PortSpec("in", dtype="config", role="artifact", multi=True)
OUT = PortSpec("out", dtype="config", role="artifact")
DESIGN_HELP = ("A project file (path), sample:<name>.json or embedded:<key> (a design "
               "stored in the pipeline)")
EPOCHS = P("epochs", "int", 0, lo=0, hi=100_000, help="Epochs (0: the design's own)")

# stage kind -> what it produces
MODEL_STAGES = ("pipeline.train", "pipeline.finetune", "pipeline.distill", "pipeline.prune",
                "pipeline.quantize", "pipeline.qat")
PASS_THROUGH = ("pipeline.evaluate",)          # passes its input model on
NEEDS_INPUT = ("pipeline.finetune", "pipeline.distill", "pipeline.prune",
               "pipeline.quantize", "pipeline.qat", "pipeline.evaluate", "pipeline.export",
               "pipeline.deploy")
SOURCES = ("pipeline.train", "pipeline.cross_validate")
SHIP = ("pipeline.export", "pipeline.deploy")
# stages that work inside a trained PyTorch network (load, fine-tune, prune, ...)
TORCH_STAGES = ("pipeline.finetune", "pipeline.distill", "pipeline.prune",
                "pipeline.quantize", "pipeline.qat", "pipeline.evaluate", "pipeline.ensemble")
FREEZE = ("none", "all_but_last", "first_half")


def _stage(type_id: str, name: str, params: tuple, desc: str, *, inputs=(IN,),
           outputs=(OUT,)) -> BlockDefinition:
    return BlockDefinition(
        type_id=type_id, display_name=name, category=CATEGORY, color=family_color("training"),
        params=params, inputs=inputs, outputs=outputs, library="AI Made Easy",
        description=desc, meta={"pipeline": type_id.split(".", 1)[1]})


def _blocks() -> list[BlockDefinition]:
    design = P("design", "str", "", help=DESIGN_HELP)
    return [
        _stage("pipeline.train", "Train", (design, EPOCHS),
               "Trains a design (any family) and passes the trained model on.", inputs=()),
        _stage("pipeline.cross_validate", "Cross-validate",
               (design, P("k", "int", 5, lo=2, hi=50, help="Folds"), EPOCHS,
                P("refit", "bool", False,
                  help="Then train the final model and pass it on (stacking ensembles "
                       "need this: they learn from the out-of-fold predictions)")),
               "k-fold cross-validation of a design: the mean and spread of every metric, "
               "and every training example's out-of-fold prediction. With refit, the final "
               "model is trained and passed on.", inputs=()),
        _stage("pipeline.finetune", "Fine-tune",
               (P("design", "str", "", help="Optional: a design with the same architecture "
                                            "(e.g. on new data); empty: the input's own"),
                P("epochs", "int", 5, lo=1, hi=100_000),
                P("lr_scale", "float", 0.1, lo=1e-6, hi=10.0,
                  help="Multiplies the design's learning rate"),
                P("freeze", "enum", "none", options=FREEZE,
                  help="Layers kept fixed at first"),
                P("unfreeze_after", "int", 0, lo=0, hi=100_000,
                  help="Unfreeze everything from this epoch (0: never)"),
                P("lr_decay", "float", 1.0, lo=0.01, hi=1.0,
                  help="Discriminative learning rates: each layer group before the last "
                       "learns this many times slower (1: one rate for all)"),
                P("layer_groups", "int", 0, lo=0, hi=1000,
                  help="Layer groups for lr_decay (0: every layer is its own group)")),
               "Continues training from the input model's weights, optionally freezing "
               "layers, with a smaller learning rate and discriminative (per-layer-group) "
               "learning rates."),
        _stage("pipeline.distill", "Distill",
               (P("student", "str", "", help="The smaller design to train: " + DESIGN_HELP),
                P("temperature", "float", 4.0, lo=0.1, hi=100.0,
                  help="Softens the teacher's predictions"),
                P("alpha", "float", 0.5, lo=0.0, hi=1.0,
                  help="Weight of the true labels (1 - alpha: the teacher)"),
                EPOCHS),
               "Trains a student design to match the input (teacher) model's predictions as "
               "well as the labels: knowledge distillation."),
        _stage("pipeline.ensemble", "Ensemble",
               (P("method", "enum", "average", options=("average", "vote", "stack"),
                  help="average: mean probabilities / values; vote: majority class; stack: "
                       "a meta-learner (logistic regression / ridge) trained on the "
                       "members' out-of-fold predictions"),),
               "Combines two or more models trained on the same data and evaluates the "
               "ensemble on the test split. Stacking learns how to weigh the members from "
               "Cross-validate stages' out-of-fold predictions.", inputs=(MEMBERS,)),
        _stage("pipeline.prune", "Prune",
               (P("amount", "float", 0.5, lo=0.0, hi=0.99,
                  help="Share of the weights set to zero (smallest first, across layers)"),
                P("epochs", "int", 2, lo=0, hi=100_000,
                  help="Fine-tuning epochs with the pruning mask (0: none)"),
                P("lr_scale", "float", 0.1, lo=1e-6, hi=10.0)),
               "Magnitude pruning: zeroes the smallest weights of every linear and "
               "convolution layer, then fine-tunes to recover accuracy."),
        _stage("pipeline.quantize", "Quantize",
               (P("mode", "enum", "dynamic_int8", options=("dynamic_int8",),
                  help="dynamic_int8: 8-bit weights for linear layers (CPU)"),),
               "Post-training quantization: smaller weights for CPU serving; reports the "
               "size and the test metrics after quantization."),
        _stage("pipeline.qat", "Quantization-aware training",
               (P("epochs", "int", 3, lo=1, hi=100_000,
                  help="Fine-tuning epochs with fake-quantized weights and activations"),
                P("lr_scale", "float", 0.1, lo=1e-6, hi=10.0,
                  help="Multiplies the design's learning rate")),
               "Prepares the input model for int8 (fake quantization), fine-tunes it so it "
               "learns to tolerate the rounding, then converts it to int8. Reports the "
               "float, post-training dynamic int8 and QAT int8 metrics and sizes."),
        _stage("pipeline.evaluate", "Evaluate",
               (P("bins", "int", 10, lo=2, hi=100, help="Reliability-diagram bins"),),
               "Test-set metrics of the input model and, for classifiers, its calibration "
               "(expected calibration error and a reliability table)."),
        _stage("pipeline.export", "Export",
               (P("formats", "str", "onnx, torchscript",
                  help="Comma-separated: onnx, onnx_int8, torchscript, coreml, openvino, "
                       "executorch, executorch_int8 (PyTorch); onnx, saved_model, litert, "
                       "litert_int8 (Keras)"),
                P("out_dir", "str", "", help="Folder for the package (empty: "
                                             "~/.aime/pipelines/<id>/<stage>)")),
               "Builds a serving package (FastAPI app, Dockerfile, exported model files) "
               "from the input model. Each export is checked against it: the stage's "
               "metrics give the largest output difference, top-1 agreement (int8) and "
               "the file size per format."),
        _stage("pipeline.deploy", "Register",
               (P("name", "str", "", help="Model name in the registry (empty: the design's)"),
                P("stage", "enum", "staging", options=("none", "staging", "production")),
                P("package", "bool", False, help="Also build a serving package")),
               "Registers the input model in the model registry (and optionally builds a "
               "serving package)."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)
