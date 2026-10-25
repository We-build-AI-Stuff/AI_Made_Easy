"""The aime command's arguments: one subcommand per task."""
from __future__ import annotations

import argparse

from ai_made_easy.core.codegen import FRAMEWORKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aime", description="AI Made Easy — validate projects and generate code headlessly")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("blocks", help="list registered blocks")
    p_val = sub.add_parser("validate", help="validate a project graph")
    p_val.add_argument("project", help="path to project .json")
    for name, help_text, choices, default in (
            ("gen", "generate model code", FRAMEWORKS, "pytorch"),
            ("train", "generate a training script (auto: the design's framework)", None,
             "auto")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-f", "--framework", choices=choices, default=default,
                       help="pytorch, keras, sklearn, pgmpy, pymc, ... (train: auto)")
        p.add_argument("-o", "--out", default="exports", help="output directory")
    p_run = sub.add_parser("run", help="train headlessly and stream events")
    p_run.add_argument("project", help="path to project .json")
    p_run.add_argument("-f", "--framework", default="auto",
                       help="auto (the design's framework), pytorch, keras, sklearn, ...")
    on_help = ("where to train: local (default), ssh:<host>, kaggle, colab, modal, "
               "container or a saved backend (see `aime backends`)")
    p_run.add_argument("--on", default=None, metavar="BACKEND", help=on_help)
    p_runs = sub.add_parser("runs", help="run history")
    p_runs.add_argument("action", nargs="?", default="list",
                        choices=("list", "show", "compare", "delete", "attach", "import",
                                 "export", "resume", "more"))
    p_runs.add_argument("ids", nargs="*", help="run id(s); import: the results .zip")
    p_runs.add_argument("-o", "--out", default="runs.csv",
                        help="export: .csv or .parquet file (default runs.csv)")
    p_runs.add_argument("--epochs", action="store_true",
                        help="export: one row per epoch instead of per run")
    p_runs.add_argument("-n", "--more", type=int, default=10, metavar="N",
                        help="more: epochs for a network, trees for a boosted model "
                             "(default 10)")
    p_lr = sub.add_parser("lr-find", help="find a learning rate for a design (a short "
                                          "range test on this computer)")
    p_lr.add_argument("project", help="path to project .json")
    p_lr.add_argument("-f", "--framework", default="auto", help="auto, pytorch or keras")
    p_lr.add_argument("--steps", type=int, default=100, help="batches to test (default 100)")
    p_lr.add_argument("--apply", action="store_true",
                      help="write the suggested rate into the project's optimizer block")
    p_be = sub.add_parser("backends", help="where runs train: list, add, test, remove")
    p_be.add_argument("action", nargs="?", default="list",
                      choices=("list", "add", "test", "remove"))
    p_be.add_argument("name", nargs="?", default="",
                      help="backend name (test: also ssh:<host>, kaggle, …)")
    p_be.add_argument("--kind", default="", help="add: ssh, kaggle, colab, modal, container")
    p_be.add_argument("-s", "--set", action="append", default=[], metavar="OPTION=VALUE",
                      help="add: a backend option, e.g. host=gpu-box (repeatable; "
                           "`aime backends list --options` shows them)")
    p_be.add_argument("--options", action="store_true", help="list: show every kind's options")
    p_runs.add_argument("--project", default=None, help="filter by project name")
    p_sw = sub.add_parser("sweep", help="hyperparameter sweep (trials go to the run history)")
    p_sw.add_argument("project", help="path to project .json")
    p_sw.add_argument("-p", "--param", action="append", default=[], metavar="NODE.PARAM=SPACE",
                      help="float:LOW:HIGH | log:LOW:HIGH | int:LOW:HIGH[:STEP] | "
                           "choice:A,B,C (repeatable)")
    p_sw.add_argument("--metric", default="val_loss")
    p_sw.add_argument("--direction", choices=("min", "max"), default="")
    p_sw.add_argument("--strategy", choices=("tpe", "random", "grid"), default="tpe")
    p_sw.add_argument("-n", "--trials", type=int, default=10)
    p_sw.add_argument("--points", type=int, default=3, help="grid points per range")
    p_sw.add_argument("--seed", type=int, default=0)
    p_sw.add_argument("--on", default=None, metavar="BACKEND", help="where trials train")
    p_sw.add_argument("--list-params", action="store_true",
                      help="print the parameters that can be swept and exit")
    p_dep = sub.add_parser("deploy", help="build a model-server package from a finished run")
    p_dep.add_argument("run_id")
    p_dep.add_argument("-o", "--out", default=None, help="output folder (must be empty)")
    p_dep.add_argument("--formats", default="", help="comma-separated: onnx, onnx_int8, "
                       "torchscript, coreml, openvino, executorch, executorch_int8, "
                       "gguf (PyTorch); onnx, "
                       "saved_model, litert, litert_int8 (Keras); or 'suggested' for the "
                       "formats the project's budget device runs")
    p_dep.add_argument("--list-formats", action="store_true",
                       help="list the run's export formats (availability, suggestions)")
    p_dep.add_argument("--name", default=None)
    p_mod = sub.add_parser("models", help="model registry")
    p_mod.add_argument("action", nargs="?", default="list",
                       choices=("list", "register", "stage", "deploy", "delete"))
    p_mod.add_argument("args", nargs="*")
    p_mod.add_argument("-o", "--out", default=None)
    p_mod.add_argument("--formats", default="")
    p_mod.add_argument("--also", default="", metavar="mlflow,wandb",
                       help="register / stage: also register the version in MLflow's model "
                            "registry and / or as a W&B model artifact")
    p_srv = sub.add_parser("serve", help="serve a deployment package with uvicorn")
    p_srv.add_argument("package")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_imp = sub.add_parser("import", help="import a PyTorch / ONNX / Keras model as a project")
    p_imp.add_argument("kind", choices=("pytorch", "onnx", "keras", "mlflow"))
    p_imp.add_argument("source", help=".py file, module or saved .pt module (pytorch), .onnx, "
                       ".keras / .h5, or an MLflow model URI (models:/name/1, runs:/id/model)")
    p_imp.add_argument("--attr", default="", help="model class or factory (pytorch)")
    p_imp.add_argument("--shape", default="", help="per-sample input shape, e.g. 3,224,224")
    p_imp.add_argument("--dtype", default="float32", choices=("float32", "int64"))
    p_imp.add_argument("--kwargs", default="{}", help="constructor arguments as JSON")
    p_imp.add_argument("-o", "--out", required=True, help="project .json to write")
    p_data = sub.add_parser("data", help="profile datasets, preview splits and augmentation")
    p_data.add_argument("action", choices=("profile", "check", "split", "augment",
                                           "fingerprint", "fetch"))
    p_data.add_argument("source", help="project .json, or a data file / folder (profile); "
                        "fetch: kaggle:owner/name, kaggle-competition:name[/file] or "
                        "hf:owner/repo[:split]")
    p_data.add_argument("--force", action="store_true", help="fetch: download again")
    p_data.add_argument("--target", default=None, help="target column (profiling a table)")
    p_data.add_argument("--json", action="store_true", help="print JSON")
    p_data.add_argument("-o", "--out", default="augmentation_preview",
                        help="output folder (augment)")
    p_hub = sub.add_parser("hub", help="the Hugging Face Hub: push a run, pull it back, "
                                       "deploy a Space (sign in with `hf auth login`)")
    p_hub.add_argument("action", choices=("push", "pull", "space"))
    p_hub.add_argument("args", nargs="+", help="push RUN_ID REPO | pull REPO | "
                                               "space RUN_ID SPACE")
    p_hub.add_argument("--model", action="store_true",
                       help="push / space: the first argument is a registered model "
                            "(NAME or NAME/VERSION) instead of a run id")
    p_hub.add_argument("--public", action="store_true", help="make the repo public")
    p_hub.add_argument("--name", default="", help="pull: registry name (default: repo name)")
    p_hub.add_argument("--formats", default="", help="space: export formats to include")
    p_kag = sub.add_parser("kaggle", help="submit predictions to a Kaggle competition")
    p_kag.add_argument("action", choices=("submit",))
    p_kag.add_argument("competition")
    p_kag.add_argument("file", help="predictions file (the competition's format)")
    p_kag.add_argument("-m", "--message", required=True)
    p_kag.add_argument("--yes", action="store_true",
                       help="don't ask (a submission counts against the daily limit)")
    p_trk = sub.add_parser("tracking", help="MLflow / W&B experiment tracking status")
    p_trk.add_argument("project", nargs="?", default="",
                       help="project .json (shows its meta.tracking setting)")
    p_web = sub.add_parser("web", help="serve the browser UI and REST API")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--open", action="store_true", help="open the browser")
    p_sum = sub.add_parser("summary", help="print the analytic model summary as JSON")
    p_sum.add_argument("project", help="path to project .json")
    p_bud = sub.add_parser("budget", help="estimate FLOPs / memory / latency on a device")
    p_bud.add_argument("project", nargs="?", default="", help="path to project .json")
    p_bud.add_argument("--device", default=None, help="device profile id (see --list)")
    p_bud.add_argument("--list", action="store_true", help="list device profiles")
    p_bud.add_argument("--json", action="store_true", help="print JSON")
    p_rec = sub.add_parser("recipes", help="list design recipes")
    p_rec.add_argument("--task", default=None, help="only recipes for this task")

    def wizard_args(p: argparse.ArgumentParser, task_required: bool = True) -> None:
        p.add_argument("--task", required=task_required,
                       default="", help="task id (see `aime recipes`)"
                       + ("" if task_required else "; default: the first task --data fits"))
        p.add_argument("--data", default="", help="a table file or dataset folder "
                                                  "(default: each recipe's demo data)")
        p.add_argument("--target", default="", help="target column of a table")
        p.add_argument("--modality", default="", help="tabular, image, text, audio, ... "
                                                      "(demo data only)")
        p.add_argument("--device", default="", help="budget: device profile id")
        p.add_argument("--max-memory-gb", type=float, default=0.0,
                       help="budget: training memory")
        p.add_argument("--max-params-m", type=float, default=0.0,
                       help="budget: parameters (millions)")
        p.add_argument("--max-latency-ms", type=float, default=0.0,
                       help="budget: inference latency on the device")

    p_new = sub.add_parser("new", help="create a project from the best recipe for a task "
                                       "(or for your data: aime new --data table.csv)")
    wizard_args(p_new, task_required=False)
    p_new.add_argument("--recipe", default="", help="use this recipe instead of the best")
    p_new.add_argument("--list", action="store_true", help="rank the recipes and exit")
    p_new.add_argument("-o", "--out", default="", help="project .json to write")
    p_aml = sub.add_parser("automl", help="search recipes and settings for a task")
    wizard_args(p_aml)
    p_aml.add_argument("-n", "--trials", type=int, default=12)
    p_aml.add_argument("--epochs", type=int, default=0, help="epochs per trial (0: default)")
    p_aml.add_argument("--recipes", default="", help="comma-separated candidate recipes")
    p_aml.add_argument("--metric", default="", help="metric to optimise (default: the task's)")
    p_aml.add_argument("--no-tune", action="store_true", help="every recipe once, defaults")
    p_aml.add_argument("--no-prune", action="store_true", help="never stop trials early")
    p_aml.add_argument("--seed", type=int, default=0)
    p_aml.add_argument("--objectives", default="",
                       help="also minimise these design costs on --device, comma-separated: "
                            "latency_ms, params_m, model_mb (a Pareto front, not one best)")
    p_aml.add_argument("--on", default="", metavar="BACKEND",
                       help="where trials train: ssh:<host>, kaggle, modal, … (see "
                            "`aime backends`)")
    p_aml.add_argument("--concurrency", type=int, default=1,
                       help="trials training at once (default 1; a few on remote backends)")
    p_aml.add_argument("-o", "--out", default="", help="write the best design here")
    p_pipe = sub.add_parser("pipeline", help="run multi-stage pipelines")
    p_pipe.add_argument("action", choices=("run", "status", "list", "resume"))
    p_pipe.add_argument("target", nargs="?", default="",
                        help="pipeline project .json (run) or pipeline id (status, resume)")
    p_pipe.add_argument("--no-wait", action="store_true", help="start and return the id")
    p_pipe.add_argument("--on", default="", metavar="BACKEND",
                        help="where training stages run (the other stages run here)")
    p_llm = sub.add_parser("llm", help="generate an LLM workflow script")
    p_llm.add_argument("project", help="path to project .json")
    p_llm.add_argument("-o", "--out", default="exports", help="output directory")
    for name, help_text in (("onnx", "generate ONNX export script"),
                            ("jit", "generate TorchScript export script")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("project", help="path to project .json")
        p.add_argument("-o", "--out", default="exports", help="output directory")
        p.add_argument("--run", action="store_true",
                       help="execute the generated script immediately")
    return parser
