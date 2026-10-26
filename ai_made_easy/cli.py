"""Headless CLI over core — proves the app is drivable without the GUI.

This is the same JSON-in/JSON-out surface a future MCP agent will use.

Usage:
  aime blocks                       # list registered blocks (JSON schemas)
  aime validate <project.json>      # check a graph, print issues
  aime gen <project.json> -f pytorch [-o DIR]   # generate model code
  aime train <project.json> -f pytorch [-o DIR] # generate training script
  aime onnx <project.json> [-o DIR]   # generate (and with --run, execute) an ONNX export script
  aime jit <project.json> [-o DIR]    # generate (and with --run, execute) a TorchScript export script
  aime run <project.json> [-f auto]   # train headlessly (recorded in the run history)
  aime runs [list | show ID | compare ID ID… | delete ID]
  aime deploy RUN_ID -o DIR [--formats onnx,executorch_int8|suggested]   # model server package
  aime deploy RUN_ID --list-formats    # export formats: available, suggested for the device
  aime models [list | register RUN_ID NAME | stage NAME VERSION STAGE | deploy NAME VERSION -o DIR]
  aime serve DIR [--port 8000]       # run a deployment package locally
  aime import pytorch model.py --attr Net --shape 3,32,32 -o project.json
  aime import onnx model.onnx -o project.json      (also: keras model.keras)
  aime sweep <project.json> -p opt.lr=log:1e-4:1e-1 -p d1.units=int:16:128 \
             -p opt.nesterov=choice:true,false --metric accuracy --strategy tpe -n 20
  aime recipes [--task TASK]          # design recipes per task
  aime new --task binary --data churn.csv -o churn.json [--recipe ID] [--list]
  aime automl --task binary --data churn.csv -n 12 --epochs 20 -o best.json
  aime pipeline run pipeline.json | status ID | list | resume ID
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ai_made_easy.core.codegen import FRAMEWORKS, export, export_training
from ai_made_easy.core.codegen import sanitize_identifier as sanitize_name
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry


def main(argv: list[str] | None = None) -> int:
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

    args = parser.parse_args(argv)

    if args.command == "blocks":
        print(json.dumps(get_registry().list_blocks(), indent=2))
        return 0

    if args.command == "runs":
        return _runs_command(args)
    if args.command == "lr-find":
        return _lr_find_command(args)
    if args.command == "backends":
        return _backends_command(args)
    if args.command in ("deploy", "models", "serve"):
        return _deploy_command(args)
    if args.command == "import":
        return _import_command(args)
    if args.command == "data":
        return _data_command(args)
    if args.command in ("recipes", "new", "automl"):
        return _wizard_command(args)
    if args.command == "pipeline":
        return _pipeline_command(args)
    if args.command in ("hub", "kaggle", "tracking"):
        return _tools_command(args)
    if args.command == "web":
        try:
            from ai_made_easy.server.app import main as serve_web
        except ImportError as exc:
            print(f"error: {exc} — pip install 'ai-made-easy[web]'", file=sys.stderr)
            return 1
        serve_web(args.host, args.port, args.open)
        return 0

    if args.command == "budget" and (args.list or not args.project):
        return _budget_command(args, None)

    with open(args.project, encoding="utf-8") as fh:
        graph = Graph.from_dict(json.load(fh))

    if args.command == "validate":
        from ai_made_easy.core.data.lints import data_issues, project_base

        issues = graph.validate() + data_issues(graph, base=project_base(args.project))
        for issue in issues:
            print(issue)
        print(f"{len(issues)} issue(s)")
        return 1 if any(i.severity == "error" for i in issues) else 0

    if args.command in ("gen", "train"):
        try:
            if args.command == "train":
                from ai_made_easy.core.runner.manager import resolve_framework

                args.framework = resolve_framework(graph, args.framework)
            path = (
                export(graph, args.framework, args.out)
                if args.command == "gen"
                else export_training(graph, args.framework, args.out)
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(path)
        return 0

    if args.command == "sweep":
        return _sweep_command(args, graph)

    if args.command == "budget":
        return _budget_command(args, graph)

    if args.command == "summary":
        from ai_made_easy.core.summary import summarize

        s = summarize(graph)
        print(json.dumps({
            "total_params": s.total_params,
            "layers": [
                {"name": L.name, "type": L.type_id,
                 "output_shape": L.output_shape, "params": L.params}
                for L in s.layers
            ],
        }, indent=2))
        return 0

    if args.command == "run":
        from ai_made_easy.core.runner.manager import RunManager

        mgr = RunManager()
        try:
            run_id = mgr.start(graph, args.framework, on=args.on)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"type": "run_started", "run_id": run_id}), flush=True)
        import time as _time

        seen = 0
        while True:
            st = mgr.status(run_id)
            events, seen = mgr.get(run_id).events_since(seen)
            for event in events:
                print(json.dumps(event), flush=True)
            if st["state"] in ("finished", "failed", "stopped"):
                print(json.dumps({"type": "run_" + st["state"],
                                  "returncode": st["returncode"]}), flush=True)
                return 0 if st["state"] == "finished" else 1
            _time.sleep(0.1)

    if args.command == "llm":
        from ai_made_easy.core.codegen.llm_gen import generate_llm_script

        try:
            code = generate_llm_script(graph)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        from pathlib import Path

        script = Path(args.out) / f"{sanitize_name(graph.name)}_llm.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code, encoding="utf-8")
        print(script)
        return 0

    if args.command in ("onnx", "jit"):
        from ai_made_easy.core.codegen.runtime_export import (
            generate_onnx_export,
            generate_torchscript_export,
        )

        try:
            stem = graph.name.replace("-", "_")
            out_model = f"{args.out}/{stem}.{'onnx' if args.command == 'onnx' else 'torchscript.pt'}"
            code = (generate_onnx_export(graph, out_model)
                    if args.command == "onnx"
                    else generate_torchscript_export(graph, out_model))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        from pathlib import Path

        script = Path(args.out) / f"{stem}_export_{'onnx' if args.command == 'onnx' else 'jit'}.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code, encoding="utf-8")
        print(script)
        if args.run:
            from ai_made_easy.core.runner.manager import python_executable

            result = subprocess.run([python_executable(), str(script)], cwd=".")
            return result.returncode
        return 0

    return 2


def parse_space(text: str, points: int = 3) -> dict:
    """``node.param=kind:...`` → a sweep dimension dict."""
    if "=" not in text or "." not in text.split("=", 1)[0]:
        raise ValueError(f"expected NODE.PARAM=SPACE, got {text!r}")
    key, space = text.split("=", 1)
    node, param = key.rsplit(".", 1)
    kind, _, rest = space.partition(":")
    if kind == "choice":
        def conv(v: str):
            low = v.strip().lower()
            if low in ("true", "false"):
                return low == "true"
            for cast in (int, float):
                try:
                    return cast(v)
                except ValueError:
                    pass
            return v.strip()

        return {"node": node, "param": param, "kind": "choice",
                "values": [conv(v) for v in rest.split(",") if v.strip()]}
    parts = rest.split(":")
    if kind in ("float", "log") and len(parts) == 2:
        return {"node": node, "param": param, "kind": "float", "low": float(parts[0]),
                "high": float(parts[1]), "log": kind == "log", "points": points}
    if kind == "int" and len(parts) in (2, 3):
        return {"node": node, "param": param, "kind": "int", "low": int(parts[0]),
                "high": int(parts[1]), "step": int(parts[2]) if len(parts) == 3 else 1}
    raise ValueError(f"cannot parse the search space {space!r}")


def _budget_command(args, graph) -> int:  # noqa: ANN001
    from ai_made_easy.core import api, budget

    if graph is None:
        if not args.list:
            print("error: give a project, or --list to see device profiles", file=sys.stderr)
            return 2
        for d in api.list_devices()["devices"]:
            print(f"{d['id']:<18} {d['label']:<40} {d['memory_gb']:>5g} GB  "
                  f"{d['tflops_fp32']:>6g} TFLOPs fp32")
        return 0
    try:
        report = api.estimate_budget(graph, args.device)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        est = report["estimate"]
        device = est["device"]["label"] if est["device"] else "no device (use --device)"
        print(f"device            {device}")
        print(f"parameters        {est['params']:,} ({est['trainable_params']:,} trainable)")
        print(f"forward FLOPs     {budget.human_flops(est['flops'])} per sample")
        print(f"model size        {est['model_mb']:.1f} MB")
        print(f"training memory   {est['train_memory_gb']:.2f} GB at batch {est['batch_size']}"
              + (" (mixed precision)" if est["mixed_precision"] else ""))
        if est["latency_ms"] is not None:
            print(f"latency           {est['latency_ms']:.2f} ms per sample")
            print(f"training step     {est['step_time_ms']:.1f} ms")
        for c in report["checks"]:
            mark = "OVER" if c["over"] else "ok"
            print(f"budget {c['kind']:<13}{c['used']:.2f} / {c['limit']:g} {c['unit']}  {mark}")
    return 1 if any(c["over"] for c in report["checks"]) else 0


def _sweep_command(args, graph) -> int:  # noqa: ANN001
    import time as _time

    from ai_made_easy.core import api

    if args.list_params:
        for p in api.sweepable_params(graph)["params"]:
            space = (",".join(map(str, p["values"])) if p["kind"] == "choice"
                     else f"{p['low']}:{p['high']}")
            print(f"{p['key']:<32} {p['kind']:<7} current={p['current']!r:<10} {space}")
        return 0
    try:
        spec = {"dimensions": [parse_space(t, args.points) for t in args.param],
                "metric": args.metric, "direction": args.direction,
                "strategy": args.strategy, "max_trials": args.trials, "seed": args.seed,
                "on": args.on or ""}
        sweep_id = api.start_sweep(graph, spec)["sweep_id"]
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"type": "sweep_started", "sweep_id": sweep_id}), flush=True)
    reported = 0
    while True:
        record = api.get_sweep(sweep_id)
        done = [t for t in record["trials"] if t["state"] not in ("pending", "running")]
        for t in done[reported:]:
            print(json.dumps({"type": "trial", **t}), flush=True)
        reported = len(done)
        if record["state"] in ("finished", "stopped", "failed"):
            print(json.dumps({"type": "sweep_" + record["state"], "best": record["best"]}),
                  flush=True)
            return 0 if record["state"] == "finished" and record["best"] else 1
        _time.sleep(0.5)


def _budget_args(args) -> dict:  # noqa: ANN001
    budget = {"device": args.device, "max_train_memory_gb": args.max_memory_gb,
              "max_params_m": args.max_params_m, "max_latency_ms": args.max_latency_ms}
    return {k: v for k, v in budget.items() if v}


def _wizard_command(args) -> int:  # noqa: ANN001
    import time as _time

    from ai_made_easy.core import api

    try:
        if args.command == "recipes":
            for r in api.list_recipes(args.task)["recipes"]:
                missing = f"  [needs {', '.join(r['missing'])}]" if r["missing"] else ""
                print(f"{r['id']:<26} {r['tier']:<10} {r['modality']:<12} "
                      f"{','.join(r['tasks'])}{missing}")
            return 0
        budget = _budget_args(args)
        if args.command == "new" and not args.task:
            if not args.data:
                print("error: give --task, or --data to pick the task from your data",
                      file=sys.stderr)
                return 1
            facts = api.detect_data(args.data, args.target)
            if not facts.get("tasks"):
                print(f"error: {facts.get('summary') or args.data}: no task fits this data; "
                      "give --task", file=sys.stderr)
                return 1
            args.task = facts["tasks"][0]
            print(f"task: {args.task} (from the data; --task to choose another of "
                  f"{', '.join(facts['tasks'])})", file=sys.stderr)
        if args.command == "new":
            if args.recipe:
                graph = api.build_recipe(args.recipe, args.task, path=args.data or None,
                                         target=args.target, budget=budget)
            else:
                ranked = api.recommend_recipes(args.task, path=args.data or None,
                                               target=args.target, budget=budget,
                                               modality=args.modality)
                print(f"data: {ranked['facts']['summary']}", file=sys.stderr)
                rows = ranked["suggestions"]
                for i, sug in enumerate(rows):
                    r = sug["recipe"]
                    mark = "ready" if sug["ready"] else "not ready"
                    print(f"{i + 1:>2}. {r['title']:<28} {r['tier']:<10} {mark:<9} "
                          f"{'; '.join(sug['reasons'] + sug['cautions'] + sug['errors'])[:110]}",
                          file=sys.stderr)
                if args.list:
                    return 0
                best = next((s for s in rows if s["ready"]), None)
                if best is None:
                    print("error: no recipe is ready for this task and data", file=sys.stderr)
                    return 1
                graph = best["graph"]
                print(f"using {best['recipe']['title']}", file=sys.stderr)
            text = json.dumps(graph, indent=2)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                print(args.out)
            else:
                print(text)
            return 0
        spec = {"task": args.task, "path": args.data or None, "target": args.target,
                "modality": args.modality, "max_trials": args.trials, "epochs": args.epochs,
                "recipes": [r for r in args.recipes.split(",") if r.strip()],
                "metric": args.metric, "tune": not args.no_tune, "prune": not args.no_prune,
                "budget": budget, "seed": args.seed,
                "objectives": [o.strip() for o in args.objectives.split(",") if o.strip()],
                "on": args.on, "concurrency": args.concurrency}
        started = api.start_automl({k: v for k, v in spec.items() if v is not None})
    except (api.ApiError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sweep_id = started["sweep_id"]
    print(json.dumps({"type": "automl_started", **started}), flush=True)
    reported = 0
    while True:
        record = api.get_sweep(sweep_id)
        done = [t for t in record["trials"] if t["state"] not in ("pending", "running")]
        for t in done[reported:]:
            print(json.dumps({"type": "trial", **t}), flush=True)
        reported = len(done)
        if record["state"] in ("finished", "stopped", "failed"):
            break
        _time.sleep(0.5)
    print(json.dumps({"type": "automl_" + record["state"], "best": record["best"]}), flush=True)
    if started.get("objectives"):
        front = [t for t in api.automl_leaderboard(sweep_id)["trials"] if t["pareto"]]
        print(json.dumps({"type": "pareto_front", "trials": [
            {k: t[k] for k in ("number", "recipe", "score", "objectives", "values")}
            for t in front]}), flush=True)
    if record["best"] and args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(api.sweep_best_graph(sweep_id), fh, indent=2)
        print(args.out)
    return 0 if record["state"] == "finished" and record["best"] else 1


def _pipeline_command(args) -> int:  # noqa: ANN001
    import time as _time
    from pathlib import Path

    from ai_made_easy.core import api

    try:
        if args.action == "list":
            for row in api.list_pipelines()["pipelines"]:
                done = sum(s["state"] in ("finished", "cached") for s in row["stages"].values())
                print(f"{row['pipeline_id']:<28} {row['state']:<9} {done}/{len(row['stages'])} "
                      f"stages  {row['name']}")
            return 0
        if not args.target:
            print("error: give a pipeline .json (run) or a pipeline id", file=sys.stderr)
            return 1
        if args.action == "status":
            print(json.dumps({k: v for k, v in api.get_pipeline(args.target).items()
                              if k != "graph"}, indent=2, default=str))
            return 0
        if args.action == "run":
            path = Path(args.target)
            graph = json.loads(path.read_text(encoding="utf-8"))
            started = api.start_pipeline(graph, base=str(path.resolve().parent), on=args.on)
        else:
            started = api.resume_pipeline(args.target)
    except (api.ApiError, KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pipeline_id = started["pipeline_id"]
    print(json.dumps({"type": "pipeline_started", **started}), flush=True)
    if args.no_wait:
        return 0
    reported: dict[str, str] = {}
    while True:
        record = api.get_pipeline(pipeline_id)
        for sid in record["order"]:
            stage = record["stages"][sid]
            if stage["state"] != reported.get(sid) and stage["state"] != "pending":
                reported[sid] = stage["state"]
                print(json.dumps({"type": "stage", "id": sid, "state": stage["state"],
                                  "metrics": stage["metrics"], "message": stage["message"]},
                                 default=str), flush=True)
        if record["state"] in ("finished", "failed", "stopped"):
            print(json.dumps({"type": "pipeline_" + record["state"]}), flush=True)
            return 0 if record["state"] == "finished" else 1
        _time.sleep(0.5)


def _import_command(args) -> int:  # noqa: ANN001
    from pathlib import Path

    from ai_made_easy.core import api

    try:
        shape = [int(d) for d in args.shape.replace("x", ",").split(",") if d.strip()] or None
        if args.kind == "mlflow":
            result = api.open_mlflow_model(args.source, input_shape=shape)
        else:
            result = api.import_model(args.kind, args.source, attr=args.attr,
                                      input_shape=shape, dtype=args.dtype,
                                      kwargs=json.loads(args.kwargs))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result["summary"])
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    if result["unsupported"]:
        return 1
    Path(args.out).write_text(json.dumps(result["graph"], indent=1), encoding="utf-8")
    print(f"project: {args.out}")
    return 0 if result["ok"] else 1


def _data_command(args) -> int:  # noqa: ANN001
    from pathlib import Path

    from ai_made_easy.core import api
    from ai_made_easy.core.data.lints import project_base

    if args.action == "fetch":
        try:
            result = api.fetch_dataset(args.source, force=args.force)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=1, default=str))
            return 0
        facts = result["facts"]
        print(f"{args.source} -> {result['path']}")
        print(f"{facts.get('kind', '')}: {facts.get('rows', 0):,} rows; fits "
              f"{', '.join(facts.get('tasks') or []) or 'no task detected'}")
        print(f"start a project: aime new <task> --data {result['path']}")
        return 0
    source = Path(args.source).expanduser()
    is_project = source.suffix == ".json" and source.is_file() and \
        "nodes" in json.loads(source.read_text(encoding="utf-8") or "{}")
    try:
        if args.action == "profile" and not args.json:
            from ai_made_easy.core.data.lints import dataset_nodes
            from ai_made_easy.core.data.profile import profile_dataset, profile_path

            if not is_project:
                print(profile_path(source, target=args.target).text())
                return 0
            nodes = dataset_nodes(Graph.from_dict(json.loads(source.read_text(encoding="utf-8"))))
            if not nodes:
                print("error: the design has no dataset block", file=sys.stderr)
                return 1
            print(profile_dataset(nodes[0][1], nodes[0][2], project_base(source)).text())
            return 0
        if args.action == "profile" and not is_project:
            result = api.profile_data(path=str(source), target=args.target)
        else:
            if not is_project:
                print("error: give a project .json", file=sys.stderr)
                return 2
            graph = json.loads(source.read_text(encoding="utf-8"))
            base = str(project_base(source))
            result = {
                "profile": lambda: api.profile_data(graph, base=base),
                "check": lambda: api.data_issues(graph, base=base),
                "split": lambda: api.split_preview(graph, base=base),
                "fingerprint": lambda: api.data_fingerprint(graph, base=base),
                "augment": lambda: api.augmentation_preview(graph, args.out, base=base),
            }[args.action]()
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json or args.action == "profile":
        print(json.dumps(result, indent=2, default=str))
    elif args.action == "check":
        for issue in result["issues"]:
            print(f"{issue['severity']}: {issue['message']}")
        print(f"{len(result['issues'])} data issue(s)")
    elif args.action == "split":
        print(f"split: {result['method']}")
        print("            " + "".join(f"{s:>9}" for s in ("train", "val", "test")))
        for cls, counts in result["per_class"].items():
            print(f"{cls[:12]:<12}" + "".join(f"{counts[s]:>9,}" for s in
                                                ("train", "val", "test")))
        print(f"{'total':<12}" + "".join(f"{result['totals'][s]:>9,}" for s in
                                         ("train", "val", "test")))
        for note in result["notes"]:
            print(f"note: {note}")
    elif args.action == "augment":
        for item in result["images"]:
            print(f"{item['source']}: eval {item['eval']}, {len(item['train'])} training views")
        print(f"preview images in {args.out}")
    else:
        print(result["fingerprint"] or "(no local data)")
    return 0


def _deploy_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    formats = [f.strip() for f in getattr(args, "formats", "").split(",") if f.strip()]
    also = [t.strip() for t in getattr(args, "also", "").split(",") if t.strip()]
    try:
        if args.command == "serve":
            from ai_made_easy.core.runner.manager import python_executable

            result = subprocess.run([python_executable(), "-m", "uvicorn", "app:app", "--host",
                                     args.host, "--port", str(args.port)], cwd=args.package)
            return result.returncode
        if args.command == "deploy":
            options = api.export_options(args.run_id)["formats"]
            if args.list_formats:
                for row in options:
                    mark = "suggested" if row["suggested"] else \
                        "available" if row["available"] else "no"
                    tail = f"  ({row['device']})" if row["suggested"] else \
                        f"  {row['reason']}" if not row["available"] else ""
                    print(f"{row['id']:<16} {mark:<10}{tail}")
                return 0
            if not args.out:
                print("error: -o/--out is required", file=sys.stderr)
                return 2
            if "suggested" in formats:
                formats = [f for f in formats if f != "suggested"] + [
                    r["id"] for r in options if r["suggested"]]
            result = api.deploy_run(args.run_id, args.out, formats, args.name)
            print(result["log"].replace("EXPORT-DONE", "").strip())
            print(f"package: {result['path']}")
            return 0
        a = args.args
        if args.action == "list":
            for m in api.list_models()["models"]:
                metrics = " ".join(f"{k}={v:.4g}" for k, v in list(m["metrics"].items())[:3]
                                   if isinstance(v, (int, float)))
                print(f"{m['name']:<24} v{m['version']:<4} {m['stage']:<11} "
                      f"{m['framework']:<8} {metrics}")
        elif args.action == "register" and len(a) == 2:
            m = api.register_model(a[0], a[1], also=also)
            print(f"registered {m['name']} v{m['version']}")
            _print_mirrors(m)
        elif args.action == "stage" and len(a) == 3:
            m = api.set_model_stage(a[0], int(a[1]), a[2], also=also)
            print(f"{m['name']} v{m['version']} -> {m['stage']}")
            _print_mirrors(m)
        elif args.action == "deploy" and len(a) in (1, 2) and args.out:
            result = api.deploy_model(a[0], a[1] if len(a) == 2 else "latest", args.out,
                                      formats)
            print(f"package: {result['path']}")
        elif args.action == "delete" and len(a) == 2:
            api.delete_model(a[0], int(a[1]))
            print(f"deleted {a[0]} v{a[1]}")
        else:
            print("usage: aime models [list | register RUN_ID NAME | stage NAME VERSION "
                  "STAGE | deploy NAME [VERSION] -o DIR | delete NAME VERSION]",
                  file=sys.stderr)
            return 2
        return 0
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _print_mirrors(model: dict) -> None:
    for where in model.get("mirrors") or []:
        print(f"  also in {where['target']}: {where['uri']}")


def _tools_command(args) -> int:  # noqa: ANN001
    """``aime hub``, ``aime kaggle submit`` and ``aime tracking``."""
    from pathlib import Path

    from ai_made_easy.core import api

    try:
        if args.command == "tracking":
            graph = json.loads(Path(args.project).read_text(encoding="utf-8")) if args.project else None
            status = api.tracking_status(graph)
            chosen = ", ".join(status["selected"]) or "off"
            print(f"tracking: {chosen}  (AIME_TRACKING={status['env'] or '-'}; per project: "
                  f"meta.tracking)")
            print(f"MLflow store: {status['mlflow_uri']}")
            for t in status["trackers"]:
                print(f"  {t['label']:<18} {'installed' if t['installed'] else 'not installed'}")
            return 0
        if args.command == "kaggle":
            if not args.yes:
                answer = input(f"Submit {args.file} to {args.competition}? It counts against "
                               "the daily limit. [y/N] ")
                if answer.strip().lower() not in ("y", "yes"):
                    print("not submitted")
                    return 1
            result = api.submit_to_kaggle(args.competition, args.file, args.message,
                                          confirm=True)
            print(f"submitted: {result['result'] or 'ok'}\n{result['url']}")
            return 0
        a = args.args

        def source(ref: str) -> dict:
            if not args.model:
                return {"run_id": ref}
            name, _, version = ref.partition("/")
            return {"model": name, "version": version or "latest"}

        if args.action == "push" and len(a) == 2:
            result = api.push_to_hub(a[1], private=not args.public, **source(a[0]))
            print(f"pushed: {result['url']}")
        elif args.action == "pull" and len(a) == 1:
            m = api.pull_from_hub(a[0], name=args.name)
            print(f"registered {m['name']} v{m['version']}; package it with: "
                  f"aime models deploy {m['name']} {m['version']} -o server")
        elif args.action == "space" and len(a) == 2:
            formats = [f.strip() for f in args.formats.split(",") if f.strip()]
            result = api.deploy_to_space(a[1], formats=formats, private=not args.public,
                                         **source(a[0]))
            print(f"Space: {result['url']} (it builds the Docker image on the Hub)")
        else:
            print("usage: aime hub push RUN_ID REPO | pull REPO | space RUN_ID SPACE",
                  file=sys.stderr)
            return 2
        return 0
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _runs_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    try:
        if args.action == "export":
            result = api.export_runs(args.out, project=args.project, epochs=args.epochs)
            print(f"wrote {result['rows']} row(s), {result['columns']} column(s) to "
                  f"{result['path']}")
            return 0
        if args.action == "list":
            rows = api.list_runs(args.project)["runs"]
            for r in rows:
                metrics = " ".join(f"{k}={v:.4g}" for k, v in
                                   list(r["final_metrics"].items())[:4])
                print(f"{r['run_id']}  {r['status']:<9} {r['framework']:<8} "
                      f"{r['name']:<24} {metrics}")
            print(f"{len(rows)} run(s)")
            return 0
        if not args.ids:
            print("error: give a run id", file=sys.stderr)
            return 2
        if args.action in ("attach", "import", "resume", "more"):
            if args.action == "attach":
                status = api.attach_run(args.ids[0])
            elif args.action == "import":
                status = api.import_run_results(args.ids[0])
            elif args.action == "resume":
                resumed = api.resume_run(args.ids[0])
                print(f"{resumed['run_id']} continues after epoch {resumed['after_epoch']}")
                status = api.manager().status(resumed["run_id"])
            else:
                started = api.train_more(args.ids[0], args.more, trees=args.more)
                print(f"{started['run_id']} continues {args.ids[0]}")
                status = api.manager().status(started["run_id"])
            run_id = status["run_id"]
            if status["state"] not in ("finished", "failed", "stopped"):
                status = api.manager().wait(run_id, timeout=7 * 24 * 3600)
            print(f"{run_id}  {status['state']}  {status.get('latest_metrics') or ''}")
            return 0 if status["state"] == "finished" else 1
        if args.action == "show":
            print(json.dumps(api.get_run(args.ids[0]), indent=2, default=str))
        elif args.action == "compare":
            print(json.dumps(api.compare_runs(args.ids), indent=2, default=str))
        elif args.action == "delete":
            for run_id in args.ids:
                api.delete_run(run_id)
                print(f"deleted {run_id}")
        return 0
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _lr_find_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    path = Path(args.project)
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
        run_id = api.find_learning_rate(graph, steps=args.steps,
                                        framework=args.framework)["run_id"]
        api.manager().wait(run_id, timeout=24 * 3600)
        out = api.learning_rate_result(run_id)
    except (api.ApiError, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if out.get("state") != "finished" or not out.get("suggested"):
        print(f"no suggestion: {out.get('reason') or out.get('error') or out.get('state')}",
              file=sys.stderr)
        return 1
    print(f"suggested learning rate {out['suggested']:.3g} ({out['reason']}); the loss was "
          f"lowest at {out['min_loss_lr']:.3g}")
    if args.apply:
        path.write_text(json.dumps(api.apply_learning_rate(graph, out["suggested"]), indent=2),
                        encoding="utf-8")
        print(f"wrote it into {path}")
    return 0


def _option_value(text: str):  # noqa: ANN202
    """``true`` / numbers / JSON lists become values; anything else stays a string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _backends_command(args) -> int:  # noqa: ANN001
    from ai_made_easy.core import api

    try:
        if args.action == "list":
            listed = api.list_backends()
            for b in listed["backends"]:
                options = " ".join(f"{k}={v}" for k, v in b["options"].items())
                live = "" if b["live"] else "  (curves per poll)"
                print(f"{b['name']:<16} {b['kind']:<10} {options}{live}")
            if args.options:
                for kind, options in listed["options"].items():
                    print(f"\n{kind}:")
                    for name, text in options.items():
                        print(f"  {name:<16} {text}")
            return 0
        if not args.name:
            print("error: give a backend name", file=sys.stderr)
            return 2
        if args.action == "add":
            options = {}
            for item in args.set:
                key, sep, value = item.partition("=")
                if not sep:
                    print(f"error: --set needs OPTION=VALUE, got {item!r}", file=sys.stderr)
                    return 2
                options[key.strip()] = _option_value(value)
            entry = api.save_backend(args.name, args.kind, options)
            print(f"saved {entry['name']} ({entry['kind']}); test it: aime backends test "
                  f"{entry['name']}")
        elif args.action == "test":
            result = api.test_backend(args.name)
            accel = result.get("accelerator") or {}
            gpu = f"  gpu: {accel['gpu']}" if accel.get("gpu") else ""
            device = f"  budget device: {accel['device']}" if accel.get("device") else ""
            print(f"{'ok' if result['ok'] else 'FAILED'}: {result['message']}{gpu}{device}")
            return 0 if result["ok"] else 1
        elif args.action == "remove":
            api.delete_backend(args.name)
            print(f"removed {args.name}")
        return 0
    except (api.ApiError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
