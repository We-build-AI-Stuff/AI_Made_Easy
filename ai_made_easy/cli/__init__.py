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
  aime sweep <project.json> -p opt.lr=log:1e-4:1e-1 -p d1.units=int:16:128              -p opt.nesterov=choice:true,false --metric accuracy --strategy tpe -n 20
  aime recipes [--task TASK]          # design recipes per task
  aime new --task binary --data churn.csv -o churn.json [--recipe ID] [--list]
  aime automl --task binary --data churn.csv -n 12 --epochs 20 -o best.json
  aime pipeline run pipeline.json | status ID | list | resume ID
"""
from __future__ import annotations

import json
import subprocess
import sys

from ai_made_easy.cli.deploy import (_data_command, _deploy_command, _import_command,
                                     _tools_command)
from ai_made_easy.cli.design import _budget_command, _sweep_command, _wizard_command, parse_space
from ai_made_easy.cli.parser import build_parser
from ai_made_easy.cli.runs import (_backends_command, _lr_find_command, _pipeline_command,
                                   _runs_command)
from ai_made_easy.core.codegen import export, export_training
from ai_made_easy.core.codegen import sanitize_identifier as sanitize_name
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
    if args.command in ("deploy", "models", "serve", "gguf"):
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


__all__ = ["main", "parse_space"]
