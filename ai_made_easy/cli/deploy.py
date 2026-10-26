"""Deploying, importing, data and your tools from the command line."""
from __future__ import annotations

import json
import subprocess
import sys

from ai_made_easy.core.graph import Graph


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
