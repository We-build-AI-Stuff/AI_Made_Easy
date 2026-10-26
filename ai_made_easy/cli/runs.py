"""Runs, pipelines, learning rates and backends from the command line."""
from __future__ import annotations

import json
import sys
from pathlib import Path


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
