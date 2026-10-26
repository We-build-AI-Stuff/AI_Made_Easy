"""Designs from the command line: budgets, sweeps, recipes and the wizard."""
from __future__ import annotations

import json
import sys


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
