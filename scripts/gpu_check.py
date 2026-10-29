"""Check training on several real GPUs, through the Modal backend.

CI has no GPUs, so DistributedDataParallel and sharding (FSDP) over NCCL, and float16
loss scaling, only run on CPU processes there. This trains a small network on 2 GPUs
in a Modal sandbox, each way, and checks each run finished and learned:

    python scripts/gpu_check.py --gpu L4 --out gpu_check.json

Sign-in is Modal's own: ``modal token new``, or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in
the environment (the GPU check workflow takes them from the repository's secrets). This
script never reads them.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# name -> the Trainer's settings, and what the run's log says when it trains that way
CASES = {
    "ddp_float16": ({"sharding": False, "mixed_precision": True, "precision": "float16"},
                    "training on 2 GPUs (DistributedDataParallel)"),
    "sharded_float16": ({"sharding": True, "mixed_precision": True, "precision": "float16"},
                        "training on 2 GPUs (FSDP"),
    "sharded_bfloat16": ({"sharding": True, "mixed_precision": True, "precision": "bfloat16"},
                         "training on 2 GPUs (FSDP"),
}
NO_BFLOAT16 = {"T4"}   # GPUs older than Ampere


def design(settings: dict) -> dict:
    """A small classifier on synthetic data, on 2 GPUs."""
    nodes = [
        ("in", "core.input", {"shape": "32"}),
        ("d1", "core.dense", {"units": 256}),
        ("act", "core.relu", {}),
        ("d2", "core.dense", {"units": 256}),
        ("act2", "core.relu", {}),
        ("d3", "core.dense", {"units": 4}),
        ("out", "core.output", {}),
        ("data", "data.synthetic", {"kind": "classification", "n_samples": 4000,
                                    "n_features": 32, "n_classes": 4, "seed": 3}),
        ("opt", "train.adam", {"lr": 0.003}),
        ("loss", "train.loss_cross_entropy", {}),
        ("acc", "eval.accuracy", {}),
        ("trainer", "train.trainer", {"epochs": 6, "batch_size": 64, "device": "cuda",
                                      "gpus": 2, "seed": 0, **settings}),
    ]
    chain = ["in", "d1", "act", "d2", "act2", "d3", "out"]
    return {"schema_version": 3, "name": "gpu_check", "meta": {},
            "nodes": [{"id": i, "type": t, "params": p, "position": [0, 0]}
                      for i, t, p in nodes],
            "edges": [{"from": f"{a}/out", "to": f"{b}/in"}
                      for a, b in zip(chain, chain[1:], strict=False)]}


def check(gpu: str, cases: list[str], timeout: float) -> dict:
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.core.runner import backends
    from ai_made_easy.core.runner.manager import RunManager

    backends.save("gpu-check", "modal", gpu=gpu, gpu_count=2)
    mgr = RunManager()
    results = {}
    for name in cases:
        settings, expect = CASES[name]
        if name.endswith("bfloat16") and gpu in NO_BFLOAT16:
            results[name] = {"ok": None, "why": f"{gpu} has no bfloat16"}
            continue
        run_id = mgr.start(Graph.from_dict(design(settings)), on="gpu-check")
        problems = []
        try:
            state = mgr.wait(run_id, timeout)["state"]
        except TimeoutError:   # stop it, so the sandbox doesn't keep the GPUs
            mgr.stop(run_id)
            state = "timed out"
            problems.append(f"it did not finish within {timeout:.0f}s; stopped")
        record = mgr.history.get(run_id)
        logs = list(mgr.get(run_id).logs)
        accuracy = (record.final_metrics or {}).get("accuracy")
        if state not in ("finished", "timed out"):
            problems.append(f"the run {state}: {record.error or logs[-5:]}")
        if not any(expect in line for line in logs):
            problems.append(f"no {expect!r} in its log")
        if accuracy is None or accuracy < 0.6:
            problems.append(f"accuracy {accuracy}: it did not learn")
        results[name] = {"ok": not problems, "run_id": run_id, "accuracy": accuracy,
                         "why": "; ".join(problems)}
        print(f"{name}: {'ok' if not problems else 'FAILED: ' + '; '.join(problems)}",
              flush=True)
    return results


def summary(report: dict) -> str:
    lines = [f"## GPU check: 2 × {report['gpu']} on Modal ({report['date']})", "",
             "| Case | Result | Accuracy | Notes |", "|---|---|---|---|"]
    for name, r in report["results"].items():
        mark = {True: "passed", False: "**failed**", None: "skipped"}[r["ok"]]
        acc = f"{r['accuracy']:.3f}" if r.get("accuracy") is not None else ""
        lines.append(f"| {name} | {mark} | {acc} | {r.get('why', '')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", default="L4", help="a Modal GPU type (default L4)")
    parser.add_argument("--cases", default=",".join(CASES),
                        help=f"comma-separated, among {', '.join(CASES)}")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per run")
    parser.add_argument("--out", default="gpu_check.json")
    args = parser.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = sorted(set(cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases {unknown}")
    report = {"gpu": args.gpu, "count": 2,
              "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
              "results": check(args.gpu, cases, args.timeout)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary(report))
    return 0 if all(r["ok"] is not False for r in report["results"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
