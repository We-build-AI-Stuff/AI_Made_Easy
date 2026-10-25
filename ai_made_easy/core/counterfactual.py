"""What would change the prediction (3.11): counterfactuals for a finished run that reads
table rows.

Given one row and its prediction, the search asks the run's own predictor (the Try It
worker, on this machine) about edited copies of the row:

1. **One change.** Every column you didn't lock is set, in turn, to values the training
   table really has: its deciles for numbers, its most common values for categories.
   Each column whose change reaches the goal is listed, with its smallest such change.
2. **A few changes together**, when no single change is enough: the change that moves the
   prediction furthest towards the goal is kept, then the next, up to three; changes the
   result doesn't need are dropped again.

The goal is another class for a classifier (by default the runner-up), or a value to
reach for a regressor. A change's size is measured in the column's spread (its
interquartile range) so columns compare; a new category counts as 1. Models are never
retrained or opened up, so this works for networks and classic models alike, and its
answers are what the model does, not why.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MAX_CHANGES = 3          # changes combined at most
SHOWN = 5                # single changes listed
DECILES = [i / 10 for i in range(1, 10)]
CATEGORIES = 12          # most common values tried for a category
_BATCH = 64              # rows per request to the worker (its limit)


class CounterfactualError(ValueError):
    """A run or a request the search can't answer."""


def _table(record: dict) -> tuple[Path, str]:
    node = next((n for n in (record.get("graph") or {}).get("nodes", [])
                 if n.get("type") == "data.csv"), None)
    if node is None:
        raise CounterfactualError("what would change the prediction is worked out for "
                                  "models trained on a table (data.csv)")
    params = node.get("params") or {}
    path = Path(params.get("path") or "")
    if not path.is_file():
        raise CounterfactualError(f"the run's table is no longer at {path}: the search "
                                  "tries values from it")
    return path, params.get("format") or "csv"


def candidates(path: Path, fmt: str, columns: list[str]) -> dict:
    """For each column: the values to try, whether it is numeric, and its spread."""
    import pandas as pd

    readers = {"parquet": pd.read_parquet, "excel": pd.read_excel,
               "tsv": lambda p: pd.read_csv(p, sep="\t"),
               "json": lambda p: pd.read_json(p, lines=str(p).endswith(".jsonl"))}
    frame = readers.get(fmt, pd.read_csv)(path)
    out = {}
    for name in columns:
        if name not in frame.columns:
            continue
        values = frame[name].dropna()
        if values.empty:
            continue
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            whole = bool((values == values.round()).all())
            tried = sorted({float(v) for v in values.quantile(DECILES)})
            if whole:
                tried = sorted({float(round(v)) for v in tried})
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            spread = float(q3 - q1) or float(values.std() or 0) or 1.0
            out[name] = {"numeric": True, "whole": whole, "values": tried, "spread": spread}
        else:
            common = values.astype(str).value_counts().index[:CATEGORIES].tolist()
            out[name] = {"numeric": False, "values": common, "spread": 1.0}
    return out


def _value(info: dict, v):  # noqa: ANN001, ANN202
    if not info["numeric"]:
        return str(v)
    return int(v) if info["whole"] else round(float(v), 6)


def _distance(info: dict, old, new) -> float:  # noqa: ANN001
    if not info["numeric"]:
        return 0.0 if str(old) == str(new) else 1.0
    try:
        return abs(float(new) - float(old)) / info["spread"]
    except (TypeError, ValueError):
        return 1.0


class _Goal:
    """How close a prediction is to the goal, and whether it reached it."""

    def __init__(self, base: dict, desired: str | None, target: float | None) -> None:
        if isinstance(base.get("probabilities"), dict):
            self.kind = "class"
            ranked = sorted(base["probabilities"], key=base["probabilities"].get,
                            reverse=True)
            if desired is None:
                desired = next((c for c in ranked if c != base.get("label")), None)
            if desired is None or str(desired) not in base["probabilities"]:
                raise CounterfactualError(f"pick a class to reach among {ranked}")
            self.desired = str(desired)
            self.text = f"predicted {self.desired} (now {base.get('label')})"
        elif isinstance(base.get("prediction"), (int, float)):
            self.kind = "value"
            if target is None or not math.isfinite(float(target)):
                raise CounterfactualError("give a target value for the prediction to reach")
            self.now, self.target = float(base["prediction"]), float(target)
            self.up = self.target >= self.now
            self.text = (f"a prediction {'at least' if self.up else 'at most'} "
                         f"{self.target:g} (now {self.now:g})")
        else:
            raise CounterfactualError("counterfactuals are worked out for classifiers and "
                                      "single-value regressors")

    def score(self, p: dict) -> float:
        if self.kind == "class":
            return float(p["probabilities"].get(self.desired, 0.0))
        value = float(p["prediction"])
        return value if self.up else -value

    def reached(self, p: dict) -> bool:
        if self.kind == "class":
            return str(p.get("label")) == self.desired
        value = float(p["prediction"])
        return value >= self.target if self.up else value <= self.target


def search(predict, row: dict, columns: dict, *, desired: str | None = None,  # noqa: ANN001
           target: float | None = None, locked=()) -> dict:
    """Counterfactuals of ``row``. ``predict(rows) -> predictions`` is the run's own."""
    asked = [0]

    def ask(rows: list[dict]) -> list[dict]:
        out = []
        for i in range(0, len(rows), _BATCH):
            out.extend(predict(rows[i:i + _BATCH]))
        asked[0] += len(rows)
        return out

    base = ask([row])[0]
    goal = _Goal(base, desired, target)
    if goal.reached(base):
        raise CounterfactualError(
            f"it already predicts {goal.desired}: pick another class" if goal.kind == "class"
            else f"the prediction already reaches {goal.target:g}: pick another value")
    free = {c: info for c, info in columns.items() if c not in set(locked)}
    if not free:
        raise CounterfactualError("every column is locked: unlock one for the search to change")

    def edits(current: dict, skip=()) -> list[tuple[str, object, float]]:
        found = []
        for name, info in free.items():
            if name in skip:
                continue
            for v in info["values"]:
                new = _value(info, v)
                gap = _distance(info, current.get(name), new)
                if gap > 0:
                    found.append((name, new, gap))
        return found

    def entry(changes: dict, prediction: dict) -> dict:
        return {"changes": [{"column": c, "from": row.get(c), "to": v}
                            for c, v in changes.items()],
                "distance": round(sum(_distance(free[c], row.get(c), v)
                                      for c, v in changes.items()), 4),
                "prediction": prediction}

    singles = edits(row)
    answers = ask([{**row, name: new} for name, new, _ in singles])
    best: dict[str, dict] = {}
    for (name, new, gap), p in zip(singles, answers, strict=True):
        if goal.reached(p) and (name not in best or gap < best[name]["distance"]):
            best[name] = entry({name: new}, p) | {"distance": round(gap, 4)}
    single = sorted(best.values(), key=lambda e: e["distance"])[:SHOWN]
    combined = None
    if not single:
        current, changes, prediction = dict(row), {}, base
        tried = dict(zip([(n, v) for n, v, _ in singles], answers, strict=True))
        for step in range(MAX_CHANGES):
            options = edits(current, skip=changes) if step else singles
            if step:
                got = ask([{**current, name: new} for name, new, _ in options])
                tried = dict(zip([(n, v) for n, v, _ in options], got, strict=True))
            if not options:
                break
            name, new, _ = max(options, key=lambda o: (goal.score(tried[(o[0], o[1])]), -o[2]))
            changes[name], current[name] = new, new
            prediction = tried[(name, new)]
            if goal.reached(prediction):
                break
        if goal.reached(prediction):
            for name in list(changes):        # drop the changes it doesn't need
                if len(changes) == 1:
                    break
                trimmed = {**current, name: row.get(name)}
                p = ask([trimmed])[0]
                if goal.reached(p):
                    current, prediction = trimmed, p
                    del changes[name]
            combined = entry(changes, prediction)
    return {"prediction": base, "goal": goal.text, "single": single, "combined": combined,
            "locked": sorted(set(locked)), "predictions_checked": asked[0],
            "found": bool(single or combined)}


def what_would_change(service, history, run_id: str, row: dict, *,  # noqa: ANN001
                      desired: str | None = None, target: float | None = None,
                      locked=()) -> dict:
    """Counterfactuals of one table row for a finished run (see the module docstring)."""
    from ai_made_easy.core import tryit

    if not isinstance(row, dict) or not row:
        raise CounterfactualError("give one row: {column: value, ...}")
    history.get(run_id)
    record = json.loads((history.path(run_id) / "run.json").read_text(encoding="utf-8"))
    path, fmt = _table(record)
    worker, meta = tryit._target(service, history, run_id)
    if meta.get("input_kind") != "records":
        raise CounterfactualError("counterfactuals are worked out for models that read "
                                  "table rows")
    names = list(worker.signature.get("columns") or row)
    missing = [c for c in names if c not in row]
    if missing:
        raise CounterfactualError(f"the row is missing {missing}")
    unknown = [c for c in locked if c not in names]
    if unknown:
        raise CounterfactualError(f"no such column to lock: {unknown}")
    columns = candidates(path, fmt, names)
    out = search(lambda rows: worker.ask(rows, timeout=120), {c: row[c] for c in names},
                 columns, desired=desired, target=target, locked=locked)
    return {"run_id": run_id, **out}
