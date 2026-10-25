"""One-click fixes: turn known validation issues into graph edits.

Pure functions over the IR (Qt-free, unit-tested). ``fix_for_issue`` mirrors
core/suggestions.py's pattern table: each recognised issue maps to a
transform that returns a NEW Graph — the UI reloads it onto the canvas.
"""
from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ai_made_easy.core.graph import Graph, ValidationIssue

import uuid


def _fresh_id(base: str, graph: "Graph") -> str:
    while base in graph.nodes:
        base = f"{base}_{uuid.uuid4().hex[:3]}"
    return base


def _clamp_params(graph: "Graph", node_id: str) -> "Graph | None":
    node = graph.nodes[node_id]
    resolved = node.resolved_params()
    changed = False
    for spec in node.definition().params:
        if spec.type not in ("int", "float"):
            continue
        v = resolved.get(spec.name)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if spec.minimum is not None and v < float(spec.minimum):
            node.params[spec.name] = spec.minimum
            changed = True
        elif spec.maximum is not None and v > float(spec.maximum):
            node.params[spec.name] = spec.maximum
            changed = True
    return graph if changed else None


def fix_for_issue(graph: "Graph", issue: "ValidationIssue"):
    """Return (button_label, description, fixed_graph) or None."""
    msg, nid = issue.message, issue.node_id
    g = copy.deepcopy(graph)
    node = g.nodes.get(nid) if nid else None

    # Dense on a spatial tensor -> insert Flatten between
    if node is not None and "Flatten" in msg and node.type_id == "core.dense":
        in_edge = next((e for e in g.edges
                        if e.target_id == nid), None)
        if in_edge is not None:
            src = g.nodes[in_edge.source_id]
            flat_id = _fresh_id("flatten", g)
            from ai_made_easy.core.graph import Edge, NodeInstance
            mid_x = (src.position[0] + node.position[0]) / 2
            mid_y = (src.position[1] + node.position[1]) / 2
            g.add_node(NodeInstance(flat_id, "core.flatten",
                                    {}, (mid_x, mid_y)))
            g.edges.remove(in_edge)
            g.add_edge(Edge(in_edge.source_id, in_edge.source_port,
                            flat_id, "in"))
            g.add_edge(Edge(flat_id, "out", nid, in_edge.target_port))
            return ("Insert Flatten", "insert a Flatten block between "
                    f"{src.definition().display_name} and Dense", g)

    # out-of-range parameter -> clamp to the nearest bound
    if node is not None and re.search(r"below the minimum|exceeds the maximum", msg):
        fixed = _clamp_params(g, nid)
        if fixed is not None:
            return ("Clamp to range", "move the out-of-range value to the nearest "
                    "allowed bound", fixed)

    # MultiheadAttention width mismatch -> switch to auto
    if node is not None and node.type_id == "core.multihead_attention" and (
            "divisible" in msg or "embed_dim" in msg):
        node.params["embed_dim"] = 0
        return ("Infer embed_dim", "set embed_dim = 0 so it is inferred from the "
                "input width", g)

    # Text Splitter overlap >= chunk -> halve it
    if node is not None and node.type_id == "llm.text_splitter" \
            and "chunk_overlap" in msg:
        chunk = node.resolved_params().get("chunk_size", 500)
        node.params["chunk_overlap"] = max(chunk // 4, 0)
        return ("Reduce overlap", "set chunk_overlap to a quarter of "
                "chunk_size", g)

    # dataset vs Input shape mismatch -> reshape the Input to the dataset
    m = re.search(r"produces (\d+) features per sample but the Input", msg)
    if m:
        inputs = [n for n in g.nodes.values() if n.type_id == "core.input"]
        if inputs:
            inputs[0].params["shape"] = m.group(1)
            return ("Match input", "set the Input shape to the dataset's "
                    f"{m.group(1)} features", g)

    # a probability table whose columns do not sum to 1 -> rescale them
    if node is not None and msg.endswith(": Normalize the table") \
            and str(node.params.get("cpd", "")).strip():
        from ai_made_easy.core.pgm.network import normalized

        node.params["cpd"] = normalized(node.params["cpd"])
        return ("Normalize the table", "rescale every column of the probability table to "
                "sum to 1", g)

    # a tabular model whose categorical positions differ from the data -> copy them
    if node is not None and msg.endswith(": Use the data's categories"):
        from ai_made_easy.core.tabular.rules import fix_categories

        if fix_categories(g, node):
            return ("Use the data's categories", "set the categorical positions and "
                    "cardinalities from the dataset", g)

    # "... set <param> to <number>" on a block that has that parameter -> set it
    m = re.search(r"[Ss]et (\w+) to (-?\d+(?:\.\d+)?)\b", msg)
    if node is not None and m and m.group(1) in {p.name for p in node.definition().params}:
        spec = next(p for p in node.definition().params if p.name == m.group(1))
        value = int(float(m.group(2))) if spec.type == "int" else float(m.group(2))
        node.params[spec.name] = value
        return (f"Set {spec.name} = {value}", f"set {spec.name} to {value}", g)

    # "... set <param> to <option>" for an enum parameter -> pick that option
    m = re.search(r"[Ss]et (\w+) to ([A-Za-z_][\w-]*)\b", msg)
    if node is not None and m:
        spec = next((p for p in node.definition().params if p.name == m.group(1)), None)
        if spec is not None and spec.type == "enum" and m.group(2) in spec.options:
            node.params[spec.name] = m.group(2)
            return (f"Set {spec.name} = {m.group(2)}", f"set {spec.name} to {m.group(2)}", g)
        if spec is not None and spec.type == "bool" and m.group(2) in ("true", "false"):
            node.params[spec.name] = m.group(2) == "true"
            return (f"Set {spec.name} = {m.group(2)}", f"set {spec.name} to {m.group(2)}", g)

    # training memory over budget -> mixed precision, else a batch size that fits
    if msg.startswith("Training needs about"):
        fixed = _fit_training_memory(g)
        if fixed is not None:
            return fixed

    # "... Set the Input shape to '28, 28'" -> set it
    m = re.search(r"Set the Input shape to '([\d, ]+)'", msg)
    if m:
        inputs = [n for n in g.nodes.values() if n.type_id == "core.input"]
        if len(inputs) == 1:
            inputs[0].params["shape"] = m.group(1)
            return ("Match input", f"set the Input shape to {m.group(1)}", g)

    # disconnected input with an obvious nearest predecessor -> wire it
    if node is not None and "is not connected" in msg:
        candidates = [n for n in g.nodes.values()
                      if n.instance_id != nid
                      and n.definition().outputs
                      and n.type_id not in ("core.output",)]
        nearby = [n for n in candidates
                  if abs(n.position[1] - node.position[1]) < 120
                  and n.position[0] < node.position[0] + 80]
        if nearby:
            nearest = max(nearby, key=lambda n: n.position[0])
            from ai_made_easy.core.graph import Edge
            defn = node.definition()
            port = next((p for p in defn.inputs if p.dtype == "tensor"), None)
            out = nearest.definition().outputs[0]
            g.add_edge(Edge(nearest.instance_id, out.name,
                            nid, port.name if port else "in"))
            return ("Connect", f"connect {nearest.definition().display_name} "
                    "to this block", g)

    # off-path block -> offer deletion
    if issue.severity == "warning" and "is not on the path" in msg \
            and node is not None:
        g.edges = [e for e in g.edges
                   if e.source_id != nid and e.target_id != nid]
        del g.nodes[nid]
        return ("Delete block", "delete the disconnected block", g)

    return None


def _fit_training_memory(g: "Graph"):
    from ai_made_easy.core import budget

    trainer = next((n for n in g.nodes.values() if n.type_id == "train.trainer"), None)
    limits = [c for c in budget.check(g) if c.kind == "train_memory"]
    if trainer is None or not limits:
        return None
    limit = limits[0].limit * budget.GB
    est = budget.estimate(g)
    if not est.mixed_precision and est.train_memory_bytes(mixed_precision=True) <= limit:
        trainer.params["mixed_precision"] = True
        return ("Use mixed precision", "turn on mixed precision: activations take half the "
                "memory and the batch size stays the same", g)
    p = trainer.resolved_params()
    averaging = float(p.get("ema_decay") or 0) or int(p.get("swa_start") or 0)
    if est.gpus > 1 and not est.sharded and not averaging:   # sharding keeps no averages
        est.sharded = True
        if est.train_memory_bytes() <= limit:
            trainer.params["sharding"] = True
            return ("Shard the model", f"split the weights, gradients and optimizer state "
                    f"between the {est.gpus} GPUs (FSDP): each holds its share", g)
        est.sharded = False
    batch = est.fitting_batch(limit)
    if not batch or batch >= est.batch_size:
        return None
    steps = max(1, round(est.accumulation_steps * est.batch_size / batch))
    trainer.params["batch_size"] = batch
    trainer.params["accumulation_steps"] = steps
    for loader in (n for n in g.nodes.values() if n.type_id == "prep.dataloader"):
        if int(loader.resolved_params().get("batch_size") or 0):
            loader.params["batch_size"] = batch
    return ("Fit batch size", f"use batch {batch} with {steps} accumulation steps: same "
            "effective batch, less memory", g)
