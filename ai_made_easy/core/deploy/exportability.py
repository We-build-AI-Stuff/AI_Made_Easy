"""Which export formats a design supports, and why not when it doesn't.

``options(graph, framework)`` lists every format of the framework with ``available`` and
a ``reason`` when the design can't be exported to it, plus ``suggested`` for the formats
the project's budget device runs (``devices.json`` ``formats``). The design rule
``export_targets`` warns when the device's formats are blocked.

Reasons come from what each converter can't follow; the round-trip tests
(``tests/test_export.py``) check that every sample either exports or is listed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from ai_made_easy.core.deploy.exporters import FORMATS, LABELS

ON_DEVICE = ("executorch", "executorch_int8")
FIXED_SHAPE = ("coreml", *ON_DEVICE)
ALL_TORCH = FORMATS["pytorch"]


@dataclass(frozen=True)
class Blocker:
    formats: tuple[str, ...]
    reason: str


_POOL = ("TopK / SAG pooling sorts each graph's node scores with an in-place sort, which "
         "the ONNX exporter can't follow: export to TorchScript, or use global or "
         "attention pooling")

# block types -> what they stop (PyTorch formats)
_BLOCKS = {
    "bayes.linear": Blocker(FIXED_SHAPE, "Bayesian layers draw new weights on every call "
                                         "(random sampling), which ExecuTorch and Core ML "
                                         "can't run; ONNX keeps the sampling"),
    "bayes.conv2d": Blocker(FIXED_SHAPE, "Bayesian layers draw new weights on every call "
                                         "(random sampling), which ExecuTorch and Core ML "
                                         "can't run; ONNX keeps the sampling"),
    "audio.mel_spectrogram": Blocker(ON_DEVICE, "the spectrogram front end uses a complex "
                                                "STFT, which ExecuTorch can't lower: export "
                                                "to ONNX, or compute features before the "
                                                "model"),
    "audio.mfcc": Blocker(ON_DEVICE, "the MFCC front end uses a complex STFT, which "
                                     "ExecuTorch can't lower: export to ONNX, or compute "
                                     "features before the model"),
    "flow.invconv1x1": Blocker(("coreml", "openvino"), "the invertible 1x1 convolution builds "
                               "its weight with diag, which Core ML and OpenVINO can't "
                               "convert: export to ONNX or ExecuTorch"),
    "flow.spline_coupling": Blocker((*ON_DEVICE, "coreml", "openvino"), "the spline is "
                                    "evaluated only where inputs fall inside its bounds "
                                    "(data-dependent shapes), which ExecuTorch, Core ML and "
                                    "OpenVINO can't follow: export to ONNX"),
    "tab.ft_transformer": Blocker(("executorch_int8",), "categorical embeddings read ids "
                                                        "from the float input row, and the "
                                                        "int8 quantizer can't follow that "
                                                        "cast: use ExecuTorch float or ONNX "
                                                        "int8"),
    "vision.hf_mask2former": Blocker(("executorch_int8",), "the int8 quantizer can't "
                                                           "follow Mask2Former's integer "
                                                           "index tensors: use ExecuTorch "
                                                           "float or ONNX"),
    # 3.7
    "seq.rwkv_block": Blocker(ON_DEVICE, "the RWKV recurrence runs as a loop with running "
                              "maxima that the ExecuTorch runtime rejects: export to ONNX or "
                              "TorchScript"),
    "seq.hyena": Blocker((*ON_DEVICE, "coreml", "openvino"), "the Hyena long convolution "
                         "uses FFTs (complex numbers), which ExecuTorch, Core ML and "
                         "OpenVINO's CPU runtime can't run: export to ONNX or TorchScript"),
    "vision.deform_conv2d": Blocker(("onnx", "onnx_int8", *ON_DEVICE, "coreml"), "deformable "
                                    "convolution is a torchvision operator that only "
                                    "TorchScript can carry"),
    "vision.involution": Blocker(("coreml",), "Core ML's conversion of the patch "
                                 "extraction (unfold) gives different numbers than PyTorch: "
                                 "export to ONNX, TorchScript or ExecuTorch"),
    # 3.8
    "graph.topk_pool": Blocker(("onnx", "onnx_int8"), _POOL),
    "graph.sag_pool": Blocker(("onnx", "onnx_int8"), _POOL),
}
_OPENVINO_ONE_TENSOR = Blocker(("openvino",), "OpenVINO export takes models with one input "
                               "tensor: export to ONNX, which OpenVINO also reads")
# trainer kinds -> what they stop
_KINDS = {
    "detection": Blocker(FIXED_SHAPE, "detectors keep their post-processing (NMS), whose "
                                      "number of boxes depends on the image; ExecuTorch "
                                      "and Core ML need fixed shapes: export to ONNX or "
                                      "TorchScript"),
    "graph": Blocker(FIXED_SHAPE, "graph networks take a different number of nodes and "
                                  "edges per request; ExecuTorch and Core ML need fixed "
                                  "shapes: export to ONNX"),
    "seq2seq": Blocker(ON_DEVICE, "the decoder generates token by token and stops at the "
                                  "end token (a data-dependent loop), which ExecuTorch "
                                  "can't follow: export to ONNX"),
    "text_to_image": Blocker(ALL_TORCH, "Stable Diffusion LoRA adapters are served with "
                                        "diffusers; there is no single model graph to "
                                        "export"),
}


def _task(graph):  # noqa: ANN001, ANN202
    from ai_made_easy.core.tasks import task_of

    try:
        return task_of(graph)
    except Exception:  # noqa: BLE001 — incomplete designs
        return None


def export_kind(graph) -> str:  # noqa: ANN001 — Graph
    """``detector``, ``graph`` or ``tensor`` (see exporters.py)."""
    task = _task(graph)
    kind = task.trainer_kind if task is not None else ""
    return {"detection": "detector", "graph": "graph"}.get(kind, "tensor")


def framework_of(graph) -> str:  # noqa: ANN001 — Graph
    from ai_made_easy.core.families import resolve_framework

    try:
        return resolve_framework(graph)
    except ValueError:
        return ""


def blockers(graph, framework: str | None = None) -> dict[str, tuple[str, str | None]]:  # noqa: ANN001
    """format -> (reason, node id or None) for the formats the design can't use."""
    framework = framework or framework_of(graph)
    if framework != "pytorch":
        return {}
    out: dict[str, tuple[str, str | None]] = {}
    task = _task(graph)
    kinds = [_KINDS.get(task.trainer_kind)] if task is not None else []
    if task is not None and task.trainer_kind in ("detection", "graph"):
        kinds.append(_OPENVINO_ONE_TENSOR)
    for kind in filter(None, kinds):
        for fmt in kind.formats:
            out.setdefault(fmt, (kind.reason, None))
    for node in graph.nodes.values():
        blocker = _BLOCKS.get(node.type_id)
        if blocker is None:
            continue
        for fmt in blocker.formats:
            out.setdefault(fmt, (blocker.reason, node.instance_id))
    return out


def suggested_formats(device_id: str, framework: str) -> list[str]:
    """The formats a device runs, best first, among the framework's."""
    from ai_made_easy.core import budget

    if not device_id:
        return []
    try:
        device = budget.get_device(device_id)
    except KeyError:
        return []
    return [f for f in device.formats if f in FORMATS.get(framework, ())]


def options(graph, framework: str | None = None, device: str | None = None) -> list[dict]:  # noqa: ANN001
    """Every export format of the framework, with availability, a reason and whether
    the project's budget device suggests it."""
    from ai_made_easy.core import budget

    framework = framework or framework_of(graph)
    if device is None:
        device = budget.budget_of(graph).get("device", "")
    blocked = blockers(graph, framework)
    suggested = set(suggested_formats(device, framework))
    label = ""
    if device:
        try:
            label = budget.get_device(device).label
        except KeyError:
            label = ""
    rows = []
    for fmt in FORMATS.get(framework, ()):
        reason = blocked.get(fmt, ("", None))[0]
        rows.append({"id": fmt, "label": LABELS[fmt], "available": not reason,
                     "reason": reason, "suggested": fmt in suggested and not reason,
                     "device": label if fmt in suggested else ""})
    return rows


def options_for_run(run_dir) -> list[dict]:  # noqa: ANN001 — a run or registry folder
    """Export options for a trained run, from its run.json (design and framework)."""
    import json
    from pathlib import Path

    from ai_made_easy.core.graph import Graph

    record = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
    framework = record.get("framework", "pytorch")
    try:
        graph = Graph.from_dict(record.get("graph") or {})
    except Exception:  # noqa: BLE001 — an old record: every format, nothing suggested
        return [{"id": f, "label": LABELS[f], "available": True, "reason": "",
                 "suggested": False, "device": ""} for f in FORMATS.get(framework, ())]
    return options(graph, framework)
