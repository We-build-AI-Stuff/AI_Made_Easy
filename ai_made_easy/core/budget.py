"""Resource budgets: FLOPs, memory and latency estimates against a target device.

Everything here is analytic (no torch import) and works from the IR: shapes
come from shape inference, parameter counts from the block ``param_fn``s or
the model zoo. Estimates are deliberately simple and documented so the
Summary can show estimate vs measured once a real run has calibrated them.

* FLOPs: 2 × multiply-accumulates. Linear / conv layers cost
  ``2 × weights × output positions``; recurrent and attention layers add their
  sequence terms; element-wise blocks cost one op per output value.
* Training memory: weights + gradients + optimizer state for trainable
  weights, plus saved activations × batch (halved by mixed precision), plus the
  device's runtime overhead. Gradient checkpointing keeps about two layers'
  activations instead of every layer's, and adds a forward pass to each step.
* Inference latency (batch 1): compute time at a realistic fraction of the
  device's peak + memory traffic over its bandwidth + a small launch cost per
  kernel (one per block; composite blocks declare ``meta["kernels"]``).

The project's budget lives in ``graph.meta["budget"]``::

    {"device": "rtx_3060", "max_train_memory_gb": 0, "max_latency_ms": 20,
     "max_params_m": 0, "max_model_mb": 0}

Zero means "no limit" (for memory: the device's memory). Pure Python, Qt-free.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.spec import shape_volume

GB = 1024 ** 3
COMPUTE_EFFICIENCY = 0.35   # share of peak FLOPs real models reach
SAVED_ACTIVATION_FACTOR = 1.0   # calibrated on MPS against measured peak allocations
# activations whose backward needs only their output: the producer's output is freed
OUTPUT_SAVING = {"core.relu", "core.sigmoid", "core.tanh", "core.softmax", "core.log_softmax"}
VIEW_OPS = {"core.flatten", "core.reshape", "core.permute", "core.transpose", "core.squeeze",
            "core.unsqueeze", "core.identity", "core.slice", "core.select_timestep",
            "core.output"}
RECURRENT_STATES = {"core.lstm": 5, "core.gru": 4}  # gates + cell kept per step
LAYER_LAUNCH_MS = {"cpu": 0.002, "mps": 0.02, "cuda": 0.008, "mobile": 0.01, "edge": 0.01}
OPTIMIZER_STATES = {"train.sgd": 1, "train.adam": 2, "train.adamw": 2, "train.nadam": 2,
                    "train.radam": 2, "train.adamax": 2, "train.adagrad": 1,
                    "train.adadelta": 2, "train.rmsprop": 1, "train.adafactor": 1,
                    "train.asgd": 1, "train.rprop": 2}
BUDGET_KEYS = ("device", "max_train_memory_gb", "max_latency_ms", "max_params_m",
               "max_model_mb")
BACKBONE_ACTIVATIONS_PER_FLOP = 1 / 320   # measured on torchvision ResNets / ConvNeXts


# ------------------------------------------------------------------ devices

@dataclass(frozen=True)
class Device:
    id: str
    label: str
    kind: str              # cpu, mps, cuda, mobile, edge
    memory_gb: float       # memory available to one training / inference process
    tflops_fp32: float
    tflops_fp16: float
    bandwidth_gbs: float
    overhead_gb: float = 0.4   # framework / driver context
    formats: tuple = ()        # export formats the device runs, best first

    def __post_init__(self) -> None:
        object.__setattr__(self, "formats", tuple(self.formats or ()))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["formats"] = list(self.formats)
        return out


def user_devices_path() -> Path:
    from ai_made_easy.core.paths import aime_home

    return aime_home() / "devices.json"


def devices() -> dict[str, Device]:
    """Built-in profiles plus the user's own (``~/.aime/devices.json``)."""
    out = dict(_builtin_devices())
    path = user_devices_path()
    if path.is_file():
        try:
            for row in json.loads(path.read_text(encoding="utf-8")):
                out[row["id"]] = Device(**row)
        except (ValueError, TypeError, KeyError):
            pass  # a broken user file never hides the built-ins
    return out


@lru_cache(maxsize=1)
def _builtin_devices() -> dict[str, Device]:
    rows = json.loads((Path(__file__).with_name("devices.json")).read_text(encoding="utf-8"))
    return {r["id"]: Device(**r) for r in rows}


def get_device(device_id: str) -> Device:
    table = devices()
    if device_id not in table:
        raise KeyError(f"unknown device {device_id!r}; one of {sorted(table)}")
    return table[device_id]


def save_user_device(device: Device) -> None:
    path = user_devices_path()
    rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    rows = [r for r in rows if r.get("id") != device.id] + [device.to_dict()]
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ settings

def budget_of(graph) -> dict:  # noqa: ANN001 — Graph
    """The project's budget with defaults filled in ({} values mean none)."""
    raw = dict((getattr(graph, "meta", None) or {}).get("budget") or {})
    out = {"device": str(raw.get("device") or "")}
    for key in BUDGET_KEYS[1:]:
        try:
            out[key] = max(float(raw.get(key) or 0), 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def set_budget(graph, **values) -> dict:  # noqa: ANN001 — Graph
    unknown = set(values) - set(BUDGET_KEYS)
    if unknown:
        raise ValueError(f"unknown budget setting(s) {sorted(unknown)}; one of {BUDGET_KEYS}")
    if values.get("device"):
        get_device(values["device"])
    budget = {**budget_of(graph), **values}
    graph.meta = {**(graph.meta or {}), "budget": budget}
    return budget


# ------------------------------------------------------------------ estimates

@dataclass
class LayerCost:
    node_id: str
    name: str
    type_id: str
    output_shape: list[int]
    params: int
    flops: int
    activations: int        # values saved for backward (per sample)
    trainable: bool = True
    kernels: int = 1        # kernel launches in one forward pass


@dataclass
class Estimate:
    params: int = 0
    trainable_params: int = 0
    flops: int = 0                  # per sample, forward
    activations: int = 0            # values per sample
    batch_size: int = 1
    mixed_precision: bool = False
    accumulation_steps: int = 1
    checkpointing: bool = False     # the Trainer's gradient checkpointing
    optimizer_states: int = 2
    layers: list[LayerCost] = field(default_factory=list)
    device: Device | None = None
    # structured designs (classic ML, PGMs, PyMC, GPs; core.budget_structured)
    family: str = "neural"
    value_bytes: int = 4            # bytes per stored value (float64 outside networks)
    work_values: int = 0            # training workspace (data copies, kernel matrices, draws)
    efficiency: float = COMPUTE_EFFICIENCY
    assumed_data: str = ""          # why the size is a guess (no dataset size)

    # ---------------------------------------------------------- memory
    @property
    def bytes_per_value(self) -> int:
        return 2 if self.mixed_precision else 4

    def model_bytes(self) -> int:
        return self.params * self.value_bytes

    def train_memory_bytes(self, batch_size: int | None = None,
                           mixed_precision: bool | None = None) -> int:
        batch = self.batch_size if batch_size is None else batch_size
        amp = self.mixed_precision if mixed_precision is None else mixed_precision
        if self.family != "neural":
            overhead = (self.device.overhead_gb if self.device else 0.0) * GB
            return int(self.model_bytes() + self.work_values * 8 + overhead)
        weights = self.params * 4 + (self.params * 2 if amp else 0)  # fp16 copy under AMP
        grads = self.trainable_params * 4
        states = self.trainable_params * 4 * self.optimizer_states
        saved = self.activations
        if self.checkpointing:   # one layer recomputed at a time, plus its neighbours' inputs
            saved = min(saved, 2 * max((layer.activations for layer in self.layers), default=0))
        acts = saved * batch * (2 if amp else 4) * SAVED_ACTIVATION_FACTOR
        overhead = (self.device.overhead_gb if self.device else 0.0) * GB
        return int(weights + grads + states + acts + overhead)

    def inference_memory_bytes(self) -> int:
        peak = max((layer.activations for layer in self.layers), default=0)
        return int(self.model_bytes() + 2 * peak * 4)

    # ---------------------------------------------------------- time
    def latency_ms(self, device: Device | None = None) -> float | None:
        device = device or self.device
        if device is None:
            return None
        tflops = device.tflops_fp16 if self.mixed_precision else device.tflops_fp32
        compute = self.flops / (tflops * 1e12 * self.efficiency)
        traffic = (self.model_bytes() + 2 * self.activations * 4) / (device.bandwidth_gbs * 1e9)
        launch = LAYER_LAUNCH_MS.get(device.kind, 0.01) * self.kernels / 1e3
        return (compute + traffic + launch) * 1e3

    @property
    def kernels(self) -> int:
        return sum(layer.kernels for layer in self.layers)

    def step_time_ms(self, device: Device | None = None) -> float | None:
        """One training step (forward + backward ≈ 3× forward; 4× when checkpointing
        recomputes the forward) at the batch size."""
        device = device or self.device
        if device is None:
            return None
        tflops = device.tflops_fp16 if self.mixed_precision else device.tflops_fp32
        return (4 if self.checkpointing else 3) * self.flops * self.batch_size / (tflops * 1e12 * COMPUTE_EFFICIENCY) * 1e3

    def fitting_batch(self, limit_bytes: float, mixed_precision: bool | None = None) -> int:
        """Largest power-of-two batch whose training memory fits ``limit_bytes`` (0: none)."""
        batch = 1
        if self.train_memory_bytes(1, mixed_precision) > limit_bytes:
            return 0
        while batch < self.batch_size and \
                self.train_memory_bytes(batch * 2, mixed_precision) <= limit_bytes:
            batch *= 2
        return batch

    def to_dict(self) -> dict:
        device = self.device
        return {
            "params": self.params, "trainable_params": self.trainable_params,
            "flops": self.flops, "activations": self.activations,
            "batch_size": self.batch_size, "mixed_precision": self.mixed_precision,
            "accumulation_steps": self.accumulation_steps,
            "model_mb": round(self.model_bytes() / 2**20, 2),
            "train_memory_gb": round(self.train_memory_bytes() / GB, 3),
            "inference_memory_mb": round(self.inference_memory_bytes() / 2**20, 2),
            "latency_ms": _round(self.latency_ms()), "step_time_ms": _round(self.step_time_ms()),
            "device": device.to_dict() if device else None,
            "layers": [asdict(layer) for layer in self.layers],
            "family": self.family, "assumed_data": self.assumed_data,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _rows(shape: list[int]) -> int:
    """Positions a per-feature (last-dim) weight is applied to."""
    return shape_volume(shape[:-1]) if len(shape) > 1 else 1


def _spatial(shape: list[int]) -> int:
    """Positions a per-channel (channel-first) kernel is applied to."""
    return shape_volume(shape[1:]) if len(shape) > 1 else 1


def _backbone_cost(params: dict, in_shape: list[int]) -> tuple[int, int, int]:
    from ai_made_easy.core.zoo import entry

    row = entry("torchvision_image", params["architecture"]) or {}
    count = int(float(row.get("params_m") or 0) * 1e6)
    gmacs, side = row.get("gmacs"), row.get("gmacs_side") or 224
    flops = 0
    if gmacs and len(in_shape) == 3:  # 1 multiply-accumulate = 2 FLOPs
        flops = int(2 * float(gmacs) * 1e9 * (in_shape[1] * in_shape[2]) / (side * side))
    return count, flops, int(flops * BACKBONE_ACTIVATIONS_PER_FLOP)


def layer_flops(type_id: str, params: int, in_shapes: list[list[int]],
                out_shape: list[int]) -> int:
    """Forward FLOPs of one block (per sample)."""
    out_volume = shape_volume(out_shape) if out_shape else 0
    name = type_id.split(".", 1)[-1]
    if not params:
        return out_volume
    if "conv" in name or name in ("squeeze_excite",):
        return 2 * params * _spatial(out_shape)
    if name in ("lstm", "gru") or "rnn" in name:
        steps = in_shapes[0][0] if in_shapes and len(in_shapes[0]) >= 2 else 1
        return 2 * params * steps
    if "attention" in name or name.startswith("transformer"):
        s = in_shapes[0] if in_shapes else out_shape
        steps, width = (s[0], s[-1]) if len(s) >= 2 else (1, s[-1] if s else 1)
        return 2 * params * steps + 4 * steps * steps * width
    if name in ("embedding", "learned_positional"):
        return out_volume
    if "norm" in name:
        return 5 * out_volume
    return 2 * params * _rows(out_shape)


def estimate(graph, device: str | Device | None = None) -> Estimate:  # noqa: ANN001
    """Per-layer and total costs of a design under its trainer settings (classic ML and
    probabilistic designs: ``core.budget_structured``)."""
    from ai_made_easy.core import budget_structured

    if budget_structured.applies(graph):
        est = budget_structured.estimate(graph)
        if device is None:
            device = budget_of(graph)["device"] or None
        est.device = get_device(device) if isinstance(device, str) else device
        return est
    shapes = graph.infer_shapes()
    est = Estimate()
    for node in graph.model_nodes():
        defn = node.definition()
        resolved = node.resolved_params()
        in_shapes = [shapes[e.source_id] for e in
                     (graph.input_edge_for(node.instance_id, p.name) for p in defn.inputs)
                     if e is not None and e.source_id in shapes]
        out_shape = list(shapes.get(node.instance_id) or [])
        params = int(defn.param_fn(in_shapes, resolved)) if defn.param_fn else 0
        trainable = True
        cost = (defn.meta or {}).get("cost") if isinstance(defn.meta, dict) else None
        if node.type_id == "core.pretrained_backbone" and in_shapes:
            params, flops, acts = _backbone_cost(resolved, in_shapes[0])
            trainable = not resolved.get("freeze", False)
        elif callable(cost) and in_shapes:
            # composite blocks (detectors, segmenters, U-Nets) report their own cost
            params, flops = cost(in_shapes, resolved)
            acts = int(flops * BACKBONE_ACTIVATIONS_PER_FLOP)
            # a frozen pretrained encoder (timm, Hugging Face) neither learns nor keeps
            # activations for backward
            if "freeze" in resolved and resolved.get("weights", "pretrained") != "none":
                trainable = not resolved["freeze"]
        else:
            flops = layer_flops(node.type_id, params, in_shapes, out_shape)
            acts = _saved_values(graph, node, resolved, in_shapes, out_shape)
        kernels = (defn.meta or {}).get("kernels") if isinstance(defn.meta, dict) else None
        kernels = int(kernels(resolved)) if callable(kernels) and in_shapes else 1
        est.layers.append(LayerCost(node.instance_id, defn.display_name, node.type_id,
                                    out_shape, params, flops, acts, trainable, kernels))
        est.params += params
        est.trainable_params += params if trainable else 0
        est.flops += flops
        # frozen layers keep no activations for backward
        est.activations += acts if trainable or node.type_id == "core.input" else 0
    _apply_training_settings(graph, est)
    if device is None:
        device = budget_of(graph)["device"] or None
    if isinstance(device, str):
        device = get_device(device)
    est.device = device
    return est


def _saved_values(graph, node, resolved: dict, in_shapes: list[list[int]],  # noqa: ANN001
                  out_shape: list[int]) -> int:
    """Values of this block's output that stay alive for the backward pass."""
    if not out_shape or node.type_id in VIEW_OPS:
        return 0
    consumers = [graph.nodes[e.target_id].type_id for e in graph.outgoing(node.instance_id)
                 if e.target_id in graph.nodes]
    if consumers and all(t in OUTPUT_SAVING for t in consumers):
        return 0
    if node.type_id in RECURRENT_STATES and in_shapes and len(in_shapes[0]) >= 2:
        hidden = int(resolved.get("hidden_size") or 0)
        dirs = 2 if resolved.get("bidirectional") else 1
        layers = int(resolved.get("num_layers") or 1)
        return in_shapes[0][0] * hidden * dirs * layers * RECURRENT_STATES[node.type_id]
    return shape_volume(out_shape)


def _apply_training_settings(graph, est: Estimate) -> None:  # noqa: ANN001
    from ai_made_easy.core.training import catalog as cat

    for node in graph.nodes.values():
        if node.type_id == "train.trainer":
            p = node.resolved_params()
            est.batch_size = int(p.get("batch_size") or 1)
            est.mixed_precision = bool(p.get("mixed_precision"))
            est.accumulation_steps = int(p.get("accumulation_steps") or 1)
            est.checkpointing = bool(p.get("gradient_checkpointing"))
        elif node.type_id == "prep.dataloader":
            batch = int(node.resolved_params().get("batch_size") or 0)
            if batch:
                est.batch_size = batch
        elif node.type_id in cat.OPTIMIZER_IDS:
            states = OPTIMIZER_STATES.get(node.type_id, 2)
            if node.type_id == "train.sgd" and not float(
                    node.resolved_params().get("momentum") or 0):
                states = 0
            est.optimizer_states = states


# ------------------------------------------------------------------ checks

@dataclass
class BudgetCheck:
    kind: str          # train_memory, latency, params, model_size
    used: float
    limit: float
    unit: str

    @property
    def over(self) -> bool:
        return self.limit > 0 and self.used > self.limit


def check(graph, est: Estimate | None = None) -> list[BudgetCheck]:  # noqa: ANN001
    """Every budget the project sets, with the estimated usage."""
    budget = budget_of(graph)
    if not budget["device"] and not any(budget[k] for k in BUDGET_KEYS[1:]):
        return []
    est = est or estimate(graph)
    out: list[BudgetCheck] = []
    memory_limit = budget["max_train_memory_gb"] or (est.device.memory_gb if est.device else 0)
    if memory_limit:
        out.append(BudgetCheck("train_memory", est.train_memory_bytes() / GB, memory_limit, "GB"))
    if budget["max_latency_ms"] and est.device:
        out.append(BudgetCheck("latency", est.latency_ms() or 0.0, budget["max_latency_ms"],
                               "ms"))
    if budget["max_params_m"]:
        out.append(BudgetCheck("params", est.params / 1e6, budget["max_params_m"], "M"))
    if budget["max_model_mb"]:
        out.append(BudgetCheck("model_size", est.model_bytes() / 2**20, budget["max_model_mb"],
                               "MB"))
    return out


def budget_report(graph) -> dict:  # noqa: ANN001
    """Estimate + budget checks as JSON (API, web, CLI)."""
    est = estimate(graph)
    return {"budget": budget_of(graph), "estimate": est.to_dict(),
            "checks": [{**asdict(c), "over": c.over} for c in check(graph, est)]}


def human_bytes(n: float) -> str:
    for unit, size in (("GB", GB), ("MB", 2**20), ("KB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{int(n)} B"


def human_flops(n: float) -> str:
    for unit, size in (("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{int(n)} FLOPs"
