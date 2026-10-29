"""More layers (3.12): the RWKV-7 stack.

RWKV-7 ("Goose", Peng et al., 2025) is the RWKV that llama.cpp reads, so a language
model built from it exports to GGUF. Its layers pass the first layer's values on to the
others (a value residual), so it is one block holding the whole stack rather than one
block per layer. Its math follows llama.cpp's graph, step for step, and its weights are
named after llama.cpp's tensors. Checked by the codegen harness and against PyTorch's
parameter count; the Keras form computes the same function (tests/test_v312_features.py).
"""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

reg = get_registry()
_BEFORE = {b.type_id for b in reg.all()}

SEQUENCE = "Sequence Models"


def lora_ranks(dim: int) -> tuple[int, int, int, int]:
    """RWKV-7's low-rank sizes for the decay, the in-context learning rate, the value
    residual and the gate, as its reference code sets them."""
    def rank(factor: float, power: float) -> int:
        return max(32, int(round(factor * dim ** power / 32)) * 32)
    return rank(1.8, 0.5), rank(1.8, 0.5), rank(1.3, 0.5), rank(0.6, 0.8)


TORCH = {
    "RWKV7": '''\
class RWKV7Layer(nn.Module):
    """One RWKV-7 layer: time mixing (a state per head updated by a generalised delta
    rule, with a decay and an in-context learning rate per channel) and channel mixing,
    each after a layer norm, with token shift and residuals."""

    def __init__(self, dim: int, index: int, count: int, head_size: int, hidden: int,
                 ranks: tuple) -> None:
        super().__init__()
        c, self.heads, self.head_size = dim, dim // head_size, head_size
        dw, da, dv, dg = ranks
        r0, r1 = index / max(count - 1, 1), 1.0 - index / count
        ddd = torch.arange(c) / c

        def small(rows: int, cols: int) -> nn.Parameter:
            return nn.Parameter(nn.init.orthogonal_(torch.empty(rows, cols), gain=0.1))

        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        # token shift for r, w, k, v, a and g
        self.lerp = nn.Parameter(torch.stack([1 - ddd ** (e * r1)
                                              for e in (0.2, 0.9, 0.7, 0.7, 0.9, 0.2)]))
        speed = -7 + 5 * (torch.arange(c) / max(c - 1, 1)) ** (0.85 + r0 ** 0.5)
        self.w0, self.w1, self.w2 = (nn.Parameter(speed + 0.5),
                                     nn.Parameter(torch.zeros(dw, c)), small(c, dw))
        self.a0, self.a1, self.a2 = (nn.Parameter(torch.zeros(c)),
                                     nn.Parameter(torch.zeros(da, c)), small(c, da))
        self.v0, self.v1, self.v2 = (nn.Parameter(torch.ones(c)),
                                     nn.Parameter(torch.zeros(dv, c)), small(c, dv))
        self.g1, self.g2 = nn.Parameter(torch.zeros(dg, c)), small(c, dg)
        self.k_k = nn.Parameter(torch.full((c,), 0.85))
        self.k_a = nn.Parameter(torch.ones(c))
        self.r_k = nn.Parameter(torch.zeros(c))
        self.receptance, self.key, self.value, self.output = (
            nn.Linear(c, c, bias=False) for _ in range(4))
        nn.init.zeros_(self.output.weight)
        self.ln_x = nn.GroupNorm(self.heads, c, eps=64e-5)
        nn.init.constant_(self.ln_x.weight, ((1 + index) / count) ** 0.7)
        self.cmix_k = nn.Parameter(1 - ddd ** (r1 ** 4))
        self.ckey = nn.Linear(c, hidden, bias=False)
        self.cvalue = nn.Linear(hidden, c, bias=False)
        nn.init.zeros_(self.cvalue.weight)

    @staticmethod
    def shift(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.new_zeros(x.shape[0], 1, x.shape[2]), x[:, :-1]], 1)

    def wkv(self, r, w, k, v, a, b):  # noqa: ANN001, ANN201
        """The state [B, H, value, key]: state·diag(w) + vᵀk + (state·a)ᵀb, read by r."""
        batch, length, c = r.shape
        h, n = self.heads, self.head_size
        r, w, k, v, a, b = (z.float().reshape(batch, length, h, n) for z in (r, w, k, v, a, b))
        state = r.new_zeros(batch, h, n, n)
        out = []
        for t in range(length):
            sa = state @ a[:, t, :, :, None]
            state = (state * w[:, t, :, None, :] + v[:, t, :, :, None] * k[:, t, :, None, :]
                     + sa * b[:, t, :, None, :])
            out.append((state @ r[:, t, :, :, None])[..., 0])
        return torch.stack(out, 1).reshape(batch, length, c)

    def forward(self, x: torch.Tensor, first):  # noqa: ANN001
        batch, length, c = x.shape
        h, n = self.heads, self.head_size
        y = self.ln1(x)
        xr, xw, xk, xv, xa, xg = y + (self.shift(y) - y) * self.lerp[:, None, None, :]
        r = self.receptance(xr)
        w = torch.exp(-0.606531 * torch.sigmoid(
            self.w0 + torch.tanh(xw @ self.w1.t()) @ self.w2.t()))
        k, v = self.key(xk), self.value(xv)
        if first is None:
            first = v
        else:   # the first layer's values, mixed in
            v = v + (first - v) * torch.sigmoid(self.v0 + (xv @ self.v1.t()) @ self.v2.t())
        a = torch.sigmoid(self.a0 + (xa @ self.a1.t()) @ self.a2.t())
        g = torch.sigmoid(xg @ self.g1.t()) @ self.g2.t()
        kk = nn.functional.normalize((k * self.k_k).reshape(batch, length, h, n), dim=-1,
                                     eps=1e-12).reshape(batch, length, c)
        k = k * (1 + (a - 1) * self.k_a)
        out = self.wkv(r, w, k, v, -kk, kk * a).to(x.dtype)
        # a trailing axis of 1: the same group norm, in a shape OpenVINO can run
        out = self.ln_x(out.reshape(batch * length, c, 1)).reshape(batch, length, c)
        bonus = (r * k * self.r_k).reshape(batch, length, h, n).sum(-1, keepdim=True)
        out = out + (bonus * v.reshape(batch, length, h, n)).reshape(batch, length, c)
        x = x + self.output(out * g)
        y = self.ln2(x)
        y = y + (self.shift(y) - y) * self.cmix_k
        return x + self.cvalue(torch.relu(self.ckey(y)) ** 2), first


class RWKV7(nn.Module):
    """RWKV-7 (Peng et al., 2025): a layer norm, then ``layers`` RWKV-7 layers that share
    the first layer's values. [B, L, D] → [B, L, D]; causal."""

    def __init__(self, dim: int, layers: int = 2, head_size: int = 32, hidden: int = 0,
                 ranks: tuple = (32, 32, 32, 32)) -> None:
        super().__init__()
        self.head_size = head_size
        self.ln0 = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            RWKV7Layer(dim, i, layers, head_size, hidden or 4 * dim, ranks)
            for i in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, first = self.ln0(x), None
        for layer in self.layers:
            x, first = layer(x, first)
        return x
''',
}

KERAS = {
    "RWKV7": '''\
class RWKV7(TorchLayout):
    """RWKV-7: a layer norm, then RWKV-7 layers that share the first layer's values
    (weights in PyTorch's layout and names)."""

    def __init__(self, layers: int = 2, head_size: int = 32, hidden: int = 0,
                 ranks=(32, 32, 32, 32), **kwargs):
        super().__init__(**kwargs)
        self.count, self.head_size, self.hidden, self.ranks = (layers, head_size, hidden,
                                                               tuple(ranks))

    def build(self, input_shape):
        c = input_shape[-1]
        hidden = self.hidden or 4 * c
        dw, da, dv, dg = self.ranks
        self.ln0 = (self.vector("ln0_weight", (c,), "ones"), self.vector("ln0_bias", (c,), "zeros"))
        ddd = ops.arange(c, dtype="float32") / c
        small = keras.initializers.Orthogonal(gain=0.1)
        self.blocks = []
        for i in range(self.count):
            p = f"layers_{i}_"
            r0, r1 = i / max(self.count - 1, 1), 1.0 - i / self.count
            speed = -7 + 5 * (ops.arange(c, dtype="float32") / max(c - 1, 1)) ** (0.85 + r0 ** 0.5)
            layer = {
                "ln1": (self.vector(p + "ln1_weight", (c,), "ones"),
                        self.vector(p + "ln1_bias", (c,), "zeros")),
                "ln2": (self.vector(p + "ln2_weight", (c,), "ones"),
                        self.vector(p + "ln2_bias", (c,), "zeros")),
                "lerp": self.vector(p + "lerp", (6, c), ops.stack(
                    [1 - ddd ** (e * r1) for e in (0.2, 0.9, 0.7, 0.7, 0.9, 0.2)])),
                "w0": self.vector(p + "w0", (c,), speed + 0.5),
                "w1": self.vector(p + "w1", (dw, c), "zeros"),
                "w2": self.vector(p + "w2", (c, dw), small),
                "a0": self.vector(p + "a0", (c,), "zeros"),
                "a1": self.vector(p + "a1", (da, c), "zeros"),
                "a2": self.vector(p + "a2", (c, da), small),
                "v0": self.vector(p + "v0", (c,), "ones"),
                "v1": self.vector(p + "v1", (dv, c), "zeros"),
                "v2": self.vector(p + "v2", (c, dv), small),
                "g1": self.vector(p + "g1", (dg, c), "zeros"),
                "g2": self.vector(p + "g2", (c, dg), small),
                "k_k": self.vector(p + "k_k", (c,), 0.85),
                "k_a": self.vector(p + "k_a", (c,), 1.0),
                "r_k": self.vector(p + "r_k", (c,), "zeros"),
                "receptance": self.linear(p + "receptance", c, c, bias=False),
                "key": self.linear(p + "key", c, c, bias=False),
                "value": self.linear(p + "value", c, c, bias=False),
                "output": (self.vector(p + "output_weight", (c, c), "zeros"), None),
                "ln_x": (self.vector(p + "ln_x_weight", (c,), ((1 + i) / self.count) ** 0.7),
                         self.vector(p + "ln_x_bias", (c,), "zeros")),
                "cmix_k": self.vector(p + "cmix_k", (c,), 1 - ddd ** (r1 ** 4)),
                "ckey": self.linear(p + "ckey", c, hidden, bias=False),
                "cvalue": (self.vector(p + "cvalue_weight", (c, hidden), "zeros"), None),
            }
            self.blocks.append(layer)

    @staticmethod
    def shift(x):
        return ops.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1]

    def wkv(self, r, w, k, v, a, b):
        batch, length, c = ops.shape(r)[0], r.shape[1], r.shape[2]
        h, n = c // self.head_size, self.head_size
        r, w, k, v, a, b = (ops.reshape(z, (-1, length, h, n)) for z in (r, w, k, v, a, b))
        state = ops.zeros_like(r[:, 0])[..., None] * ops.zeros((n,))
        out = []
        for t in range(length):
            sa = ops.matmul(state, a[:, t, :, :, None])
            state = (state * w[:, t, :, None, :] + v[:, t, :, :, None] * k[:, t, :, None, :]
                     + sa * b[:, t, :, None, :])
            out.append(ops.matmul(state, r[:, t, :, :, None])[..., 0])
        return ops.reshape(ops.stack(out, 1), (-1, length, c))

    def group_norm(self, x, weight, bias):
        length, c = x.shape[1], x.shape[2]
        y = ops.reshape(x, (-1, length, c // self.head_size, self.head_size))
        mean = ops.mean(y, axis=-1, keepdims=True)
        var = ops.mean(ops.square(y - mean), axis=-1, keepdims=True)
        y = (y - mean) * ops.rsqrt(var + 64e-5)
        return ops.reshape(y, (-1, length, c)) * weight + bias

    def call(self, x):
        length, c = x.shape[1], x.shape[2]
        h, n = c // self.head_size, self.head_size
        x, first = self.layer_norm(x, *self.ln0), None
        for p in self.blocks:
            y = self.layer_norm(x, *p["ln1"])
            sx = self.shift(y) - y
            xr, xw, xk, xv, xa, xg = (y + sx * p["lerp"][j] for j in range(6))
            r = self.dense(xr, p["receptance"])
            w = ops.exp(-0.606531 * ops.sigmoid(
                p["w0"] + ops.matmul(ops.tanh(ops.matmul(xw, ops.transpose(p["w1"]))),
                                     ops.transpose(p["w2"]))))
            k, v = self.dense(xk, p["key"]), self.dense(xv, p["value"])
            if first is None:
                first = v
            else:
                mix = p["v0"] + ops.matmul(ops.matmul(xv, ops.transpose(p["v1"])),
                                           ops.transpose(p["v2"]))
                v = v + (first - v) * ops.sigmoid(mix)
            a = ops.sigmoid(p["a0"] + ops.matmul(ops.matmul(xa, ops.transpose(p["a1"])),
                                                 ops.transpose(p["a2"])))
            g = ops.matmul(ops.sigmoid(ops.matmul(xg, ops.transpose(p["g1"]))),
                           ops.transpose(p["g2"]))
            kk = ops.reshape(k * p["k_k"], (-1, length, h, n))
            kk = kk / ops.maximum(ops.sqrt(ops.sum(ops.square(kk), axis=-1, keepdims=True)),
                                  1e-12)
            kk = ops.reshape(kk, (-1, length, c))
            k = k * (1 + (a - 1) * p["k_a"])
            out = self.group_norm(self.wkv(r, w, k, v, -kk, kk * a), *p["ln_x"])
            bonus = ops.sum(ops.reshape(r * k * p["r_k"], (-1, length, h, n)), axis=-1,
                            keepdims=True)
            out = out + ops.reshape(bonus * ops.reshape(v, (-1, length, h, n)), (-1, length, c))
            x = x + self.dense(out * g, p["output"])
            y = self.layer_norm(x, *p["ln2"])
            y = y + (self.shift(y) - y) * p["cmix_k"]
            x = x + self.dense(ops.square(ops.relu(self.dense(y, p["ckey"]))), p["cvalue"])
        return x
''',
}

PYTORCH_HELPERS.update(TORCH)
KERAS_HELPERS.update(KERAS)


def _hidden(d: int, p) -> int:
    return int(p["hidden"]) or 4 * d


def _rwkv7_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"RWKV-7 expects a batch-first sequence [L, D], got {s}")
    size = int(p["head_size"])
    if size < 1 or s[1] % size:
        raise ShapeError(f"RWKV-7: the width D={s[1]} is not divisible by the head size "
                         f"{size}")
    return list(s)


def rwkv7_params(d: int, layers: int, hidden: int) -> int:
    ranks = sum(lora_ranks(d))
    return 2 * d + layers * (19 * d + 2 * d * ranks + 4 * d * d + 2 * d * hidden)


reg.register(nn_block(
    "seq.rwkv7", "RWKV-7 Stack", SEQUENCE, family="recurrent",
    params=(P("layers", "int", 2, lo=1, hi=64, help="RWKV-7 layers in the stack"),
            P("head_size", "int", 32, lo=8, hi=256, help="Channels per head; must divide D"),
            P("hidden", "int", 0, lo=0, help="Channel-mixing width; 0 = 4 × D")),
    shape=_rwkv7_shape, layout="ir",
    param_fn=lambda s, p: rwkv7_params(s[0][-1], int(p["layers"]), _hidden(s[0][-1], p)),
    torch=lambda c: (f"RWKV7({c['input_size']}, {int(c['layers'])}, {int(c['head_size'])}, "
                     f"{_hidden(c['input_size'], c)}, {lora_ranks(c['input_size'])})"),
    torch_helpers=("RWKV7",),
    keras_expr=lambda c: (f"RWKV7({int(c['layers'])}, {int(c['head_size'])}, "
                          f"{_hidden(c['input_size'], c)}, {lora_ranks(c['input_size'])})"
                          f"({c['i0']})"),
    keras_helpers=("TorchLayout", "RWKV7"),
    desc="A stack of RWKV-7 layers (Peng et al., 2025): a recurrent network that trains "
         "like a transformer, keeping one state per head that it updates with a decay and "
         "an in-context learning rate. Put it after an Embedding, then a LayerNorm and a "
         "Dense without bias, for a language model that exports to GGUF. Causal "
         "([L, D] → [L, D]).",
))


LIBRARY312_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
