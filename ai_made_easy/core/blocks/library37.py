"""More layers (3.7): attention variants and recurrent-style sequence mixers from recent
papers, all-MLP mixers, and efficient convolution blocks.

Sliding-window, ALiBi and differential attention; RetNet retention, an RWKV block, a
Hyena operator, Perceiver latents and Linformer attention; MLP-Mixer, gMLP and
PoolFormer; Ghost, Involution, deformable convolution, Fused-MBConv, ECA and GRN; and
Dynamic Tanh. Every one is checked by the codegen harness (designed shape = real shape)
and against PyTorch's parameter count; those with a Keras form are checked in Keras too.
Helpers use ``nn.functional`` (not ``F``) because the generated file imports ``F`` only
when a layer line uses it.
"""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

reg = get_registry()
_BEFORE = {b.type_id for b in reg.all()}

TRANSFORMER = "Transformer Parts"
SEQUENCE = "Sequence Models"
VISION = "Vision Blocks"

TORCH = {
    "LocalAttention": '''\
class LocalAttention(nn.Module):
    """Multi-head self-attention that can be limited to a window of nearby positions
    (Longformer / Mistral) and can add ALiBi's linear distance penalty (Press et al.,
    2022) instead of position embeddings."""

    def __init__(self, dim: int, heads: int = 4, window: int = 0, causal: bool = True,
                 alibi: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads, self.window, self.causal, self.alibi = heads, window, causal, alibi
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.register_buffer("slopes", torch.tensor(
            [2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        i = torch.arange(n, device=x.device)
        gap = i[:, None] - i[None, :]
        allowed = gap >= 0 if self.causal else torch.ones_like(gap, dtype=torch.bool)
        if self.window > 0:
            allowed = allowed & (gap.abs() < self.window)
        mask = torch.zeros(n, n, device=x.device, dtype=q.dtype).masked_fill(~allowed,
                                                                            float("-inf"))
        if self.alibi:
            mask = mask - self.slopes.to(q.dtype)[:, None, None] * gap.abs().to(q.dtype)
        h = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(h.transpose(1, 2).reshape(b, n, d))
''',
    "DiffAttention": '''\
class DiffAttention(nn.Module):
    """Differential attention (Ye et al., 2024): the difference of two softmax maps, so
    attention noise common to both cancels out; lambda is learned, each head is
    RMS-normalized."""

    def __init__(self, dim: int, heads: int = 4, causal: bool = True,
                 lambda_init: float = 0.8) -> None:
        super().__init__()
        self.heads, self.causal, self.lambda_init = heads, causal, lambda_init
        hd = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.lambdas = nn.Parameter(torch.randn(4, hd // 2) * 0.1)
        self.norm = nn.Parameter(torch.ones(hd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, hd = self.heads, d // self.heads
        q, k, v = self.qkv(x).reshape(b, n, 3, h, hd).permute(2, 0, 3, 1, 4)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        scale = (hd // 2) ** -0.5
        s1, s2 = q1 @ k1.transpose(-1, -2) * scale, q2 @ k2.transpose(-1, -2) * scale
        if self.causal:
            future = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            s1, s2 = s1.masked_fill(future, float("-inf")), s2.masked_fill(future, float("-inf"))
        # exp(lq1.lk1) - exp(lq2.lk2) + init, kept as tensors of one element: Core ML
        # folds arithmetic on 0-d constants into Python floats it then rejects
        e = torch.exp((self.lambdas[0::2] * self.lambdas[1::2]).sum(-1))
        lam = e[0:1] - e[1:2] + self.lambda_init
        att = s1.softmax(-1) - lam * s2.softmax(-1)
        o = att @ v
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm
        o = o * (1 - self.lambda_init)
        return self.out(o.transpose(1, 2).reshape(b, n, d))
''',
    "Retention": '''\
class Retention(nn.Module):
    """RetNet's multi-scale retention (Sun et al., 2023), parallel form: attention
    without softmax, each head decaying earlier positions at its own rate, then a group
    norm and a swish gate."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.gate = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.GroupNorm(heads, dim)
        self.register_buffer("gamma", 1 - torch.exp2(-5.0 - torch.arange(heads).float()),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, hd = self.heads, d // self.heads
        q, k, v = (f(x).reshape(b, n, h, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        i = torch.arange(n, device=x.device, dtype=x.dtype)
        gap = i[:, None] - i[None, :]
        decay = torch.where(gap >= 0, self.gamma.to(x.dtype)[:, None, None] ** gap.clamp(min=0),
                            torch.zeros((), device=x.device, dtype=x.dtype))
        y = (q @ k.transpose(-1, -2) * hd ** -0.5 * decay) @ v
        # a trailing axis of 1: the same group norm, in a shape OpenVINO can run
        y = self.norm(y.transpose(1, 2).reshape(b * n, d, 1)).reshape(b, n, d)
        return self.out(nn.functional.silu(self.gate(x)) * y)
''',
    "RWKVBlock": '''\
class RWKVBlock(nn.Module):
    """One RWKV-4 layer (Peng et al., 2023): time mixing (a linear-attention recurrence
    with a learned decay per channel) and channel mixing, each after a layer norm, with
    token shift and residuals. Runs as a recurrence over the sequence."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mix_k, self.mix_v, self.mix_r = (nn.Parameter(torch.full((dim,), 0.5))
                                              for _ in range(3))
        self.decay = nn.Parameter(torch.linspace(-5, 1, dim))
        self.first = nn.Parameter(torch.full((dim,), 0.3))
        self.key, self.value, self.receptance, self.output = (
            nn.Linear(dim, dim, bias=False) for _ in range(4))
        self.cmix_k, self.cmix_r = nn.Parameter(torch.full((dim,), 0.5)), nn.Parameter(
            torch.full((dim,), 0.5))
        self.ckey = nn.Linear(dim, hidden, bias=False)
        self.cvalue = nn.Linear(hidden, dim, bias=False)
        self.creceptance = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def shift(x: torch.Tensor) -> torch.Tensor:
        # (pad by -1 is the same, but Core ML can't convert a negative pad)
        return torch.cat([x.new_zeros(x.shape[0], 1, x.shape[2]), x[:, :-1]], 1)

    def wkv(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, n, d = k.shape
        w, u = -torch.exp(self.decay.float()), self.first.float()
        k, v = k.float(), v.float()
        num = torch.zeros(b, d, device=k.device)
        den = torch.zeros(b, d, device=k.device)
        top = torch.full((b, d), -1e38, device=k.device)
        out = []
        for t in range(n):
            kt, vt = k[:, t], v[:, t]
            ww = u + kt
            p = torch.maximum(top, ww)
            e1, e2 = torch.exp(top - p), torch.exp(ww - p)
            out.append((e1 * num + e2 * vt) / (e1 * den + e2))
            ww = top + w
            p = torch.maximum(ww, kt)
            e1, e2 = torch.exp(ww - p), torch.exp(kt - p)
            num, den, top = e1 * num + e2 * vt, e1 * den + e2, p
        return torch.stack(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        prev = self.shift(h)
        k = self.key(h * self.mix_k + prev * (1 - self.mix_k))
        v = self.value(h * self.mix_v + prev * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(h * self.mix_r + prev * (1 - self.mix_r)))
        x = x + self.output(r * self.wkv(k, v).to(x.dtype))
        h = self.ln2(x)
        prev = self.shift(h)
        k = torch.square(torch.relu(self.ckey(h * self.cmix_k + prev * (1 - self.cmix_k))))
        r = torch.sigmoid(self.creceptance(h * self.cmix_r + prev * (1 - self.cmix_r)))
        return x + r * self.cvalue(k)
''',
    "HyenaOperator": '''\
class HyenaOperator(nn.Module):
    """Hyena (Poli et al., 2023), order 2: a long convolution whose filter comes from a
    small network of the position (computed with FFTs, so it spans the whole sequence),
    between two data-controlled gates. Causal."""

    def __init__(self, dim: int, filter_hidden: int = 32) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.short = nn.Conv1d(3 * dim, 3 * dim, 3, padding=2, groups=3 * dim)
        self.filter = nn.Sequential(nn.Linear(3, filter_hidden), nn.SiLU(),
                                    nn.Linear(filter_hidden, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        u = self.short(self.in_proj(x).transpose(1, 2))[..., :n]
        x1, x2, v = u.split(d, dim=1)
        t = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)[:, None]
        z = torch.cat([t, torch.sin(6.283 * t), torch.cos(6.283 * t)], dim=-1)
        h = (self.filter(z) * torch.exp(-4 * t)).t()               # [d, n]
        g = (v * x1).float()
        size = 2 * n
        y = torch.fft.irfft(torch.fft.rfft(g, n=size) * torch.fft.rfft(h.float(), n=size),
                            n=size)[..., :n].to(x.dtype)
        y = (y + g.to(x.dtype) * self.bias[:, None]) * x2
        return self.out(y.transpose(1, 2))
''',
    "PerceiverLatents": '''\
class PerceiverLatents(nn.Module):
    """Perceiver / Perceiver-Resampler cross-attention (Jaegle et al., 2021): a fixed set
    of learned latent vectors attends to an input of any length."""

    def __init__(self, dim: int, latents: int = 16, heads: int = 4) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latents, dim) * 0.02)
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.latents.expand(x.shape[0], -1, -1)
        kv = self.norm_kv(x)
        return q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
''',
    "LinformerAttention": '''\
class LinformerAttention(nn.Module):
    """Linformer self-attention (Wang et al., 2020): keys and values are projected from
    the sequence length down to k, so the cost grows linearly with the length."""

    def __init__(self, dim: int, seq_len: int, k: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj_k = nn.Linear(seq_len, k, bias=False)
        self.proj_v = nn.Linear(seq_len, k, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k = self.proj_k(k.transpose(1, 2)).transpose(1, 2)
        v = self.proj_v(v.transpose(1, 2)).transpose(1, 2)
        split = lambda t: t.reshape(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)
        h = nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
        return self.out(h.transpose(1, 2).reshape(b, n, d))
''',
    "MixerBlock": '''\
class MixerBlock(nn.Module):
    """MLP-Mixer block (Tolstikhin et al., 2021): an MLP across positions, then an MLP
    across features, each after a layer norm and with a residual."""

    def __init__(self, seq_len: int, dim: int, token_hidden: int, channel_hidden: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.tokens = nn.Sequential(nn.Linear(seq_len, token_hidden), nn.GELU(),
                                    nn.Linear(token_hidden, seq_len))
        self.channels = nn.Sequential(nn.Linear(dim, channel_hidden), nn.GELU(),
                                      nn.Linear(channel_hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channels(self.norm2(x))
''',
    "GMLPBlock": '''\
class GMLPBlock(nn.Module):
    """gMLP block (Liu et al., 2021): a feed-forward whose hidden half is gated by a
    linear projection across positions (the spatial gating unit)."""

    def __init__(self, seq_len: int, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, hidden)
        self.gate_norm = nn.LayerNorm(hidden // 2)
        self.spatial = nn.Linear(seq_len, seq_len)
        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.spatial.bias)
        self.down = nn.Linear(hidden // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = nn.functional.gelu(self.up(self.norm(x))).chunk(2, dim=-1)
        v = self.spatial(self.gate_norm(v).transpose(1, 2)).transpose(1, 2)
        return x + self.down(u * v)
''',
    "PoolFormerBlock": '''\
class PoolFormerBlock(nn.Module):
    """PoolFormer block (Yu et al., 2022, MetaFormer): average pooling as the token
    mixer, then a 1x1-convolution MLP, each after a group norm and with a residual."""

    def __init__(self, channels: int, pool_size: int = 3, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.GroupNorm(1, channels), nn.GroupNorm(1, channels)
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size // 2, count_include_pad=False)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.pool(h) - h
        return x + self.mlp(self.norm2(x))
''',
    "GhostModule": '''\
class GhostModule(nn.Module):
    """GhostNet's module (Han et al., 2020): a few channels from an ordinary 1x1
    convolution, the rest from cheap depthwise 3x3 convolutions of those."""

    def __init__(self, in_channels: int, out_channels: int, ratio: int = 2) -> None:
        super().__init__()
        self.out_channels = out_channels
        init = -(-out_channels // ratio)
        self.primary = nn.Sequential(nn.Conv2d(in_channels, init, 1, bias=False),
                                     nn.BatchNorm2d(init), nn.ReLU(inplace=True))
        self.cheap = nn.Sequential(nn.Conv2d(init, init * (ratio - 1), 3, padding=1,
                                             groups=init, bias=False),
                                   nn.BatchNorm2d(init * (ratio - 1)), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.primary(x)
        return torch.cat([h, self.cheap(h)], dim=1)[:, :self.out_channels]
''',
    "Involution": '''\
class Involution(nn.Module):
    """Involution (Li et al., 2021): the inverse of a convolution. Its kxk kernel is
    generated at every pixel from that pixel's features, and shared across the channels
    of a group."""

    def __init__(self, channels: int, kernel_size: int = 7, groups: int = 4,
                 reduction: int = 4) -> None:
        super().__init__()
        self.k, self.groups = kernel_size, groups
        mid = max(channels // reduction, 1)
        self.reduce = nn.Sequential(nn.Conv2d(channels, mid, 1, bias=False),
                                    nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.span = nn.Conv2d(mid, kernel_size * kernel_size * groups, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g, k = self.groups, self.k
        kernel = self.span(self.reduce(x)).reshape(b, g, 1, k * k, h, w)
        patches = nn.functional.unfold(x, k, padding=k // 2).reshape(b, g, c // g, k * k, h, w)
        return (kernel * patches).sum(3).reshape(b, c, h, w)
''',
    "DeformConv2d": '''\
class DeformConv2d(nn.Module):
    """Deformable convolution v1 (Dai et al., 2017): a small convolution predicts where
    each kernel tap samples (an offset per tap and pixel), so the kernel follows the
    object's shape. torchvision.ops.DeformConv2d; the offsets start at zero."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        from torchvision.ops import DeformConv2d as _Deform

        self.offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size,
                                stride, padding)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.conv = _Deform(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x, self.offset(x))
''',
    "FusedMBConv": '''\
class FusedMBConv(nn.Module):
    """EfficientNetV2's Fused-MBConv (Tan & Le, 2021): MBConv with the 1x1 expansion and
    the depthwise convolution fused into one regular kxk convolution; faster on GPUs in
    early stages."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4,
                 kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        mid = in_channels * expand_ratio
        pad = kernel_size // 2
        if expand_ratio == 1:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(out_channels), nn.SiLU())
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, mid, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(mid), nn.SiLU(),
                nn.Conv2d(mid, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return x + h if self.residual else h
''',
    "ECA": '''\
class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020): channel weights from a 1-D
    convolution across the pooled channels, with no dimensionality reduction."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.mean((2, 3))[:, None, :])
        return x * torch.sigmoid(y)[:, 0, :, None, None]
''',
    "GRN": '''\
class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2, Woo et al., 2023): each channel is
    scaled by its share of the total response, which keeps channels from collapsing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x
''',
    "DyT": '''\
class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al., 2025): weight * tanh(alpha * x) + bias on the last axis,
    a drop-in for LayerNorm in transformers without computing any statistics."""

    def __init__(self, features: int, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.tanh(self.alpha * x) + self.bias
''',
}

KERAS = {
    "local_attention": '''\
def local_attention(x, heads: int, window: int = 0, causal: bool = True,
                    alibi: bool = False, dropout: float = 0.0):
    """Multi-head self-attention, optionally windowed and with ALiBi distance bias."""
    n, d = x.shape[1], x.shape[2]
    hd = d // heads
    qkv = ops.transpose(ops.reshape(layers.Dense(3 * d)(x), (-1, n, 3, heads, hd)),
                        (2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    s = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / hd ** 0.5
    i = ops.arange(n)
    gap = ops.expand_dims(i, 1) - ops.expand_dims(i, 0)
    allowed = gap >= 0 if causal else ops.ones((n, n), dtype="bool")
    if window > 0:
        allowed = ops.logical_and(allowed, ops.abs(gap) < window)
    if alibi:
        slopes = ops.convert_to_tensor([2.0 ** (-8.0 * (h + 1) / heads) for h in range(heads)])
        s = s - ops.reshape(slopes, (heads, 1, 1)) * ops.cast(ops.abs(gap), s.dtype)
    s = ops.where(allowed, s, -1e9)
    a = layers.Dropout(dropout)(ops.softmax(s, axis=-1))
    h = ops.reshape(ops.transpose(ops.matmul(a, v), (0, 2, 1, 3)), (-1, n, d))
    return layers.Dense(d)(h)
''',
    "PerceiverLatents": '''\
class PerceiverLatents(layers.Layer):
    """A set of learned latents cross-attends to the input."""

    def __init__(self, latents: int, heads: int, **kwargs):
        super().__init__(**kwargs)
        self.n, self.heads = latents, heads

    def build(self, input_shape):
        d = input_shape[-1]
        self.latents = self.add_weight(shape=(self.n, d), initializer="random_normal")
        self.norm_q = layers.LayerNormalization(epsilon=1e-5)
        self.norm_kv = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(self.heads, d // self.heads)

    def call(self, x):
        q = ops.zeros_like(x[:, :1, :1]) + self.latents[None]
        kv = self.norm_kv(x)
        return q + self.attn(self.norm_q(q), kv)
''',
    "mixer_block": '''\
def mixer_block(x, token_hidden: int, channel_hidden: int):
    """MLP-Mixer block: an MLP across positions, then across features."""
    n, d = x.shape[1], x.shape[2]
    h = layers.Permute((2, 1))(layers.LayerNormalization(epsilon=1e-5)(x))
    h = layers.Dense(n)(layers.Dense(token_hidden, activation="gelu")(h))
    x = x + layers.Permute((2, 1))(h)
    h = layers.Dense(channel_hidden, activation="gelu")(layers.LayerNormalization(epsilon=1e-5)(x))
    return x + layers.Dense(d)(h)
''',
    "gmlp_block": '''\
def gmlp_block(x, hidden: int):
    """gMLP block with a spatial gating unit."""
    n, d = x.shape[1], x.shape[2]
    h = layers.Dense(hidden, activation="gelu")(layers.LayerNormalization(epsilon=1e-5)(x))
    u, v = h[..., :hidden // 2], h[..., hidden // 2:]
    v = layers.Permute((2, 1))(layers.LayerNormalization(epsilon=1e-5)(v))
    v = layers.Permute((2, 1))(layers.Dense(n, kernel_initializer="zeros",
                                            bias_initializer="ones")(v))
    return x + layers.Dense(d)(u * v)
''',
    "poolformer_block": '''\
def poolformer_block(x, pool_size: int = 3, mlp_ratio: int = 4):
    """PoolFormer block (channels-last)."""
    c = x.shape[-1]
    h = layers.GroupNormalization(groups=1, epsilon=1e-5)(x)
    x = x + layers.AveragePooling2D(pool_size, strides=1, padding="same")(h) - h
    h = layers.GroupNormalization(groups=1, epsilon=1e-5)(x)
    h = layers.Conv2D(c * mlp_ratio, 1, activation="gelu")(h)
    return x + layers.Conv2D(c, 1)(h)
''',
    "ghost_module": '''\
def ghost_module(x, out_channels: int, ratio: int = 2):
    """GhostNet module (channels-last)."""
    init = -(-out_channels // ratio)
    h = layers.ReLU()(layers.BatchNormalization()(layers.Conv2D(init, 1, use_bias=False)(x)))
    cheap = layers.DepthwiseConv2D(3, padding="same", depth_multiplier=ratio - 1,
                                   use_bias=False)(h)
    cheap = layers.ReLU()(layers.BatchNormalization()(cheap))
    return layers.Concatenate()([h, cheap])[..., :out_channels]
''',
    "fused_mbconv": '''\
def fused_mbconv(x, out_channels: int, expand_ratio: int = 4, kernel_size: int = 3,
                 stride: int = 1):
    """Fused-MBConv (channels-last)."""
    c = x.shape[-1]
    if expand_ratio == 1:
        h = layers.Conv2D(out_channels, kernel_size, stride, padding="same", use_bias=False)(x)
        h = layers.Activation("silu")(layers.BatchNormalization()(h))
    else:
        h = layers.Conv2D(c * expand_ratio, kernel_size, stride, padding="same",
                          use_bias=False)(x)
        h = layers.Activation("silu")(layers.BatchNormalization()(h))
        h = layers.BatchNormalization()(layers.Conv2D(out_channels, 1, use_bias=False)(h))
    return x + h if stride == 1 and c == out_channels else h
''',
    "eca": '''\
def eca(x, kernel_size: int = 3):
    """Efficient Channel Attention (channels-last)."""
    c = x.shape[-1]
    y = ops.reshape(layers.GlobalAveragePooling2D()(x), (-1, c, 1))
    y = layers.Conv1D(1, kernel_size, padding="same", use_bias=False)(y)
    return x * ops.reshape(ops.sigmoid(y), (-1, 1, 1, c))
''',
    "GRN": '''\
class GRN(layers.Layer):
    """Global Response Normalization (channels-last)."""

    def build(self, input_shape):
        c = input_shape[-1]
        self.gamma = self.add_weight(shape=(c,), initializer="zeros")
        self.beta = self.add_weight(shape=(c,), initializer="zeros")

    def call(self, x):
        gx = ops.sqrt(ops.sum(ops.square(x), axis=(1, 2), keepdims=True))
        nx = gx / (ops.mean(gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x
''',
    "DyT": '''\
class DyT(layers.Layer):
    """Dynamic Tanh on the last axis."""

    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha0 = alpha

    def build(self, input_shape):
        f = input_shape[-1]
        self.alpha = self.add_weight(shape=(), initializer=keras.initializers.Constant(
            self.alpha0))
        self.w = self.add_weight(shape=(f,), initializer="ones")
        self.b = self.add_weight(shape=(f,), initializer="zeros")

    def call(self, x):
        return self.w * ops.tanh(self.alpha * x) + self.b
''',
}

PYTORCH_HELPERS.update(TORCH)
KERAS_HELPERS.update(KERAS)


# ------------------------------------------------------------------ shape helpers

def _seq(s, name: str) -> None:
    if len(s) != 2:
        raise ShapeError(f"{name} expects a batch-first sequence [L, D], got {s}")


def _heads(d: int, heads: int, name: str) -> None:
    if heads < 1 or d % heads:
        raise ShapeError(f"{name}: the width D={d} is not divisible by {heads} heads")


def _image(s, name: str) -> None:
    if len(s) != 3:
        raise ShapeError(f"{name} expects feature maps [C, H, W], got {s}")


def _odd(k: int, name: str) -> None:
    if k % 2 == 0:
        raise ShapeError(f"{name}: use an odd kernel size (3, 5, 7)")


def _down(d: int, stride: int) -> int:
    return (d - 1) // stride + 1          # "same" padding


def _attn_params(d: int) -> int:
    return 4 * d * d + 4 * d              # qkv + output projections


def _seq_shape(name: str, check=None):
    def fn(in_shapes, p):
        s = in_shapes[0]
        _seq(s, name)
        if "heads" in p:
            _heads(s[1], int(p["heads"]), name)
        if check:
            check(s, p)
        return list(s)

    return fn


_HEADS = P("heads", "int", 4, lo=1, help="Attention heads; must divide the width D")
_CAUSAL = P("causal", "bool", True, help="Each position sees only itself and earlier ones")


# ------------------------------------------------------------------ attention variants

reg.register(nn_block(
    "core.sliding_window_attention", "Sliding-Window Attention", TRANSFORMER,
    family="attention",
    params=(_HEADS, P("window", "int", 16, lo=1, help="Positions each one can see"),
            _CAUSAL, P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
    shape=_seq_shape("Sliding-Window Attention"), layout="ir",
    param_fn=lambda s, p: _attn_params(s[0][-1]),
    torch=lambda c: (f"LocalAttention({c['input_size']}, {int(c['heads'])}, "
                     f"window={int(c['window'])}, causal={bool(c['causal'])}, "
                     f"dropout={float(c['dropout'])})"),
    torch_helpers=("LocalAttention",),
    keras_expr=lambda c: (f"local_attention({c['i0']}, {int(c['heads'])}, "
                          f"window={int(c['window'])}, causal={bool(c['causal'])}, "
                          f"dropout={float(c['dropout'])})"),
    keras_helpers=("local_attention",),
    desc="Self-attention where each position only sees the last few (Longformer, Mistral): "
         "memory grows with the window, not the whole sequence, and stacked layers still "
         "reach far ([L, D] → [L, D]).",
))

reg.register(nn_block(
    "core.alibi_attention", "ALiBi Attention", TRANSFORMER, family="attention",
    params=(_HEADS, _CAUSAL, P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
    shape=_seq_shape("ALiBi Attention"), layout="ir",
    param_fn=lambda s, p: _attn_params(s[0][-1]),
    torch=lambda c: (f"LocalAttention({c['input_size']}, {int(c['heads'])}, "
                     f"causal={bool(c['causal'])}, alibi=True, dropout={float(c['dropout'])})"),
    torch_helpers=("LocalAttention",),
    keras_expr=lambda c: (f"local_attention({c['i0']}, {int(c['heads'])}, "
                          f"causal={bool(c['causal'])}, alibi=True, "
                          f"dropout={float(c['dropout'])})"),
    keras_helpers=("local_attention",),
    desc="Self-attention with no position embeddings: each head penalizes distant "
         "positions linearly, at its own rate (Press et al., 2022; BLOOM, MPT). Works on "
         "sequences longer than those it trained on ([L, D] → [L, D]).",
))


def _diff_check(s, p):
    hd = s[1] // int(p["heads"])
    if hd % 2:
        raise ShapeError(f"Differential Attention: the head width {hd} must be even (it is "
                         "split into two halves)")


reg.register(nn_block(
    "core.diff_attention", "Differential Attention", TRANSFORMER, family="attention",
    params=(_HEADS, _CAUSAL, P("lambda_init", "float", 0.8, lo=0.0, hi=1.0,
                               help="Starting weight of the subtracted map")),
    shape=_seq_shape("Differential Attention", _diff_check), layout="ir",
    param_fn=lambda s, p: _attn_params(s[0][-1]) + 3 * (s[0][-1] // int(p["heads"])),
    torch=lambda c: (f"DiffAttention({c['input_size']}, {int(c['heads'])}, "
                     f"causal={bool(c['causal'])}, lambda_init={float(c['lambda_init'])})"),
    torch_helpers=("DiffAttention",),
    keras_expr=lambda c: (f"DiffAttention({int(c['heads'])}, causal={bool(c['causal'])}, "
                          f"lambda_init={float(c['lambda_init'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "DiffAttention"),
    desc="Differential Transformer attention (Ye et al., 2024): two attention maps, one "
         "subtracted from the other, so noise both share cancels and the model focuses on "
         "what matters; better on long contexts and retrieval ([L, D] → [L, D]).",
))


# ------------------------------------------------------------------ recurrent-style mixers

reg.register(nn_block(
    "seq.retention", "Retention (RetNet)", SEQUENCE, family="recurrent",
    params=(_HEADS,),
    shape=_seq_shape("Retention"), layout="ir",
    param_fn=lambda s, p: 5 * s[0][-1] ** 2 + 2 * s[0][-1],
    torch=lambda c: f"Retention({c['input_size']}, {int(c['heads'])})",
    torch_helpers=("Retention",),
    keras_expr=lambda c: f"Retention({int(c['heads'])})({c['i0']})",
    keras_helpers=("TorchLayout", "Retention"),
    desc="RetNet's multi-scale retention (Sun et al., 2023): attention without softmax, "
         "where each head forgets the past at its own rate. Trains in parallel like a "
         "transformer and can run step by step like an RNN. Causal ([L, D] → [L, D]).",
))


def _rwkv_hidden(d: int, p) -> int:
    return int(p["hidden"]) or 4 * d


reg.register(nn_block(
    "seq.rwkv_block", "RWKV Block", SEQUENCE, family="recurrent",
    params=(P("hidden", "int", 0, lo=0, help="Channel-mixing width; 0 = 4 × D"),),
    shape=_seq_shape("RWKV Block"), layout="ir",
    param_fn=lambda s, p: (5 * s[0][-1] ** 2 + 2 * s[0][-1] * _rwkv_hidden(s[0][-1], p)
                           + 11 * s[0][-1]),
    torch=lambda c: f"RWKVBlock({c['input_size']}, {_rwkv_hidden(c['input_size'], c)})",
    torch_helpers=("RWKVBlock",),
    keras_expr=lambda c: f"RWKVBlock({_rwkv_hidden(c['input_size'], c)})({c['i0']})",
    keras_helpers=("TorchLayout", "RWKVBlock"),
    desc="One RWKV layer (Peng et al., 2023): a recurrent network that trains like a "
         "transformer, with time mixing (attention-like, but linear and decaying) and "
         "channel mixing. Stack a few after an Embedding for a language model. Causal "
         "([L, D] → [L, D]).",
))


def _hyena_params(s, p) -> int:
    d, f = s[0][-1], int(p["filter_hidden"])
    return 4 * d * d + 18 * d + 4 * f + f * d


reg.register(nn_block(
    "seq.hyena", "Hyena Operator", SEQUENCE, family="recurrent",
    params=(P("filter_hidden", "int", 32, lo=4, hi=512,
              help="Width of the network that generates the long filter"),),
    shape=_seq_shape("Hyena Operator"), layout="ir", param_fn=_hyena_params,
    torch=lambda c: f"HyenaOperator({c['input_size']}, {int(c['filter_hidden'])})",
    torch_helpers=("HyenaOperator",),
    keras_expr=lambda c: f"HyenaOperator({int(c['filter_hidden'])})({c['i0']})",
    keras_helpers=("TorchLayout", "HyenaOperator"),
    desc="Hyena (Poli et al., 2023): a sequence-long convolution, computed with FFTs, "
         "between two learned gates; an attention replacement whose cost grows almost "
         "linearly with the length. Causal ([L, D] → [L, D]).",
))


def _perceiver_shape(in_shapes, p):
    s = in_shapes[0]
    _seq(s, "Perceiver Latents")
    _heads(s[1], int(p["heads"]), "Perceiver Latents")
    return [int(p["latents"]), s[1]]


reg.register(nn_block(
    "core.perceiver_latents", "Perceiver Latents", TRANSFORMER, family="attention",
    params=(P("latents", "int", 16, lo=1, help="Number of output vectors"), _HEADS),
    shape=_perceiver_shape, layout="ir",
    param_fn=lambda s, p: (int(p["latents"]) * s[0][-1] + _attn_params(s[0][-1])
                           + 4 * s[0][-1]),
    torch=lambda c: (f"PerceiverLatents({c['input_size']}, latents={int(c['latents'])}, "
                     f"heads={int(c['heads'])})"),
    torch_helpers=("PerceiverLatents",),
    keras_expr=lambda c: (f"PerceiverLatents({int(c['latents'])}, "
                          f"{int(c['heads'])})({c['i0']})"),
    keras_helpers=("PerceiverLatents",),
    desc="A fixed number of learned vectors read the whole input through cross-attention "
         "(Perceiver, Flamingo's resampler): a long sequence becomes a short summary "
         "([L, D] → [latents, D]), so later layers cost the same at any length.",
))


def _linformer_check(s, p):
    if int(p["k"]) > s[0]:
        raise ShapeError(f"Linformer Attention: k={p['k']} is longer than the sequence "
                         f"({s[0]})")


reg.register(nn_block(
    "core.linformer_attention", "Linformer Attention", TRANSFORMER, family="attention",
    params=(_HEADS, P("k", "int", 32, lo=1, help="Projected length of keys and values")),
    shape=_seq_shape("Linformer Attention", _linformer_check), layout="ir",
    param_fn=lambda s, p: _attn_params(s[0][-1]) + 2 * s[0][0] * int(p["k"]),
    torch=lambda c: (f"LinformerAttention({c['input_size']}, {c['seq_len']}, "
                     f"k={int(c['k'])}, heads={int(c['heads'])})"),
    torch_helpers=("LinformerAttention",),
    keras_expr=lambda c: (f"LinformerAttention(k={int(c['k'])}, "
                          f"heads={int(c['heads'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "LinformerAttention"),
    desc="Self-attention with keys and values squeezed along the sequence to k positions "
         "(Wang et al., 2020): cost grows linearly with the length. Needs a fixed length "
         "([L, D] → [L, D]).",
))


# ------------------------------------------------------------------ all-MLP mixers

def _mixer_hidden(s, p) -> tuple[int, int]:
    n, d = s
    return int(p["token_hidden"]) or 2 * n, int(p["channel_hidden"]) or 4 * d


def _mixer_params(s, p) -> int:
    n, d = s[0]
    t, c = _mixer_hidden(s[0], p)
    return 4 * d + (n * t + t + t * n + n) + (d * c + c + c * d + d)


reg.register(nn_block(
    "core.mixer_block", "MLP-Mixer Block", TRANSFORMER, family="attention",
    params=(P("token_hidden", "int", 0, lo=0, help="Width of the across-positions MLP; "
                                                   "0 = 2 × L"),
            P("channel_hidden", "int", 0, lo=0, help="Width of the across-features MLP; "
                                                     "0 = 4 × D")),
    shape=_seq_shape("MLP-Mixer Block"), layout="ir", param_fn=_mixer_params,
    torch=lambda c: (f"MixerBlock({c['seq_len']}, {c['input_size']}, "
                     f"{_mixer_hidden(c['input_shape'], c)[0]}, "
                     f"{_mixer_hidden(c['input_shape'], c)[1]})"),
    torch_helpers=("MixerBlock",),
    keras_expr=lambda c: (f"mixer_block({c['i0']}, {_mixer_hidden(c['input_shape'], c)[0]}, "
                          f"{_mixer_hidden(c['input_shape'], c)[1]})"),
    keras_helpers=("mixer_block",),
    desc="MLP-Mixer (Tolstikhin et al., 2021): attention replaced by an MLP that mixes "
         "positions, followed by one that mixes features. Use on image patches or any "
         "fixed-length sequence ([L, D] → [L, D]).",
))


def _gmlp_hidden(d: int, p) -> int:
    return int(p["hidden"]) or 4 * d


def _gmlp_check(s, p):
    if _gmlp_hidden(s[1], p) % 2:
        raise ShapeError("gMLP Block: the hidden width must be even (it is split in two)")


def _gmlp_params(s, p) -> int:
    n, d = s[0]
    f = _gmlp_hidden(d, p)
    return 2 * d + d * f + f + f + n * n + n + (f // 2) * d + d


reg.register(nn_block(
    "core.gmlp_block", "gMLP Block", TRANSFORMER, family="attention",
    params=(P("hidden", "int", 0, lo=0, help="Feed-forward width; 0 = 4 × D"),),
    shape=_seq_shape("gMLP Block", _gmlp_check), layout="ir", param_fn=_gmlp_params,
    torch=lambda c: (f"GMLPBlock({c['seq_len']}, {c['input_size']}, "
                     f"{_gmlp_hidden(c['input_size'], c)})"),
    torch_helpers=("GMLPBlock",),
    keras_expr=lambda c: f"gmlp_block({c['i0']}, {_gmlp_hidden(c['input_size'], c)})",
    keras_helpers=("gmlp_block",),
    desc="gMLP (Liu et al., 2021): a feed-forward block whose hidden units are gated by a "
         "learned mix of all positions, close to a transformer's quality with no attention "
         "([L, D] → [L, D]).",
))


def _poolformer_shape(in_shapes, p):
    _image(in_shapes[0], "PoolFormer Block")
    _odd(int(p["pool_size"]), "PoolFormer Block")
    return list(in_shapes[0])


def _poolformer_params(s, p) -> int:
    c = s[0][0]
    h = c * int(p["mlp_ratio"])
    return 4 * c + c * h + h + h * c + c


reg.register(nn_block(
    "vision.poolformer_block", "PoolFormer Block", VISION, family="conv",
    params=(P("pool_size", "int", 3, lo=1, hi=9), P("mlp_ratio", "int", 4, lo=1, hi=8)),
    shape=_poolformer_shape, layout="cl", param_fn=_poolformer_params,
    torch=lambda c: (f"PoolFormerBlock({c['in_channels']}, pool_size={int(c['pool_size'])}, "
                     f"mlp_ratio={int(c['mlp_ratio'])})"),
    torch_helpers=("PoolFormerBlock",),
    keras_expr=lambda c: (f"poolformer_block({c['i0']}, pool_size={int(c['pool_size'])}, "
                          f"mlp_ratio={int(c['mlp_ratio'])})"),
    keras_helpers=("poolformer_block",),
    desc="PoolFormer (Yu et al., 2022): a transformer block for images whose attention is "
         "replaced by plain average pooling. Shows how much of a vision transformer's "
         "strength is the block structure; cheap ([C, H, W] → same).",
))


# ------------------------------------------------------------------ convolution blocks

def _ghost_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Ghost Module")
    return [int(p["out_channels"]), s[1], s[2]]


def _ghost_params(s, p) -> int:
    c, o, r = s[0][0], int(p["out_channels"]), int(p["ratio"])
    init = -(-o // r)
    return c * init + 2 * init + init * (r - 1) * 9 + 2 * init * (r - 1)


reg.register(nn_block(
    "vision.ghost_module", "Ghost Module", VISION, family="conv",
    params=(P("out_channels", "int", 32, lo=1),
            P("ratio", "int", 2, lo=2, hi=8, help="1 / ratio of the channels come from the "
                                                  "full convolution")),
    shape=_ghost_shape, layout="cl", param_fn=_ghost_params,
    torch=lambda c: (f"GhostModule({c['in_channels']}, {int(c['out_channels'])}, "
                     f"ratio={int(c['ratio'])})"),
    torch_helpers=("GhostModule",),
    keras_expr=lambda c: (f"ghost_module({c['i0']}, {int(c['out_channels'])}, "
                          f"ratio={int(c['ratio'])})"),
    keras_helpers=("ghost_module",),
    desc="GhostNet's replacement for a 1×1 convolution (Han et al., 2020): half the "
         "channels from a real convolution, the other half cheaply derived from them. "
         "About half the cost; for phones ([C, H, W] → [out, H, W]).",
))


def _involution_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Involution")
    _odd(int(p["kernel_size"]), "Involution")
    if s[0] % int(p["groups"]):
        raise ShapeError(f"Involution: {p['groups']} groups must divide the {s[0]} channels")
    return list(s)


def _involution_params(s, p) -> int:
    c, k, g = s[0][0], int(p["kernel_size"]), int(p["groups"])
    mid = max(c // int(p["reduction"]), 1)
    return c * mid + 2 * mid + mid * k * k * g + k * k * g


reg.register(nn_block(
    "vision.involution", "Involution", VISION, family="conv",
    params=(P("kernel_size", "int", 7, lo=1, hi=11), P("groups", "int", 4, lo=1),
            P("reduction", "int", 4, lo=1)),
    shape=_involution_shape, layout="cl", param_fn=_involution_params,
    torch=lambda c: (f"Involution({c['in_channels']}, kernel_size={int(c['kernel_size'])}, "
                     f"groups={int(c['groups'])}, reduction={int(c['reduction'])})"),
    torch_helpers=("Involution",),
    keras_expr=lambda c: (f"involution({c['i0']}, kernel_size={int(c['kernel_size'])}, "
                          f"groups={int(c['groups'])}, reduction={int(c['reduction'])})"),
    keras_helpers=("involution",),
    desc="Involution (Li et al., 2021): a convolution turned inside out. The kernel is "
         "different at every pixel (generated from it) and shared across channels; large "
         "kernels for few parameters ([C, H, W] → same).",
))


def _deform_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Deformable Conv2D")
    k, st, pad = int(p["kernel_size"]), int(p["stride"]), int(p["padding"])
    h, w = (s[1] + 2 * pad - k) // st + 1, (s[2] + 2 * pad - k) // st + 1
    if h < 1 or w < 1:
        raise ShapeError("Deformable Conv2D: the kernel is larger than the padded input")
    return [int(p["out_channels"]), h, w]


def _deform_params(s, p) -> int:
    c, o, k = s[0][0], int(p["out_channels"]), int(p["kernel_size"])
    return c * 2 * k * k * k * k + 2 * k * k + c * o * k * k + o


reg.register(replace(nn_block(
    "vision.deform_conv2d", "Deformable Conv2D", VISION, family="conv",
    params=(P("out_channels", "int", 32, lo=1), P("kernel_size", "int", 3, lo=1),
            P("stride", "int", 1, lo=1), P("padding", "int", 1, lo=0)),
    shape=_deform_shape, layout="cl", param_fn=_deform_params,
    torch=lambda c: (f"DeformConv2d({c['in_channels']}, {int(c['out_channels'])}, "
                     f"{int(c['kernel_size'])}, stride={int(c['stride'])}, "
                     f"padding={int(c['padding'])})"),
    torch_helpers=("DeformConv2d",),
    desc="Deformable convolution (Dai et al., 2017): the network learns where each kernel "
         "tap looks, so the receptive field bends to fit objects; detection and "
         "segmentation of things with varied shapes. Needs torchvision."),
    requires=("torchvision",), extra="vision"))


def _fused_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Fused-MBConv")
    _odd(int(p["kernel_size"]), "Fused-MBConv")
    st = int(p["stride"])
    return [int(p["out_channels"]), _down(s[1], st), _down(s[2], st)]


def _fused_params(s, p) -> int:
    c, o, e, k = s[0][0], int(p["out_channels"]), int(p["expand_ratio"]), int(p["kernel_size"])
    if e == 1:
        return c * o * k * k + 2 * o
    mid = c * e
    return c * mid * k * k + 2 * mid + mid * o + 2 * o


reg.register(nn_block(
    "vision.fused_mbconv", "Fused-MBConv", VISION, family="conv",
    params=(P("out_channels", "int", 32, lo=1), P("expand_ratio", "int", 4, lo=1, hi=8),
            P("kernel_size", "int", 3, lo=1, hi=9), P("stride", "int", 1, lo=1, hi=4)),
    shape=_fused_shape, layout="cl", param_fn=_fused_params,
    torch=lambda c: (f"FusedMBConv({c['in_channels']}, {int(c['out_channels'])}, "
                     f"expand_ratio={int(c['expand_ratio'])}, "
                     f"kernel_size={int(c['kernel_size'])}, stride={int(c['stride'])})"),
    torch_helpers=("FusedMBConv",),
    keras_expr=lambda c: (f"fused_mbconv({c['i0']}, {int(c['out_channels'])}, "
                          f"expand_ratio={int(c['expand_ratio'])}, "
                          f"kernel_size={int(c['kernel_size'])}, stride={int(c['stride'])})"),
    keras_helpers=("fused_mbconv",),
    desc="EfficientNetV2's early-stage block (Tan & Le, 2021): MBConv with the expansion "
         "and depthwise convolution fused into one regular convolution, faster on GPUs; a "
         "shortcut when the shape stays.",
))


def _eca_shape(in_shapes, p):
    _image(in_shapes[0], "ECA")
    _odd(int(p["kernel_size"]), "ECA")
    return list(in_shapes[0])


reg.register(nn_block(
    "vision.eca", "ECA (efficient channel attention)", "Attention", family="attention",
    params=(P("kernel_size", "int", 3, lo=1, hi=9, help="Neighbouring channels each weight "
                                                        "looks at"),),
    shape=_eca_shape, layout="cl", param_fn=lambda s, p: int(p["kernel_size"]),
    torch=lambda c: f"ECA({int(c['kernel_size'])})", torch_helpers=("ECA",),
    keras_expr=lambda c: f"eca({c['i0']}, {int(c['kernel_size'])})", keras_helpers=("eca",),
    desc="Efficient Channel Attention (Wang et al., 2020): squeeze-and-excitation with a "
         "tiny 1-D convolution instead of an MLP, a handful of parameters. Drop it after "
         "any convolution; the shape stays.",
))


def _grn_shape(in_shapes, p):
    _image(in_shapes[0], "GRN")
    return list(in_shapes[0])


reg.register(nn_block(
    "vision.grn", "Global Response Norm (GRN)", "Normalization", family="normalization",
    shape=_grn_shape, layout="cl", param_fn=lambda s, p: 2 * s[0][0],
    torch="GRN({in_channels})", torch_helpers=("GRN",),
    keras_expr="GRN()({i0})", keras_helpers=("GRN",),
    desc="ConvNeXt V2's Global Response Normalization (Woo et al., 2023): scales each "
         "channel by its share of the total response so channels stay diverse; put it "
         "inside a block's MLP. Starts as the identity ([C, H, W] → same).",
))


def _dyt_shape(in_shapes, p):
    s = in_shapes[0]
    if not s:
        raise ShapeError("Dynamic Tanh needs features on the last axis")
    return list(s)


reg.register(nn_block(
    "core.dyt", "Dynamic Tanh (DyT)", "Normalization", family="normalization",
    params=(P("alpha", "float", 0.5, lo=0.01, hi=10.0, help="Starting steepness"),),
    shape=_dyt_shape, layout="ir", param_fn=lambda s, p: 2 * s[0][-1] + 1,
    torch=lambda c: f"DyT({c['features']}, alpha={float(c['alpha'])})",
    torch_helpers=("DyT",),
    keras_expr=lambda c: f"DyT(alpha={float(c['alpha'])})({c['i0']})",
    keras_helpers=("DyT",),
    desc="Dynamic Tanh (Zhu et al., 2025): a learned tanh squashing in place of LayerNorm "
         "in transformers, with no mean or variance to compute; matches LayerNorm's "
         "results in the paper ([…, D] → same).",
))


LIBRARY37_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
