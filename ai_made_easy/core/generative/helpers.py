"""PyTorch helpers for generative models, emitted into generated code.

- ``Reparameterize``: the VAE bottleneck. Splits mean / log-variance halves, samples z and
  keeps the KL term in ``.kl``; setting ``.prior_z`` makes it emit prior samples instead
  (the decoder half of a canvas-designed VAE then generates).
- ``DCGANGenerator`` / ``Discriminator``: DCGAN-style image GAN pieces.
- ``DiffusionUNet``: a time- (and optionally class-) conditioned U-Net that predicts noise.
- ``CausalTransformer`` / ``GPT``: decoder-only language models (causal self-attention).
- ``Seq2SeqTransformer``: an encoder-decoder transformer with greedy decoding.

``*_cost`` twins count parameters / multiply-accumulates without torch.
"""
from __future__ import annotations

import math

REPARAM = '''\
class Reparameterize(nn.Module):
    """[.., 2Z] or [2C, H, W] (mean, log-variance halves) -> sampled z; keeps the KL term."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.kl = None
        self.prior_z = None        # set to a [n, *latent] tensor to decode prior samples
        self.latent_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = x.chunk(2, dim=-1 if x.dim() == 2 else 1)
        self.latent_shape = tuple(mean.shape[1:])
        if self.prior_z is not None:
            return self.prior_z.to(device=x.device, dtype=x.dtype)
        logvar = logvar.clamp(-30.0, 20.0)
        self.kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).flatten(1).sum(1).mean()
        if not self.training:
            return mean
        return mean + torch.randn_like(mean) * (0.5 * logvar).exp()
'''

DCGAN_INIT = '''\
def dcgan_init(m: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) convolutions, N(1, 0.02) batch-norm scales."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(getattr(m, "weight_orig", m.weight), 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
'''

DCGAN = '''\
import math


class DCGANGenerator(nn.Module):
    """Latent [B, Z] -> image [B, C, S, S] in [-1, 1] (Radford et al., 2016)."""

    def __init__(self, latent: int, channels: int, image_size: int, width: int = 64) -> None:
        super().__init__()
        ups = int(math.log2(image_size // 4))
        ch = width * 2 ** (ups - 1)
        layers = [nn.ConvTranspose2d(latent, ch, 4, 1, 0, bias=False), nn.BatchNorm2d(ch),
                  nn.ReLU(True)]
        for _ in range(ups - 1):
            layers += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ConvTranspose2d(ch, channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.apply(dcgan_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.reshape(z.shape[0], -1, 1, 1))
'''

DISCRIMINATOR = '''\
class Discriminator(nn.Module):
    """Image [B, C, H, W] -> realness score [B] (strided convolutions, LeakyReLU)."""

    def __init__(self, channels: int, height: int, width_px: int, width: int = 64,
                 norm: str = "batch", spectral: bool = False) -> None:
        super().__init__()
        sn = nn.utils.spectral_norm if spectral else (lambda m: m)
        layers, ch, h, w, out, first = [], channels, height, width_px, width, True
        while min(h, w) >= 8:
            layers.append(sn(nn.Conv2d(ch, out, 4, 2, 1, bias=first or norm == "none")))
            if not first and norm == "batch":
                layers.append(nn.BatchNorm2d(out))
            elif not first and norm == "layer":
                layers.append(nn.GroupNorm(1, out))
            layers.append(nn.LeakyReLU(0.2, True))
            ch, h, w, first = out, h // 2, w // 2, False
            out = min(out * 2, width * 8)
        self.features = nn.Sequential(*layers)
        self.head = sn(nn.Linear(ch * h * w, 1))
        self.apply(dcgan_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1)).squeeze(1)
'''

UNET = '''\
import math


def _groups(c: int) -> int:
    return math.gcd(c, 8) or 1


class DiffusionResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        h = h + self.time(nn.functional.silu(emb))[:, :, None, None]
        h = self.conv2(self.drop(nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.heads = c // 32 if c % 32 == 0 and c >= 64 else 1
        self.norm = nn.GroupNorm(_groups(c), c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = (t.transpose(-1, -2) for t in qkv.unbind(1))
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class DiffusionUNet(nn.Module):
    """Noisy image [B, C, H, W] + timestep (+ class) -> predicted noise [B, C, H, W]."""

    is_denoiser = True

    def __init__(self, channels: int, base: int = 32, depth: int = 3, attention: bool = True,
                 num_classes: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        tdim = base * 4
        self.base, self.num_classes = base, num_classes
        self.time = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.label = nn.Embedding(num_classes + 1, tdim) if num_classes else None
        self.inc = nn.Conv2d(channels, base, 3, padding=1)
        mults = [min(2 ** i, 4) for i in range(depth)]
        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        ch, skips = base, []
        for i, m in enumerate(mults):
            self.down.append(DiffusionResBlock(ch, base * m, tdim, dropout))
            ch = base * m
            skips.append(ch)
            self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                                   if i < depth - 1 else nn.Identity())
        self.mid1 = DiffusionResBlock(ch, ch, tdim, dropout)
        self.attn = SelfAttention2d(ch) if attention else nn.Identity()
        self.mid2 = DiffusionResBlock(ch, ch, tdim, dropout)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            self.up.append(DiffusionResBlock(ch + skips[i], base * m, tdim, dropout))
            ch = base * m
            self.upsample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                               nn.Conv2d(ch, ch, 3, padding=1))
                                 if i > 0 else nn.Identity())
        self.out = nn.Sequential(nn.GroupNorm(_groups(ch), ch), nn.SiLU(),
                                 nn.Conv2d(ch, channels, 3, padding=1))

    def embed(self, t: torch.Tensor, y) -> torch.Tensor:
        half = self.base // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device)
                          / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None]
        emb = self.time(torch.cat([args.sin(), args.cos()], 1))
        if self.label is not None:
            if y is None:
                y = torch.full_like(t, self.num_classes)   # the "no class" token
            emb = emb + self.label(y)
        return emb

    def forward(self, x: torch.Tensor, t=None, y=None) -> torch.Tensor:
        if t is None:
            t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        emb = self.embed(t, y)
        h, skips = self.inc(x), []
        for block, down in zip(self.down, self.downsample):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.attn(self.mid1(h, emb)), emb)
        for block, up in zip(self.up, self.upsample):
            h = up(block(torch.cat([h, skips.pop()], 1), emb))
        return self.out(h)
'''

CAUSAL = '''\
class CausalBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention: position i sees 0..i only."""

    def __init__(self, d: int, heads: int, dropout: float = 0.1, mlp: int = 4) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, length, 3, self.heads,
                                                d // self.heads).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(b, length, d))
        return x + self.mlp(self.ln2(x))


class CausalTransformer(nn.Module):
    """[B, L, D] -> [B, L, D] through causal blocks and a final LayerNorm."""

    def __init__(self, d: int, layers: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(CausalBlock(d, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.ln(x)
'''

GPT = '''\
class GPT(nn.Module):
    """Tokens [B, L] -> next-token logits [B, L, V] (decoder-only transformer)."""

    is_language_model = True

    def __init__(self, vocab: int, context: int, d_model: int = 128, layers: int = 4,
                 heads: int = 4, dropout: float = 0.1, tie: bool = True) -> None:
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.drop = nn.Dropout(dropout)
        self.body = CausalTransformer(d_model, layers, heads, dropout)
        self.head = nn.Linear(d_model, vocab, bias=False)
        if tie:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        """GPT-2 initialisation: small weights keep the first logits near uniform."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.long()[:, -self.context:]
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.head(self.body(self.drop(self.tok(idx) + self.pos(pos))))
'''

SEQ2SEQ = '''\
class Seq2SeqTransformer(nn.Module):
    """Source tokens [B, S] -> target logits [B, T, V] (encoder-decoder transformer).

    The last three vocabulary ids are PAD, BOS and EOS. ``forward(src, tgt_in)`` uses
    teacher forcing; without ``tgt_in`` it returns logits for a BOS-only target (the
    designed shape); ``generate(src)`` decodes greedily.
    """

    is_seq2seq = True

    def __init__(self, vocab: int, max_source: int, max_target: int, d_model: int = 128,
                 encoder_layers: int = 3, decoder_layers: int = 3, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.pad, self.bos, self.eos = vocab - 3, vocab - 2, vocab - 1
        self.max_source, self.max_target = max_source, max_target
        self.embed = nn.Embedding(vocab, d_model)
        self.src_pos = nn.Embedding(max_source, d_model)
        self.tgt_pos = nn.Embedding(max_target, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                         batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, encoder_layers, nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, decoder_layers, nn.LayerNorm(d_model))
        self.head = nn.Linear(d_model, vocab)
        nn.init.normal_(self.embed.weight, 0.0, d_model ** -0.5)

    def encode(self, src: torch.Tensor):
        src = src.long()[:, :self.max_source]
        pos = torch.arange(src.shape[1], device=src.device)
        mask = src == self.pad
        return self.encoder(self.embed(src) + self.src_pos(pos), src_key_padding_mask=mask), mask

    def decode(self, memory, src_mask, tgt: torch.Tensor) -> torch.Tensor:
        tgt = tgt.long()[:, :self.max_target]
        n = tgt.shape[1]
        pos = torch.arange(n, device=tgt.device)
        # bool, like the padding masks (True: not attended to)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt.device), 1)
        out = self.decoder(self.embed(tgt) + self.tgt_pos(pos), memory, tgt_mask=causal,
                           tgt_key_padding_mask=tgt == self.pad,
                           memory_key_padding_mask=src_mask)
        return self.head(out)

    def forward(self, src: torch.Tensor, tgt_in=None) -> torch.Tensor:
        if tgt_in is None:
            tgt_in = torch.full((src.shape[0], self.max_target), self.bos, device=src.device)
        memory, mask = self.encode(src)
        return self.decode(memory, mask, tgt_in)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len=None) -> torch.Tensor:
        memory, mask = self.encode(src)
        out = torch.full((src.shape[0], 1), self.bos, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(min(max_len or self.max_target, self.max_target)):
            nxt = self.decode(memory, mask, out)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad), nxt)
            out = torch.cat([out, nxt[:, None]], 1)
            done |= nxt == self.eos
            if done.all():
                break
        return out[:, 1:]
'''

GENERATIVE_HELPERS: dict[str, str] = {
    "Reparameterize": REPARAM, "DCGANGenerator": DCGAN_INIT + "\n\n" + DCGAN,
    "Discriminator": DCGAN_INIT + "\n\n" + DISCRIMINATOR,
    "DiffusionUNet": UNET, "CausalTransformer": CAUSAL, "GPT": GPT,
    "Seq2SeqTransformer": SEQ2SEQ,
}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(GENERATIVE_HELPERS)


# ------------------------------------------------------------- analytic costs

def _conv(cin: int, cout: int, k: int, bias: bool = True) -> int:
    return k * k * cin * cout + (cout if bias else 0)


def dcgan_cost(latent: int, channels: int, image_size: int, width: int) -> tuple[int, int]:
    ups = int(math.log2(image_size // 4))
    ch = width * 2 ** (ups - 1)
    params = _conv(latent, ch, 4, False) + 2 * ch
    macs = latent * ch * 16                       # 1x1 -> 4x4
    side = 4
    for _ in range(ups - 1):
        params += _conv(ch, ch // 2, 4, False) + ch
        macs += ch * (ch // 2) * 16 * side * side
        ch //= 2
        side *= 2
    params += _conv(ch, channels, 4, False)
    macs += ch * channels * 16 * side * side
    return params, macs


def _groups(c: int) -> int:
    return math.gcd(c, 8) or 1


def _res_cost(cin: int, cout: int, tdim: int, side: int) -> tuple[int, int]:
    params = (2 * cin + _conv(cin, cout, 3) + tdim * cout + cout + 2 * cout
              + _conv(cout, cout, 3) + (_conv(cin, cout, 1) if cin != cout else 0))
    pixels = side * side
    macs = (9 * cin * cout + 9 * cout * cout + (cin * cout if cin != cout else 0)) * pixels \
        + tdim * cout
    return params, macs


def unet_cost(channels: int, height: int, width: int, base: int, depth: int, attention: bool,
              num_classes: int) -> tuple[int, int]:
    tdim = 4 * base
    params = base * tdim + tdim + tdim * tdim + tdim
    macs = base * tdim + tdim * tdim
    if num_classes:
        params += (num_classes + 1) * tdim
    params += _conv(channels, base, 3)
    macs += 9 * channels * base * height * width
    mults = [min(2 ** i, 4) for i in range(depth)]
    ch, skips, h, w = base, [], height, width
    for i, m in enumerate(mults):
        p, a = _res_cost(ch, base * m, tdim, 1)
        params += p
        macs += (a - tdim * base * m) * h * w + tdim * base * m
        ch = base * m
        skips.append(ch)
        if i < depth - 1:
            h, w = (h + 1) // 2, (w + 1) // 2
            params += _conv(ch, ch, 3)
            macs += 9 * ch * ch * h * w
    for _ in range(2):
        p, a = _res_cost(ch, ch, tdim, 1)
        params += p
        macs += (a - tdim * ch) * h * w + tdim * ch
    if attention:
        params += 2 * ch + _conv(ch, 3 * ch, 1) + _conv(ch, ch, 1)
        n = h * w
        macs += 4 * ch * ch * n + 2 * n * n * ch
    for i in reversed(range(depth)):
        p, a = _res_cost(ch + skips[i], base * mults[i], tdim, 1)
        params += p
        macs += (a - tdim * base * mults[i]) * h * w + tdim * base * mults[i]
        ch = base * mults[i]
        if i > 0:
            h, w = h * 2, w * 2
            params += _conv(ch, ch, 3)
            macs += 9 * ch * ch * h * w
    params += 2 * ch + _conv(ch, channels, 3)
    macs += 9 * ch * channels * h * w
    return params, macs


def causal_cost(d: int, layers: int, length: int) -> tuple[int, int]:
    params = layers * (12 * d * d + 13 * d) + 2 * d
    macs = layers * (12 * d * d * length + 2 * length * length * d)
    return params, macs


def gpt_cost(vocab: int, context: int, d: int, layers: int, length: int,
             tie: bool) -> tuple[int, int]:
    params, macs = causal_cost(d, layers, length)
    params += vocab * d + context * d + (0 if tie else vocab * d)
    return params, macs + length * d * vocab


def seq2seq_cost(vocab: int, max_source: int, max_target: int, d: int, enc: int,
                 dec: int) -> tuple[int, int]:
    ff = 4 * d
    enc_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    dec_layer = 8 * d * d + 2 * d * ff + 15 * d + ff
    params = (vocab * d + max_source * d + max_target * d + enc * enc_layer + dec * dec_layer
              + 4 * d + d * vocab + vocab)
    s, t = max_source, max_target
    macs = (enc * (s * (4 * d * d + 2 * d * ff) + 2 * s * s * d)
            + dec * (t * (4 * d * d + 2 * d * ff) + 2 * t * t * d          # self-attn + MLP
                     + t * 2 * d * d + s * 2 * d * d + 2 * t * s * d)      # cross-attention
            + t * d * vocab)
    return params, macs
