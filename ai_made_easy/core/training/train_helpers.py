"""Source of the PyTorch optimizers and losses that are not in torch.

``TRAIN_HELPERS`` maps a helper name (a component's ``meta["torch_helper"]``) to class
source that code generation emits into the training script; the catalog package re-exports it.
"""
from __future__ import annotations

_PROTOTYPES = '''\
class ClassPrototypes(nn.Module):
    """Running mean embedding of each class; scores embeddings by cosine similarity."""

    def __init__(self, embed_dim: int, num_classes: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.register_buffer("seen", torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, emb: torch.Tensor, target: torch.Tensor) -> None:
        emb = nn.functional.normalize(emb.detach().float(), dim=1)
        for c in target.unique():
            mean = emb[target == c].mean(0)
            old = self.prototypes[c]
            self.prototypes[c] = mean if self.seen[c] == 0 else (
                self.momentum * old + (1 - self.momentum) * mean)
            self.seen[c] = 1

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        protos = nn.functional.normalize(self.prototypes, dim=1)
        emb = nn.functional.normalize(emb.to(protos.device).float(), dim=1)
        return 10.0 * emb @ protos.t()
'''

# PyTorch optimizers and losses that are not in torch: emitted into the training script
TRAIN_HELPERS = {
    "MarginSoftmaxLoss": '''\
class MarginSoftmaxLoss(nn.Module):
    """ArcFace (Deng et al., 2019) / CosFace (Wang et al., 2018): softmax over the scaled
    cosine to learned class centres, with a margin on the true class."""

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0,
                 margin: float = 0.5, kind: str = "arcface") -> None:
        super().__init__()
        self.centres = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        self.scale, self.margin, self.kind = scale, margin, kind

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        centres = nn.functional.normalize(self.centres, dim=1)
        return nn.functional.normalize(emb.to(centres.device).float(), dim=1) @ centres.t()

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * self.cosine(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(emb)
        target = target.long().reshape(-1)
        true = nn.functional.one_hot(target, cos.shape[1]).bool()
        if self.kind == "arcface":
            angle = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            shifted = torch.cos(angle + self.margin)
        else:
            shifted = cos - self.margin
        logits = torch.where(true, shifted, cos) * self.scale
        return nn.functional.cross_entropy(logits, target)
''',
    "TripletLoss": _PROTOTYPES + '''

class TripletLoss(ClassPrototypes):
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalised embeddings. Labelled
    triplets [batch, 3, embedding] (anchor, positive, negative) use their own triplets."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.3) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    @staticmethod
    def _gaps(emb: torch.Tensor) -> torch.Tensor:
        """d(anchor, negative) - d(anchor, positive) per labelled triplet."""
        e = nn.functional.normalize(emb.float(), dim=-1)
        return (e[:, 0] - e[:, 2]).norm(dim=1) - (e[:, 0] - e[:, 1]).norm(dim=1)

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:   # labelled triplets: a logit that says "the positive is closer"
            return 10.0 * self._gaps(emb)[:, None]
        return super().scores(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:
            return torch.relu(self.margin - self._gaps(emb)).mean()
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        dist = torch.cdist(e, e)
        same = target[:, None] == target[None, :]
        hardest_pos = (dist * same).max(dim=1).values
        hardest_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
        has_neg = torch.isfinite(hardest_neg)
        if self.training:
            self.update(e, target)
        if not has_neg.any():
            return (dist * 0).sum()
        return torch.relu(hardest_pos - hardest_neg + self.margin)[has_neg].mean()
''',
    "NTXentLoss": _PROTOTYPES + '''

class NTXentLoss(ClassPrototypes):
    """NT-Xent (SimCLR, Chen et al., 2020). A paired batch holds two views of each image,
    the first views then the second; each view's positive is its twin. Unpaired batches
    (the test set) score same-class examples as positives instead."""

    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.5) -> None:
        super().__init__(embed_dim, num_classes)
        self.temperature = temperature
        self.paired = True

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)      # class averages: only to measure the embedding
        n = len(e)
        eye = torch.eye(n, dtype=torch.bool, device=e.device)
        sim = (e @ e.t() / self.temperature).masked_fill(eye, float("-inf"))
        if self.paired and n % 2 == 0 and n > 1:
            twin = (torch.arange(n, device=e.device) + n // 2) % n
            return nn.functional.cross_entropy(sim, twin)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        positive = (target[:, None] == target[None, :]) & ~eye
        count = positive.sum(1)
        anchors = count > 0
        if not anchors.any():
            return (sim.masked_fill(eye, 0) * 0).sum()
        return -(log_prob.masked_fill(~positive, 0).sum(1)[anchors] / count[anchors]).mean()
''',
    "SupConLoss": _PROTOTYPES + '''

class SupConLoss(ClassPrototypes):
    """Supervised contrastive loss (Khosla et al., 2020): InfoNCE where every same-class
    example in the batch is a positive."""

    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.1) -> None:
        super().__init__(embed_dim, num_classes)
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        sim = e @ e.t() / self.temperature
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        sim = sim.masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        positive = (target[:, None] == target[None, :]) & ~eye
        count = positive.sum(1)
        if self.training:
            self.update(e, target)
        anchors = count > 0
        if not anchors.any():
            return (sim.masked_fill(eye, 0) * 0).sum()
        mean_pos = log_prob.masked_fill(~positive, 0).sum(1)[anchors] / count[anchors]
        return -mean_pos.mean()
''',
    "CosineEmbeddingPairsLoss": _PROTOTYPES + '''

class CosineEmbeddingPairsLoss(ClassPrototypes):
    """CosineEmbeddingLoss over every pair in the batch: 1 - cos for same-class pairs,
    max(0, cos - margin) for the others. Each kind is averaged on its own, so a batch with
    few same-class pairs still learns from them. Labelled pairs use their own labels."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.2) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:   # labelled pairs: a logit that says "same"
            cos = nn.functional.cosine_similarity(emb[:, 0].float(), emb[:, 1].float(), dim=1)
            return 10.0 * (cos - (1 + self.margin) / 2)[:, None]
        return super().scores(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        if emb.ndim == 3:   # labelled pairs [batch, 2, embedding]: 1 = same, 0 = different
            cos = nn.functional.cosine_similarity(emb[:, 0].float(), emb[:, 1].float(), dim=1)
            return torch.where(target > 0, 1 - cos, torch.relu(cos - self.margin)).mean()
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)
        cos = e @ e.t()
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        same = (target[:, None] == target[None, :]) & ~eye
        other = target[:, None] != target[None, :]
        loss = (cos * 0).sum()
        if same.any():
            loss = loss + (1 - cos[same]).mean()
        if other.any():
            loss = loss + torch.relu(cos[other] - self.margin).mean()
        return loss
''',
    "MarginRankingPairsLoss": _PROTOTYPES + '''

class MarginRankingPairsLoss(ClassPrototypes):
    """MarginRankingLoss on cosine similarity: around each example, every same-class
    neighbour should be more similar than every other-class one by the margin. On labelled
    pairs, it ranks the items' scores the way the labels say."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.2) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:   # labelled pairs: a logit that says "a ranks higher"
            return (emb[:, 0, :1] - emb[:, 1, :1]).float()
        return super().scores(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        if emb.ndim == 3:   # labelled pairs: each item's score; 1 = a ranks above b
            sign = target.float() * 2 - 1
            return nn.functional.margin_ranking_loss(
                emb[:, 0, 0].float(), emb[:, 1, 0].float(), sign, margin=self.margin)
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)
        cos = e @ e.t()
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        same = (target[:, None] == target[None, :]) & ~eye
        other = target[:, None] != target[None, :]
        # [anchor, positive, negative]: the positive should win by the margin
        gap = torch.relu(self.margin - cos[:, :, None] + cos[:, None, :])
        valid = same[:, :, None] & other[:, None, :]
        if not valid.any():
            return (cos * 0).sum()
        return gap[valid].mean()
''',
    "ScheduleFreeAdamW": '''\
class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-free AdamW (Defazio et al., 2024). Training runs at y, an interpolation of
    the fast iterate z and the running average x; eval() switches the weights to x and
    train() back to y."""

    def __init__(self, params, lr: float = 2.5e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, warmup_steps: int = 0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      warmup_steps=warmup_steps, k=0, weight_sum=0.0,
                                      training=True))

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if group["training"]:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(self.state[p]["z"], 1 - 1 / beta1)    # y -> x
                group["training"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if not group["training"]:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(self.state[p]["z"], 1 - beta1)        # x -> y
                group["training"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            k = group["k"]
            warm = min(1.0, (k + 1) / group["warmup_steps"]) if group["warmup_steps"] else 1.0
            lr = group["lr"] * warm * (1 - beta2 ** (k + 1)) ** 0.5
            group["weight_sum"] += lr ** 2
            c = lr ** 2 / group["weight_sum"] if group["weight_sum"] else 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state["z"], state["v"] = p.detach().clone(), torch.zeros_like(p)
                z, v = state["z"], state["v"]
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                update = p.grad / v.sqrt().add_(group["eps"])
                if group["weight_decay"]:
                    update.add_(p, alpha=group["weight_decay"])
                p.lerp_(z, 1 - 1 / beta1)        # y -> x
                z.sub_(update, alpha=lr)
                p.lerp_(z, c)                    # the average moves toward z
                p.lerp_(z, 1 - beta1)            # x -> y
            group["k"] = k + 1
        return loss
''',
    "SAM": '''\
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021) around SGD or AdamW: first_step()
    climbs to the worst weights within rho, second_step() restores them and steps with
    the gradient taken there."""

    def __init__(self, params, base: str = "sgd", rho: float = 0.05, adaptive: bool = False,
                 lr: float = 0.05, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(rho=rho, adaptive=adaptive, lr=lr))
        if base == "adamw":
            self.base_optimizer = torch.optim.AdamW(self.param_groups, lr=lr,
                                                    weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.SGD(self.param_groups, lr=lr, momentum=momentum,
                                                  weight_decay=weight_decay)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self) -> torch.Tensor:
        norms = [((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                 for group in self.param_groups for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0)

    @torch.no_grad()
    def first_step(self) -> None:
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e)
                self.state[p]["e_w"] = e

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()

    def step(self, closure=None):
        raise RuntimeError("SAM steps with first_step() and second_step()")
''',
    "Muon": '''\
class Muon(torch.optim.Optimizer):
    """Muon (Jordan et al., 2024): SGD momentum, orthogonalized by Newton-Schulz steps,
    for every weight with two or more axes (conv kernels flattened); AdamW for the rest."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, adamw_scale=adamw_lr / max(lr, 1e-12),
                                      betas=betas, eps=eps, weight_decay=weight_decay))

    @staticmethod
    def orthogonalize(g: torch.Tensor, steps: int) -> torch.Tensor:
        a, b, c = 3.4445, -4.7750, 2.0315
        x = g / (g.norm() + 1e-7)
        tall = x.size(0) > x.size(1)
        if tall:
            x = x.T
        for _ in range(steps):
            m = x @ x.T
            x = a * x + (b * m + c * m @ m) @ x
        return x.T if tall else x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if p.ndim >= 2:
                    buf = state.setdefault("momentum", torch.zeros_like(g))
                    buf.mul_(group["momentum"]).add_(g)
                    g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    flat = g.reshape(g.size(0), -1)
                    update = self.orthogonalize(flat, group["ns_steps"]).view_as(p)
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr * max(1.0, flat.size(0) / flat.size(1)) ** 0.5)
                else:
                    state["step"] = state.get("step", 0) + 1
                    m = state.setdefault("exp_avg", torch.zeros_like(g))
                    v = state.setdefault("exp_avg_sq", torch.zeros_like(g))
                    b1, b2 = group["betas"]
                    m.lerp_(g, 1 - b1)
                    v.lerp_(g.square(), 1 - b2)
                    alr = lr * group["adamw_scale"]
                    m_hat = m / (1 - b1 ** state["step"])
                    v_hat = v / (1 - b2 ** state["step"])
                    p.mul_(1 - alr * wd)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]), value=-alr)
        return loss
''',
    "Lion": '''\
class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023): steps by the sign of an interpolated momentum."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(torch.sign(m * beta1 + p.grad * (1 - beta1)), alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss
''',
    "Lamb": '''\
class Lamb(torch.optim.Optimizer):
    """LAMB (You et al., 2020): Adam scaled per layer by ||w|| / ||update||."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-6,
                 weight_decay: float = 0.01) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                update = (m / (1 - beta1 ** t)) / ((v / (1 - beta2 ** t)).sqrt() + group["eps"])
                update.add_(p, alpha=group["weight_decay"])
                w_norm, u_norm = p.norm(), update.norm()
                trust = torch.where((w_norm > 0) & (u_norm > 0), w_norm / u_norm,
                                    torch.ones_like(w_norm))
                p.add_(update * trust, alpha=-group["lr"])
        return loss
''',
    "AdaBelief": '''\
class AdaBelief(torch.optim.Optimizer):
    """AdaBelief (Zhuang et al., 2020): Adam whose second moment tracks (g - m)^2, the
    gradient's distance from its running mean; decoupled weight decay."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-16,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"], state["s"] = torch.zeros_like(p), torch.zeros_like(p)
                state["step"] += 1
                t, m, s = state["step"], state["m"], state["s"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                diff = p.grad - m
                s.mul_(beta2).addcmul_(diff, diff, value=1 - beta2).add_(group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                denom = (s / (1 - beta2 ** t)).sqrt().add_(group["eps"])
                p.addcdiv_(m, denom, value=-group["lr"] / (1 - beta1 ** t))
        return loss
''',
    "Adan": '''\
class Adan(torch.optim.Optimizer):
    """Adan (Xie et al., 2022), the paper's notation: betas weigh the new gradient, its
    change since the last step and the new squared term."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.02, 0.08, 0.01), eps: float = 1e-8,
                 weight_decay: float = 0.02) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2, b3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"], state["v"], state["n"] = (torch.zeros_like(p) for _ in range(3))
                    state["prev"] = g.clone()
                state["step"] += 1
                t, m, v, n = state["step"], state["m"], state["v"], state["n"]
                diff = g - state["prev"]
                m.mul_(1 - b1).add_(g, alpha=b1)
                v.mul_(1 - b2).add_(diff, alpha=b2)
                n.mul_(1 - b3).addcmul_(g + (1 - b2) * diff, g + (1 - b2) * diff, value=b3)
                m_hat = m / (1 - (1 - b1) ** t)
                v_hat = v / (1 - (1 - b2) ** t)
                n_hat = n / (1 - (1 - b3) ** t)
                step = group["lr"] / (n_hat.sqrt() + group["eps"])
                p.sub_(step * (m_hat + (1 - b2) * v_hat))
                p.div_(1 + group["lr"] * group["weight_decay"])
                state["prev"].copy_(g)
        return loss
''',
    "AdEMAMix": '''\
class AdEMAMix(torch.optim.Optimizer):
    """AdEMAMix (Pagliardini et al., 2024): AdamW plus alpha x a slow EMA of the gradients
    (beta3), with optional warm-ups for alpha and beta3."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999, 0.9999),
                 alpha: float = 5.0, warmup_steps: int = 0, eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, alpha=alpha, eps=eps,
                                      warmup_steps=warmup_steps, weight_decay=weight_decay))

    @staticmethod
    def _beta3(beta1: float, beta3: float, frac: float) -> float:
        """The paper's scheduler: from beta1 to beta3, linear in 1 / log(beta)."""
        if frac >= 1 or beta3 <= beta1:
            return beta3
        lb1, lb3 = math.log(beta1), math.log(beta3)
        return min(math.exp(lb1 * lb3 / ((1 - frac) * lb3 + frac * lb1)), beta3)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state["step"] = 0
                    state["m1"], state["m2"], state["nu"] = (torch.zeros_like(p) for _ in range(3))
                state["step"] += 1
                t = state["step"]
                frac = t / group["warmup_steps"] if group["warmup_steps"] else 1.0
                alpha = group["alpha"] * min(frac, 1.0)
                b3 = self._beta3(beta1, beta3, frac)
                state["m1"].mul_(beta1).add_(g, alpha=1 - beta1)
                state["m2"].mul_(b3).add_(g, alpha=1 - b3)
                state["nu"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m1_hat = state["m1"] / (1 - beta1 ** t)
                denom = (state["nu"] / (1 - beta2 ** t)).sqrt().add_(group["eps"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.addcdiv_(m1_hat + alpha * state["m2"], denom, value=-group["lr"])
        return loss
''',
    "LookaheadAdamW": '''\
class LookaheadAdamW(torch.optim.AdamW):
    """Lookahead (Zhang et al., 2019) around AdamW: every k steps the slow weights move
    alpha of the way to the fast ones, and the fast weights restart from there."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, k: int = 5, alpha: float = 0.5) -> None:
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.k, self.alpha, self.fast_steps = k, alpha, 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.fast_steps += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "slow" not in state:
                    state["slow"] = p.detach().clone()
                elif self.fast_steps % self.k == 0:
                    state["slow"].add_(p - state["slow"], alpha=self.alpha)
                    p.copy_(state["slow"])
        return loss
''',
    # 3.7
    "AsymmetricLoss": '''\
class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label logits (Ridnik et al., 2021): focal terms with
    separate gammas for positives and negatives, and a probability margin that ignores
    easy negatives."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        target = target.to(p.dtype)
        neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        pos_term = target * (1 - p) ** self.gamma_pos * torch.log(p.clamp(min=1e-8))
        neg_term = (1 - target) * neg ** self.gamma_neg * torch.log((1 - neg).clamp(min=1e-8))
        return -(pos_term + neg_term).mean()
''',
    "SSIMLoss": '''\
class SSIMLoss(nn.Module):
    """alpha x (1 - SSIM) + (1 - alpha) x L1 (Zhao et al., 2017). SSIM uses uniform
    windows on images [N, C, H, W]; other shapes compare each example's whole signal."""

    def __init__(self, alpha: float = 0.84, window: int = 7) -> None:
        super().__init__()
        self.alpha, self.window = alpha, window

    def ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lo = torch.minimum(x.min(), y.min()).detach()
        rng = (torch.maximum(x.max(), y.max()).detach() - lo).clamp(min=1e-6)
        c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
        if x.dim() == 4 and min(x.shape[-2:]) >= self.window:
            pool = lambda t: nn.functional.avg_pool2d(t, self.window, 1)
        else:
            x, y = x.reshape(len(x), 1, -1), y.reshape(len(y), 1, -1)
            pool = lambda t: t.mean(-1, keepdim=True)
        mx, my = pool(x), pool(y)
        vx, vy, cov = pool(x * x) - mx ** 2, pool(y * y) - my ** 2, pool(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
        return s.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(pred.dtype).reshape(pred.shape)
        l1 = (pred - target).abs().mean()
        return self.alpha * (1 - self.ssim(pred, target)) + (1 - self.alpha) * l1
''',
    "VICRegLoss": _PROTOTYPES + '''

def twin_views(emb: torch.Tensor, target: torch.Tensor, paired: bool):
    """The two views of a paired batch (first half, second half). An unpaired batch (the
    test set) pairs each example with its class's mean embedding instead."""
    n = len(emb)
    if paired and n % 2 == 0 and n > 1:
        return emb[: n // 2], emb[n // 2:]
    means = torch.zeros_like(emb)
    for c in target.unique():
        means[target == c] = emb[target == c].mean(0)
    return emb, means


class VICRegLoss(ClassPrototypes):
    """VICReg (Bardes et al., 2022): invariance (MSE between the views), variance (each
    feature's std across the batch kept above 1) and covariance (off-diagonal covariance
    pushed to 0)."""

    def __init__(self, embed_dim: int, num_classes: int, invariance: float = 25.0,
                 variance: float = 25.0, covariance: float = 1.0) -> None:
        super().__init__(embed_dim, num_classes)
        self.weights = (invariance, variance, covariance)
        self.paired = True

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        emb = emb.float()
        if self.training:
            self.update(emb, target)
        a, b = twin_views(emb, target, self.paired)
        inv = nn.functional.mse_loss(a, b)
        var = cov = emb.new_zeros(())
        if len(a) > 1:
            for z in (a, b):
                z = z - z.mean(0)
                var = var + torch.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean() / 2
                c = z.t() @ z / (len(z) - 1)
                off = c - torch.diag(torch.diag(c))
                cov = cov + off.pow(2).sum() / z.shape[1] / 2
        wi, wv, wc = self.weights
        return wi * inv + wv * var + wc * cov
''',
    "BarlowTwinsLoss": _PROTOTYPES + '''

def twin_views(emb: torch.Tensor, target: torch.Tensor, paired: bool):
    """The two views of a paired batch (first half, second half). An unpaired batch (the
    test set) pairs each example with its class's mean embedding instead."""
    n = len(emb)
    if paired and n % 2 == 0 and n > 1:
        return emb[: n // 2], emb[n // 2:]
    means = torch.zeros_like(emb)
    for c in target.unique():
        means[target == c] = emb[target == c].mean(0)
    return emb, means


class BarlowTwinsLoss(ClassPrototypes):
    """Barlow Twins (Zbontar et al., 2021): the cross-correlation of the standardized
    embeddings of the two views is pushed towards the identity matrix."""

    def __init__(self, embed_dim: int, num_classes: int, off_diagonal: float = 5e-3) -> None:
        super().__init__(embed_dim, num_classes)
        self.off_diagonal = off_diagonal
        self.paired = True

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        emb = emb.float()
        if self.training:
            self.update(emb, target)
        a, b = twin_views(emb, target, self.paired)
        if len(a) < 2:
            return (emb * 0).sum()
        a = (a - a.mean(0)) / (a.std(0) + 1e-5)
        b = (b - b.mean(0)) / (b.std(0) + 1e-5)
        c = a.t() @ b / len(a)
        diag = torch.diagonal(c)
        return (1 - diag).pow(2).sum() + self.off_diagonal * (c.pow(2).sum() - diag.pow(2).sum())
''',
    "Prodigy": '''\
class Prodigy(torch.optim.Optimizer):
    """Prodigy (Mishchenko & Defazio, 2023): Adam whose step size d is estimated on the
    fly from how far the weights have moved along the gradients. Keep lr at 1."""

    def __init__(self, params, lr: float = 1.0, betas=(0.9, 0.999), d_coef: float = 1.0,
                 eps: float = 1e-8, weight_decay: float = 0.0, d0: float = 1e-6) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, d_coef=d_coef, eps=eps,
                                      weight_decay=weight_decay, d=d0, d0=d0, d_numerator=0.0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            b3 = b2 ** 0.5
            d, d0, lr = group["d"], group["d0"], group["lr"]
            dlr = d * lr
            numerator = group["d_numerator"] * b3
            denom = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state["p0"] = p.detach().clone()
                    state["s"], state["m"], state["v"] = (torch.zeros_like(p) for _ in range(3))
                numerator += (d / d0) * dlr * float(torch.dot(g.flatten(),
                                                              (state["p0"] - p).flatten()))
                state["m"].mul_(b1).add_(g, alpha=d * (1 - b1))
                state["v"].mul_(b2).addcmul_(g, g, value=d * d * (1 - b2))
                state["s"].mul_(b3).add_(g, alpha=(d / d0) * dlr)
                denom += float(state["s"].abs().sum())
            group["d_numerator"] = numerator
            if denom > 0:
                group["d"] = d = max(d, group["d_coef"] * numerator / denom)
            dlr = d * lr
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["weight_decay"]:
                    p.add_(p, alpha=-group["weight_decay"] * dlr)
                p.addcdiv_(state["m"], state["v"].sqrt().add_(d * group["eps"]), value=-dlr)
        return loss
''',
    "CautiousAdamW": '''\
class CautiousAdamW(torch.optim.Optimizer):
    """Cautious AdamW (Liang et al., 2024): AdamW, but each coordinate of the update is
    applied only where it agrees in sign with the current gradient (rescaled so the
    average step size is kept)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state["step"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                update = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + group["eps"])
                mask = (update * g > 0).to(g.dtype)
                mask = mask / mask.mean().clamp(min=1e-3)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update * mask, alpha=-group["lr"])
        return loss
''',
    "Yogi": '''\
class Yogi(torch.optim.Optimizer):
    """Yogi (Zaheer et al., 2018): Adam with an additive second-moment update,
    v <- v - (1 - beta2) * sign(v - g^2) * g^2."""

    def __init__(self, params, lr: float = 1e-2, betas=(0.9, 0.999), eps: float = 1e-3,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if group["weight_decay"]:
                    g = g.add(p, alpha=group["weight_decay"])
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.full_like(p, 1e-6)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                g2 = g * g
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.addcmul_(torch.sign(v - g2), g2, value=-(1 - b2))
                denom = (v / (1 - b2 ** t)).sqrt().add_(group["eps"])
                p.addcdiv_(m, denom, value=-group["lr"] / (1 - b1 ** t))
        return loss
''',
    "ADOPT": '''\
class ADOPT(torch.optim.Optimizer):
    """ADOPT (Taniguchi et al., 2024): the gradient is divided by the second moment from
    the step before (clipped to +-step^0.25), then averaged into momentum; the first step
    only records g^2. Weight decay is decoupled, as in AdamW."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.9999), eps: float = 1e-6,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g, state = p.grad, self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = g * g
                    continue
                state["step"] += 1
                m, v = state["m"], state["v"]
                normed = g / v.sqrt().clamp(min=group["eps"])
                limit = state["step"] ** 0.25
                m.mul_(b1).add_(normed.clamp(-limit, limit), alpha=1 - b1)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(m, alpha=-group["lr"])
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
        return loss
''',
}
