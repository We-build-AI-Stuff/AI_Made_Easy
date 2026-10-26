"""Optimizers."""
from __future__ import annotations

from ai_made_easy.core.spec import P
from ai_made_easy.core.training.catalog import Component


def _wd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


def _kwd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


_LR = lambda default: P("lr", "float", default, lo=0.0, help="Learning rate")  # noqa: E731
_B1 = P("beta1", "float", 0.9, lo=0.0, hi=0.999999)
_B2 = P("beta2", "float", 0.999, lo=0.0, hi=0.999999)
_EPS = lambda d: P("eps", "float", d, lo=0.0)  # noqa: E731
_WD = lambda d=0.0: P("weight_decay", "float", d, lo=0.0)  # noqa: E731


def _sgd_checks(p):
    if p.get("nesterov") and float(p.get("momentum", 0)) <= 0:
        return [("error", "Nesterov momentum requires momentum > 0")]
    return []


OPTIMIZERS: list[Component] = [
    Component(
        "train.sgd", "SGD",
        (_LR(1e-2), P("momentum", "float", 0.9, lo=0.0, hi=0.999999),
         P("nesterov", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.SGD(model.parameters(), lr={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.SGD(learning_rate={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_kwd(p)})"),
        desc="Stochastic gradient descent with optional (Nesterov) momentum.",
        checks=_sgd_checks),
    Component(
        "train.adam", "Adam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.Adam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.Adam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + f"{_kwd(p)})"),
        desc="Adaptive moment estimation."),
    Component(
        "train.adamw", "AdamW",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(1e-2), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.AdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.AdamW(learning_rate={p['lr']}, "
                         f"weight_decay={p['weight_decay']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        desc="Adam with decoupled weight decay."),
    Component(
        "train.nadam", "NAdam",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.NAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Nadam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with Nesterov momentum."),
    Component(
        "train.radam", "RAdam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.RAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        desc="Rectified Adam (variance-adaptive warmup)."),
    Component(
        "train.adamax", "Adamax",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.Adamax(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adamax(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam variant based on the infinity norm."),
    Component(
        "train.adagrad", "Adagrad",
        (_LR(1e-2), _EPS(1e-10), _WD()),
        torch=lambda p: (f"torch.optim.Adagrad(model.parameters(), lr={p['lr']}, "
                         f"eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adagrad(learning_rate={p['lr']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Per-parameter learning rates from accumulated squared gradients."),
    Component(
        "train.adadelta", "Adadelta",
        (_LR(1.0), P("rho", "float", 0.9, lo=0.0, hi=1.0), _EPS(1e-6), _WD()),
        torch=lambda p: (f"torch.optim.Adadelta(model.parameters(), lr={p['lr']}, "
                         f"rho={p['rho']}, eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adadelta(learning_rate={p['lr']}, rho={p['rho']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Adagrad extension with a moving window of gradient updates."),
    Component(
        "train.rmsprop", "RMSprop",
        (_LR(1e-2), P("alpha", "float", 0.99, lo=0.0, hi=1.0), _EPS(1e-8),
         P("momentum", "float", 0.0, lo=0.0, hi=0.999999), P("centered", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.RMSprop(model.parameters(), lr={p['lr']}, "
                         f"alpha={p['alpha']}, eps={p['eps']}, momentum={p['momentum']}, "
                         f"centered={p['centered']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.RMSprop(learning_rate={p['lr']}, rho={p['alpha']}, "
                         f"momentum={p['momentum']}, epsilon={p['eps']}, "
                         f"centered={p['centered']}{_kwd(p)})"),
        desc="Divides the gradient by a running RMS of recent magnitudes."),
    Component(
        "train.adafactor", "Adafactor",
        (_LR(1e-2), _WD()),
        torch=lambda p: f"torch.optim.Adafactor(model.parameters(), lr={p['lr']}{_wd(p)})",
        keras=lambda p: f"keras.optimizers.Adafactor(learning_rate={p['lr']}{_kwd(p)})",
        desc="Memory-efficient adaptive optimizer with factored second moments."),
    Component(
        "train.asgd", "ASGD",
        (_LR(1e-2), P("lambd", "float", 1e-4, lo=0.0), P("alpha", "float", 0.75, lo=0.0),
         P("t0", "float", 1e6, lo=0.0), _WD()),
        torch=lambda p: (f"torch.optim.ASGD(model.parameters(), lr={p['lr']}, "
                         f"lambd={p['lambd']}, alpha={p['alpha']}, t0={p['t0']}{_wd(p)})"),
        desc="Averaged stochastic gradient descent."),
    Component(
        "train.rprop", "Rprop",
        (_LR(1e-2), P("eta_minus", "float", 0.5, lo=0.0, hi=1.0),
         P("eta_plus", "float", 1.2, lo=1.0)),
        torch=lambda p: (f"torch.optim.Rprop(model.parameters(), lr={p['lr']}, "
                         f"etas=({p['eta_minus']}, {p['eta_plus']}))"),
        desc="Resilient backpropagation (sign-based, full-batch friendly)."),
    Component(
        "train.lion", "Lion",
        (_LR(1e-4), _B1, P("beta2", "float", 0.99, lo=0.0, hi=0.999999), _WD(1e-2)),
        torch=lambda p: (f"Lion(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lion(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}{_kwd(p)})"),
        desc="Evolved sign momentum (Chen et al., 2023): one state per weight, half Adam's "
             "memory. Use a learning rate about 10x smaller than AdamW's.",
        meta={"torch_helper": "Lion"}),
    Component(
        "train.muon", "Muon",
        (_LR(0.02), P("momentum", "float", 0.95, lo=0.0, hi=0.999),
         P("nesterov", "bool", True), P("ns_steps", "int", 5, lo=1, hi=10,
                                        help="Newton–Schulz steps per update"),
         P("adamw_lr", "float", 3e-4, lo=0.0, help="Learning rate of the AdamW part "
                                                    "(biases, norms, other 1-D weights)"),
         _WD(0.0)),
        torch=lambda p: (f"Muon(model.parameters(), lr={p['lr']}, momentum={p['momentum']}, "
                         f"nesterov={bool(p['nesterov'])}, ns_steps={int(p['ns_steps'])}, "
                         f"adamw_lr={p['adamw_lr']}, weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Muon(learning_rate={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={bool(p['nesterov'])}, "
                         f"ns_steps={int(p['ns_steps'])}, "
                         f"adam_lr_ratio={float(p['adamw_lr']) / max(float(p['lr']), 1e-12)}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Muon (Jordan et al., 2024): momentum whose update is orthogonalized by a few "
             "Newton–Schulz steps for every weight matrix (and conv kernel), AdamW for the "
             "rest. Fast on transformers and CNNs; Keras 3.10 or later.",
        meta={"torch_helper": "Muon"}),
    Component(
        "train.lbfgs", "L-BFGS",
        (_LR(1.0), P("max_iter", "int", 20, lo=1, hi=200, help="Iterations per batch"),
         P("history_size", "int", 100, lo=1, hi=1000, help="Past updates kept to estimate "
                                                           "the curvature"),
         P("line_search", "bool", True, help="Strong-Wolfe line search (steadier)")),
        torch=lambda p: (f"torch.optim.LBFGS(model.parameters(), lr={p['lr']}, "
                         f"max_iter={int(p['max_iter'])}, "
                         f"history_size={int(p['history_size'])}, line_search_fn="
                         + ("\"strong_wolfe\")" if p.get("line_search") else "None)")),
        desc="Limited-memory BFGS: a quasi-Newton method that estimates the curvature from "
             "recent gradients. Converges in few steps on small, smooth problems (tabular "
             "regression, physics fits); use large batches or the whole dataset as one "
             "batch. Re-evaluates the loss several times per step; full precision.",
        meta={"closure": True}),
    Component(
        "train.lamb", "LAMB",
        (_LR(1e-3), _B1, _B2, _EPS(1e-6), _WD(1e-2)),
        torch=lambda p: (f"Lamb(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lamb(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with a per-layer trust ratio (You et al., 2020): stays stable at very "
             "large batch sizes.",
        meta={"torch_helper": "Lamb"}),
    Component(
        "train.lookahead_adamw", "Lookahead (AdamW)",
        (_LR(1e-3), _B1, _B2, _WD(1e-2), P("k", "int", 5, lo=1, help="Fast steps per sync"),
         P("alpha", "float", 0.5, lo=0.0, hi=1.0, help="How far the slow weights move")),
        torch=lambda p: (f"LookaheadAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']}, k={int(p['k'])}, "
                         f"alpha={p['alpha']})"),
        desc="AdamW whose weights are pulled toward a slow moving copy every k steps "
             "(Zhang et al., 2019): less sensitive to the learning rate.",
        meta={"torch_helper": "LookaheadAdamW"}),
    Component(
        "train.sam", "SAM (Sharpness-Aware)",
        (P("base", "enum", "sgd", options=("sgd", "adamw"), help="The optimizer SAM wraps"),
         _LR(0.05), P("rho", "float", 0.05, lo=0.0, help="Neighbourhood size"),
         P("momentum", "float", 0.9, lo=0.0, hi=1.0, help="SGD momentum"), _WD(5e-4),
         P("adaptive", "bool", False, help="ASAM: scale the step by each weight's size")),
        torch=lambda p: (f"SAM(model.parameters(), base=\"{p['base']}\", rho={p['rho']}, "
                         f"adaptive={bool(p['adaptive'])}, lr={p['lr']}, "
                         f"momentum={p['momentum']}, weight_decay={p['weight_decay']})"),
        desc="Sharpness-Aware Minimization (Foret et al., 2021): each step first moves to "
             "the worst weights nearby, then descends from there, which finds flatter minima "
             "that generalize better. Two forward/backward passes per step; runs in full "
             "precision.",
        meta={"torch_helper": "SAM", "sam": True}),
    Component(
        "train.schedule_free_adamw", "Schedule-Free AdamW",
        (_LR(2.5e-3), _B1, _B2, _EPS(1e-8), _WD(0.0),
         P("warmup_steps", "int", 0, lo=0, help="Linear warm-up steps")),
        torch=lambda p: (f"ScheduleFreeAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}, "
                         f"warmup_steps={int(p['warmup_steps'])})"),
        desc="AdamW without a learning-rate schedule (Defazio et al., 2024): it averages "
             "its iterates instead, and evaluates the average. Leave out the scheduler.",
        meta={"torch_helper": "ScheduleFreeAdamW", "schedule_free": True}),
]

# 3.6: newer adaptive optimizers
OPTIMIZERS += [
    Component(
        "train.adabelief", "AdaBelief",
        (_LR(1e-3), _B1, _B2, _EPS(1e-16), _WD(0.0)),
        torch=lambda p: (f"AdaBelief(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adam that scales each step by how far the gradient is from its running "
             "average, its \"belief\" (Zhuang et al., 2020): big steps when gradients agree, "
             "small ones when they are noisy. Adam's speed, closer to SGD's generalization.",
        meta={"torch_helper": "AdaBelief"}),
    Component(
        "train.adan", "Adan",
        (_LR(1e-3), P("beta1", "float", 0.02, lo=0.0, hi=1.0, help="Momentum weight of the "
                                                                    "new gradient"),
         P("beta2", "float", 0.08, lo=0.0, hi=1.0, help="Weight of the gradient difference"),
         P("beta3", "float", 0.01, lo=0.0, hi=1.0, help="Weight of the new squared term"),
         _EPS(1e-8), _WD(0.02)),
        torch=lambda p: (f"Adan(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}, {p['beta3']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adaptive Nesterov momentum (Xie et al., 2022): estimates where the gradient is "
             "heading from the change between steps. Strong on vision transformers, CNNs and "
             "language models; tolerates large learning rates and batches.",
        meta={"torch_helper": "Adan"}),
    Component(
        "train.ademamix", "AdEMAMix",
        (_LR(1e-3), _B1, _B2, P("beta3", "float", 0.9999, lo=0.0, hi=0.999999,
                                help="Decay of the slow momentum (remembers ~10k steps)"),
         P("alpha", "float", 5.0, lo=0.0, hi=20.0, help="Weight of the slow momentum"),
         P("warmup_steps", "int", 0, lo=0, help="Steps over which alpha and beta3 ramp up "
                                                "(0 = off; the paper uses the whole run)"),
         _EPS(1e-8), _WD(0.0)),
        torch=lambda p: (f"AdEMAMix(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}, {p['beta3']}), "
                         f"alpha={p['alpha']}, warmup_steps={int(p['warmup_steps'])}, "
                         f"eps={p['eps']}, weight_decay={p['weight_decay']})"),
        desc="AdamW with a second, very slow momentum mixed in (Pagliardini et al., 2024): "
             "old gradients keep helping for thousands of steps, so long runs reach a lower "
             "loss. Use a warm-up for alpha and beta3 when training from scratch.",
        meta={"torch_helper": "AdEMAMix"}),
]
# 3.7
OPTIMIZERS += [
    Component(
        "train.prodigy", "Prodigy (learning-rate free)",
        (_LR(1.0), _B1, _B2, P("d_coef", "float", 1.0, lo=0.01, hi=100.0,
                               help="Scales the learning rate Prodigy finds"),
         _EPS(1e-8), _WD(0.0)),
        torch=lambda p: (f"Prodigy(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), d_coef={p['d_coef']}, "
                         f"eps={p['eps']}, weight_decay={p['weight_decay']})"),
        desc="Adam that estimates its own step size while it trains (Mishchenko & Defazio, "
             "2023): leave lr at 1 and it finds the right scale, so there is no learning "
             "rate to tune. Pair it with a cosine schedule.",
        meta={"torch_helper": "Prodigy"}),
    Component(
        "train.cautious_adamw", "Cautious AdamW",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(0.01)),
        torch=lambda p: (f"CautiousAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="AdamW that skips the parts of a step pointing against the current gradient "
             "(Liang et al., 2024): a one-line change that speeds up training of language "
             "and vision models.",
        meta={"torch_helper": "CautiousAdamW"}),
    Component(
        "train.yogi", "Yogi",
        (_LR(1e-2), _B1, _B2, _EPS(1e-3), _WD(0.0)),
        torch=lambda p: (f"Yogi(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="Adam whose second moment changes additively, not multiplicatively (Zaheer et "
             "al., 2018), so a burst of large gradients does not shrink the step for long. "
             "Steadier on sparse or noisy gradients.",
        meta={"torch_helper": "Yogi"}),
    Component(
        "train.adopt", "ADOPT",
        (_LR(1e-3), _B1, P("beta2", "float", 0.9999, lo=0.0, hi=0.99999), _EPS(1e-6),
         _WD(0.0)),
        torch=lambda p: (f"ADOPT(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        desc="ADOPT (Taniguchi et al., 2024): Adam that normalizes each gradient by the "
             "second moment from before it and applies momentum after that, which makes it "
             "converge for any beta2. A drop-in for Adam and AdamW, often steadier.",
        meta={"torch_helper": "ADOPT"}),
]

# PyTorch optimizers and losses that are not in torch: see train_helpers.py
