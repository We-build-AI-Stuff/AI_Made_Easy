"""Learning-rate schedulers."""
from __future__ import annotations

from ai_made_easy.core.spec import P
from ai_made_easy.core.training.catalog import Component, Render, _ints


# meta: step = epoch | batch | metric

def _milestones_check(p):
    try:
        ms = _ints(p.get("milestones", ""))
    except ValueError:
        return [("error", "milestones must be comma-separated integers")]
    if ms != sorted(ms) or len(set(ms)) != len(ms):
        return [("error", "milestones must be strictly increasing")]
    return []


def _cyclic_check(p):
    if float(p["base_lr"]) >= float(p["max_lr"]):
        return [("error", "base_lr must be smaller than max_lr")]
    return []


def _keras_closed_form(expr: str) -> Render:
    return lambda p: f"lambda epoch, lr: {expr.format(**p)}"


SCHEDULERS: list[Component] = [
    Component(
        "train.step_lr", "Step LR",
        (P("step_size", "int", 10, lo=1, help="Epochs between decays"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.StepLR(optimizer, step_size={p['step_size']}, "
                         f"gamma={p['gamma']})"),
        keras=_keras_closed_form("LR0 * {gamma} ** (epoch // {step_size})"),
        desc="Multiply the LR by gamma every step_size epochs.", meta={"step": "epoch"}),
    Component(
        "train.multistep_lr", "Multi-Step LR",
        (P("milestones", "str", "30, 60", help="Epochs at which to decay"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.MultiStepLR(optimizer, "
                         f"milestones={_ints(p['milestones'])}, gamma={p['gamma']})"),
        keras=lambda p: (f"lambda epoch, lr: LR0 * {p['gamma']} ** "
                         f"sum(epoch >= m for m in {_ints(p['milestones'])})"),
        desc="Decay by gamma at the given epochs.", meta={"step": "epoch"},
        checks=_milestones_check),
    Component(
        "train.exponential_lr", "Exponential LR",
        (P("gamma", "float", 0.95, lo=0.0, hi=1.0),),
        torch=lambda p: f"torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma={p['gamma']})",
        keras=_keras_closed_form("LR0 * {gamma} ** epoch"),
        desc="Multiply the LR by gamma every epoch.", meta={"step": "epoch"}),
    Component(
        "train.cosine_annealing_lr", "Cosine Annealing",
        (P("T_max", "int", 50, lo=1, help="Epochs per half cosine"),
         P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
                         f"T_max={p['T_max']}, eta_min={p['eta_min']})"),
        keras=_keras_closed_form(
            "{eta_min} + (LR0 - {eta_min}) * (1 + math.cos(math.pi * min(epoch, {T_max}) "
            "/ {T_max})) / 2"),
        desc="Cosine decay from the initial LR to eta_min.", meta={"step": "epoch"}),
    Component(
        "train.warm_restarts_lr", "Cosine Warm Restarts",
        (P("T_0", "int", 10, lo=1), P("T_mult", "int", 1, lo=1),
         P("eta_min", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, "
                         f"T_0={p['T_0']}, T_mult={p['T_mult']}, eta_min={p['eta_min']})"),
        desc="SGDR: cosine cycles that restart, optionally growing.", meta={"step": "epoch"}),
    Component(
        "train.warmup_cosine_lr", "Warmup + Cosine",
        (P("warmup_epochs", "int", 5, lo=0), P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (
            "torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=["
            f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, "
            f"total_iters={max(int(p['warmup_epochs']), 1)}), "
            f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
            f"T_max=max(EPOCHS - {p['warmup_epochs']}, 1), eta_min={p['eta_min']})], "
            f"milestones=[{max(int(p['warmup_epochs']), 1)}])"),
        keras=lambda p: (
            f"lambda epoch, lr: LR0 * (epoch + 1) / {max(int(p['warmup_epochs']), 1)} "
            f"if epoch < {p['warmup_epochs']} else {p['eta_min']} + (LR0 - {p['eta_min']}) * "
            f"(1 + math.cos(math.pi * (epoch - {p['warmup_epochs']}) / "
            f"max(EPOCHS - {p['warmup_epochs']}, 1))) / 2"),
        desc="Linear warmup, then cosine decay over the remaining epochs.",
        meta={"step": "epoch"}),
    Component(
        "train.linear_lr", "Linear LR",
        (P("start_factor", "float", 0.1, lo=1e-8, hi=1.0),
         P("end_factor", "float", 1.0, lo=0.0, hi=1.0),
         P("total_iters", "int", 5, lo=1, help="Epochs of the linear ramp")),
        torch=lambda p: (f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor="
                         f"{p['start_factor']}, end_factor={p['end_factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form(
            "LR0 * ({start_factor} + ({end_factor} - {start_factor}) * "
            "min(epoch, {total_iters}) / {total_iters})"),
        desc="Linearly ramp the LR factor (warmup or decay).", meta={"step": "epoch"}),
    Component(
        "train.constant_lr", "Constant LR (warm phase)",
        (P("factor", "float", 0.333, lo=0.0, hi=1.0), P("total_iters", "int", 5, lo=1)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ConstantLR(optimizer, factor={p['factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form("LR0 * ({factor} if epoch < {total_iters} else 1.0)"),
        desc="Scale the LR by a constant factor for the first epochs.", meta={"step": "epoch"}),
    Component(
        "train.polynomial_lr", "Polynomial LR",
        (P("total_iters", "int", 50, lo=1), P("power", "float", 1.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.PolynomialLR(optimizer, "
                         f"total_iters={p['total_iters']}, power={p['power']})"),
        keras=_keras_closed_form(
            "LR0 * (1 - min(epoch, {total_iters}) / {total_iters}) ** {power}"),
        desc="Polynomial decay to zero over total_iters epochs.", meta={"step": "epoch"}),
    Component(
        "train.plateau_lr", "Reduce on Plateau",
        (P("factor", "float", 0.5, lo=0.0, hi=0.999), P("patience", "int", 5, lo=0),
         P("min_lr", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\"min\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        keras=lambda p: (f"keras.callbacks.ReduceLROnPlateau(monitor=\"val_loss\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        desc="Reduce the LR when validation loss stops improving.", meta={"step": "metric"}),
    Component(
        "train.one_cycle_lr", "One-Cycle",
        (P("max_lr", "float", 1e-2, lo=1e-8), P("pct_start", "float", 0.3, lo=0.0, hi=1.0),
         P("anneal_strategy", "enum", "cos", options=("cos", "linear"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={p['max_lr']}, "
                         f"total_steps=EPOCHS * len(train_loader), pct_start={p['pct_start']}, "
                         f"anneal_strategy=\"{p['anneal_strategy']}\")"),
        desc="Super-convergence schedule stepped every batch.", meta={"step": "batch"}),
    Component(
        "train.cyclic_lr", "Cyclic LR",
        (P("base_lr", "float", 1e-4, lo=0.0), P("max_lr", "float", 1e-2, lo=0.0),
         P("step_size_up", "int", 2000, lo=1, help="Batches per half cycle"),
         P("mode", "enum", "triangular", options=("triangular", "triangular2", "exp_range"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr={p['base_lr']}, "
                         f"max_lr={p['max_lr']}, step_size_up={p['step_size_up']}, "
                         f"mode=\"{p['mode']}\")"),
        desc="Cycle the LR between two bounds every batch.", meta={"step": "batch"},
        checks=_cyclic_check),
    # 3.6
    Component(
        "train.inverse_sqrt_lr", "Inverse Square Root",
        (P("warmup_epochs", "int", 4, lo=1, help="Linear warm-up, then decay as 1/sqrt"),),
        torch=lambda p: (f"torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: "
                         f"min((e + 1) / {int(p['warmup_epochs'])}, "
                         f"({int(p['warmup_epochs'])} / (e + 1)) ** 0.5))"),
        keras=lambda p: (f"lambda epoch, lr: LR0 * min((epoch + 1) / {int(p['warmup_epochs'])}, "
                         f"({int(p['warmup_epochs'])} / (epoch + 1)) ** 0.5)"),
        desc="The original Transformer's schedule (Vaswani et al., 2017): a linear warm-up, "
             "then the learning rate falls as 1/√epoch. Never reaches zero, so training can "
             "go on.", meta={"step": "epoch"}),
    Component(
        "train.wsd_lr", "Warmup-Stable-Decay",
        (P("warmup_epochs", "int", 2, lo=0), P("decay_fraction", "float", 0.2, lo=0.01, hi=1.0,
                                                help="Share of the run spent decaying"),
         P("final_factor", "float", 0.0, lo=0.0, hi=1.0, help="LR at the end, as a fraction")),
        torch=lambda p: (
            "torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e, w="
            f"{int(p['warmup_epochs'])}, d=max(1, round(EPOCHS * {float(p['decay_fraction'])})), "
            f"f={float(p['final_factor'])}: (e + 1) / w if e < w else 1.0 if e < EPOCHS - d "
            "else f + (1 - f) * (EPOCHS - e - 1) / d)"),
        keras=lambda p: (
            f"lambda epoch, lr, w={int(p['warmup_epochs'])}, "
            f"d=max(1, round(EPOCHS * {float(p['decay_fraction'])})), "
            f"f={float(p['final_factor'])}: LR0 * ((epoch + 1) / w if epoch < w else 1.0 "
            "if epoch < EPOCHS - d else f + (1 - f) * (EPOCHS - epoch - 1) / d)"),
        desc="Warm up, hold the learning rate, and decay only at the end (used to train "
             "MiniCPM, DeepSeek and other LLMs): you can stop or extend a run without "
             "planning its length in advance.", meta={"step": "epoch"}),
]
