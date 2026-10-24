"""PyTorch training-script template: the entry point, k-fold and final report (part of
``torch_template``)."""

TORCH_MAIN = r'''

def _inputs(loader: DataLoader, limit: int) -> tuple:
    xs, ys, seen = [], [], 0
    for xb, yb in loader:
        if not torch.is_tensor(xb):
            break
{% if two_views %}
        xb, yb, _paired = views(xb, yb)
{% endif %}
        xs.append(xb.cpu().numpy())
        ys.append(yb.cpu().numpy() if torch.is_tensor(yb) else np.asarray(yb))
        seen += len(xb)
        if seen >= limit:
            break
    if not xs:
        return None, None
    x, y = np.concatenate(xs), np.concatenate(ys)
    keep = max(4, min(limit, int(4e6 // max(x[0].nbytes, 1))))   # about 4 MB at most
    return x[:keep], y[:keep]


def save_calibration(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """A few training inputs (int8 calibration for exports) and labelled test inputs
    (exports are checked against the model on them)."""
    try:
        x, _ = _inputs(train_loader, 128)
        x_eval, y_eval = _inputs(test_loader, 256)
        if x is None:
            return
        arrays = {"x": x}
        if x_eval is not None:
            arrays["x_eval"] = x_eval
{% if spec.task == "multiclass" %}
            if y_eval.ndim == 1:
                arrays["y_eval"] = y_eval.astype(np.int64)
{% endif %}
        np.savez_compressed("calibration.npz", **arrays)
    except Exception as exc:  # noqa: BLE001 — exports fall back to random inputs
        print(f"(no calibration sample: {exc})")


def main() -> None:
    if GPUS > 1 and WORLD == 1:
        train_together(GPUS)
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
{% if spec.trainer.deterministic %}
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
{% endif %}
    device = pick_device()
    print(f"device: {device}")
{% if spec.kfold %}
    cv = cross_validate(device)
    if not REFIT:
        return
    print("refit: training the final model")
{% else %}
    cv = {}
{% endif %}
    train_loader, val_loader, test_loader = make_loaders()
    print(f"samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")
    model = {{ spec.class_name }}().to(device)
    if INIT_HOOK is not None:
        INIT_HOOK(model)
    if PREPARE_HOOK is not None:
        model = PREPARE_HOOK(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if CHECKPOINTING:
        print(f"gradient checkpointing: {checkpoint_layers(model)} layers recompute their "
              "activations")
    model, loss_fn = fit(model, train_loader, val_loader, device)
    if RANK:   # the first process saves and reports
        return
    if BEFORE_SAVE is not None:
        BEFORE_SAVE(model)
    torch.save(model.state_dict(), CHECKPOINT)
{% if metric_loss %}
    # the loss's class centres / averages score embeddings for serving (see infer)
    INFERENCE_STATE["scorer"] = {k: v.detach().cpu() for k, v in loss_fn.state_dict().items()}
{% endif %}
    save_inference_state()
    save_calibration(train_loader, test_loader)
    print(f"saved best weights to {CHECKPOINT}")
    test_loss, test_metrics = evaluate(model, loss_fn, test_loader, device)
    report = {"loss": test_loss, **{k: v for k, v in test_metrics.items() if isinstance(v, float)}}
{% if prob %}
    report.update(probabilistic_report(model, train_loader, val_loader, test_loader, device))
    save_inference_state()      # temperature, conformal quantiles, Laplace posterior
{% endif %}
    print("test: " + " ".join(f"{k}={v:.4f}" for k, v in report.items()))
    final = {**report, **test_metrics, **{f"cv_{m}": v for m, v in cv.items()}}
    Path("metrics.json").write_text(json.dumps(
        {k: v for k, v in final.items() if isinstance(v, (float, list))}, indent=1), encoding="utf-8")
    if RESUME_FILE:             # trained and scored: nothing to resume
        Path(RESUME_FILE).unlink(missing_ok=True)
        Path(RESUME_FILE).with_suffix(".json").unlink(missing_ok=True)
{% if spec.predict %}
    preview_predictions(model, test_loader, device)
{% endif %}
{% if spec.explain %}
    try:
        explain_features(model, loss_fn, test_loader if len(test_loader.dataset) else val_loader,
                         device{{ ", background_loader=train_loader" if spec.explain.method == "shap" else "" }})
    except Exception as exc:  # noqa: BLE001 — an explanation is optional
        print(f"(could not explain the features: {exc})")
{% endif %}
{% if spec.gradcam %}
    try:
        save_grad_cams(model, test_loader if len(test_loader.dataset) else val_loader, device)
    except Exception as exc:  # noqa: BLE001 — heatmaps are optional
        print(f"(could not draw Grad-CAM heatmaps: {exc})")
{% endif %}
{% if spec.attention %}
    try:
        save_attention_maps(model, test_loader if len(test_loader.dataset) else val_loader,
                            device)
    except Exception as exc:  # noqa: BLE001 — maps are optional
        print(f"(could not draw attention maps: {exc})")
{% endif %}
{% if spec.saliency %}
    try:
        save_audio_saliency(model, test_loader if len(test_loader.dataset) else val_loader,
                            device)
    except Exception as exc:  # noqa: BLE001 — maps are optional
        print(f"(could not draw audio saliency: {exc})")
{% endif %}
{% if spec.is_classification %}
    try:
        dump_test_results(model, test_loader, device)
    except Exception as exc:  # noqa: BLE001 — artefacts are optional
        print(f"(could not save predictions: {exc})")
{% endif %}
{% if spec.explain %}


def feature_names(n: int) -> list[str]:
    """Input column names after the table preprocessing, else feature 0..n-1."""
    state = INFERENCE_STATE.get("table")
    if state:
        cats = state.get("categories", {})
        names = [c for c in state["columns"] if c not in cats]
        names += [f"{c}={v}" for c, values in cats.items() for v in values]
        if len(names) == n:
            return names
    return [f"feature {i}" for i in range(n)]


{% if spec.explain.method == "shap" %}
def explain_features(model: nn.Module, loss_fn, loader: DataLoader, device: torch.device,
                     background_loader: DataLoader | None = None):
    """SHAP values by expected gradients (shap.GradientExplainer) against training rows."""
    import shap

    model.eval()
    xs = [xb for xb, _ in loader]
    if not xs:
        return
    x = torch.cat(xs)[:{{ spec.explain.max_samples }}].float().to(device)
    pool = torch.cat([xb for xb, _ in background_loader]) if background_loader else x
    gen = torch.Generator().manual_seed(SEED)
    background = pool[torch.randperm(len(pool), generator=gen)[:{{ spec.explain.background }}]]
    background = background.float().to(device)
{% if metric_loss %}
    explained = nn.Sequential(model, ScoreHead())
{% else %}
    explained = model
{% endif %}
    values = shap.GradientExplainer(explained, background).shap_values(x)
    values = np.asarray(values)
    with torch.no_grad():
        out = explained(x)
    if values.ndim == 3:   # (rows, features, outputs): take each row's predicted output
        picked = out.argmax(1).cpu().numpy() if out.shape[1] > 1 else np.zeros(len(x), int)
        values = values[np.arange(len(x)), :, picked]
    magnitude = np.abs(values)
    rows = []
    names = feature_names(x.shape[1])
    for j in range(x.shape[1]):
        rows.append({"name": names[j], "importance": float(magnitude[:, j].mean()),
                     "std": float(magnitude[:, j].std(ddof=1)) if len(x) > 1 else 0.0})
    rows.sort(key=lambda r: r["importance"], reverse=True)
    Path("explain.json").write_text(json.dumps(
        {"method": "shap", "metric": "mean |SHAP value|", "background": len(background),
         "rows": len(x), "features": rows}, indent=1), encoding="utf-8")
    print("SHAP: " + ", ".join(f"{r['name']}={r['importance']:.4f}" for r in rows[:5]))
{% if metric_loss %}


class ScoreHead(nn.Module):
    """Class scores of an embedding, so SHAP explains the predicted class."""

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return SCORER(emb)
{% endif %}
{% elif spec.explain.method == "integrated_gradients" %}
def explain_features(model: nn.Module, loss_fn, loader: DataLoader, device: torch.device):
    """Integrated gradients (Sundararajan et al., 2017): the gradient of the explained
    output, averaged along the straight path from a baseline row to each test row."""
    model.eval()
    xs = [xb for xb, _ in loader]
    if not xs:
        return
    x = torch.cat(xs)[:{{ spec.explain.max_samples }}].float().to(device)
    baseline = {{ "x.mean(0, keepdim=True)" if spec.explain.baseline == "mean" else "torch.zeros_like(x[:1])" }}
    with torch.no_grad():
        out = {{ "SCORER(model(x))" if metric_loss else "model(x)" }}
{% if spec.is_classification %}
    picked = out.argmax(1) if out.shape[1] > 1 else None   # explain the predicted class
{% endif %}
    total = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, {{ spec.explain.steps }}, device=device):
        point = (baseline + alpha * (x - baseline)).requires_grad_(True)
        y = {{ "SCORER(model(point))" if metric_loss else "model(point)" }}
{% if spec.is_classification %}
        y = y.gather(1, picked[:, None]).sum() if picked is not None else y.sum()
{% else %}
        y = y.sum()
{% endif %}
        total += torch.autograd.grad(y, point)[0]
    attributions = ((x - baseline) * total / {{ spec.explain.steps }}).abs().detach().cpu()
    rows = []
    names = feature_names(x.shape[1])
    for j in range(x.shape[1]):
        column = attributions[:, j]
        rows.append({"name": names[j], "importance": float(column.mean()),
                     "std": float(column.std()) if len(column) > 1 else 0.0})
    rows.sort(key=lambda r: r["importance"], reverse=True)
    Path("explain.json").write_text(json.dumps(
        {"method": "integrated_gradients", "metric": "mean |attribution|",
         "baseline": "{{ spec.explain.baseline }}", "rows": len(x), "features": rows}, indent=1), encoding="utf-8")
    print("feature attribution: " + ", ".join(f"{r['name']}={r['importance']:.4f}"
                                              for r in rows[:5]))
{% else %}
@torch.no_grad()
def explain_features(model: nn.Module, loss_fn, loader: DataLoader, device: torch.device):
    """Permutation importance: how much the loss rises when one feature is shuffled."""
    model.eval()
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb)
    if not xs:
        return
    x, y = torch.cat(xs)[:{{ spec.explain.max_samples }}], torch.cat(ys)[:{{ spec.explain.max_samples }}]

    def loss_of(inputs: torch.Tensor) -> float:
        out, target = adapt(model(inputs.to(device)), y.to(device))
        return float(loss_fn(out, target))

    base = loss_of(x)
    rng = torch.Generator().manual_seed(SEED)
    rows = []
    names = feature_names(x.shape[1])
    for j in range(x.shape[1]):
        drops = []
        for _ in range({{ spec.explain.n_repeats }}):
            shuffled = x.clone()
            shuffled[:, j] = x[torch.randperm(len(x), generator=rng), j]
            drops.append(loss_of(shuffled) - base)
        mean = sum(drops) / len(drops)
        std = (sum((d - mean) ** 2 for d in drops) / max(len(drops) - 1, 1)) ** 0.5
        rows.append({"name": names[j], "importance": mean, "std": std})
    rows.sort(key=lambda r: r["importance"], reverse=True)
    Path("explain.json").write_text(json.dumps(
        {"method": "permutation", "metric": "test loss increase", "baseline_loss": base,
         "rows": len(x), "features": rows}, indent=1), encoding="utf-8")
    print("feature importance: " + ", ".join(f"{r['name']}={r['importance']:.4f}"
                                             for r in rows[:5]))
{% endif %}
{% endif %}


if __name__ == "__main__":
    main()
'''
