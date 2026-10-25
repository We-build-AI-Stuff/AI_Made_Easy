"""Classic ML projects: detection, validation and scikit-learn script generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment

from ai_made_easy.core.classic import catalog as ml
from ai_made_easy.core.codegen import CodegenError, sanitize_identifier
from ai_made_easy.core.graph import Graph, ValidationIssue
from ai_made_easy.core.training import catalog as tcat
from ai_made_easy.core.training import data_catalog as dcat
from ai_made_easy.core.training.data_template import GROUP_WEIGHTS

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
_env.filters["repr"] = repr

SUPPORTED_DATA = ("data.csv", "data.sklearn", "data.synthetic", "data.numpy", "data.json",
                  "data.huggingface", "data.text_csv", "data.text_folder", "data.image_folder",
                  "data.timeseries_csv")
TABLE_PREP = ("prep.drop_columns", "prep.impute", "prep.one_hot", "prep.ordinal_encode",
              "prep.log_transform", "prep.clip_outliers", "prep.normalize", "prep.robust_scale",
              "prep.minmax", "prep.variance_filter")
FEATURE_ORDER = ("ml.max_abs_scale", "ml.normalizer", "ml.binarizer", "ml.variance_threshold",
                 "ml.polynomial", "ml.spline", "ml.power_transform", "ml.quantile_transform",
                 "ml.kbins", "ml.select_k_best", "ml.select_percentile", "ml.select_fpr",
                 "ml.select_fdr", "ml.select_fwe", "ml.pca", "ml.incremental_pca",
                 "ml.sparse_pca", "ml.factor_analysis", "ml.nmf", "ml.lda_topics",
                 "ml.feature_agglomeration", "ml.truncated_svd", "ml.kernel_pca", "ml.fast_ica",
                 "ml.gaussian_projection", "ml.sparse_projection", "ml.nystroem",
                 "ml.rbf_sampler", "ml.skewed_chi2_sampler", "ml.additive_chi2_sampler",
                 "ml.polynomial_sketch", "ml.random_trees_embedding",
                 "ml.dictionary_learning", "ml.minibatch_nmf", "ml.isomap", "ml.lle",
                 "ml.umap")
NN_PREFIXES = ("core.", "arch.", "llm.")


def is_classic(graph: Graph) -> bool:
    return any(n.type_id in ml.ESTIMATOR_IDS for n in graph.nodes.values())


@dataclass
class ClassicSpec:
    name: str
    estimator: ml.Estimator
    est_params: dict
    task: str
    dataset: dict
    modality: str
    steps: dict = field(default_factory=dict)
    features: list = field(default_factory=list)       # (Transformer, params)
    vectorizer: tuple | None = None
    encoder: tuple | None = None        # (Transformer, params): replaces one-hot for categories
    sampler: tuple | None = None        # (Transformer, params): e.g. SMOTE, while fitting only
    search: dict | None = None
    kfold: dict | None = None
    metrics: list = field(default_factory=list)        # (id, params, key)
    split: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    seed: int = 42


def _metric_tasks(task: str) -> set[str]:
    return {"classification": {"multiclass", "binary"}, "regression": {"regression"},
            "clustering": {"clustering"}, "anomaly": set()}[task]


def collect_classic(graph: Graph) -> ClassicSpec:
    nodes = list(graph.nodes.values())
    ests = [n for n in nodes if n.type_id in ml.ESTIMATOR_IDS]
    if len(ests) != 1:
        raise CodegenError(f"a classic ML pipeline needs exactly one estimator, found {len(ests)}")
    est_node = ests[0]
    est = ml.BY_ID[est_node.type_id]
    datasets = [n for n in nodes if n.type_id in dcat.DATASET_IDS]
    if len(datasets) > 1:
        raise CodegenError("expected at most one dataset block")
    if datasets:
        dataset = {"block": datasets[0].type_id, **datasets[0].resolved_params()}
    else:
        dataset = {"block": "data.synthetic",
                   "kind": "regression" if est.task == ml.REGRESSION else "classification",
                   "n_samples": 1000, "n_features": 20, "n_classes": 3, "noise": 0.1, "seed": 42}
    if dataset["block"] not in SUPPORTED_DATA:
        raise CodegenError(f"{dcat.BLOCKS[dataset['block']].name} is not supported by classic "
                           "ML pipelines; use a neural network for this data")
    for key in ("root", "path"):
        raw = dataset.get(key)
        if raw and not Path(str(raw)).expanduser().is_absolute() and (Path.cwd() / raw).exists():
            dataset[key] = str(Path.cwd() / raw)
    spec = ClassicSpec(name=sanitize_identifier(graph.name), estimator=est,
                       est_params=dict(est_node.resolved_params()), task=est.task,
                       dataset=dataset, modality=dcat.BLOCKS[dataset["block"]].modality)
    for n in nodes:
        tid = n.type_id
        params = dict(n.resolved_params())
        if tid in dcat.PREP_IDS:
            blk = dcat.BLOCKS[tid]
            if tid in TABLE_PREP or tid in ("prep.split", "prep.class_balance", "prep.text_clean"):
                spec.steps[tid] = params
            else:
                spec.warnings.append(f"{blk.name} is a neural-network data step and is ignored "
                                     "by the scikit-learn pipeline")
        elif tid in ml.TRANSFORMER_IDS:
            tr = ml.BY_ID[tid]
            if tr.stage == "text":
                if spec.vectorizer:
                    raise CodegenError("use one text vectorizer (TF-IDF or Count)")
                spec.vectorizer = (tr, params)
            elif tr.stage == "categorical":
                spec.encoder = (tr, params)
            elif tr.stage == "sampler":
                spec.sampler = (tr, params)
            else:
                spec.features.append((tr, params))
        elif tid == "ml.hyperparameter_search":
            spec.search = params
        elif tid == "train.kfold":
            spec.kfold = params
        elif tid in tcat.METRIC_IDS or tid in {m[0] for m in ml.CLUSTER_METRICS}:
            defn = n.definition()
            meta = (tcat.COMPONENTS[tid].meta if tid in tcat.COMPONENTS else defn.meta)
            tasks = set(meta["tasks"])
            if tasks & (_metric_tasks(est.task) | ({"clustering"} if est.task == "clustering"
                                                   else set())):
                spec.metrics.append((tid, params, meta["key"]))
            else:
                spec.warnings.append(f"{defn.display_name} does not apply to {est.task}")
        elif tid in tcat.LOSS_IDS or tid in tcat.OPTIMIZER_IDS or tid in tcat.SCHEDULER_IDS:
            spec.warnings.append(f"{n.definition().display_name} only applies to neural "
                                 "networks and is ignored")
        elif tid == "train.trainer":
            spec.seed = int(params.get("seed", 42))
    spec.features.sort(key=lambda f: FEATURE_ORDER.index(f[0].type_id))
    spec.split = spec.steps.pop("prep.split", None) or {"test_fraction": 0.2, "stratify": True,
                                                         "seed": 42, "shuffle": True}
    if spec.modality == "text" and spec.vectorizer is None:
        spec.vectorizer = (ml.BY_ID["ml.tfidf"], {p.name: p.default
                                                  for p in ml.BY_ID["ml.tfidf"].params})
    if spec.vectorizer and spec.modality != "text":
        raise CodegenError(f"{spec.vectorizer[0].name} needs a text dataset")
    if not spec.metrics:
        defaults = {"classification": ["eval.accuracy", "eval.f1"],
                    "regression": ["eval.mae", "eval.r2"],
                    "clustering": ["eval.silhouette"], "anomaly": []}[est.task]
        for m in defaults:
            meta = (tcat.COMPONENTS[m].meta if m in tcat.COMPONENTS
                    else {"key": next(c[2] for c in ml.CLUSTER_METRICS if c[0] == m)})
            spec.metrics.append((m, {p.name: p.default for p in
                                     (tcat.COMPONENTS[m].params if m in tcat.COMPONENTS else ())},
                                 meta["key"]))
    if "prep.class_balance" in spec.steps and est.task != ml.CLASSIFICATION:
        raise CodegenError("Class Balancing applies to classification estimators only")
    if spec.sampler and est.task != ml.CLASSIFICATION:
        raise CodegenError(f"{spec.sampler[0].name} balances classes: it needs a "
                           "classification estimator")
    if spec.sampler and (spec.steps.get("prep.class_balance") or {}).get(
            "strategy") == "oversample":
        raise CodegenError(f"use one way to oversample: {spec.sampler[0].name} or Class "
                           "Balancing's oversample")
    if spec.encoder and (spec.modality != "tabular" or est.task not in (
            ml.CLASSIFICATION, ml.REGRESSION)):
        spec.warnings.append(f"{spec.encoder[0].name} encodes table categories with the "
                             "target; it is ignored here")
        spec.encoder = None
    if spec.search and spec.task in (ml.CLUSTERING, ml.ANOMALY):
        raise CodegenError("hyperparameter search needs a supervised estimator")
    return spec


def classic_issues(graph: Graph) -> list[ValidationIssue]:
    """Design-time rules for classic pipelines (used by Graph.validate)."""
    issues: list[ValidationIssue] = []
    nodes = list(graph.nodes.values())
    nn = [n for n in nodes if n.type_id.startswith(NN_PREFIXES)]
    for n in nn:
        issues.append(ValidationIssue(
            "error", f"{n.definition().display_name} is a neural-network block; a project is "
                     "either a neural network or a classic ML pipeline", n.instance_id))
    ests = [n for n in nodes if n.type_id in ml.ESTIMATOR_IDS]
    for n in ests[1:]:
        issues.append(ValidationIssue(
            "error", "only one estimator per pipeline; compare models in separate projects "
                     "or use Hyperparameter Search", n.instance_id))
    if nn or len(ests) != 1:
        return issues
    est_node = ests[0]
    est = ml.BY_ID[est_node.type_id]
    anchor = est_node.instance_id
    try:
        spec = collect_classic(graph)
    except CodegenError as exc:
        return [ValidationIssue("error", str(exc), anchor)]
    for w in spec.warnings:
        issues.append(ValidationIssue("warning", w, anchor))
    if est.nonnegative and ({"prep.normalize", "prep.robust_scale"} & set(spec.steps)
                            or any(t.type_id in ("ml.pca", "ml.power_transform", "ml.kernel_pca",
                                                 "ml.fast_ica") for t, _ in spec.features)):
        issues.append(ValidationIssue(
            "error", f"{est.name} needs non-negative features, but the pipeline can produce "
                     "negative values (standardization / PCA / power transform). Use Min-Max "
                     "scaling or remove the step.", anchor))
    for tr, _ in spec.features:
        if tr.type_id in ml.NONNEGATIVE_TRANSFORMERS and (
                {"prep.normalize", "prep.robust_scale"} & set(spec.steps)
                or "prep.minmax" not in spec.steps):
            issues.append(ValidationIssue(
                "warning", f"{tr.name} needs non-negative features: add Min-Max scaling (and "
                           "no standardization) before it", anchor))
    if est.task == ml.CLUSTERING and spec.dataset["block"] in ("data.text_csv",):
        pass
    if spec.search:
        try:
            grid = ml.parse_grid(spec.search["param_grid"])
        except ValueError:
            grid = {}
        known = {p.name for p in est.params}
        unknown = [k for k in grid if k not in known]
        if unknown:
            issues.append(ValidationIssue(
                "warning", f"search parameters {unknown} are not exposed on {est.name}; they are "
                           "passed to the estimator as-is", anchor))
    k = spec.est_params.get("n_neighbors")
    if k is not None and spec.dataset["block"] == "data.synthetic" \
            and int(k) >= int(spec.dataset.get("n_samples", 1000)):
        issues.append(ValidationIssue("error", "n_neighbors must be smaller than the number of "
                                               "samples", anchor))
    for tr, params in spec.features:
        if tr.type_id in ml.UNIVARIATE_SELECTORS:
            sf = params["score_func"]
            if (est.task == ml.REGRESSION) != sf.endswith("regression"):
                if sf != "chi2" or est.task == ml.REGRESSION:
                    issues.append(ValidationIssue(
                        "error", f"score_func '{sf}' does not match the {est.task} estimator",
                        anchor))
        if tr.type_id == "ml.power_transform" and params["method"] == "box-cox" \
                and "prep.minmax" not in spec.steps:
            issues.append(ValidationIssue(
                "warning", "Box-Cox needs strictly positive features; prefer Yeo-Johnson unless "
                           "the data is positive", anchor))
    return issues


# ---------------------------------------------------------------- rendering

def _imports(spec: ClassicSpec) -> list[str]:
    imports: dict[str, set[str]] = {}

    def need(module: str, name: str) -> None:
        imports.setdefault(module, set()).add(name)

    est = spec.estimator
    need(est.module, est.cls)
    for tr, _ in [*spec.features, *([spec.encoder] if spec.encoder else []),
                  *([spec.sampler] if spec.sampler else [])]:
        need(tr.module, tr.cls)
        if tr.type_id in ml.UNIVARIATE_SELECTORS:
            need("sklearn.feature_selection", _score_func(tr, _))
    if spec.vectorizer:
        need(spec.vectorizer[0].module, spec.vectorizer[0].cls)
    steps = spec.steps
    if "prep.impute" in steps or spec.modality == "tabular":
        need("sklearn.impute", "SimpleImputer")
    if "prep.normalize" in steps:
        need("sklearn.preprocessing", "StandardScaler")
    if "prep.minmax" in steps:
        need("sklearn.preprocessing", "MinMaxScaler")
    if "prep.robust_scale" in steps:
        need("sklearn.preprocessing", "RobustScaler")
    if "prep.variance_filter" in steps:
        need("sklearn.feature_selection", "VarianceThreshold")
    if "prep.log_transform" in steps:
        need("sklearn.preprocessing", "FunctionTransformer")
    if "prep.clip_outliers" in steps:
        need("sklearn.base", "BaseEstimator")
        need("sklearn.base", "TransformerMixin")
    if spec.dataset["block"] == "data.csv":
        need("sklearn.compose", "ColumnTransformer")
        need("sklearn.compose", "make_column_selector")
        need("sklearn.preprocessing", "OrdinalEncoder" if "prep.ordinal_encode" in steps
             else "OneHotEncoder")
    if spec.task in (ml.CLASSIFICATION, ml.REGRESSION):
        need("sklearn.model_selection", "train_test_split")
    if spec.kfold:
        need("sklearn.model_selection", "cross_validate")
        need("sklearn.model_selection",
             "StratifiedKFold" if spec.task == ml.CLASSIFICATION else "KFold")
    if spec.search:
        method = spec.search["method"]
        if method == "halving":
            imports.setdefault("sklearn.experimental", set()).add("enable_halving_search_cv")
            need("sklearn.model_selection", "HalvingGridSearchCV")
        else:
            need("sklearn.model_selection",
                 "GridSearchCV" if method == "grid" else "RandomizedSearchCV")
    if "prep.class_balance" in steps and steps["prep.class_balance"]["strategy"] != \
            "group weights" and not (est.balanced and steps["prep.class_balance"]
                                     ["strategy"] == "class weights"):
        need("sklearn.utils.class_weight", "compute_sample_weight")
    order = ["sklearn.experimental"]
    lines = []
    for module in sorted(imports, key=lambda m: (m not in order, m)):
        names = sorted(imports[module])
        if module == "sklearn.experimental":
            lines.append("from sklearn.experimental import enable_halving_search_cv  # noqa: F401")
        else:
            lines.append(f"from {module} import {', '.join(names)}")
    return lines


def _score_func(tr, params) -> str:
    return params["score_func"]


def _numeric_steps(spec: ClassicSpec) -> list[str]:
    s = spec.steps
    out = []
    imp = s.get("prep.impute")
    if imp and imp["strategy"] != "drop rows":
        strategy = {"mode": "most_frequent"}.get(imp["strategy"], imp["strategy"])
        fill = f", fill_value={imp['constant']}" if strategy == "constant" else ""
        out.append(f'("impute", SimpleImputer(strategy="{strategy}"{fill}))')
    elif spec.dataset["block"] == "data.csv":
        out.append('("impute", SimpleImputer(strategy="median"))')
    if "prep.log_transform" in s:
        out.append('("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))')
    if "prep.clip_outliers" in s:
        c = s["prep.clip_outliers"]
        out.append(f'("clip", OutlierClipper(method="{c["method"]}", threshold={c["threshold"]}))')
    if "prep.variance_filter" in s:
        out.append(f'("variance", VarianceThreshold({s["prep.variance_filter"]["threshold"]}))')
    if "prep.normalize" in s:
        out.append('("scale", StandardScaler())')
    if "prep.robust_scale" in s:
        out.append('("robust", RobustScaler())')
    if "prep.minmax" in s:
        m = s["prep.minmax"]
        # test rows outside the training range would go below 0 for a transformer that
        # needs non-negative features (NMF, LDA, chi²): clip them to the range there
        clip = ", clip=True" if any(t.type_id in ml.NONNEGATIVE_TRANSFORMERS
                                    for t, _ in spec.features) else ""
        out.append(f'("minmax", MinMaxScaler(feature_range=({m["range_min"]}, '
                   f'{m["range_max"]}){clip}))')
    return out


def _categorical_step(spec: ClassicSpec) -> str:
    if spec.encoder:
        return f'("encode", {_transformer_ctor(*spec.encoder)})'
    if "prep.ordinal_encode" in spec.steps:
        return ('("encode", OrdinalEncoder(handle_unknown="use_encoded_value", '
                'unknown_value=-1))')
    max_cat = int((spec.steps.get("prep.one_hot") or {}).get("max_categories", 50))
    return (f'("encode", OneHotEncoder(handle_unknown="infrequent_if_exist", '
            f'max_categories={max_cat}, sparse_output=False))')


def _transformer_ctor(tr: ml.Transformer, params: dict) -> str:
    if tr.render:
        ctor = tr.render(params)
    else:
        args = ", ".join(f"{p.name}={ml._literal(p, params.get(p.name, p.default))}"
                         for p in tr.params)
        ctor = f"{tr.cls}({args})"
    if tr.seeded:
        ctor = ctor[:-1] + (", " if not ctor.endswith("(") else "") + "random_state=SEED)"
    return ctor


def _metric_needs_proba(spec: ClassicSpec) -> bool:
    return any(k == "log_loss" for _, _, k in spec.metrics)


def _group_column(spec: ClassicSpec, d: dict) -> str:
    column = str(spec.steps["prep.class_balance"].get("group_column") or "").strip()
    if not column:
        raise CodegenError("group weights: set Class Balancing's group column (e.g. region)")
    if d["block"] != "data.csv":
        raise CodegenError("group weights read a column of the table: they work for designs "
                           "trained on a CSV / Parquet table (data.csv)")
    if column == d.get("target_column"):
        raise CodegenError(f"group weights: {column!r} is the target; pick another column")
    return column


def classic_context(spec: ClassicSpec) -> dict:
    est = spec.estimator
    extra = {}
    balance = (spec.steps.get("prep.class_balance") or {}).get("strategy")
    if balance == "class weights" and est.balanced:
        extra["class_weight"] = '"balanced"'
    if est.cls == "SVC" and _metric_needs_proba(spec):
        extra["probability"] = "True"
    feature_steps = [f'("{tr.type_id.split(".")[-1]}", {_transformer_ctor(tr, params)})'
                     for tr, params in spec.features]
    grid = {}
    if spec.search:
        grid = {f"model__{k}": v for k, v in ml.parse_grid(spec.search["param_grid"]).items()}
    scoring = (spec.search or {}).get("scoring", "auto")
    if scoring == "auto":
        scoring = {"classification": "accuracy", "regression": "r2"}.get(spec.task, "accuracy")
    d = spec.dataset
    tok_clean = spec.steps.get("prep.text_clean") or {"lowercase": True,
                                                     "strip_punctuation": False,
                                                     "strip_digits": False, "strip_html": True}
    reqs = ["scikit-learn", "numpy", "joblib"]
    if d["block"] in ("data.csv", "data.text_csv", "data.timeseries_csv"):
        reqs.append("pandas")
    if d["block"] == "data.image_folder":
        reqs.append("pillow")
    if d["block"] == "data.huggingface":
        reqs.append("datasets")
    if est.package != "scikit-learn":
        reqs.append(est.package)
    reqs += sorted({tr.package for tr, _ in [*spec.features,
                                             *([spec.sampler] if spec.sampler else [])]
                    if tr.package})
    return {
        "spec": spec, "d": d, "est": est,
        "est_ctor": est.constructor(spec.est_params,
                                    seed="SEED" if est.seeded else None, extra=extra),
        "imports": _imports(spec),
        "numeric_steps": _numeric_steps(spec),
        "categorical_step": _categorical_step(spec),
        "feature_steps": feature_steps + ([
            f'("{spec.sampler[0].type_id.split(".")[-1]}", '
            f'{_transformer_ctor(*spec.sampler)})'] if spec.sampler else []),
        "sampler": bool(spec.sampler),
        "vectorizer": (spec.vectorizer[0].render(spec.vectorizer[1])
                       if spec.vectorizer else None),
        "grid": grid, "scoring": scoring,
        "task": spec.task, "supervised": spec.task in (ml.CLASSIFICATION, ml.REGRESSION),
        "balance": balance, "group_column": _group_column(spec, d) if balance == "group weights"
        else "", "group_weights_code": GROUP_WEIGHTS, "sample_weight": balance and not (balance == "class weights"
                                                              and est.balanced),
        "table": d["block"] == "data.csv", "text": spec.modality == "text",
        "clean": tok_clean,
        "feature_columns": [c.strip() for c in str(d.get("feature_columns", "")).split(",")
                            if c.strip()],
        "drop_columns": [c.strip() for c in str((spec.steps.get("prep.drop_columns") or {})
                                                .get("columns", "")).split(",") if c.strip()],
        "drop_rows": (spec.steps.get("prep.impute") or {}).get("strategy") == "drop rows",
        "metric_keys": [k for _, _, k in spec.metrics],
        "metric_params": {k: p for _, p, k in spec.metrics},
        "proba": est.proba or (est.cls == "SVC" and _metric_needs_proba(spec)),
        "requirements": ", ".join(reqs),
        "sk_loader": dcat.SKLEARN[d["dataset"]][0] if d["block"] == "data.sklearn" else "",
        "image_size": 32,
        "test_fraction": spec.split.get("test_fraction", 0.2) or 0.2,
        "stratify": bool(spec.split.get("stratify", True)) and spec.task == ml.CLASSIFICATION,
        "ts_targets": [c.strip() for c in str(d.get("target_columns", "")).split(",") if c.strip()],
    }


def generate_classic(graph: Graph) -> str:
    if errors := [i for i in graph.validate() if i.severity == "error"]:
        raise CodegenError("graph has validation errors:\n"
                           + "\n".join(f"  - {e}" for e in errors))
    spec = collect_classic(graph)
    from ai_made_easy.core.classic.template import SKLEARN_TEMPLATE

    return _env.from_string(SKLEARN_TEMPLATE).render(**classic_context(spec))
