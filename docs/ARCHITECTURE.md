# Architecture

```
ai_made_easy/
├── core/                 pure Python, no Qt (enforced by tests/test_structure.py)
│   ├── spec.py           BlockDefinition / ParamSpec / PortSpec contracts
│   ├── registry.py       block registry (self-describing JSON schemas)
│   ├── graph.py          graph IR: nodes, edges, shape inference, validation
│   ├── lints.py          architecture, data and training design rules
│   ├── migrate.py        project-file schema migrations
│   ├── blocks/           model-flow blocks (one module per family)
│   ├── codegen/          PyTorch + Keras emitters, helpers, ONNX/TorchScript
│   ├── training/         dataset/preprocessing catalog, catalog/ (losses, optimizers,
│   │                     schedulers, metrics, trainer), spec, PyTorch + Keras
│   │                     training-script templates, metrics code
│   ├── classic/          scikit-learn / boosting estimators and pipeline codegen
│   ├── targets.py        the single export-target table (UI, CLI, MCP)
│   ├── api/              headless facade used by the CLI, MCP server and web server
│   │                     (one module per area; everything importable from core.api)
│   ├── paths.py          $AIME_HOME (default ~/.aime) and its sub-folders
│   ├── runs/             persistent run history: run.json, epochs.jsonl, artefacts
│   ├── runner/           RunManager, run backends (local, SSH, Kaggle, Colab, Modal, Docker)
│   ├── sweeps.py         grid / random / Optuna sweeps over design parameters
│   ├── deploy/           serving packages (FastAPI + Dockerfile), exports, model registry
│   ├── importers/        PyTorch (torch.fx), ONNX and Keras importers (run in a subprocess)
│   ├── block_packs.py    versioned custom blocks and .aimeblocks packs
│   ├── data/             dataset profiles, data lints, fingerprints, split and
│   │                     augmentation previews
│   └── …                 summary, fixes, suggestions, model card, bundles
├── ui/                   PySide6 application
│   ├── context.py        composition root: builds and wires everything
│   ├── workbench.py      main window shell (layout only)
│   ├── actions_catalog.py  every action declared as data
│   ├── theme.py, icons.py  design system
│   ├── canvas/           the only package importing OdenGraphQt
│   ├── features/         panels and dialogs (presentation only)
│   └── services/         graph settle pipeline, processes, exports, projects
├── worker/               subprocess wrapper streaming JSON training events
├── mcp/                  MCP server over core/api
├── server/               FastAPI REST + WebSocket server and the built web UI (static/)
└── cli/                  headless command line: parser, dispatch, commands by area
web/                      React + TypeScript + Vite + React Flow frontend (built into server/static)
```

## Design rules

| Rule | Enforced by |
|---|---|
| `core/` never imports Qt | `test_core_stays_qt_free` |
| OdenGraphQt is only imported in `ui/canvas/` | `test_odengraphqt_only_imported_inside_canvas_package` |
| Every action spec resolves to a context slot | `test_every_catalog_slot_resolves_on_context` |
| The workbench stays layout-only | `test_workbench_is_layout_only` |
| Every model block generates runnable PyTorch and Keras code with the designed output shape | `tests/test_codegen_runtime.py` |
| Every dataset modality trains end to end | `tests/test_training_e2e.py` |
| Every classic estimator fits end to end | `tests/test_classic.py` |
| Every block can be placed and edited on the canvas | `tests/test_ui.py` |
| Imported models reproduce the original outputs | `tests/test_import.py` |
| Serving packages answer `/predict` like the checkpoint | `tests/test_deploy.py` |
| Every sample exports (ONNX, TorchScript, ExecuTorch, int8) and matches PyTorch, or a rule says why not | `tests/test_export.py` |
| Runs mirrored to MLflow / W&B match the local record; Hub push → pull → predict; Kaggle / Hub datasets detect like folders | `tests/test_integrations.py` |
| Every family's AutoML ranks its trials; classic and neural share one Pareto front; parallel trials | `tests/test_automl_families.py` |
| Glow inverts exactly and beats RealNVP; OGB official splits / evaluators; Atari and custom environments; QAT with embeddings | `tests/test_deferred.py` |
| The split preview equals the generated split | `tests/test_data.py` |
| Every REST endpoint and the run WebSocket | `tests/test_server.py` |
| The browser UI loads, validates and edits | `web/e2e/smoke.spec.ts` (Playwright) |

## Data flow

```
canvas edit ─► GraphService (debounced settle) ─► Graph IR ─► validate() + lints
      │                                                 ├─► Problems / card outlines / inspector
      │                                                 ├─► Summary, Code preview
      ▼                                                 └─► status bar
Train ─► training script (core/training or core/classic) ─► worker subprocess
      ─► epoch events ─► Training panel ─► run folder (checkpoint, metrics,
         predictions) ─► Error analysis · Saliency · Model card · Web demo
```

## Families, tasks and port roles

- `core/families.py`: every design belongs to one **family** (`neural`, `classic`,
  `llm`, ...). A family declares how it is detected, its validation rules (run
  after the generic graph checks), the frameworks that train it, its codegen
  targets and pip extras. Use `family_of(graph)` instead of block-prefix checks.
- `core/tasks.py`: a **task** (multiclass, regression, detection, ...) declares
  its target format, output role, losses, metrics, default metrics, trainer
  kind and serving schema. `TrainingSpec.task` resolves through it.
- Trainer kinds (`supervised`, `adversarial`, `diffusion`, `vae`, ...) select
  the training loop: `register_trainer(kind, renderer)` in
  `core/training/generate.py`.
- Ports carry a semantic `role` (`core.spec.ROLES`); `tensor` matches anything,
  two specific roles must agree. `multi=True` input ports take many wires.
- Blocks declare optional `requires` (modules) and `extra` (pip extra); missing
  modules are warnings with an install hint, shown in both libraries.
- `core/zoo/*.json`: pretrained catalogs generated offline by
  `scripts/build_zoo.py` (measured feature widths, parameters, accuracy,
  license).
- [FAMILIES.md](FAMILIES.md) is generated from these registries by
  `scripts/gen_families.py`: families, tasks, blocks by category, and every message the
  rule modules can raise (found by parsing their `_issue(...)` calls).
  `tests/test_release.py` fails when it is stale.

### Project files

- `Graph.to_dict()` writes `schema_version` = `core.migrate.CURRENT_SCHEMA` (3);
  `Graph.from_dict` runs `migrate()` first, one step per schema version.
- Schema 3 needed no rewrite: roles and multi-input ports belong to block definitions,
  not to project files. Saving (desktop, `PUT /api/projects`, `.aime` archives) goes
  through `migrate.stamp()`, which adds `family` and `task` ids. Readers such as the web
  project list use them without loading a design (`migrate.kind_of`), and they are
  recomputed on every save, so they are never trusted on load.
- A file from a newer schema is refused with "please update AI Made Easy".

## Vision tasks

`core/vision` adds detection, instance / semantic segmentation and keypoints
without touching the classification pipeline:

- `blocks.py`: task heads in the "Vision Tasks" category declare
  `meta["task"]` and a `meta["cost"]` hook (parameters, FLOPs) for the summary
  and budgets. Every wrapper (`helpers.py`) takes [0, 1] images and normalizes
  internally; detectors expose `losses()` / `detect()` and return padded
  detections `[D, 6]` from `forward`, so designed shapes still hold.
- `tasks.py`: registers the tasks (trainer kinds `detection` / `segmentation`)
  and a task resolver: a head block decides the task; a vision dataset plus a
  `[C, H, W]` output means semantic segmentation.
- `template.py`: the training script, registered with
  `register_trainer(kind, render, needs_spec=False)`, so it reads the graph
  directly instead of a `TrainingSpec`. It embeds `runtime.py` (dataset
  readers, metrics, RLE / overlays), which the app also executes for
  profiling and tests, so training and the Data workspace read data
  identically.
- `rules.py`: design rules registered with `lints.register_rule`. Tasks with
  their own training loop skip the classification-pipeline lints
  (`lints.PIPELINE_RULES`).
- `profile.py`: `register_profiler` hooks vision datasets into the Data
  workspace and data lints.

Heads can also sit on layers designed on the canvas. The YOLO-style head (`YOLOHead`)
reads a feature map from convolution blocks or the FPN neck. The template calls
`attach(model)` on it, so `losses()` / `detect()` run the whole network, and a
forward pre-hook records the image size, which gives the head its stride. Box-loss
blocks (GIoU / DIoU / CIoU) configure it through `configure()`. A segmenter with its
own loss (Mask2Former) exposes `segmentation_loss(images, masks)`, which the
template uses instead of the pixel loss blocks.

Loading a trained model sets `AIME_SKIP_PRETRAINED=1` so wrappers rebuild the
same architecture without downloading pretrained weights.

## Forecasting and speech

`core/forecast` and `core/speech` follow the vision layout (blocks, tasks and a
resolver, `runtime.py` shared by scripts and the app, `template.py`, `rules.py`,
`profile.py`) with trainer kinds `forecasting` and `speech`.

- Forecasters subclass `Forecaster` (`forecast/helpers.py`): `forward(x)` reads the
  history window `[L, C]` (channel 0 is the target, then past and future
  covariates) and `forecast(x, future)` also takes known-future covariates for the
  horizon. A shared `ForecastHead` maps features to the chosen distribution's
  parameters; the training script owns per-window scaling and the head's loss.
- A dataset block decides the task: forecasting data → `forecasting`; a speech
  dataset's `task` → keyword spotting, speech recognition or audio tagging.
  Generic layer stacks (e.g. TCN + Dense) train as forecasters when they end in
  `[horizon]`.
- `sequence/` holds the reusable TCN / Mamba / S4D / sLSTM / mel front-end
  helpers; their `meta["cost"]` feeds the summary and budgets.

## Generative models

`core/generative` adds five trainer kinds (`vae`, `adversarial`, `diffusion`,
`language_model`, `seq2seq`) rendered by two templates (`image_template.py`,
`text_template.py`). Blocks decide the kind (`tasks.generative_kind`): a Noise
Scheduler or Diffusion U-Net → diffusion, a Discriminator → GAN, a Reparameterize
block → VAE, text pairs / Seq2Seq Transformer → seq2seq, text / GPT / Causal
Transformer → language model.

- The canvas chain is always the model that is served: the VAE (generation sets
  `Reparameterize.prior_z` and runs the decoder half), the GAN generator, the
  denoiser, or the language model. Discriminator and Noise Scheduler are
  configuration blocks off the chain.
- Scripts print `samples: <png | txt>` after each epoch; the worker inlines the file
  into a `samples` event that the desktop and web training panels show, and the
  run manager replays the latest one to late subscribers.
- `runtime.py` holds the data generators / readers and the metrics (FID, KID,
  Inception Score, BLEU, chrF, ROUGE-L) shared by scripts, profiles and tests, and
  the byte-level BPE tokenizer a BPE Tokenizer block adds to language-model scripts.
- **Stable Diffusion LoRA** is its own family (`text_to_image`, detected by the
  `gen.sd_lora` block, rendered by `sd_template.py`): there is no layer chain, since
  diffusers builds the pretrained UNet, VAE and text encoder. The script adds peft
  adapters to the UNet attention, saves them with `save_lora_weights` and serves
  prompts. `weights = none` builds a tiny random Stable Diffusion with the same code
  path, so tests never download.

## Probabilistic models

Three families have their own scripts (`Family.generators`) and validators and are
trained in the app like any other run:

- `pgm` (framework `pgmpy`): variables are nodes, edges are dependencies; a
  `pgm.model` block picks the network kind (`acyclic=False` for Markov networks).
  CPDs are `table` parameters whose columns follow the parents in sorted order.
- `ppl` (framework `pymc`): `model.py` resolves distribution ports into a sandboxed
  expression graph and emits literal PyMC code; `runtime.py` holds the convergence
  diagnostics (compared with ArviZ in the tests).
- `gp` (frameworks `gpytorch` / `sklearn`, chosen by the model's `library`):
  `kernels.py` turns the Sum / Product tree into either library's expression.
- `ssm` (framework `statsmodels`): component blocks feed a multi-input
  `ssm.model`; `template.spec_of` maps them to `UnobservedComponents` or `SARIMAX`
  keyword arguments. Serving re-runs the Kalman filter (`smooth(params)`) on the
  request's history.

Two more live inside neural designs:

- `core/bayes`: stochastic layers mark themselves `stochastic`, KL layers expose
  `kl_divergence()`. `probabilistic_context(graph)` switches on a `PROBABILISTIC`
  section of the supervised PyTorch script (MC averaging, the MDN likelihood,
  Laplace, ECE, temperature scaling, conformal) and a `serve()` used by `infer()`.
- `core/flows`: every flow layer sets `self.logdet` in `forward` and has
  `inverse`; the `flow` trainer adds the log-determinants to a standard-normal base
  log-density and samples by running the layers backwards.

## Graphs, tables, recommenders and reinforcement learning

All four live in the neural family with their own trainer kinds, so the canvas chain is
always the network that is trained and served:

- `core/gnn` (trainer `graph`): models containing graph layers are generated with
  `forward(x, edge_index=None, batch=None)`; the first line builds a mutable context `g`
  that message-passing layers read and TopK / SAG pooling rewrite. `tasks.graph_kind`
  picks node classification, graph classification (global pooling or a multi-graph
  dataset) or link prediction (a Link Predictor block).
- `core/tabular`: blocks for the supervised pipeline. `layout.layout_of(graph)` mirrors
  `preprocess_tables` to give the input width and the positions / cardinalities of
  ordinal-encoded columns; `_raw_columns` keeps those positions unscaled.
- `core/recsys` (trainer `recommendation`): models score `[user, item, *features]` rows;
  ids are clamped into the embedding tables so random inputs (export checks) still run.
- `core/rl` (trainer `rl`): the generated model becomes a stable-baselines3
  `BaseFeaturesExtractor`; `ServedPolicy` exposes the trained policy as a module for
  exports, `infer()` uses `algorithm.predict`. Image observations (CarRacing, the built-in
  PixelChase, which the script defines and registers) reach the extractor channels-first
  and scaled to 0–1 by stable-baselines3, so the Input is `[C, H, W]`. `infer()` takes a
  frame as the environment gives it (`[H, W, C]`, 0–255).


`core/budget.py` estimates a neural design's cost from the IR alone (no
torch import): `estimate(graph, device)` walks `model_nodes()` and gives each
layer its parameters, forward FLOPs and the activation values it keeps for the
backward pass. Linear and conv layers cost `2 × weights × output positions`,
recurrent and attention layers add their sequence terms, pretrained backbones
use the GMACs recorded in the model zoo. Views (flatten, reshape, ...) keep no
memory, and an output consumed only by ReLU / sigmoid / tanh / softmax is freed
because those activations save their own output.

Training memory = weights (+ fp16 copy under AMP) + gradients + optimizer state
(Adam 2, SGD-momentum 1, plain SGD 0 per weight) + activations × batch + the
device's runtime overhead. Latency (batch 1) = FLOPs at 35% of the device's peak
+ memory traffic over its bandwidth + a launch cost per kernel. Composite blocks give
their own `meta["cost"]` (parameters, FLOPs) and `meta["kernels"]` (launches per
forward), so a transformer counts every token and layer. Device profiles
are `core/devices.json` plus `~/.aime/devices.json`.

The project's budget is `graph.meta["budget"]` (device and limits; 0 = none).
`check()` compares the estimate with it; the `resource_budget` lint reports
overruns and `fixes.py` offers mixed precision or a fitting batch size with
gradient accumulation. The PyTorch training template prints a `resources:`
line after the first epoch (peak memory, step time); the worker turns it into
a `resources` event stored on the run record for calibration.

## Recipes, the wizard and AutoML

`core/recipes/` holds parameterized design templates. A `Recipe` declares the tasks it
trains, a tier (small / medium / large / pretrained), the data kinds it reads, its knobs
(learning rate, width, depth, ...) and its optional requirements; its builder turns
`DataFacts` and knob values into a `Draft` (nodes, wires and a note on why each block is
there). `facts.detect(path)` reads a table or dataset folder. `build()` constructs the
graph, writes knobs that target sample nodes, applies the budget, then `autofit()` applies
the safe Quick Fixes (input width, sizes the data dictates, categories, memory) until none
is left; the recipe, its settings and the notes go to `graph.meta["recipe"]`, which
`explain()` reads. `recommend()` builds every recipe for the task and data kind and ranks
them by data size fit, installed requirements, validation errors and budget checks.
`tabular.py` and `media.py` define recipes from scratch; `starters.py` builds recipes from
the shipped samples and swaps in a dataset block for the user's data.

`core/automl.py` runs an AutoML search as a sweep (`AutoMLRunner` subclasses
`SweepRunner`, overriding the trial graph and the wait): a baseline round trains every
candidate recipe with its defaults, then TPE proposes a recipe and its conditional
settings; the median rule stops trials that fall behind at the same epoch. Records are
sweep records with `spec["automl"]`, so the Experiments page, `best_graph()` and the CLI
handle them like sweeps.

With `objectives` the search is multi-objective. `SweepRunner` has two hooks for this:
- `_measure` records each trial's design costs (budget estimate on the target device)
  before it trains.
- `_told` passes `[metric, *costs]` to a multi-direction TPE study.

Median pruning is off in this mode. `leaderboard()` marks the Pareto front
(`pareto_front()`) and lists it first. The desktop and web Sweeps views draw the
metric against the first cost.

## Pipelines

`core/pipelines/` adds the `pipeline` family. `blocks.py` defines the stage blocks (wired
through `artifact` ports); `plan.py` orders the stages and resolves the design each one
trains (a project path, `sample:<name>` or `embedded:<key>` from `graph.meta["designs"]`)
and the design behind the model each one passes on; `rules.py` validates the pipeline.
`PipelineRunner` (`runner.py`) runs the stages on a thread: Train and Cross-validate are
ordinary `RunManager.start` runs (Cross-validate adds a `train.kfold` block); the stages
that work inside a network use `RunManager.start_stage` with a writer from `stages.py`,
which copies (or generates) the model's training script into the stage folder and writes a
`stage_<kind>.py` that imports it and sets its hooks (`INIT_HOOK`, `LOSS_HOOK`,
`EPOCH_HOOK`, `LR_SCALE`, `BEFORE_SAVE`, `QUANTIZE`) before calling its `main()`, so
the folder holds the usual script, weights, inference state and metrics. Export and
Register call the deploy packager and the model registry. Stage fingerprints (stage,
settings, canonical design, data fingerprint, input fingerprints, version) let a new
attempt reuse finished stages of any earlier one. Records live in `$AIME_HOME/pipelines`.

## Runs, sweeps and deployment

```
RunManager.start(graph) ─► RunHistory.create (design snapshot, params, data fingerprint)
      ─► training script in $AIME_HOME/runs/<run_id>/ ─► worker subprocess
      ─► env / epoch / log / done events ─► epochs.jsonl + listeners
         (desktop panels, MCP, WebSocket clients)
SweepRunner ─► for each trial: apply values ─► validate ─► RunManager.start(parent=sweep)
build_package(run dir) ─► app.py + model + inference state + Dockerfile
      ─► export_formats.py + exporters.py: ONNX / TorchScript / Core ML / ExecuTorch / LiteRT,
         each checked against the model on the run's calibration.npz (export_report.json)
ModelRegistry.register(run dir) ─► $AIME_HOME/models/<name>/<version>/ with a stage
```

**Where a run trains** is a backend (`core/runner/backends/`): `local` (the worker as a
subprocess), `ssh`, `kaggle`, `colab`, `modal` or `container`. A remote backend gets a
**bundle** (`bundle.py`), which holds:
- the training script, with local data paths rewritten to `data/…`
- the worker
- `requirements.txt`
- the data files

On the far side it runs `worker/aime_launch.py` (standard library only). The launcher
unpacks the bundle, runs the worker, and writes `events.jsonl` and an `exit_code`.

The backend feeds those event lines into the same `TrainingRun`, so listeners cannot
tell where a run trains. A run's `done` event is held back until its outputs are
downloaded, so the record finalises with `metrics.json`. `RunRecord.backend` and
`RunRecord.remote` let `RunManager.attach` follow a remote run again after a restart.
`RunManager.import_results` completes a Colab hand-off run from its results zip.
See [REMOTE.md](REMOTE.md).

**The export layer** (`core/deploy/`) has three parts:
- `exportability.py` decides, from the design alone, which formats it can reach and why
  not (`blockers`), and which the budget device suggests. The Deploy and Share dialogs,
  the Problems panel and `aime deploy --list-formats` all read it.
- `package.py` writes the serving package. `_describe` picks what `/predict` takes for
  the design's task (records, images, series, graphs, observations, …).
- `exporters.py` runs inside the package (`export_formats.py`): each format is exported,
  then run against the source model on `calibration.npz`, and only formats that match
  are kept (`export_report.json`).

[DEPLOY.md](DEPLOY.md) is generated from these modules (`scripts/gen_deploy.py`).

**Your tools** sit beside the run, not inside it:
- `core/tracking.py`: a `RunMirror` per run, on its own thread, fed from
  `TrainingRun.record`. It replays parameters, per-epoch metrics, the design and the
  run's files to MLflow (`MlflowClient`) and W&B. Because it listens to the same events
  as the panels, it works for every backend. Where each run went is kept on
  `RunRecord.tracking`; a tracker that fails writes to the run's log and never stops
  training.
- `ModelRegistry.mirror()` registers a version in MLflow's registry or as a W&B
  artifact (`ModelVersion.mirrors`).
- `core/integrations/hub.py` pushes and pulls Hub model repositories (with an
  `aime.json`) and deploys a package as a Docker Space.
- `core/integrations/datasets.py` turns `kaggle:` and `hf:` sources into cached local
  folders, which data detection then reads.
- `core/runs/export.py` writes the run history as CSV or Parquet.

No module reads or stores a credential: each library uses its own sign-in. See
[TOOLS.md](TOOLS.md).

Generated training scripts save `inference_state.pkl` (the preprocessing fitted
on the training split) and define `load_predictor()` / `infer()`, which the
serving package imports, so served predictions use exactly the training
pipeline.

## Data workspace

`core/data/profile.py` reads datasets the way the generated scripts do (same
readers, label encoding and folder order). `core/data/lints.py` turns profile
findings into Problems-panel warnings in the context of the pipeline,
suppressing what the design already handles. Profiles are cached by file
signature; the desktop computes them on a background thread
(`ui/services/data_service.py`). `core/data/splits.py` reproduces the generated
split exactly, and `core/data/augment.py` runs the design's torchvision
pipeline in the training environment.

## Web server

`server/app.py` exposes `core/api` over REST (`/api/...`) and streams run
events over `WS /api/runs/{id}/events`. It also serves the built frontend with
an SPA fallback. When `AIME_WEB_TOKEN` is set, every request needs the bearer
token. The frontend keeps the project as React Flow nodes / edges, converts to
the project JSON for every API call, and validates with a debounce.

## Intermediate representation

- Shapes are per sample, without the batch dimension, channels-first:
  `[F]`, sequences `[L, C]`, signals `[C, L]`, images `[C, H, W]`, volumes
  `[C, D, H, W]` — exactly PyTorch's conventions.
- The Keras emitter tracks a layout per tensor (channels-last for convolutional
  tensors, IR order for sequences) and inserts the minimal `Permute` layers.
- Blocks declare `shape_fn`, optional `param_fn` (parameter counts) and
  `checks_fn` (cross-parameter rules); configuration blocks carry `meta`
  describing modality, stage and task.

## Training scripts

Generated scripts are self-contained and readable:
data loading → split → preprocessing fitted on the training split → model →
loss / optimizer / scheduler → training loop with early stopping → test
evaluation → artefacts (`*_best.pt`, `metrics.json`, `predictions.json`,
`mistakes.json`, `classes.json`). Progress lines follow
`epoch e/E key=value …`, which the worker turns into JSON events.
