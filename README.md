# AI Made Easy

[![CI](https://github.com/ManoharPaturi/AI_Made_Easy/actions/workflows/ci.yml/badge.svg)](https://github.com/ManoharPaturi/AI_Made_Easy/actions/workflows/ci.yml)

**Visual model builder for practitioners.** Design neural networks, classic
machine-learning pipelines, probabilistic models and multi-stage training pipelines by
connecting blocks, with validation as you build.
Inspect your data, train, compare experiments and tune hyperparameters, then
deploy the result as a web service. The app exports clean, runnable
**PyTorch**, **Keras 3** and **scikit-learn** code, so you never write
boilerplate. It runs on the desktop or in the browser.

![Workbench](docs/images/workbench_dark_full.png)

## Highlights

- **Train anywhere:** this computer, any SSH host (rented GPU machines from RunPod,
  Lambda, Vast.ai or Paperspace too), Kaggle, Colab, Modal or a Docker container. Pick
  it in **Run on**; live curves, Stop, sweeps and pipelines work the same, and a dropped
  connection picks up where it left off. Sign-in stays with each tool: AI Made Easy never
  asks for a password, token or key ([docs/REMOTE.md](docs/REMOTE.md)).
- **Export everywhere:** ONNX, TorchScript, Core ML, OpenVINO, ExecuTorch, LiteRT and GGUF, with int8
  quantization calibrated on the run's own samples. Every export is checked against the
  trained model before it is packaged. Detectors keep their post-processing (NMS) and
  graph networks take graphs of any size. The project's budget device suggests the
  format, and a design rule names why a design can't reach one
  ([docs/EXPORT.md](docs/EXPORT.md); what every task serves and exports:
  [docs/DEPLOY.md](docs/DEPLOY.md)).
- **Work with your tools:** runs mirror to MLflow and Weights & Biases as they train,
  on every backend; registered models go to MLflow's registry, W&B or the Hugging Face
  Hub (as a model repo or a Docker Space); Kaggle and Hub datasets download and are
  detected like a local folder; the run history exports to CSV / Parquet
  ([docs/TOOLS.md](docs/TOOLS.md)).
- **More models:** Glow image flows (exact likelihoods in bits per dimension), Open
  Graph Benchmark datasets scored by their official evaluators, Atari games for
  pixel-based RL ([docs/RL.md](docs/RL.md)), your own Gymnasium environments checked
  while you design, and quantization-aware training for text models with embeddings.
- **Pipelines:** chain Train, Cross-validate, Fine-tune, Distill, Prune, Quantize,
  quantization-aware training, Evaluate, Ensemble, Export and Register stages; every stage
  is a tracked run, and re-running a pipeline picks up where it stopped. Fine-tune with
  discriminative learning rates, stack models on their out-of-fold predictions, distill a
  random forest into a network, and expand any stage on the canvas to see its design.
- **Start from a task, not a blank canvas:** **File ▸ New from Task** asks what you want
  to do (35 tasks, from classification and forecasting to clustering, anomaly detection
  and retrieval-augmented answering) and where your data is, detects its format, target
  and classes, and ranks
  ready-to-train recipes for your data size and hardware budget, each explaining why
  every block is there. Or let **AutoML** try the recipes and tune them for you, for
  every family (networks, classic ML, graphical models, PyMC, Gaussian processes,
  state-space models, flows), optionally trading accuracy against latency or model size
  on your target device: the result is one Pareto front of networks and classic models
  to choose from. Trials can train several at a time, on any backend.
- **762 blocks in 9 families** ([docs/FAMILIES.md](docs/FAMILIES.md) lists every
  family, task, block and design rule) across every major library: `torch.nn` layers, Keras 3
  equivalents, torchvision / `keras.applications` / timm pretrained backbones,
  Hugging Face text and image encoders, scikit-learn, XGBoost, LightGBM and CatBoost,
  plus datasets, preprocessing, augmentation, losses, optimizers, schedulers
  and metrics.
- **Modern network parts (3.3).** Rotary embeddings, grouped-query attention, SwiGLU,
  Llama decoder blocks, Mixture of Experts, LoRA, DropPath, LayerScale, FiLM, ViT patch
  embedding, ConvNeXt, Swin and Conformer blocks; Lion, LAMB and Lookahead optimizers;
  ArcFace, CosFace, triplet and supervised contrastive losses for embeddings; SAM and
  Schedule-Free AdamW; weight averaging (EMA, SWA); Integrated Gradients; target encoding
  and SMOTE for classic models; Tiny Llama, Vision Transformer, ConvNeXt and ArcFace recipes.
- **Bigger models, fairer models, GGUF (3.11).** PyTorch networks train in bfloat16,
  shard across GPUs (FSDP) and fine-tune for int8 (quantization-aware training). Several
  GPUs work on Windows and with k-fold and metric losses, and Keras compiles.
  - **Fairness:** group weights keep a group column from predicting the class in
    training, and a threshold per group evens out a classifier's error rates, shown
    before and after.
  - **GGUF:** Llama-style language models export to GGUF for llama.cpp and Ollama.
  - **What would change it?** Try it shows the smallest edits to a table row that change
    the prediction.

  ![A threshold per group: each group's rates before and after](docs/images/web_threshold_per_group.png)
  ![What would change it: the smallest edits that change a prediction](docs/images/web_what_would_change.png)
- **Several GPUs, resume anywhere, fairness by group (3.10).** PyTorch networks can
  train on several GPUs, compile with `torch.compile` and checkpoint activations to save
  memory. Keras runs and runs on SSH, Modal, containers, Kaggle and Colab resume where
  they stopped. Results by group name the groups treated differently (four-fifths rule,
  true- and false-positive rates), models export to OpenVINO, and both apps let a design
  choose PyTorch or Keras.

  ![Fairness by group: the groups treated differently](docs/images/web_fairness_by_group.png)
- **Resume a run, train boosted models more, faster sequence models (3.9).** A run that
  stopped or crashed continues from its last saved epoch, as the same run. XGBoost,
  LightGBM, CatBoost and gradient-boosting models can grow more trees, and the
  learning-rate finder runs Keras designs. Mamba-2, Gated DeltaNet, gated linear
  attention and Griffin train 2-10 times faster on long sequences; 15 more layers have
  Keras versions that compute exactly what the PyTorch ones do; Core ML exports are
  checked against PyTorch on macOS.

  ![Train a boosted model more: how many trees to add](docs/images/web_train_more_trees.png)
- **Find a learning rate, train more, results by group (3.8).** From an optimizer block,
  a short range test plots loss against learning rate and suggests one to use. A finished
  network can train for more epochs from its best weights, and any run trained on a table
  shows its test results per value of a column, flagging groups where it is clearly
  worse. 15 more blocks: NLinear, iTransformer, TSMixer and TimesNet forecasters, TabM,
  latent / QK-norm / gated attention, Mamba-2, Gated DeltaNet, GLA, Griffin, MobileNetV4's
  UIB and StarNet blocks, ADOPT; iTransformer and uncertainty-regression recipes. Python
  3.13 and 3.14 are tested.

  ![Find a learning rate: loss against learning rate, with the suggestion](docs/images/web_lr_finder.png)
- **Windows, and which settings matter together (3.7).** Windows is now tested on every
  change, and the Windows desktop app is built and started in CI. Sweeps with 10 or more
  trials show which pairs of settings matter together, with a heatmap of the strongest
  pair, and can be swept again near the best trials in one click. 31 more blocks:
  sliding-window, ALiBi and differential attention, RetNet, RWKV and Hyena, MLP-Mixer and
  gMLP, Ghost and deformable convolutions, Dynamic Tanh; PolyLoss, VICReg and Barlow
  Twins; Prodigy, Cautious AdamW and Yogi; PSNR and SSIM; Tiny RWKV and Barlow Twins
  recipes.

  ![Which settings matter together: a sweep's pairs and heatmap](docs/images/web_sweep_pairs.png)
- **Try a whole file, and see which settings mattered (3.6).** Try it predicts every
  row of a CSV or every image in a zip and gives back a CSV; image classifiers show a
  Grad-CAM heatmap with each prediction. Sweeps and AutoML draw a parallel-coordinates
  chart and rank the settings by how much they mattered. 41 more blocks: ResNet, MBConv,
  CBAM and DenseNet building blocks, KAN, AdaBelief / Adan / AdEMAMix, warmup-stable-decay,
  quantile and Tweedie losses, a Gaussian (mean + variance) head, L-BFGS, new metrics,
  augmentations and scikit-learn models.

  ![Which settings mattered: parallel coordinates and importance of a sweep](docs/images/web_sweep_insights.png)
- **Try it, and see what it attended to (3.5).** Run a finished model on your own input
  from the Experiments page: a form built from what it takes (table fields, text, an
  image or a WAV clip), answered on your machine. Attention maps of transformer models
  and saliency maps of audio classifiers are saved with each run. 86 more blocks:
  `torch.nn` layers such as the full encoder-decoder Transformer, ConvLSTM2D, EmbeddingBag
  and spectral normalization; Keras losses; the Muon optimizer; and 56 more scikit-learn
  estimators and transformers.

  ![Try it: a finished run's prediction on your own input](docs/images/web_try_it.png)
- **Explain, embed and compare (3.4).** Grad-CAM heatmaps of where an image model
  looked, saved with each run; SHAP values beside permutation importance and Integrated
  Gradients; SimCLR self-supervised pretraining; Cosine Embedding and Margin Ranking
  losses, on pairs from the class labels or on a table of labelled pairs (siamese
  training, with a *Pairs — same or different* recipe). Groups fold into one card in
  both apps.
- **A canvas that keeps up (3.3).** Copy, paste and duplicate blocks between designs and
  between the two apps, frame them in named groups, pin sticky notes, drop a CSV on the
  window to start a project from it, see which features a model relies on, and compare
  a run's design with the canvas. Both apps have a command palette (Ctrl/⌘ K), find
  (Ctrl/⌘ F), a shortcuts sheet, a minimap and a short first-run tour.
- **Design-time validation.** Shapes, ranks and dtypes are inferred live across
  branches and merges; every card shows its output shape. Incompatible wiring,
  parameter constraints and common architecture mistakes (double softmax with
  cross-entropy, class count vs dataset, missing positional encodings, …) are
  reported instantly in the Problems panel, with Quick Fixes.
- **Code you can ship.** Generated model files, training scripts and pipelines
  are executed by the test suite for every block and every modality — the real
  output shape must equal the designed shape, in both PyTorch and Keras.
- **Complete training pipelines.** Image, tabular, text, audio and time-series
  datasets; preprocessing fitted on the training split only (no leakage);
  chronological time-series splits; class balancing, MixUp/CutMix, mixed
  precision, gradient clipping and accumulation, early stopping, k-fold CV.
- **Vision tasks.** Object detection (Faster R-CNN, RetinaNet, FCOS, SSD, DETR,
  RT-DETR), instance segmentation (Mask R-CNN), keypoints and semantic
  segmentation (DeepLabV3, SegFormer, U-Net family) on COCO, YOLO, Pascal VOC or
  mask-folder datasets, with box / mask-aware augmentation and COCO mAP.
- **Forecasting and speech.** Probabilistic forecasting of one or many series
  (DLinear, N-BEATS, N-HiTS, PatchTST, TiDE, TCN, DeepAR-style RNN, Informer-style
  Transformer) with quantile / Student-t / negative-binomial heads, covariates,
  rolling-origin backtests and MASE / CRPS. Keyword spotting, CTC speech
  recognition and audio tagging with in-model mel features, wav2vec 2.0 / HuBERT /
  WavLM / Whisper encoders, and Mamba, S4D, sLSTM and TCN sequence blocks.
- **Generative models.** VAEs designed layer by layer with a Reparameterize
  bottleneck, DCGAN / WGAN-GP GANs, class-conditional diffusion (DDPM / DDIM,
  classifier-free guidance, EMA), GPT-style language models and encoder-decoder
  transformers trained from scratch, with live sample grids, FID / KID, perplexity,
  BLEU / chrF / ROUGE-L, and `/generate` endpoints in deploy packages.
- **Probabilistic models.** Bayesian networks, Markov networks, HMMs and
  structure learning (pgmpy) with CPD table editing; hierarchical models in PyMC
  with R-hat / ESS diagnostics; Gaussian processes with kernel algebra (GPyTorch,
  scikit-learn); MC dropout, Bayes-by-Backprop, mixture density networks and Laplace
  for neural networks; ECE, temperature scaling and conformal prediction;
  normalizing flows (RealNVP, MAF, neural splines); and structural time series /
  SARIMAX with the Kalman filter (statsmodels).
- **Graphs, tables, recommenders and agents.** Graph neural networks (GCN, GAT,
  GraphSAGE, GIN, …) for node / graph classification and link prediction; FT-Transformer,
  TabTransformer, TabNet and ResNet for tables with learned category embeddings; matrix
  factorisation, NCF, two-tower and DLRM recommenders with NDCG@K; and reinforcement
  learning with PPO, DQN, SAC and more, where the policy network is designed on the canvas.
- **Resource budgets.** FLOPs, training memory and latency estimates for 11
  devices (laptop CPU to H100, iPhone, Jetson, Raspberry Pi), with warnings and
  Quick Fixes when a design will not fit.
- **Analysis.** Live training curves, error analysis (with prediction overlays
  for vision tasks), Grad-CAM saliency maps and generated model cards.
- **Experiments and tuning.** Every run is recorded with its design, parameters,
  metrics, environment and data fingerprint. Compare runs side by side,
  restore any of them, and sweep hyperparameters with grid, random or
  Bayesian (Optuna) search.
- **Data workspace.** Profile a dataset before training. Missing values,
  identifier and constant columns, target leakage, duplicates and class
  imbalance are flagged in the Problems panel. Preview the exact
  train / validation / test split and the augmentation pipeline.
- **Deploy.** Turn a trained run into a serving package: a FastAPI service
  with the fitted preprocessing, a Dockerfile, and ONNX / TorchScript / Core ML /
  ExecuTorch / LiteRT exports, each checked against the model. Promote versions
  through a model registry.
- **Import.** Open existing PyTorch modules, ONNX files and Keras models as
  editable designs, verified numerically against the original.
- **Exports.** PyTorch and Keras model files and training scripts,
  scikit-learn pipelines, ONNX, TorchScript, a single-file web demo, LLM
  workflow scripts and portable `.aime` project archives.

![Data workspace](docs/images/data_workspace_full.png)

Multi-objective AutoML: two spirals searched for accuracy against latency on a
Raspberry Pi 5. The front runs from a 0.06 ms MLP at 63% to a 0.23 ms ResNet at 99%.

![AutoML Pareto front](docs/images/automl_pareto_full.png)

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/ManoharPaturi/AI_Made_Easy.git
cd AI_Made_Easy
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

Install only what you need with the optional extras:

| Extra | Adds |
|---|---|
| `torch` | In-app training, ONNX / TorchScript export |
| `vision` | torchvision datasets, transforms and pretrained backbones |
| `vision-tasks` | timm backbones, Hugging Face DETR / SegFormer / DINOv2, COCO RLE masks |
| `audio` | Hugging Face speech encoders (wav2vec 2.0, HuBERT, WavLM, Whisper) |
| `text` | Hugging Face text encoders (DistilBERT, BERT, RoBERTa, MiniLM, ...) and tokenizers |
| `probabilistic` | pgmpy, hmmlearn, PyMC, GPyTorch and statsmodels (graphical models, probabilistic programs, Gaussian processes, state-space models) |
| `graph` | PyTorch Geometric (graph neural networks) and the Open Graph Benchmark |
| `generative` | diffusers and peft (Stable Diffusion LoRA fine-tuning) |
| `rl` | Gymnasium and stable-baselines3 (reinforcement learning) |
| `rl-box2d` | The Box2D environments (LunarLander, BipedalWalker, CarRacing); wheels exist up to Python 3.12 |
| `data` | pandas-backed table files (CSV, TSV, Parquet, Excel, JSON) |
| `classic` | scikit-learn, XGBoost, LightGBM, CatBoost |
| `keras` | Keras 3 for running exported Keras code |
| `llm` | Hugging Face generation, LoRA fine-tuning and RAG scripts |
| `mcp-server` | Model Context Protocol server for AI agents |
| `web` / `serve` | Browser UI (`aime web`) and serving packages (FastAPI, uvicorn) |
| `export` | ONNX / onnxruntime / skl2onnx exports and verification |
| `edge` | On-device exports: ExecuTorch (`.pte`) and the LiteRT interpreter |
| `coreml` | Core ML exports (`.mlpackage`), checked against PyTorch when exported on a Mac |
| `openvino` | OpenVINO exports (Intel CPUs, GPUs and NPUs), checked against PyTorch on the CPU |
| `gguf` | GGUF exports of Llama-style language models (llama.cpp, Ollama, LM Studio); add `llama-cpp-python` to check them |
| `tracking` | Mirror runs to MLflow and Weights & Biases |
| `hub` | The Hugging Face Hub: push runs and Spaces, load Hub datasets |
| `kaggle` / `modal` | Train on Kaggle / Modal; Kaggle datasets and competitions |
| `atari` | Atari games (ale-py, OpenCV); AI Made Easy ships no ROMs ([docs/RL.md](docs/RL.md)) |
| `tuning` | Bayesian hyperparameter search (Optuna) |
| `dev` | Test and lint tooling |

## Quick start

```bash
ai-made-easy            # or: python -m ai_made_easy
```

1. Start with **File ▸ New from Task** (<kbd>Ctrl+Shift+N</kbd>) to get a ranked recipe
   for your task and data, or drag blocks from the **Block Library** (or press
   <kbd>⌘K</kbd> / <kbd>Ctrl+K</kbd>) and connect them on the canvas.
2. Edit parameters in the **Inspector**; shapes and problems update live.
3. Add a dataset, loss, optimizer and Trainer block, then press **Train**.
4. Export from the **Export** menu, or browse **File ▸ Open Example** for
   complete projects (CNNs, transfer learning, tabular networks, gradient
   boosting with grid search, clustering, LLM workflows).

## In the browser

```bash
pip install -e ".[web,torch,vision,data,classic]"
aime web --open                 # http://127.0.0.1:8765
docker build -t ai-made-easy . && docker run -p 8765:8765 ai-made-easy
```

The web version shares the engine, run history and model registry with the
desktop app, and has the same tools: **Run on** and **Backends**, export formats,
**Tools** (tracking, Kaggle / Hub datasets, run history) and each model's **Share** menu. Hosting, authentication and the REST API are covered in
[docs/WEB.md](docs/WEB.md).

![Web designer](docs/images/web_designer.png)

| New from a task | Graphical model with its CPD table | A train → distill → quantize pipeline |
|---|---|---|
| ![Wizard](docs/images/web_wizard.png) | ![CPD editor](docs/images/web_probabilistic.png) | ![Pipeline](docs/images/web_pipeline.png) |

| Command palette (Ctrl/⌘ K) | Groups and sticky notes on the canvas |
|---|---|
| ![Command palette](docs/images/web_palette.png) | ![Groups and notes](docs/images/web_canvas_groups.png) |

## Block library

| Section | Blocks | Contents |
|---|---|---|
| Data | 28 | Input / Output, torchvision benchmarks, image / text / audio folders, table files, scikit-learn datasets, NumPy, JSON, Hugging Face, time series, forecasting tables, speech manifests, text corpora and pairs, COCO, YOLO, Pascal VOC, mask folders, synthetic |
| Preprocessing | 41 | Split, loader, class balancing, imputation, encoding, scaling, outlier clipping, tokenization, audio features, SpecAugment, 24 image transforms incl. RandAugment, AutoAugment, AugMix, MixUp / CutMix |
| Layers | 132 | Linear, convolution (1-3D, transposed, depthwise, separable), pooling, padding / resizing, recurrent, attention & transformers, embeddings, 27 activations, normalization, regularization, merges, tensor ops |
| Sequences & Forecasting | 14 | TCN, Mamba, S4D, sLSTM; mel spectrogram / MFCC front-ends; 8 forecasters with point, quantile, Student-t and negative-binomial heads |
| Generative | 8 | Reparameterize (VAE), causal transformer, DCGAN generator, diffusion U-Net, GPT, seq2seq transformer, discriminator, noise scheduler |
| Models | 20 | Detectors, instance / semantic segmenters, keypoint detector, U-Net family, 39 torchvision + 87 timm backbones, Hugging Face text, image and speech encoders, architecture templates |
| Training | 91 | 19 losses (incl. Dice, Tversky, Lovász, CTC), 12 optimizers, 12 LR schedulers, trainer, k-fold, 46 metrics (incl. COCO mAP, mean IoU, PCK, MASE, CRPS, WER / CER, FID, KID, perplexity, BLEU, chrF) |
| Classic ML | 68 | 55 estimators (linear, SVM, neighbors, Bayes, trees, ensembles, boosting, clustering, anomaly detection), feature engineering, TF-IDF, hyperparameter search |
| LLM | 15 | Model, tokenizer, prompts, LoRA / QLoRA fine-tuning, embeddings, vector store, retrieval, RAG |

Save any selection of blocks as a reusable custom block (**Model ▸ Save Selection as Block**).
Every block, grouped by family, and the rules that check each family:
[docs/FAMILIES.md](docs/FAMILIES.md).

Project files are plain JSON (schema 3, recording the design's family and task). Projects
saved by 1.x and 2.0 open unchanged.

## Command line

```bash
aime validate project.json                     # design + data checks; exit 1 on errors
aime gen project.json -f pytorch -o exports    # model file (pytorch | keras)
aime train project.json -f keras -o exports    # training script (pytorch | keras | sklearn)
aime run project.json                          # train headlessly, stream JSON events
aime run project.json --on ssh:gpu-box         # … or on an SSH host, kaggle, colab, modal, container
aime backends list | add NAME --kind ssh -s host=gpu-box | test NAME   # where runs train
aime runs list | show ID | compare A B         # run history
aime runs attach ID | import results.zip       # follow a remote run again; finish a Colab run
aime sweep project.json -p opt.lr=log:1e-4:1e-1 -p d1.units=choice:32,64 -n 12
aime new --task binary --data churn.csv -o churn.json   # best recipe for the data
aime automl --task binary --data churn.csv -n 12 -o best.json   # search the recipes
aime automl --task binary --data churn.csv --objectives latency_ms --device raspberry_pi_5
aime recipes --task forecasting                # recipes per task
aime pipeline run pipeline.json                # run a pipeline (resumes finished stages)
aime data profile data.csv --target label      # dataset profile and findings
aime data split project.json                   # samples per class in each split
aime deploy RUN_ID -o serving --formats onnx   # serving package from a run
aime deploy RUN_ID --list-formats              # export formats: available, suggested, why not
aime hub push RUN_ID you/model                  # a private Hugging Face model repo
aime data fetch kaggle-competition:titanic      # download, then detect like a local folder
aime runs export -o runs.parquet                # the run history for notebooks
aime serve serving                             # run it with uvicorn
aime models list | register RUN NAME | stage NAME VERSION production
aime import onnx model.onnx -o project.json    # also: pytorch (--attr, --shape), keras
aime web                                       # browser UI + REST API
aime blocks                                    # every block as a JSON schema
```

## Agents and assistants

The same engine is available to AI agents through an MCP server (`aime-mcp`),
with tools to list blocks, validate graphs, generate code, train, run sweeps,
deploy, import models and profile data.
The in-app **Assistant** panel connects to any OpenAI-compatible endpoint,
including local servers such as Ollama or LM Studio
(`AIME_ASSISTANT_BASE_URL`, `AIME_ASSISTANT_API_KEY`, `AIME_ASSISTANT_MODEL`).

## Desktop bundle

```bash
scripts/build_app.sh      # PyInstaller: macOS .app, Windows / Linux folder in dist/
```

The bundle contains the designer; training and export runs use a Python
environment with the frameworks installed (`$AIME_PYTHON`, or `python3` on the
`PATH`).

## Development

```bash
pip install -e ".[all,dev]"
ruff check ai_made_easy scripts tests
pytest                                  # ~1,800 tests on four workers, about five minutes
AIME_TESTS=all pytest                   # also the ~300 slow ones (real training), as CI does
cd web && npm ci && npm run build && npx playwright test   # web UI
UPDATE_GOLDEN=1 pytest tests/test_golden.py
python scripts/build_examples.py        # regenerate example projects
python scripts/verify_codegen.py --keras
python scripts/verify_codegen.py --coverage   # which blocks the codegen harness runs, per family
python scripts/gen_families.py          # regenerate docs/FAMILIES.md
AIME_ONLINE=1 pytest tests/test_online.py   # downloads real pretrained weights (nightly CI)
```

Architecture and design rules: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Adding a block:
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-block).
Web version: [docs/WEB.md](docs/WEB.md). Roadmaps: [docs/ROADMAP_3.0.md](docs/ROADMAP_3.0.md),
[docs/ROADMAP_3.1.md](docs/ROADMAP_3.1.md).
Release notes: [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
