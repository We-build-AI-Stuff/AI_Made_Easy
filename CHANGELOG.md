# Changelog

## 3.10.0

Resume Keras runs and runs on other machines, train on several GPUs, compile and
checkpoint networks, check fairness by group, and export to OpenVINO. 762 blocks, 72
recipes. See [docs/ROADMAP_3.10.md](docs/ROADMAP_3.10.md) for what is in this release.

### Added
- **Resume Keras runs:** Keras scripts save the weights, optimizer, random state and the
  early-stopping and plateau counters after each epoch; a resumed run ends exactly where
  an uninterrupted one does.
- **Resume runs on other machines:** a run trained over SSH, on Modal, in a container, on
  Kaggle or on Colab resumes on the same backend. Its saved state comes back with the
  outputs and goes out again with the resumed run.
- **Several GPUs** (Trainer `gpus`): one process per GPU with DistributedDataParallel;
  the first one reports and saves.
- **Compile** (Trainer `compile`) runs the network through `torch.compile`, falling back
  to plain PyTorch with the reason when it can't.
- **Gradient checkpointing** (Trainer `gradient_checkpointing`) recomputes activations
  instead of keeping them: the same results with less memory. The budget's memory and
  step-time estimates account for it.
- **Fairness by group:** results by group also compare how often each group is predicted
  each class and its true- and false-positive rates with all test rows, apply the
  four-fifths rule, and name the groups treated differently.
- **OpenVINO export** (`openvino` extra): an OpenVINO model folder, checked against
  PyTorch on x86 CPUs; a CI job on Linux.
- **Choose PyTorch or Keras in both apps:** the Training panel offers the frameworks
  every layer supports. A Keras design finds its learning rate with Keras.

### Changed
- The desktop's canvas actions and the web designer's bottom panels moved to their own
  modules (no change in behaviour).

### Fixed
- Downloading a remote run's outputs over SSH from a Mac no longer adds macOS `._` files.

## 3.9.0

Resume a run where it stopped, train a boosted model more, faster sequence models, Keras
versions of 15 more layers, and Core ML checked on macOS. 762 blocks, 72 recipes. See
[docs/ROADMAP_3.9.md](docs/ROADMAP_3.9.md) for what is in this release.

### Added
- **Resume an interrupted run** (both apps, `POST /api/runs/{id}/resume`,
  `aime runs resume`): PyTorch runs save what training needs after each epoch (weights,
  optimizer, schedule, averaged weights, the best weights so far and the random state). A
  run that was stopped, failed or cut off by a crash continues after its last saved
  epoch, as the same run, and ends exactly where an uninterrupted run would.
- **Train a boosted model more** (both apps, `aime runs more`): XGBoost, LightGBM,
  CatBoost and scikit-learn's gradient boosting keep their trees and add more, on the same
  training rows and class balancing; the run reports the new tree count.
- **Find a learning rate for Keras designs** (`framework` in `POST /api/lr-find`), and
  `aime lr-find project.json [--apply]` on the command line.
- **Keras versions of 15 layers** that were PyTorch-only: differential, Linformer, latent,
  QK-norm and gated attention; retention, RWKV, Hyena, Mamba-2, Gated DeltaNet, gated
  linear attention and Griffin; involution, MobileNetV4's UIB and StarNet's block. With
  the same weights they compute the same outputs as PyTorch.
- **Core ML on macOS:** exports run on the Mac and must match PyTorch (within float16
  precision); a `coreml` extra and a macOS CI job. 24 of the 27 layers from 3.7 and 3.8,
  and 33 samples, convert and match.

### Changed
- Mamba-2, Gated DeltaNet and gated linear attention train in chunks, Griffin with a
  parallel scan: the same results, 2-10 times faster on long sequences.
- TimesNet exports to ExecuTorch, float and int8. DLRM exports to ExecuTorch too.
- GitHub Actions moved to their Node 24 versions.

### Fixed
- The MCP server on Windows stopped answering at its first tool call.
- Core ML exports failed for most attention and sequence layers with NumPy 2.5.
- A pipeline stage whose run crashed without a traceback said only "the run failed"; it
  now gives the exit code and the run's last output.
- Export limits now say when Core ML can't take a layer: Hyena, deformable convolution,
  involution, Glow's invertible 1x1 convolution, the spline flow.

## 3.8.0

Find a good learning rate before training, continue a finished run, see where a model is
weak by group, and 15 more blocks; Python 3.13 and 3.14 tested. 762 blocks, 72 recipes.
See [docs/ROADMAP_3.8.md](docs/ROADMAP_3.8.md) for what is in this release and what is
still open.

### Added
- **Find a learning rate** (both apps; `POST /api/lr-find`): from an optimizer block, a
  short run raises the learning rate from 1e-7 batch by batch until the loss blows up,
  plots loss against rate and suggests where the loss falls fastest. **Use** sets it on
  the block. Nothing is saved; PyTorch designs.
- **Train more** (both apps; `POST /api/runs/{id}/more`): a finished network continues
  from its best weights for more epochs, as a new run whose note names the one it
  continues, with an optional learning-rate scale.
- **Results by group** (both apps; `POST /api/runs/{id}/groups`): pick a column of the
  run's table (numbers with many values are split into quartiles) and the run's own test
  rows are scored for each value. Groups at least 10% worse than the whole test set, with
  enough rows to trust, are flagged. Networks and classic models trained on a table.
- **Forecasting:** NLinear, iTransformer, TSMixer and TimesNet forecasters.
- **Tabular:** TabM (parameter-efficient ensembles of MLPs).
- **9 more layers** (PyTorch): multi-head latent attention, QK-norm attention and gated
  attention; Mamba-2, Gated DeltaNet, gated linear attention and Griffin's RG-LRU block;
  MobileNetV4's universal inverted bottleneck and StarNet's star block.
- **Optimizer:** ADOPT.
- **Recipes:** iTransformer forecaster (with 10-90% intervals) and Regression with
  uncertainty (a mean and a variance per row, plus a conformal 90% interval).

### Changed
- CI tests Python 3.11 to 3.14; pandas 2 on 3.11 and pandas 3 on the rest. The dev extra
  installs `httpx2`, which Starlette's test client prefers.
- The Windows test job runs as two halves in parallel.
- Internal: the largest modules are split (`training/train_helpers.py`,
  `torch_template_data.py` / `torch_template_main.py`, `ui/features/experiment_dialogs.py`);
  generated scripts are unchanged.

### Fixed
- A test worker that died could leave the test run hanging; it now fails at once with a
  traceback.
- Waiting for a run could return before its MLflow / W&B mirror was marked finished.
- A data file rewritten twice within one clock tick (same size and modified time, as on
  Windows) kept its old fingerprint.
- TimesNet designs say they can't be exported to ExecuTorch int8, instead of failing
  during export.
- Older roadmaps no longer list as open the items later releases shipped.

## 3.7.0

Windows as a tested platform, which settings matter together in a sweep, and 31 more
blocks. 747 blocks, 70 recipes. See [docs/ROADMAP_3.7.md](docs/ROADMAP_3.7.md) for what is
in this release and what is still open.

### Added
- **Windows in CI:** the test suite runs on Windows (Python 3.12), and the Windows
  desktop app is built and started on every pull request.
- **Which settings matter together** (sweeps, both apps): from 10 finished trials,
  importance comes from a random forest fitted to the trials, split into each setting's
  own effect and each pair's joint effect (functional ANOVA, the idea behind fANOVA). The
  strongest pairs are listed, with a heatmap of the predicted score over the top pair.
- **Sweep again near the best** (both apps; `GET /api/sweeps/{id}/again`): a new sweep
  with each range narrowed to where the best quarter of the trials were, opened in the
  sweep form for review before it starts.
- **18 more layers:** sliding-window, ALiBi, differential and Linformer attention;
  Perceiver latents; RetNet retention, an RWKV block and a Hyena operator; MLP-Mixer, gMLP
  and PoolFormer blocks; Ghost module, Involution, deformable convolution, Fused-MBConv,
  ECA and Global Response Normalization; Dynamic Tanh. 11 have Keras forms.
- **Training:** PolyLoss, an asymmetric loss for multi-label, Charbonnier and SSIM (+ L1)
  losses; VICReg and Barlow Twins self-supervised losses; the Prodigy, Cautious AdamW and
  Yogi optimizers.
- **Metrics:** PSNR, SSIM and concordance (Lin's CCC), in neural and classic pipelines.
- **Augmentation:** photometric distort.
- **Recipes:** Tiny RWKV (a recurrent language model) and Barlow Twins (self-supervised
  image features).

### Changed
- The web app is built with TypeScript 7.0.
- Child processes (training workers, Try it, generated scripts) write UTF-8 to the app,
  and the app reads it as UTF-8, on every OS.

### Fixed
- Windows: generated scripts failed to run when the data path (C:\Users\…) appeared in
  their header, which Python read as an escape sequence.
- Windows: `$AIME_DOCKER` and `$AIME_SSH` lost the backslashes of a Windows path.
- Windows: deleting a run right after it finished could fail while the worker was still
  exiting; export metadata recorded `exports\model.onnx` instead of `exports/model.onnx`.
- RWKV and Hyena designs say they can't be exported to ExecuTorch, and deformable
  convolution to ONNX or ExecuTorch, instead of failing during export.

## 3.6.0

Try a model on a whole file, see which settings mattered in a sweep, and 41 more blocks.
716 blocks, 68 recipes. See [docs/ROADMAP_3.6.md](docs/ROADMAP_3.6.md) for what is in
this release and what is still open.

### Added
- **Try it on a whole file** (both apps; `POST /api/runs/{id}/try/file`): a CSV table, a
  CSV of numbers, a text file, or a zip of images or WAV clips comes back as a CSV of
  predictions (up to 10,000 inputs), computed on this machine.
- **Grad-CAM in Try it:** an image classifier's prediction can come with a heatmap of
  where it looked. It is computed from the loaded network, so runs trained before 3.6
  explain too.
- **Which settings mattered** (sweeps and AutoML, both apps; `GET
  /api/sweeps/{id}/analysis`): a parallel-coordinates chart of the finished trials and
  each setting's importance, with where the best scores were.
- **Comparing runs** lists each run's design changes against the first run.
- **The blocks 3.5 left open:** a Gaussian head (mean and variance, trained with the
  Gaussian negative log-likelihood), a multi-label hinge loss, labelled triplets for the
  Triplet loss, and L-BFGS.
- **11 more layers**, PyTorch and Keras: ResNet basic and bottleneck blocks, MBConv,
  CBAM, DenseNet's dense block, CoordConv2D, a Kolmogorov-Arnold (KAN) layer, Highway,
  Maxout, Sparsemax and Snake.
- **Training:** the AdaBelief, Adan and AdEMAMix optimizers; inverse-square-root and
  warmup-stable-decay schedules; quantile (pinball) and Tweedie losses.
- **Metrics:** Brier score, Hamming loss, Jaccard index, Pearson and Spearman
  correlation, max error and mean squared log error, in neural and classic pipelines.
- **Augmentation:** random JPEG quality, elastic transform, random channel permutation,
  Gaussian noise.
- **scikit-learn:** Gaussian process classifier and regressor, radius-neighbours
  classifier and regressor, Gamma regression; random trees embedding, dictionary
  learning and mini-batch NMF.

### Changed
- Family trainers (forecasting, vision tasks, …) refuse SAM and L-BFGS with a clear
  message instead of failing during training.
- The `vision` extra needs torchvision 0.21.
- Minor dependency updates (fastapi, transformers, React, Vite, Playwright).

### Fixed
- Try it: loading one run's model no longer makes predictions from other runs wait.
- Elastic Transform with a sigma too large for the images is a design error that says
  which value fits, instead of a crash during training.
- Windows: text files (designs, predictions, model cards, generated scripts) are read
  and written as UTF-8, so names with accents no longer fail or come back garbled. A
  test keeps every text read and write explicit.

## 3.5.0

Try a trained model, see what it attended to, and 86 more blocks from the libraries you
use. 675 blocks, 68 recipes. See [docs/ROADMAP_3.5.md](docs/ROADMAP_3.5.md) for
what is in this release and what is still open.

### Added
- **Try it** (Experiments page, both apps): run a finished model on your own input. The
  form follows what the model takes: one field per column for a table, a text box, an
  image or WAV file, or a JSON list for anything else. It shows the predicted label with
  the top classes. The model is loaded once, in the run's own folder and interpreter,
  and answers on this machine; nothing is sent anywhere. Also `GET` / `POST
  /api/runs/{id}/try`.
- **Attention Maps** (`eval.attention_maps`, PyTorch): after training, the last attention
  layer's weights on a few test samples, saved with the run and shown in both apps. A
  vision transformer's patch attention is drawn over the image; text and sequences save
  the attention matrix. Works with `nn.MultiheadAttention`, the Transformer layers and
  the attention blocks (GQA, Llama, and others).
- **Audio Saliency** (`eval.audio_saliency`, PyTorch): which parts of each test clip's
  spectrogram, or of a raw waveform, moved an audio classifier's prediction.
- **21 more layers** from `torch.nn` and Keras 3: constant padding 1–3D, channel
  shuffle, cosine similarity, pairwise distance, EmbeddingBag, Unfold / Fold,
  fractional max pooling 2–3D, LP pooling 1D / 3D, Softmax2d, Unflatten, feature alpha
  dropout, the full encoder-decoder `nn.Transformer`, ConvLSTM2D, additive (Bahdanau)
  attention, and spectral-normalized Dense / Conv2D. Each one is checked against the
  designer's shapes and PyTorch's parameter count, and has a Keras form where one is
  faithful.
- **Losses:** hinge, squared hinge, soft margin, mean squared log error, mean absolute
  percentage error, and generalized cross-entropy (robust to wrong labels).
- **The Muon optimizer:** orthogonalized momentum for weight matrices, AdamW for the
  rest (Keras 3.10 or later for Keras scripts).
- **35 more scikit-learn estimators:** Nu-SVC / SVR, the cross-validated logistic,
  ridge, lasso and elastic-net models, LARS and Lasso-LARS, OMP, ARD, Theil-Sen, RANSAC,
  quantile and Tweedie regression, passive-aggressive models, nearest centroid,
  complement naive Bayes, single extra trees, bagging regression, dummy baselines, a
  calibrated linear SVM, kernel ridge, PLS, affinity propagation, mean shift, OPTICS,
  bisecting K-Means, a Bayesian Gaussian mixture, elliptic envelope and an SGD one-class
  SVM.
- **21 more scikit-learn transformers:** max-abs scaling, a row normalizer, a
  binarizer, variance threshold, percentile / FPR / FDR / FWE feature selection,
  incremental and sparse PCA, factor analysis, NMF, LDA topics, feature agglomeration,
  Gaussian and sparse random projections, random Fourier, skewed and additive chi²
  features, a polynomial kernel sketch and locally linear embedding.

### Changed
- **Web canvas:** a folded group's card can be dragged, and its blocks move with it;
  wires that end inside it are dashed stand-ins, as on the desktop.
- **Tests:** a plain `pytest` now runs on four workers and leaves out the slow tests
  (`tests/slow_tests.txt`): about three minutes, against 45–60. `AIME_TESTS=all` runs
  everything, as CI does.
- `ai_made_easy.egg-info/` is no longer tracked.

### Fixed
- Pipelines with a transformer that needs non-negative features (NMF, LDA, chi²) warn
  without Min-Max scaling, and the scaler then clips test rows to its range.
- Every univariate feature selector, not only Select K Best, checks its score function
  against the task.
- `test_tcn_receptive_field` no longer depends on the tests that ran before it.

## 3.4.1

A fix release.

### Fixed
- Desktop: a window's services and timers (validation, autosave) no longer outlive
  the window. They kept updating its widgets after it closed; the context now closes
  with its window, and closing stops autosave.
- Desktop: the running-wire animation no longer touches wires of a closed canvas.
- Tests: windows are deleted when each test ends, which removes the "shared QObject
  was deleted directly" message at the end of full runs; the canvas-PNG test runs its
  subprocess against the checkout, not an installed copy.

## 3.4.0

Explain image models, learn from pairs, and fold groups away. 589 blocks, 68 recipes.
See [docs/ROADMAP_3.4.md](docs/ROADMAP_3.4.md) for what is in this release and what is
still open.

### Added
- **SHAP values:** a third way to explain a run's features (expected gradients against
  training rows), beside permutation importance and Integrated Gradients. `pip install
  'ai-made-easy[explain]'`.
- **Collapse a group** into one card (▾ on its title, or Edit ▸ Collapse / Expand Group),
  in both apps. Wires to its blocks end on the card; the design and its code don't change,
  and a folded group stays folded when the design is reopened.
- **SimCLR:** the NT-Xent loss learns image features without labels, from two random
  augmentations of every image, and the **SimCLR (self-supervised)** recipe trains an
  encoder with it. Labels only measure the result.
- **Cosine Embedding and Margin Ranking losses** for embeddings. The pairs come from the
  class labels in each batch, so they work with any dataset that has classes.
- **Grad-CAM Heatmaps** (`eval.grad_cam`): after training, an image classifier saves
  heatmaps of where it looked for a few test predictions. The heatmaps are kept with the
  run and shown on the Experiments page in both apps; until now, Grad-CAM was only in the
  desktop's one-image Inspect dialog.
- **Labelled Pairs** (`data.pairs`): a table where each row is two items (the `a_` and
  `b_` columns) and a label. The model is an encoder that embeds both items.
  - Cosine Embedding learns whether a pair is the same or different.
  - Margin Ranking learns which item ranks higher, from one score per item.
  - Every binary metric applies to each pair's verdict. After training, the model embeds
    (or scores) single items.
  - The new **Pairs — same or different (siamese)** recipe trains on generated demo
    pairs until you point it at your own table.

  589 blocks, 68 recipes.

### Changed
- On the desktop, wires into and out of a folded group now end on the card as dashed
  stand-ins and follow it when it moves, as they do on the web.

### Fixed
- **Serving a metric-learning model** (ArcFace, CosFace, triplet, supervised contrastive,
  SimCLR) read its embedding as class scores, so served labels were wrong. The trained
  loss's class centres are now saved with the model and score each embedding, and each
  result also includes the embedding. Retrain older runs to serve them.
- Desktop: undoing the creation of a folded group left its blocks hidden. Undo now shows
  them again, and redo folds them away again.
- Desktop: a folded card that was moved, saved and reopened jumped back to where it was
  folded. It now opens where you left it.
- Feature explanations are skipped, with a warning, for labelled pairs; they explain
  single rows. Grad-CAM shows at most 12 images, the most a run displays.

## 3.3.0

More blocks, a canvas that keeps up, and a more polished web app. 583 blocks, 66 recipes.
See [docs/ROADMAP_3.3.md](docs/ROADMAP_3.3.md) for what is in this release and what moves
to the next one.

### Added
- **Modern network parts** (Transformer Parts, Vision Blocks):
  - Rotary embedding (RoPE), grouped-query attention, a SwiGLU / GeGLU feed-forward and
    a Llama decoder block (RMSNorm, GQA with rotary positions, SwiGLU).
  - Mixture of Experts: top-k routed SwiGLU experts with a load-balancing loss.
  - LayerScale, DropPath (stochastic depth), FiLM conditioning, LoRA Dense.
  - Patch Embedding (ViT), ConvNeXt block, Swin Transformer block (shifted windows with a
    relative position bias), Conformer block.
  - Every one is checked against the designer's shapes and parameter counts. Rotary
    embedding, the gated feed-forward, LayerScale and DropPath also generate Keras.
- **Optimizers:** Lion, LAMB and Lookahead (AdamW).
- **Weight averaging:** the Trainer's `ema_decay` evaluates and keeps an exponential
  moving average of the weights (PyTorch and Keras).
- **Classic ML:** Isomap, Nystroem kernel features and UMAP (`pip install
  'ai-made-easy[umap]'`).
- **Recipes:** Tiny Llama (language modelling), Vision Transformer and ConvNeXt (image
  classification, from scratch).
- **Canvas conveniences (desktop and web):**
  - Copy, paste and duplicate blocks (Ctrl/⌘ C, V, D) with their settings and the wires
    between them. The clipboard holds design JSON, so blocks copied in one app paste in
    the other, and a saved design pastes too.
  - Groups (Ctrl/⌘ G): a titled frame around blocks. On the desktop it moves them
    together; Ctrl/⌘ Shift U ungroups.
  - Sticky notes. Groups and notes are saved with the design and never reach the code.
  - Drop a data file (CSV, Parquet, Excel, JSON, text, or a zip of class folders) on the
    window to start a project from it: the wizard opens with the data detected.
- **Feature Importance** block: after training, permutation importance on the test set,
  shown with the run on the Experiments page (PyTorch, tables and feature vectors).
- **Compare with the canvas:** what changed since a run (blocks, parameters, wires).
- **Metric learning:** ArcFace, CosFace, batch-hard Triplet and Supervised Contrastive
  losses. The network learns an embedding; accuracy is measured against the class
  centres, and an **ArcFace embeddings** recipe trains one on images.
- **Optimizers:** SAM (sharpness-aware, around SGD or AdamW) and Schedule-Free AdamW.
  The Trainer's `swa_start` averages the weights of every epoch from then on (SWA).
- **Integrated Gradients:** a second way to explain a run's features, beside permutation
  importance.
- **Classic ML:** Target Encoding for categories, and SMOTE oversampling (`pip install
  'ai-made-easy[imbalance]'`), applied while fitting only, inside cross-validation too.
- `aime new --data table.csv` picks the task from the data. 583 blocks, 66 recipes.
- **Web app, matching the desktop:**
  - A command palette (Ctrl/⌘ K) for every command, and find a block on the canvas
    (Ctrl/⌘ F, or `@` in the palette).
  - A keyboard shortcuts sheet (`?`), select all (Ctrl/⌘ A) and zoom (Ctrl/⌘ =, −, 0).
  - A File menu with recent projects, and Export canvas as PNG.
  - Questions such as "Delete 3 runs?" and the access-token prompt now appear in the app
    instead of as browser pop-ups.
  - Messages stack (up to four), repeats are counted, and each has a × to close it.
  - Lists show loading, error (with Try again) and empty states.
  - A restored autosave can be discarded ("Start fresh").
  - Add a block from the library with the keyboard (Enter), and every icon button has a
    label for screen readers.
- **Desktop:** a minimap (View ▸ Minimap, Ctrl+Shift+M) that you can click or drag to move
  around, and screen-reader names for the panels.
- **Both apps:** a three-step first-run tour (Help ▸ Show the Tour) and one chart palette,
  so a run's curve has the same colour in both.

### Fixed
- The web app's Ctrl/⌘ S shortcut was registered again on every render.
- The New Project wizard kept data from Tools ▸ Download a Dataset only until a task
  was picked; it now starts on that data and keeps it.
- Dropping a file on the desktop canvas tried to open it as a canvas session file.
- On Windows, checking whether a run was still alive terminated its process. The check
  no longer sends a signal, and a reused process id no longer counts as the run.
- The web server marks runs a crash left "running" as failed when it starts, and a run's
  live view ends once no process is running it, instead of waiting forever.
- The access token in download and WebSocket links is hidden in the server's access log.
- Uploads are limited to `AIME_MAX_UPLOAD_MB` (default 2 GB) and streamed to disk. A
  results zip is checked for its unpacked size before it is extracted.
- Stopping a run that had just finished no longer marks it stopped. Following a remote
  run again from two places starts one poller.
- Long runs and long-lived servers no longer keep every event in memory.
- If a Modal job can't be stopped, the run log says so and how to stop it yourself.
- Imports, augmentation previews, export checks and `aime run` use the configured
  Python (`AIME_PYTHON`) instead of the app's own binary in the packaged app.
- Exporting run history no longer leaves a temporary folder behind, and two saves of
  one project at once no longer share a temporary file.
- Docker image builds and `pip install` on remote machines have timeouts.
- An ensemble whose member sizes can't be estimated gets a warning instead of passing
  the parameter limit silently.

## 3.2.0

3.2 makes the app solid and gets designs out of it. Runs train on SSH hosts, Kaggle,
Colab, Modal or a container as well as this computer. Trained models export to ONNX,
TorchScript, Core ML, ExecuTorch and LiteRT, each format checked against the model. Runs
and models reach the tools teams already use: MLflow, Weights & Biases, the Hugging Face
Hub and Kaggle. AutoML covers every family, and the models put off in 3.0 and 3.1 land:
Glow, the Open Graph Benchmark, Atari, your own Gymnasium environments, and QAT for text
models. It has 554 blocks, 62 recipes and 35 tasks.

AI Made Easy still never asks for, stores or logs a password, token or key. Every
integration uses the tool's own sign-in and is off until you choose it.

### Added
- **Train anywhere** ([REMOTE.md](docs/REMOTE.md)):
  - **Run on** (desktop and web): this computer, any SSH host in `~/.ssh/config`
    (including rented GPU machines), a Kaggle notebook, a Colab hand-off, a Modal sandbox,
    or a Docker container.
  - Remote runs stream the same events as local ones, so live curves, Stop, sweeps,
    pipelines and AutoML all work unchanged.
  - A dropped SSH connection resumes with no epoch lost, and `aime runs attach` follows a
    run again after a restart.
  - Each backend reports its accelerator, which picks the device the budget is checked
    against.
- **Export everywhere** ([EXPORT.md](docs/EXPORT.md), [DEPLOY.md](docs/DEPLOY.md)):
  - Formats: ONNX for detectors (post-processing included) and graph networks,
    ExecuTorch (`.pte`) and LiteRT (`.tflite`), with int8 calibrated on the run's own
    samples.
  - Every format is run against the trained model before it is packaged.
  - The budget device suggests a format, and a design rule names why a design can't
    reach one.
  - `docs/DEPLOY.md` lists, for every task, what `/predict` takes and which formats it
    exports. It is generated from the code.
- **Work with your tools** ([TOOLS.md](docs/TOOLS.md)):
  - Runs mirror to MLflow and W&B as they train, on every backend.
  - Registered models go to MLflow's registry, W&B, a Hugging Face model repository or a
    Docker Space. Publishing is confirmed every time and private by default.
  - `kaggle:`, `kaggle-competition:` and `hf:` datasets download once and are detected
    like a local folder. Kaggle submissions are confirmed every time.
  - MLflow PyTorch / ONNX models open as designs (`aime import mlflow`).
  - The run history exports to CSV or Parquet.
  - New extras: `tracking` and `hub`.
- **AutoML for every family:**
  - held-out metrics for each family: log-likelihood (Bayesian networks, HMMs), PSIS-LOO
    ELPD (PyMC), NLPD (GPs), MASE (state-space), bits per dimension (flows), ROC AUC
    (labelled anomaly detection)
  - cost estimates for classic and probabilistic designs; classic runs measure their
    predict latency and size, so one Pareto front mixes networks and classic models
  - tuning knobs for the probabilistic recipes
  - trials can train several at a time and on any backend (`--concurrency`, `--on`);
    the tuned trials start after the baseline round is scored
- **More models:**
  - **Glow:** 1×1 invertible convolutions, convolutional couplings, squeeze and
    multi-scale split blocks, image density data, and a Glow recipe scored in bits per
    dimension.
  - **Open Graph Benchmark:** node, graph and link datasets with their official splits
    and evaluators (`ogb` in the `graph` extra).
  - **Atari** games through ale-py, with the DQN-paper preprocessing and frame stacking
    blocks and an Atari PPO recipe (the `atari` extra). A design rule reports a missing
    ROM ([RL.md](docs/RL.md)).
  - **Your own Gymnasium environments:** the spaces are read in a separate process while
    you design and checked against the network.
  - **QAT for text models:** embedding tables get weight-only quantization, so
    quantization-aware training works for designs with embeddings.
- **Web parity:**
  - a **Backends** button in the header
  - **Tools** (tracking for the project, Kaggle / Hub datasets that open the wizard with
    the data detected, run-history downloads)
  - a model's **Share** dialog (export formats, MLflow, W&B, Hub, Space)
  - **Pull from the Hub**
  - MLflow models in **Import model**

### Fixed
- **CI:** every job has a timeout, and pytest-timeout fails a hung test with every
  thread's stack. A recovery prompt that hung headless test runs is skipped in tests.
- **Packages:**
  - DLRM and class-conditional diffusion packages failed verification.
  - VAE, GAN, diffusion and Stable Diffusion LoRA packages were described as image
    classifiers and had no `/generate`.
  - Stable Diffusion LoRA runs could not be packaged at all.
  - Flow packages were described as image models once flows accepted images.
- **Downloads:** Torchvision downloads had no socket timeout, so a dead network hung the
  run.
- **Remote runs:**
  - Starting an SSH run blocked until training ended.
  - Bundle files had a 1970 timestamp, which Colab's results zip refused.
  - The web Training panel missed a run's first log lines.
- **Web:** download links (serving packages, run exports) carry the access token, so
  they work when the server requires one.
- **Large designs:** 1000 blocks validate in 20 ms (was 292 ms). The desktop canvas
  opens 500 blocks in about 0.3 s (was 22 s).

### Known limitations
- A Modal run can't be followed again after the app restarts; its outputs stay in the
  `aime-runs` volume.
- Kaggle has no live log API, so a Kaggle run's curves arrive when the kernel finishes.

## 3.1.0

3.1 finishes what 3.0 put off rather than opening new model areas. Every design now has a
task. Every family's recipes train on your own data, not just the demo datasets. The
wizard offers pretrained text and audio models. Pipelines add quantization-aware
training, discriminative learning rates, stacking and classic teachers. 544 blocks fill
the gaps across vision, sequences, probabilistic models, generative models and RL. AutoML
can trade accuracy against latency or size.

### Added
- Every design now has a task, so the wizard, Explain, the status bar and the project
  lists say what it does. Classic pipelines resolve to multi-class or binary
  classification, regression, and two new tasks, **clustering** and **anomaly
  detection**. LLM workflows get **text generation**, **LLM fine-tuning** and
  **retrieval-augmented answering**. A pipeline takes the task of the design its Train
  stage trains.
- Recipes for the new tasks: k-means, Gaussian mixture and HDBSCAN clustering; isolation
  forest and local outlier factor anomaly detection; and prompting, LoRA fine-tuning and
  RAG starters from the LLM samples. Every table now also offers clustering and anomaly
  detection, and AutoML searches clusterings by silhouette.
- Clustering and anomaly detection train on tables with or without labels. A label
  column is never trained on: it scores clusters (adjusted Rand, NMI), and 0 / 1 labels
  evaluate a detector (precision, recall, ROC AUC, with the rarer value as the anomaly).
- Design rules: a classifier on a continuous target, a regressor on a categorical
  target, a label column that clustering ignores, and labels that cannot evaluate an
  anomaly detector. The target warnings now name the estimator for classic pipelines.
- **File ▸ Open Recent** shows each project's family and task; the status bar shows the
  task for every family.
- Your own data for the families that trained only on demo data. The wizard now swaps
  your data into these recipes:
  - an image folder (flat, or one sub-folder per class) into the VAE, DCGAN and diffusion
    recipes, in colour or greyscale; with class folders, the diffusion U-Net is
    class-conditional
  - a table's numeric columns into the normalizing flows
  - a table with a group column (school, store, region, ...) into the hierarchical PyMC
    models, with the group, predictor and outcome columns set and the priors scaled to
    the data
  - a time series into the structural and SARIMAX models (the season length comes from
    the timestamps) and into the hidden Markov model
- Recommenders read optional user and item feature tables beside the interaction log,
  and DLRM's input widens to use them. Data detection finds them automatically.
- **Graphs CSV** dataset: node and edge tables keyed by a graph id plus a graph table of
  labels, for graph classification. A folder with node and edge tables is detected as
  one graph or, with a graph id, as many graphs.

- **Pretrained recipes:** text classification on a Hugging Face encoder (DistilBERT,
  MiniLM, ELECTRA, BERT, RoBERTa, DeBERTa-v3), audio classification on wav2vec 2.0 /
  HuBERT / WavLM, and image classification on timm models or DINOv2. Each comes as a
  linear probe (frozen encoder) and a fine-tune. The wizard shows what each download is,
  its size and its licence, from the model zoo, and offers pretrained recipes only when
  they fit the budget.
- The Pretrained Text Encoder block gets a `weights` setting (`none` builds the same
  architecture with random weights, offline) and parameter and FLOP estimates; budgets
  now treat frozen timm and Hugging Face encoders as frozen. New `text` extra.
- A model-zoo catalog of the Hugging Face encoders (`core/zoo/hf_encoders.json`):
  feature width, parameters, download size and licence.
- **Advanced pipelines:**
  - Fine-tune gets discriminative learning rates: with `lr_decay`, each earlier layer
    group learns more slowly (geometric decay over `layer_groups`), alongside the
    freeze / unfreeze schedule.
  - New **Quantization-aware training** stage: prepare (fake quantization), fine-tune,
    then convert to int8. Its metrics compare the float model, post-training dynamic
    int8 and QAT int8, with their sizes. On digits, QAT int8 stays within a point of
    float at about a quarter of the size, and the run folder deploys like any other.
  - **Stacking ensembles:** Cross-validate saves every example's out-of-fold prediction
    and, with `refit`, trains and passes on the final model. Ensemble's new `stack`
    method fits a logistic-regression (or ridge) meta-learner on them, and reports the
    plain average for comparison.
  - **Classic teachers:** Distill accepts a scikit-learn / boosting teacher; its
    `predict_proba` (or `predict`) on the student's training rows gives the soft
    labels.
  - **Composite view:** a pipeline stage expands on the canvas to show the design it
    trains: its layers with their shapes, then its data and training blocks. On the web,
    use the ▸ toggle; on the desktop, **Model ▸ Expand** (Ctrl+E) on a selected stage.
  - New sample `pipeline_advanced.json` combines all four.
- The training script gains `PARAM_GROUPS` / `PREPARE_HOOK` pipeline hooks, and an int8
  serving path for QAT models. `train.kfold` gets `refit`.
- **Block coverage** (544 blocks, up from 531):
  - **Detectors from blocks:** an **FPN Neck** fuses three feature maps, and an anchor-free
    **YOLO-style Head** detects on features from your own convolution blocks. Together
    they make a detector you can read end to end (new sample `yolo_style_detection.json`:
    mAP@.5 0.9 on the shapes after three epochs from scratch). **GIoU**, **DIoU** and
    **CIoU** box losses train the head.
  - **Mask2Former** (Hugging Face, Swin backbone) for semantic segmentation, trained with
    its own matching loss.
  - **Temporal Fusion Transformer (lite)** forecaster: variable selection, an LSTM
    encoder-decoder and attention over past and known-future inputs, with every
    distribution head.
  - **AST** (Audio Spectrogram Transformer) joins the speech encoders. Its Kaldi
    filterbanks are computed inside the model and match Hugging Face's feature extractor.
  - LSTM `proj_size` (LSTMP), with exact parameter counts.
  - **BPE Tokenizer** for language models: byte-level merges learned on the training
    text, saved with the model. The tiny GPT sample reaches 0.81 bits per byte in three
    epochs, against 1.42 on bytes.
  - **Multivariate Normal** and **LKJ Correlation** distributions. With `group`, a vector
    per level (one plate), indexed in expressions as `ab[0]`. Correlated intercepts and
    slopes sample non-centred, and the report lists the learned correlations.
  - **MMHC** structure learning (max-min parents and children, then hill climbing).
  - **Quantile Head** trained with the pinball loss. With Conformal Prediction it becomes
    conformalized quantile regression: the interval is calibrated on the validation
    set, and the served prediction returns the quantiles and the interval.
  - **Time-varying regression coefficients** in state-space models: random-walk
    coefficients for structural models and SARIMAX, with their paths in the report.
  - **Inception Score** metric for image generators.
  - **Stable Diffusion LoRA** (new `generative` extra: diffusers and peft): adapters on
    SD 1.5 / 2.1-base learn a style or a subject from an image folder, with live sample
    grids, diffusers-format weights and prompt-to-image serving. `weights = none` builds
    a tiny random Stable Diffusion, so the whole path runs offline. New wizard recipe and
    `stable_diffusion_lora` sample.
  - Reinforcement learning on pixels: image observations reach the designed network as
    `[C, H, W]`, so convolutional policies work. New **PixelChase** environment (built
    in) with the sample `rl_pixel_chase_cnn.json`, and the **Box2D** environments
    LunarLander, BipedalWalker and CarRacing (new `rl-box2d` extra; sample
    `rl_lunar_lander_ppo.json`).
  - `scripts/verify_codegen.py --coverage` reports, per family, which blocks the codegen
    harness runs; every layer block has a case. New `CONTRIBUTING.md` with a guide to
    adding a block.
- **Multi-objective AutoML:** `objectives` (`latency_ms`, `params_m`, `model_mb`) makes a
  search trade the task's metric against design costs, estimated on the budget's device
  (or `device`).
  - TPE samples towards the trade-off curve, and median pruning is off: a weaker but
    faster trial may belong on the curve.
  - Only neural recipes take part, since they are the ones with a cost model.
  - The leaderboard marks the **Pareto front**, the trials that no other trial beats on
    every objective, and lists it first. Every trial records its costs.
  - Desktop: a trade-off menu in New from Task, a Costs column and ◆ marks in
    Experiments ▸ Sweeps, and a scatter of the metric against latency with the front
    joined.
  - Web: the same menu, column and chart.
  - CLI: `aime automl --objectives latency_ms --device raspberry_pi_5` prints the front.
  - MCP: `start_automl(objectives=...)`.
- `scripts/grab_ui.py automl` captures a multi-objective search on two spirals.

### Fixed
- Budget estimates for tabular transformers.
  - FT-Transformer and TabTransformer now count their layers once per column token, plus
    attention, which brings them within 15% of PyTorch's FLOP counter. Before, they were
    costed as one matrix-vector product, about 7× too low.
  - Launch overhead is counted per kernel rather than per block, so a composite block
    (a transformer, a ResNet, TabNet) no longer looks cheaper than a small MLP.
- A folder of audio clips preselected keyword spotting in the wizard, a task no recipe
  could train on a folder; it now preselects multi-class (or binary) classification.
- The graph recipes keep the node table's id and label columns instead of assuming `id`
  and `label`.
- Clustering and anomaly-detection scripts no longer stop when the table has no label
  column.
- `recommend()` no longer fails for a family's task when it is the first thing called.
- The run history ranks the Davies-Bouldin index lower-is-better.

## 3.0.0

The designer grows from classifiers and regressors to every major model area. It covers
29 tasks across 8 families: neural networks, classic ML, LLM workflows, graphical models,
probabilistic programs, Gaussian processes, state-space models and pipelines. That is
529 blocks, up from 345. Each family is validated while you design, generates runnable
code, trains in the app and deploys. A task-first wizard with AutoML and multi-stage
pipelines sit on top, and a resource budget is checked throughout.
[docs/FAMILIES.md](docs/FAMILIES.md) lists every family, task, block and design rule.

### Added
- Multi-stage pipelines (`core/pipelines`): Train, Cross-validate, Fine-tune (freeze /
  unfreeze, learning-rate scale), Distill (temperature, alpha), Ensemble (average / vote),
  Prune (global magnitude pruning with fine-tuning), Quantize (dynamic int8), Evaluate
  (test metrics, expected calibration error, reliability table), Export (serving package)
  and Register stages wired on the canvas. Every compute stage is a child run whose folder
  is deployable; running a pipeline again reuses finished, unchanged stages (fingerprints
  of stage, settings, design, data and inputs), so an interrupted pipeline resumes. Rules
  check wiring, stage designs, teacher / student and ensemble compatibility, fine-tune
  architectures, export formats and the budget. Train runs pipelines on desktop and web;
  Experiments ▸ Pipelines shows each stage's timeline. `aime pipeline run|status|list|
  resume`, `/api/pipelines`, MCP tools, two samples.
- Task-first wizard and AutoML (`core/recipes`, `core/automl.py`). New from Task (desktop,
  Ctrl+Shift+N) and New (web): pick one of 29 tasks, point at a table or dataset folder
  (format, target, classes and the fitting tasks are detected: CSV / Parquet / Excel, text
  tables, time series, interactions, edge lists, class folders of images / text / audio,
  COCO, YOLO, VOC, images + masks) or use demo data, set a budget, and get recipes ranked
  for the data size and budget with reasons, cautions and cost estimates. Recipes cover
  every task (MLP, Tabular ResNet, FT-Transformer, random forest, gradient boosting,
  linear baselines, CNNs, ResNet transfer learning, text and audio classifiers, plus
  designs for vision, forecasting, speech, generative, sequence, graph, recommender, RL
  and probabilistic tasks); each is fitted to the data by the safe Quick Fixes and notes
  why every block is there (Why tab, Explain This Design, Properties). AutoML trains every
  candidate once, then tunes recipes and their settings with TPE, prunes weak trials by
  the median rule and skips designs over the budget; results appear with the sweeps.
  `aime new`, `aime automl`, `aime recipes`, `/api/wizard/*`, `/api/automl`, MCP tools.
- Graph neural networks (`core/gnn`, PyTorch Geometric, `graph` extra): GCN, GAT / GATv2,
  GraphSAGE, GIN, GraphConv, EdgeConv and graph transformer layers; global, attention,
  TopK and SAG pooling; a Link Predictor. Node classification, graph classification and
  link prediction (held-out edges, negative sampling, ROC-AUC / AP) on generated community
  or motif graphs, node / edge CSVs, Planetoid and TU benchmarks. Graph models take
  `forward(x, edge_index, batch)`; deploy packages serve node, graph and link requests.
- Tabular deep learning (`core/tabular`): Tabular ResNet, FT-Transformer, TabTransformer,
  TabNet and categorical entity embeddings. A Quick Fix fills the positions and
  cardinalities of ordinal-encoded columns from the data, scaling keeps those codes as
  integers, and a generated Synthetic Table mixes numbers and categories. Modules can
  add an `aux_loss()` to the supervised loss (TabNet's sparsity penalty).
- Recommenders (`core/recsys`): matrix factorisation, NCF, two-tower retrieval and
  DLRM-lite with BPR, BCE (sampled negatives) or MSE objectives; Recall@K, NDCG@K and MRR
  over all items next to a most-popular baseline; top-K serving with a popularity
  fallback; generated interactions (implicit or ratings, with features) and CSV logs.
- Reinforcement learning (`core/rl`, `rl` extra): Gymnasium environments and PPO, A2C,
  DQN, SAC and TD3 from stable-baselines3; the canvas network is the policy's feature
  extractor; live reward curves, best-policy checkpoints, a random-policy baseline, and
  deploy packages that return actions.
- Quick Fixes now also set enum and boolean parameters ("set objective to bpr").
- Samples: GCN node classification, GIN graph classification, GraphSAGE link prediction,
  FT-Transformer churn, Tabular ResNet regression, matrix factorisation, DLRM, PPO on
  CartPole and SAC on Pendulum.
- Probabilistic models, all behind the `probabilistic` extra except the
  PyTorch parts.
  - Graphical models (`core/pgm`, pgmpy): discrete and Gaussian variables wired by a
    multi-input `parents` port; Bayesian and Markov networks, naive Bayes, dynamic
    Bayesian networks and hidden Markov models (hmmlearn). CPD tables are edited in a
    table editor (desktop and web) with a "Normalize the table" Quick Fix. Structure
    learning (GES, Hill-Climb with BIC / BDeu / K2, PC, Chow-Liu trees), MLE /
    Bayesian / EM parameter learning, variable elimination, belief propagation and
    sampling, marginal and MAP queries. Deployed models answer evidence queries.
  - Probabilistic programs (`core/ppl`, PyMC): 19 distribution blocks with a port per
    parameter, deterministic expressions, observed columns and group-level
    (hierarchical) effects, non-centred by default. NUTS (PyMC or NumPyro), ADVI,
    SMC or MAP; split R-hat and bulk / tail ESS identical to ArviZ, divergences,
    trace and posterior-predictive plots. Rules check supports, observed data types
    and improper priors.
  - Gaussian processes (`core/gp`): RBF, Matérn, periodic, linear, rational
    quadratic, spectral-mixture, white and constant kernels combined with Sum /
    Product blocks; exact or sparse variational GPs (GPyTorch) or scikit-learn;
    Gaussian, Student-t, Bernoulli and Poisson likelihoods; interval coverage
    against a constant baseline.
  - Bayesian deep learning (`core/bayes`): MC Dropout, Bayes-by-Backprop linear and
    convolution layers (KL added to the loss), a mixture density head and a
    last-layer Laplace approximation, in any supervised PyTorch design.
  - Calibration: ECE with a reliability diagram, temperature scaling and split
    conformal prediction (class sets or intervals). Served predictions add entropy,
    prediction sets, standard deviations and intervals.
  - Normalizing flows (`core/flows`): RealNVP affine couplings, masked
    autoregressive layers, neural spline couplings, ActNorm and permutations, a
    `density_estimation` task with a maximum-likelihood trainer, 2-D density data
    and CSV tables, a Gaussian baseline and live density pictures. Deployed flows
    return log-densities and samples.
  - State-space models (`core/ssm`, statsmodels): level / trend, seasonal, cycle,
    autoregressive and regression components, or a SARIMAX specification, fitted
    with the Kalman filter; held-out forecasts with intervals against seasonal
    naive, smoothed components, and serving that filters new histories.
  - Samples: student network, structure learning, market-regime HMM, hierarchical
    linear and logistic regression, GP trend + seasonality and GP classification,
    MC-dropout calibration, Bayes by Backprop, a mixture density network, spline and
    RealNVP flows, a structural time series and a seasonal ARIMA.
- Table parameters (`"table"` type, a JSON 2-D list) with grid editors, and
  multi-input ports on the desktop and web canvases.
- Generative models (`core/generative`), each with its own training loop.
  - VAEs built from layers around a Reparameterize bottleneck (β, KL warm-up; MSE
    or binary cross-entropy reconstruction); generation decodes prior samples.
  - GANs: any generator on the canvas (or a DCGAN generator block) against a
    Discriminator block with BCE, hinge or WGAN-GP losses, spectral norm and
    several critic steps.
  - Diffusion: a time- and class-conditioned U-Net with a Noise Scheduler block
    (linear / cosine, DDPM or DDIM sampling, classifier-free guidance, EMA).
  - Language models (GPT block, or Embedding + Causal Transformer) and
    encoder-decoder transformers on byte-level text (text corpora, CSV pairs,
    generated sentences and tasks).
  - Metrics: FID / KID (offline pixel features or Inception-v3), perplexity and
    bits per byte, BLEU / chrF (identical to sacrebleu) and ROUGE-L.
  - Live sample grids and generated text every epoch in the desktop and web
    training panels (a new `samples` worker event); saved samples in the run view.
  - Rules with Quick Fixes: dataset vs Input / generator shape, VAE reconstruction
    shape, latent Input for GANs, U-Net depth vs image size, class counts, causal
    language models (no bidirectional layers), byte vocabularies, context and
    target lengths, one generative model per design.
  - Deploy packages add `POST /generate` (images as base64 PNG, or text).
  - Samples: shapes VAE, DCGAN, class-conditional diffusion, tiny GPT and a
    reverse-the-word seq2seq.
- Time-series forecasting (`core/forecast`), with its own training loop.
  - Models: DLinear, N-BEATS, N-HiTS, PatchTST, TiDE, TCN, a DeepAR-style RNN and
    an Informer-style Transformer, each with a point, quantile (non-crossing),
    Student-t or negative-binomial head.
  - Data: long-format tables with several series (id column), past and
    known-future covariates (promotions, holidays), and generated seasonal demand
    series. Rolling-origin backtest split, per-window scaling.
  - Metrics: MASE (with the seasonal-naive baseline reported alongside), sMAPE,
    WAPE, CRPS, pinball loss, 80% interval coverage and error by horizon step;
    forecast plots with intervals; deploy packages serve forecasts for histories.
  - Rules with Quick Fixes: Input vs window × channels, model vs dataset horizon,
    future-covariate counts, TCN receptive field vs window, head vs data type.
    Profiles report series lengths, gaps, missing values, count data and the
    detected seasonal period.
- Speech (`core/speech`): keyword spotting, speech recognition with CTC and
  audio tagging, with their own training loop.
  - Speech manifests (CSV / TSV of WAV files with keywords, transcripts or tags)
    and generated tone-coded speech; waveform augmentation; CTC loss; accuracy /
    F1, CER / WER and tag mAP; spectrogram samples with predictions. Deploy
    packages accept base64 WAV or waveforms and resample them.
  - Rules: Input vs clip length, class / alphabet counts, CTC frames vs
    transcript length, sample rate vs mel features and 16 kHz pretrained encoders.
- Sequence blocks: TCN, Mamba (selective SSM), S4D and sLSTM (xLSTM) in pure
  PyTorch; in-model mel spectrogram / MFCC front-ends (image or sequence layout);
  wav2vec 2.0, HuBERT, WavLM and Whisper encoders (`audio` extra).
- Samples: demand forecasting (N-HiTS, quantiles), keyword spotting (mel CNN) and
  speech recognition (BiLSTM + CTC).
- Vision tasks (`core/vision`): object detection, instance segmentation, keypoint
  detection and semantic segmentation, with their own training loop and metrics.
  - Models: torchvision Faster / Mask / Keypoint R-CNN, RetinaNet, FCOS, SSD(lite),
    DeepLabV3, FCN, LR-ASPP; Hugging Face DETR, Conditional DETR, RT-DETR and
    SegFormer; a U-Net family (U-Net, U-Net++, Attention U-Net, ResUNet); 87 curated
    timm backbones and DINOv2 / CLIP / SigLIP image encoders.
  - Data: COCO, YOLO (boxes, polygons, pose), Pascal VOC, segmentation mask folders
    and generated Synthetic Shapes. Profiles in the Data workspace (objects per
    class, COCO object sizes, empty images, mask class shares, boxes outside the image).
  - Box / mask / keypoint-aware augmentation (`transforms.v2`); Dice, Tversky,
    Lovász-Softmax and pixel focal losses; COCO mAP@[.5:.95] / mAP@.5 / mask mAP
    (identical to pycocotools), PCK, mean IoU, Dice and pixel accuracy.
  - Design rules with Quick Fixes: class / keypoint counts vs the dataset,
    detectors wired straight from Input to Output, dataset ↔ task compatibility,
    anchors vs image size, SegFormer stride, non-commercial weight licenses.
  - Test-image overlays (truth vs prediction) in Error Analysis (desktop) and
    the run view (web); deploy packages serve boxes, labels, scores and RLE masks.
  - Two samples (Shapes detection, Shapes segmentation); `vision-tasks` extra and CI job.
- Resource budgets (`core/budget.py`): per-layer FLOPs, training memory
  (weights, gradients, optimizer state, saved activations, mixed precision) and
  latency estimates against 11 device profiles (laptop CPU, Apple MPS, RTX
  3060/4090, T4, A100, H100, iPhone ANE, Jetson Orin Nano, Raspberry Pi 5) plus
  your own (`~/.aime/devices.json`). FLOPs match PyTorch's counter within 7%
  and training memory matches measured MPS peaks within 7% on the samples.
- Budget settings (device, max training memory / latency / parameters / model
  size) in the Summary tab (desktop and web) and `meta.budget`; designs over
  budget get warnings with Quick Fixes (mixed precision, or a smaller batch with
  gradient accumulation that keeps the effective batch).
- Sweeps skip trials that break the budget; training runs record measured peak
  memory and step time (`resources` in the run record) to compare with the
  estimate. `aime budget`, `GET /api/devices`, `POST /api/budget`, MCP
  `estimate_budget`; the Summary shows FLOPs per layer.
- Model families and a task registry: designs are recognised, validated and
  trained through `core/families.py` / `core/tasks.py`; `/api/tasks`,
  `/api/families`, `/api/describe`, MCP `list_tasks`. The status bar shows the
  design's family and task.
- Semantic port roles (logits, boxes, masks, tokens, graph, variable, ...)
  checked on every wire; blocks can declare optional package requirements,
  reported with the pip extra that installs them.
- Trainer-kind registry for non-supervised training loops.
- Pretrained image backbones: 39 torchvision architectures (was 16) from a
  generated, measured catalog — adds ResNet-152, ResNeXt, Wide ResNet,
  EfficientNet-B4 / V2, ConvNeXt S/B, DenseNet-169/201, VGG-19, RegNet-Y
  800MF–3.2GF, ShuffleNetV2, MNASNet, ViT-B/32, ViT-L/16, Swin(V2), MaxViT.

### Changed
- Project files use schema 3. Saved projects record their family and task, and the web
  Open dialog shows both. Nothing is rewritten: 1.x and 2.0 projects open unchanged
  (port roles and multi-input ports live in the block registry).
- `docs/FAMILIES.md` is generated from the registries by `scripts/gen_families.py`, and a
  test keeps it current. Its rule lists quote the messages straight from the rule modules.
- CI gains a speech job (the `audio` extra), alongside the vision, probabilistic and
  graph / RL jobs. A nightly workflow installs every extra and runs the opt-in
  `AIME_ONLINE=1` tests, which download real pretrained weights (torchvision, timm,
  Hugging Face) and check each pretrained block's output shape.

### Fixed
- A web server that had just started could show an empty block library ("Search 0
  blocks"). The browser's first requests run in parallel, and one of them could read the
  block catalog while another was still loading it. Loading is now guarded by a lock.
- The audio recipes (Mel-spectrogram CNN, Keyword CNN, Audio tagger, BiLSTM + CTC) asked
  for torchaudio, which the `audio` extra does not install, so the wizard always showed
  them as unavailable. They only use PyTorch and the standard library. A test now checks
  that every block's and recipe's requirements come with the extra it names.
- The web app treated a project file without `schema_version` as schema 2, while the
  desktop treats it as 1. An old file saved again from the browser therefore skipped its
  1 → 2 upgrade.
- Standardization divided columns that are constant in the training data by ~1e-8, so
  any other value at test time exploded (digits: test loss ~100,000); constant columns are
  now left unscaled, as in scikit-learn.
- A training script that stopped with an error message (`SystemExit("...")`, e.g. a data
  shape mismatch) was recorded as a finished run; it is now a failed run with the message.
- Networks reading a table now check their Input width against the preprocessed row
  (one-hot columns included) for every model, not only the tabular blocks.
- The desktop canvas keeps a recipe's per-block notes when it renames blocks on load.
- Desktop training runs now record measured peak memory and step time (they
  were only recorded for runs started from the CLI, web or MCP).
- The "softmax before a logit loss" lint no longer fires on tasks that do not
  train with cross-entropy (VAE, GAN, diffusion, forecasting).
- Deploy packages of vision runs listed no PyTorch requirement: task-specific
  training scripts now declare what they need.
- RegNet backbones were mapped to Keras classes that Keras 3 does not provide;
  they are now reported as PyTorch-only.

## 2.0.0

The designer becomes a complete model workflow: track and tune experiments,
deploy trained models as services, import existing models, inspect data before
training, and work from the browser as well as the desktop.

### Added
- **Experiments.** Every run is recorded in a persistent history
  (`$AIME_HOME/runs`) with its design snapshot, parameters, per-epoch metrics,
  environment (Python, platform, package versions) and data fingerprint.
  The Experiments dock lists, filters, tags, compares (metrics, differing
  parameters, overlaid curves) and restores runs.
- **Hyperparameter sweeps** over any numeric or choice parameter in the design:
  grid, random or Bayesian (Optuna TPE) search, invalid configurations skipped,
  trials recorded as child runs, leaderboard and "Apply best". Deterministic
  training option.
- **Deploy.** Serving packages from a finished run: FastAPI app (`/predict`,
  `/health`, `/metadata`) with the exact fitted preprocessing, the model,
  requirements, Dockerfile and README. Optional ONNX (verified against
  onnxruntime), int8-quantized ONNX, TorchScript and Core ML exports. A model
  registry with versions and staging / production stages.
- **Import.** PyTorch modules (`torch.fx`), ONNX files and Keras models become
  editable designs. Weights are copied into the rebuilt model and outputs are
  compared numerically; unsupported operations are listed.
- **Custom block packs.** Versioned custom blocks, shareable as `.aimeblocks`
  files; `.aime` archives embed the custom blocks they use and restore their
  source run.
- **Data workspace.** Dataset profiles (column statistics, class balance,
  image / text / audio statistics) with findings: missing values, constant and
  identifier columns, target leakage, duplicates, high cardinality, skew,
  broken images. Pipeline-aware data warnings in the Problems panel, an exact
  split preview and an augmentation preview.
- **Web version.** `aime web` serves the designer in the browser (React +
  React Flow) with live validation, training with streamed charts, data,
  experiments and model pages, on a REST + WebSocket API with optional token
  auth. A Docker image is provided.
- Command line: `aime runs`, `sweep`, `deploy`, `models`, `serve`, `import`,
  `data`, `web`. MCP tools for runs, sweeps, deployment, import and data.

### Changed
- One headless API (`core/api.py`) backs the CLI, the MCP server and the web
  server.
- Problems-panel data checks run in the background with a cache instead of
  hashing dataset files on every edit.
- Generated training scripts save the fitted preprocessing state and include
  `load_predictor()` / `infer()` for serving.
- Settings layout version 5 (new Experiments and Data docks); saved window
  layouts from 1.x are reset once.

### Fixed
- Serving inputs that are long base64 strings no longer crash path detection
  (`File name too long`).
- ONNX import verification for models exported with a fixed batch size.
- Several native teardown crashes on CI after successful test sessions.

## 1.0.0

First production release: a professional designer for practitioners.

### Added
- Block library expanded to 345 blocks: complete `torch.nn` layer coverage with
  Keras 3 equivalents, 16 pretrained image backbones, 8 Hugging Face text
  encoders, 55 classic-ML estimators (scikit-learn, XGBoost, LightGBM,
  CatBoost), 12 feature transformers, hyperparameter search.
- Training engine: 12 dataset sources across image, tabular, text, audio and
  time-series data; 45 preprocessing and augmentation blocks; 14 task-aware
  losses, 12 optimizers, 12 schedulers, 25 metrics; mixed precision, gradient
  clipping and accumulation, class balancing, MixUp / CutMix, k-fold CV.
- scikit-learn export target with ColumnTransformer pipelines, CV and search.
- Architecture lints and data/training rules with Quick Fixes.
- Dock-based workbench with dark and light themes, Block Library, Inspector,
  Problems, Output and Training panels, command palette.
- Error analysis, Grad-CAM saliency and model card dialogs.
- Autosave with crash recovery, recent projects, project-file migrations,
  logging and an error reporter, desktop packaging, CI.

### Changed
- Code generation rewritten: independent PyTorch and Keras emitters, layout-aware
  Keras export, readable variable naming, typed documented output.
- Preprocessing statistics are fitted on the training split only.
- Professional diagnostics wording throughout.

### Removed
- Education features: missions, quizzes, prediction gate, celebrations,
  camera / microphone capture, teaching samples.

### Fixed
- Broken Keras export for padded convolutions, bidirectional and stacked RNNs,
  transformer encoders, sequence inputs and image data layout.
- Wrong `in_channels` for Conv1D, binary-classification metrics, BCE target
  shapes, dead branches emitted into `forward()`.
- Blocks with `height` / `width` parameters could not be placed.
- Opening a project marked it as modified; problem jump-to-block crashed.
