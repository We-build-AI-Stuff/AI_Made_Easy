# AI Made Easy 3.9 — plan and status

*Status: released as 3.9.0 (PRs #70–#74).*

## Phase 0 — fixes from 3.8 and coverage (#70)
- **The MCP server on Windows.** 3.8's CI captured the stall: the first tool call loaded
  numpy's extension while the transport's thread blocked reading the stdin pipe. The
  server now loads the block registry before serving and reads requests from a private
  copy of the pipe, with standard input on the null device (child processes such as
  training workers no longer inherit it). The Windows test passes without its
  expected-failure marking. **Done.**
- **The worker that died on Ubuntu 3.11.** It left no Python stack, which points to the
  process being killed outright. A failed Linux test job now prints the kernel's
  out-of-memory records, so a repeat says whether memory was the cause. It has not
  recurred so far. **Done (diagnostics).**
- **Core ML in CI** (`test-coreml`, macOS): every sample and each 3.7 / 3.8 layer is
  exported, and on macOS the converted program must match PyTorch within float16
  tolerance (`HALF_TOL`, 5% of the output scale: deep networks drift 2–3%). What it
  found, and fixed:
  - coremltools 9.0 and NumPy 2.5 broke every traced reshape that reads the input's size
    (most attention and sequence layers): a scoped fix to coremltools' cast during
    conversion.
  - Rewritten with identical outputs so Core ML converts them: RWKV's token shift,
    differential attention's lambda, rotary-embedding positions, DLRM's interaction pairs
    (which also lifts DLRM's ExecuTorch limit).
  - Export rules with the reason for what Core ML can't take: Hyena (complex FFT),
    deformable convolution (no converter), involution (Core ML's unfold gives different
    numbers), Glow's invertible 1x1 convolution (diag; rewriting it broke ONNX int8), the
    spline flow (data-dependent shapes).
  - A `coreml` extra. **Done.**
- **TimesNet on ExecuTorch.** Eval mode fixes its periods as plain integers and gives each
  period its own copy of the convolutions: float ExecuTorch (a data-dependent shape) and
  int8 (a convolution used at two call sites gave a program that couldn't load) both work,
  and the int8 rule is lifted. **Done.**

## Phase 1 — faster sequence models, Keras parity (#71)
- Mamba-2, Gated DeltaNet and gated linear attention run in chunks (within a chunk the
  outputs are one masked product; only the state between chunks loops; the delta rule's
  writes come from a small triangular system solved for every chunk at once). Griffin
  uses a parallel prefix scan. They work in log decays, so an underflowing decay stays
  finite. Outputs and gradients equal the step-by-step recurrence (float64, four chunk
  sizes, a ragged length). Forward + backward at length 256 on CPU: Mamba-2 ~10×,
  Griffin 5×, GLA and Gated DeltaNet ~2.3×. **Done.**
- Keras forms for 15 layers that were PyTorch-only. The sequence layers keep PyTorch's
  weight layout (`TorchLayout`), so with copied weights both compute the same outputs;
  the vision blocks match parameter counts and shapes, involution its values.
  Deformable convolution stays PyTorch-only (Keras has no equivalent). Forecasting and
  tabular family blocks stay PyTorch-only by design. **Done.**

## Phase 2 — features (#72)
- **Resume an interrupted run**: per-epoch `resume.pt` (model, optimizer, scheduler, AMP
  scaler, EMA / SWA, best weights, random state, the loader's shuffle generator) and
  `resume.json`; the run continues as the same run. A resumed run ends bit-for-bit where
  an uninterrupted one does. Local PyTorch runs; both apps, API, CLI. **Done.**
- **Train boosted trees more**: XGBoost, LightGBM, CatBoost, scikit-learn's
  (Hist)GradientBoosting. **Done.**
- **Learning-rate finder for Keras designs**, and `aime lr-find`. The apps train with
  PyTorch by default, so the Keras finder is reached from the API, the CLI and MCP-driven
  workflows. **Done.**

## Phase 3 — upkeep (#73)
- GitHub Actions on their Node 24 versions (CI warned about Node 20).
- The Keras forms in `core/blocks/keras_forms.py`.
- ROADMAP_3.8's open items marked fixed. Web and Python dependencies were already
  current. **Done.**

## Phase 4 — release 3.9.0 (#74)
- CHANGELOG, README, this roadmap, a screenshot of Train More for a boosted model,
  version 3.9.0. The tag and release follow your confirmation.

## Still open
- Making the Windows test jobs required checks is your call (a ruleset change).
- Resume for runs on remote backends and Keras runs: done in 3.10
  ([ROADMAP_3.10.md](ROADMAP_3.10.md)).
