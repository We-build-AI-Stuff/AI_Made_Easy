# Contributing

Thanks for helping. Setup, checks and the release flow are in the README's
[Development](README.md#development) section; how the pieces fit is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Before opening a pull request, run:

```bash
ruff check ai_made_easy scripts tests
pytest
python scripts/verify_codegen.py --coverage
cd web && npm run build && npx playwright test
```

`pytest` runs on four workers (`-n0` for one process, to debug) and leaves out the
slow tests listed in `tests/slow_tests.txt`: real training runs, subprocesses and
sweeps, about three minutes in all. `AIME_TESTS=all pytest` (or `--run-slow`) runs the
whole suite, as CI does on every pull request; a slow test named on the command line
always runs. After adding slow tests, refresh the list from a timed run:

```bash
AIME_TESTS=all pytest --durations=0 --durations-min=1 > durations.log
python scripts/find_slow_tests.py durations.log
```

## Adding a block

A block is one `BlockDefinition` in the registry. The palette, the inspector, the web
UI, validation, code generation, the budget and the docs all read the registry, so a
new block needs no UI code.

### 1. Where it goes

| Kind of block | File |
|---|---|
| A layer (dense, conv, attention, ...) | `ai_made_easy/core/blocks/<group>.py`, built with `nn_block` from `_dsl.py` |
| A family's block (vision head, forecaster, distribution, ...) | `ai_made_easy/core/<family>/blocks.py`, registered by its `register_all()` |
| A dataset, preprocessing or training block | `core/training/data_catalog.py` or `core/training/catalog/` (losses.py, optimizers.py, …) |

The type id's prefix is the family namespace (`vision.fpn`, `forecast.tft`, `ppl.mv_normal`):
the coverage report and FAMILIES.md group blocks by it.

### 2. What it declares

- **Parameters:** `P(name, type, default, lo=, hi=, options=, help=)`. The `help` text
  is the tooltip; say what the value does, with a typical range.
- **Shape:** `shape_fn(in_shapes, params) -> out_shape`. Raise `ShapeError` with a message
  that names the fix ("height and width must be multiples of 32: resize to 224×224").
- **Cost:** `param_fn` for the parameter count and, for anything that is not one layer,
  `meta["cost"] = fn(in_shapes, params) -> (parameters, forward FLOPs)`. The summary and
  `core/budget.py` use them. Check the count against PyTorch in a test.
- **Code:** `torch` (the module constructor) and `torch_expr` (the call; `{i0}`, `{i1}`
  are the inputs), plus `keras` / `keras_expr` when Keras can express it. Raise
  `KerasUnsupported` for configurations Keras cannot express. Helper classes go in the
  family's `helpers.py` and are listed in `torch_helpers`. Generated scripts must stay
  self-contained: no imports from `ai_made_easy`.
- **Optional packages:** `requires=("transformers",)` and `extra="vision-tasks"`. The
  library shows an install badge and validation names the extra. Never import an
  optional package at module level.
- **Downloads:** pretrained weights need a `weights` parameter with an offline value
  (`none`: same architecture, random weights), a model-zoo row (`scripts/build_zoo.py`)
  and an entry in `core/zoo/__init__.py::DOWNLOADS`, so the wizard shows size and
  licence.
- **Meta:** `task` for task heads, `kind` (`loss`, `metrics`, `config`) for training
  blocks, and anything your family's template reads.

### 3. Design rules

Family rules are `register_rule(fn)` functions over a `LintContext`, or a family's
`validate`. Messages follow one pattern: what is wrong, why it matters, how to fix it.
Phrases like "set <param> to <value>" and "Set the Input shape to '…'" become one-click
Quick Fixes (`core/fixes.py`). Use `error` only when training would fail, `warning` when
it would run but go wrong, and `info` for advice.

### 4. Tests

- **Shapes:** `scripts/verify_codegen.py` builds `Input → block → Output` for every layer
  block, runs the generated PyTorch code and compares the real output shape with the
  designer's. Add parameter variants to `VARIANTS`, surrounding layers to `CONTEXTS`
  (for a block that cannot follow an Input directly), and offline overrides to
  `DEFAULT_OVERRIDES`. `python scripts/verify_codegen.py --coverage` must report no
  `no_case` block. The table lists, per family, the blocks the harness runs, the ones
  waiting for an optional package, and the configuration blocks the family tests cover.
- **Behaviour:** a test with tiny synthetic data that trains through the generated script
  and checks it learns something (beats a baseline, recovers a known parameter). Gate
  optional packages with `pytest.importorskip` and add the test to the matching CI job in
  `.github/workflows/ci.yml`.
- **Downloads:** real pretrained weights belong in `tests/test_online.py`, which runs
  nightly.

### 5. Docs

Run `python scripts/gen_families.py` to regenerate `docs/FAMILIES.md`, add a CHANGELOG
line, and add a sample to `ai_made_easy/samples/` when the block makes a new kind of
design possible.
