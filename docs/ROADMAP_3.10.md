# AI Made Easy 3.10 — plan and status

*Status: released as 3.10.0 (PRs #75–#79).*

## Phase 0 — resume Keras runs and remote runs (#75)
- **Keras runs resume.** After each epoch the script saves the weights and optimizer, the
  random state (Python, NumPy, PyTorch's CPU, CUDA and MPS generators, and the layers'
  seed generators) and the counters of its early-stopping and plateau callbacks. The
  random state is restored when training begins, because `fit` draws random numbers
  while it sets up. A resumed run ends bit-for-bit where an uninterrupted one does.
  **Done.**
- **Remote runs resume** on the backend they ran on (SSH, Modal, container, Kaggle,
  Colab). The saved state comes back with the outputs and goes out with the resumed
  run's bundle; Kaggle and Colab bundles are checked against a 20 MB limit. Downloads
  over SSH from a Mac no longer include `._` files. **Done.**

## Phase 1 — compile, checkpointing, several GPUs (#76)
- **Compile:** `torch.compile` with a fallback that says why. **Done.**
- **Gradient checkpointing:** the same results with less memory. Batch-norm layers keep
  their activations (recomputing them would update their statistics twice). The budget
  counts what the script really keeps: a wrapped layer's output, plus one layer's inner
  values while it is recomputed (flat networks save little), and a third more step time.
  **Done.**
- **Several GPUs:** one forked process per GPU with DistributedDataParallel (gloo on CPU,
  NCCL on GPUs), a distributed sampler and an averaged training loss. Every process
  takes the first one's validation loss, so they stop early and change the learning rate
  together. GPUs are counted in a child process so CUDA isn't started before the fork.
  The first process reports and saves. K-fold and metric-learning runs use one GPU.
  **Done.**
- Keras scripts say these settings apply to PyTorch only. **Done.**

## Phase 2 — features (#77)
- **Fairness by group.** For classifiers, each group's rate of being predicted each
  class, and its true- and false-positive rates, are compared with all test rows. A
  group is flagged when it is predicted a class 10 points more or less often and outside
  the four-fifths rule, or its true-positive rate is 10 points lower, or its
  false-positive rate 10 points higher. Both apps show one paragraph. Groups with few
  rows are left out. **Done.**
- **OpenVINO export** (`openvino` extra). Models are compiled in float32 and must match
  PyTorch. The tests run on x86 only: OpenVINO's ARM plugin lacks group normalization and
  drifts in precision. Detection and graph models are not supported (they need more than
  one input tensor), nor is retention (its group norm over a flattened tensor has no CPU
  kernel). **Done.**
- **PyTorch or Keras in the apps.** A design's framework choice is honoured everywhere,
  and the Training panel offers the frameworks every layer supports. A saved choice that a
  later block can't follow falls back to one that every layer has. With it, the Keras
  learning-rate finder is reachable from both apps. **Done.**
- CodeRabbit's reviews of #76 and #77 found seven issues, all fixed in #77 (the items
  above on validation loss, GPU counting, the checkpointed memory estimate and the saved
  framework, plus the fairness ratio, exact 10-point gaps and a stale framework list).

## Phase 3 — upkeep (#78)
- The desktop's canvas actions move to `ui/context_canvas.py` and the web designer's
  bottom panels to `web/src/designer/BottomPanels.tsx`. Dependencies were already
  current. **Done.**

## Phase 4 — release 3.10.0 (#79)
- CHANGELOG, README, this roadmap, a fairness screenshot, version 3.10.0. The tag and
  release follow your confirmation.

## Still open
- Making the Windows test jobs required checks is your call (a ruleset change).
- OpenVINO is checked on x86 only.
