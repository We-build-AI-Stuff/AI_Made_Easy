# AI Made Easy 3.11 — plan and status

*Status: implemented in PRs #80–#84; the 3.11.0 tag and release wait for your confirmation.*

## Phase 0 — several GPUs everywhere, Keras compiles, OpenVINO on ARM (#80)
- **Several GPUs on Windows.** Windows can't fork. There the other processes are spawned
  from the script, with the settings changed since it was written. The worker's own
  process trains as the first one, so the run's live events keep coming from it.
  Settings the spawned processes can't get (hooks, which are functions) are named in a
  warning. **Done.**
- **Several GPUs with k-fold and metric losses.** Each fold's training rows are split
  between the processes, and the first scores the fold. ArcFace-style class centres
  live in the loss, outside DistributedDataParallel; their gradients are averaged too,
  so every process ends with the same centres. **Done.**
- **When a process fails, the run stops** instead of the others waiting for it until the
  backend times out (30 minutes): a watchdog ends the run when another process exits
  with an error. If the first process fails, the others are stopped before anything
  waits on them. **Done.**
- **Keras compiles** (Trainer **Compile** → `jit_compile`). If compiling fails before
  an epoch ends, training restarts without it, from the weights and optimizer state it
  had before. **Done.**
- **OpenVINO on ARM.** It is checked on macOS's ARM CPUs too, in its accuracy mode:
  - **Tolerance:** within 1e-3 there (its ARM kernels drift up to 7e-4 in float32), and
    within 1e-4 on x86.
  - **Retention:** its group norm was rewritten so OpenVINO can run it, with
    bit-identical outputs and gradients. The 3.10 rule against exporting it is gone.

  **Done.**
- Tests that waited silently for a condition now fail with what they were waiting for.
  **Done.**

## Phase 1 — bigger models (#81)
- **bfloat16** (Trainer **Precision**: auto, float16 or bfloat16). bfloat16 needs no
  loss scaling. **Done.**
- **Sharded training** (Trainer **Sharding**, 2+ GPUs). FSDP2 splits the weights,
  gradients and optimizer state between the GPUs; the run still saves whole weights.
  - **Budget:** memory checks count the split, and the whole weights each GPU holds
    before it.
  - **Memory fix:** it offers "Shard the model", except with EMA or SWA, which sharding
    can't keep.

  **Done.**
- **Quantization-aware training** (Trainer **QAT epochs**). After training, the model is
  fine-tuned with int8 rounding simulated, then saved as `qat_int8.pt2`, which the
  ExecuTorch int8 export uses. It is skipped, with the reason, for models it can't
  quantize. A late failure keeps the float results. **Done.**

## Phase 2 — fairness mitigation, GGUF, what would change it (#82)
- **Group weights** (Class Balancing). Each training row weighs P(group)·P(class) /
  P(group, class), so the group says nothing about the class: PyTorch (a weighted
  sampler), Keras and classic models (`sample_weight`). **Done.**
- **A threshold per group** (Results by group, binary classifiers). Every group gets the
  same true-positive rate, the one that leaves the rows most accurate (equal
  opportunity).
  - **Fitting and comparing:** the thresholds are fitted on the validation rows (half
    the test rows for classic models). Before and after are shown, in both apps, on
    rows they weren't fitted on.
  - **Honest summary:** when thresholds don't help, the summary says so.

  On a table where one region's incomes are recorded low, the gap in error rates drops
  from 58% to 8%. **Done.**
- **GGUF export** for Llama-style language models (llama.cpp, Ollama, LM Studio).
  - **Tokenizer:** the file includes it, as bytes or the run's BPE merges.
  - **Check:** with llama-cpp-python installed, the file is checked token by token
    against PyTorch (about 1e-6), and llama.cpp tokenizes text exactly as the run does.

  **Done.**
- **What would change the prediction** (Try it, both apps; REST and MCP). For a table
  row, the run's own model is asked about edited copies of it, using values from the
  training table:
  - It lists the smallest single changes that reach another class (or a regressor's
    target).
  - Otherwise, a few changes together.
  - You can lock columns.

  **Done.**

## Phase 3 — upkeep (#83)
- `core/api.py`, `training/catalog.py` and `cli.py` became packages, by area. They were
  split by script from the original text, and every import still works. **Done.**
- Tested with fastapi 0.162; no changes were needed. **Done.**

## Phase 4 — release 3.11.0 (#84)
- CHANGELOG, README, this roadmap, two screenshots, version 3.11.0. The tag and release
  follow your confirmation.

## Fixed along the way
- **Resumed SSH and Colab runs could replay their last attempt.** A resumed run reuses
  its folder on the host (or in Drive). The follower could read the stopped attempt's
  events and exit code before the new launcher cleared them, which ended the new
  attempt at once. They are now removed before the launcher starts. CI's SSH resume
  test caught it, and it now checks the race every time.
- **Copies of a checkpointed model ran the original's layers.** With gradient
  checkpointing, EMA's and SWA's averaged models (and the quantization-aware copy)
  called the original layers. The wrapper is now bound to its own layer.
- **A Keras compile retry restores the random state too**, so dropout draws as if the
  run had never compiled.

## Still open
- Making the Windows test jobs required checks is your call (a ruleset change).
- GGUF covers designs built from Llama blocks; other language models (Tiny GPT, RWKV,
  fine-tuned Hugging Face models) don't export to it yet.
- Sharded training is checked on CPU processes here; on GPUs it runs the same code with
  NCCL.
