# AI Made Easy 3.12 — plan and status

*Status: implemented in PRs #88–#92; the 3.12.0 tag and release wait for your confirmation.*

## Phase 0 — reliability (#88)
- **One-off CI failures, looked into:**
  - **Desktop tests crashing:** each new window set the Qt style again, deleting the one
    a closing window still used. The style is now set once. **Fixed.**
  - **Metric-loss centres:** a port race (`EADDRINUSE`), fixed in 3.11 when the
    processes started meeting through a file.
  - **The data-service crash:** fixed in 3.11.
  - **mask2former's ONNX export:** it once differed by 1.55e-2, alongside an onnxruntime
    GridSample error. It was not reproduced: 30 random initialisations all match within
    about 1e-6. **Still open** (below).
- **Several GPUs on Modal.** Modal backends take `gpu_count` (1 to 8; A10G up to 4), and
  the Trainer's GPUs trains on them. **Done.**
- **A GPU check.** `scripts/gpu_check.py` and the *GPU check* workflow train a small network
  on 2 Modal GPUs: DistributedDataParallel in float16, sharded in float16 and in bfloat16.
  The workflow needs the repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`; a
  run that takes too long is stopped and reported. **Done; not run yet** (below).
  - *Changed from the plan:* the GPU check's result is not shown on the Trainer. The
    workflow can't write to main (its rules), so the result is in the run's summary.

## Phase 1 — GGUF for more language models (#89)
- **GPT → GGUF** (llama.cpp's `gpt2`), checked within 5e-3. llama.cpp computes GELU
  through a float16 table: zeroing the MLP's output brings the gap to 4e-7, and a trained
  Tiny GPT measures 5.7e-4. **Done.**
- **RWKV-7 Stack block** (PyTorch and Keras), following llama.cpp's `rwkv7` graph step for
  step. Tiny RWKV uses it and exports to GGUF, about 1e-5 from PyTorch. **Done.**
  - It is one block for the whole stack, because RWKV-7's layers share the first layer's
    values.
  - ONNX, TorchScript, OpenVINO and ExecuTorch export too. ExecuTorch int8 doesn't: its
    quantizer finds a cycle in the state update.
- **Q8_0 and Q4_0** GGUF files: a little over a quarter and about an eighth of the float
  size, with how often the top next token still agrees. **Done.**
- **Fine-tuned Hugging Face models → GGUF**: Llama, Qwen2, Phi-3 and Gemma 3. **Done.**
  - The LoRA adapter is merged into the base model, then each family is written as
    llama.cpp's converter writes it.
  - **Tokenizers:** SentencePiece, from `tokenizer.model` or from a `tokenizer.json`
    saved by transformers 5 (its pieces scored by merge rank), and byte-level BPE (its
    pre-tokenizer recognised by its regex).
  - **Checks**, in a separate process because llama.cpp stops its process on a broken
    file: the same tokens as the tokenizer, and next-token scores within 1e-3 of
    transformers.
  - Tested on tiny random models built in the tests, so nothing is downloaded.

## Phase 2 — fairness in production, and a file's changes (#90)
- **Thresholds per group, used.** **Done.**
  - **Use These Thresholds** (Results by group) makes the scored run's Try it and its
    deployment packages decide each two-class prediction with its group's threshold.
  - Rows carry the group column, even when the model doesn't read it; without it the
    default (0.5) applies.
  - **Stop Using Them** goes back to 0.5.
  - It lives in the generated inference code, so packages need nothing else; their
    metadata records the thresholds.
- **What would change each row of a file.** Try it on a CSV can add each row's smallest
  change to another class, for the first 500 rows. **Done.**

## Phase 3 — upkeep (#91)
- torch 2.15, transformers 5 and pytest 9: a nested `approx` in one test, the
  encoder-decoder's mixed masks, and SentencePiece tokenizers saved as `tokenizer.json`.
  **Done.**
- The 3.11 roadmap says released. **Done.**

## Phase 4 — release 3.12.0 (#92)
- CHANGELOG, README, this roadmap, a screenshot, version 3.12.0. The tag and release
  follow your confirmation.

## Fixed along the way
- **Tiny RWKV's final LayerNorm normalized over the whole sequence.** The sample file is
  schema v1, which kept LayerNorm's old "every axis" default, so since 3.7 its training
  saw later tokens. A rule now flags this in any language model, with a one-click fix.
- **Hugging Face tokenizers whose vocabulary scores don't follow their merges**
  (`tokenizer.json` only) split text differently in llama.cpp. Their pieces are now
  scored by merge rank.

## Still open
- **The GPU check hasn't run.** It needs a Modal account: add `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET` as repository secrets, then run *GPU check* from Actions.
- **ONNX export** still uses torch's legacy TorchScript-based exporter, deprecated since
  torch 2.9. Moving to the `torch.export`-based one changes how layers export, so every
  sample needs re-checking.
- **TorchScript export** relies on `torch.jit.trace` and `torch.jit.load`, deprecated in
  torch 2.15. Keeping the format means keeping them until torch removes them.
- **mask2former's ONNX export** failed once in CI (1.55e-2) and was not reproduced.
- **Hugging Face → GGUF** covers four families. Others (Mistral, GPT-2, Falcon…), Gemma 1
  and 2, and rope scalings llama.cpp's families don't read say why they can't export.
