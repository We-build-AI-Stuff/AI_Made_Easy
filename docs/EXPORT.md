# Export everywhere

A trained run exports to the runtime it will run in, and **every export is checked
against the trained model** before it lands in the serving package:

| Framework | Formats |
| --- | --- |
| PyTorch | ONNX, ONNX int8, TorchScript, Core ML, OpenVINO, ExecuTorch (`.pte`), ExecuTorch int8, GGUF (Llama-style language models) |
| Keras | ONNX, TensorFlow SavedModel, LiteRT (`.tflite`), LiteRT int8 |
| scikit-learn | ONNX |

Pick formats when you build a package: **Deploy** on the Experiments page (desktop and
web), the pipeline **Export** stage, or the command line:

```bash
aime deploy <run id> --list-formats            # available, suggested, and why not
aime deploy <run id> -o server --formats onnx,executorch_int8
aime deploy <run id> -o server --formats suggested   # what the budget device runs
```

Formats can also be added to an existing package later:

```bash
cd server && python export_formats.py litert_int8
```

## How exports are checked

The package's `export_formats.py` loads the trained model and runs the source model and
each exported file on the same inputs. When the run saved samples, those inputs are real
ones: training scripts write `calibration.npz`, with up to 128 training inputs and 256
labelled test inputs, at most about 4 MB. Older runs fall back to random inputs of the
right shape.

- **Float formats** must match PyTorch or Keras within `1e-4` of the output's scale.
  Per-pixel outputs (segmentation) also pass when every pixel's class agrees and the
  drift stays below `2e-3`.
- **Detectors** must reproduce at least 90% of the detections: same label, a score within
  0.02, and IoU ≥ 0.9. Detections tied with the last kept score aren't compared, because
  which of two equal scores survives the cut is arbitrary.
- **Int8 formats** never fail the build. They report how often their top-1 agrees with
  the float model and, with labelled samples, both accuracies. The note warns when
  accuracy drops by more than one point.
- **Models that sample at inference** (MC dropout, Bayesian layers) are checked for
  output shape only.

The results are in `export_report.json` and the package's `metadata.json`. A pipeline's
**Export** stage turns them into metrics, `<format>_<metric>`:
- `onnx_max_diff`
- `executorch_int8_top1_agreement`, `executorch_int8_accuracy`
- `onnx_matched_detections`
- `<format>_mb`

## On-device

```bash
pip install 'ai-made-easy[edge]'        # ExecuTorch + the LiteRT interpreter
pip install tensorflow                  # Keras -> LiteRT
```

- **ExecuTorch** (PyTorch): lowered to XNNPACK for phone and embedded CPUs, with a fixed
  batch of one. **ExecuTorch int8** quantizes weights and activations (PT2E, symmetric
  int8), calibrated on the run's training samples. If int8 loses too much accuracy, set
  the Trainer's **QAT epochs**: after training, the run fine-tunes a copy with int8
  rounding simulated (quantization-aware training, Adam at 1e-4 on the CPU), reports its
  int8 test scores (`int8_accuracy`, ...) and saves `qat_int8.pt2`, which this export
  then uses instead of calibrating. Models with one input and output (not metric
  learning, pairs, two views or probabilistic heads).
- **LiteRT** (Keras): the `.tflite` file has a fixed batch of one. **LiteRT int8** is full
  integer quantization with float inputs and outputs, calibrated on the same samples.
  LiteRT needs Keras on TensorFlow. If the run trained on the torch backend and TensorFlow
  is installed, the package converts in a TensorFlow-backend process, which reads the
  same `.keras` file. The old name `tflite` still works.
- **Core ML** (PyTorch): for Apple devices. It isn't checked on the build machine.
- **OpenVINO** (PyTorch, the `openvino` extra): Intel's runtime for its CPUs, GPUs and
  NPUs. The export is a folder `model.openvino/` with `model.xml` and its weights
  `model.bin`, takes any batch size and is checked against PyTorch on this computer's
  CPU in float32 (within 1e-4 of the output scale; within 1e-2 on ARM CPUs such as
  Apple silicon, where its kernels drift, up to 5e-3 over Mamba-2's scan). Models with one input tensor; OpenVINO also reads the ONNX
  export.

## GGUF: language models for llama.cpp, Ollama and LM Studio

```bash
pip install 'ai-made-easy[gguf]'      # plus llama-cpp-python to check the file
aime deploy RUN_ID -o pkg --formats gguf
```

A language model built like the **Tiny Llama** recipe exports to one `model.gguf`
(float32) that llama.cpp, Ollama and LM Studio run. That means Embedding → Llama
Decoder Blocks → RMSNorm → Dense without bias. Its layers are llama.cpp's `llama`
architecture one to one:
- The rotary positions already rotate neighbouring feature pairs, as llama.cpp does.
- The fused key/value and gate/up weights are split into llama.cpp's tensors.

The tokenizer goes in the file too:
- **Bytes:** one token per byte.
- **BPE:** with a **BPE Tokenizer** block, the run's merges, as a GPT-2-style byte-level BPE.

On the synthetic text (and on a sentence with contractions, accents and runs of
spaces), llama.cpp splits text into exactly the run's tokens. Its word split follows
GPT-2's and can differ on rare characters (combining accents).

When llama-cpp-python is installed, the file is loaded on the CPU with a float32 cache
and checked token by token against PyTorch. The tests ask for a gap under 1e-4; they
see about 1e-6. llama.cpp's default float16 cache alone moves the logits by about 5e-4.

Any other design lists GGUF as unavailable, with the reason. To make the file smaller,
quantize it with llama.cpp:

```bash
llama-quantize model.gguf model-q8_0.gguf Q8_0
```

## Detectors and graph networks

**Detectors** (torchvision Faster / Mask / Keypoint R-CNN, RetinaNet, FCOS, SSD, the
YOLO-style head, and DETR / Conditional DETR / RT-DETR) export to ONNX and TorchScript
with their post-processing (NMS) included.
- Input: one image at the training size, `[1, C, H, W]` in `[0, 1]`.
- Output: padded detections `[1, D, 6]` = `x1, y1, x2, y2, score, label`. A score of 0
  is padding.

**Graph networks** export to ONNX through PyTorch Geometric's ONNX support. The file
takes any number of nodes and edges:
- Inputs: node features `x [N, F]` and edges `edge_index [2, E]`.
- Output: one row per node. Graph classification gives one row per graph; link
  prediction gives node embeddings.

The check runs the training graph, a smaller subgraph and a noisy copy.

## Device profiles suggest a format

Each device profile (`devices.json`, and your own in `~/.aime/devices.json`) lists the
formats it runs, best first. With a budget device set on the project, the deploy dialogs
pre-select those formats and mark them *suggested for <device>*:

| Device | Suggested (PyTorch) | Suggested (Keras) |
| --- | --- | --- |
| Raspberry Pi 5 | ExecuTorch int8, ONNX int8, ExecuTorch | LiteRT int8, LiteRT |
| Android phone | ExecuTorch int8, ExecuTorch, ONNX int8 | LiteRT int8, LiteRT |
| iPhone (Neural Engine) | Core ML, ExecuTorch int8, ExecuTorch | LiteRT int8, LiteRT |
| Jetson Orin Nano | ONNX (TensorRT reads it), ExecuTorch, TorchScript | LiteRT |
| Laptop CPU | ONNX, ONNX int8, TorchScript | ONNX, SavedModel |
| Apple M-series | Core ML, ONNX, TorchScript | ONNX, SavedModel |
| NVIDIA GPUs | ONNX, TorchScript | ONNX, SavedModel |

## What can't be exported, and why

A design rule warns when the budget device's formats are out of reach, and names the
block responsible. Without a device it notes when ONNX is out of reach. Formats a design
can't use are greyed out in the deploy dialogs, with the reason. The round-trip tests
check that every sample either exports to each format or is covered by one of these
rules:

| Design | Can't export to | Why |
| --- | --- | --- |
| Detectors | ExecuTorch, Core ML, OpenVINO | NMS gives a different number of boxes per image; fixed-shape runtimes can't follow it |
| Graph networks | ExecuTorch, Core ML, OpenVINO | nodes and edges vary per request |
| TopK / SAG pooling | ONNX | an in-place sort the ONNX exporter can't follow (TorchScript works) |
| Bayesian layers | ExecuTorch, Core ML | random weight sampling at inference (ONNX keeps it) |
| Mel spectrogram / MFCC | ExecuTorch | complex STFT |
| DLRM | ExecuTorch | `triu_indices` has no ExecuTorch kernel |
| Spline coupling (flows) | ExecuTorch, Core ML, OpenVINO | data-dependent shapes |
| Glow's invertible 1x1 convolution | Core ML, OpenVINO | `diag` has no converter |
| Hyena | ExecuTorch, Core ML, OpenVINO | FFTs (complex numbers) |
| Seq2seq transformer | ExecuTorch | token-by-token decoding loop |
| FT-Transformer | ExecuTorch int8 | categorical ids read from the float row |
| Mask2Former | ExecuTorch int8 | integer index tensors |
| Stable Diffusion LoRA | all | adapters are served with diffusers |
| Anything but a Llama-style decoder | GGUF | llama.cpp's `llama` architecture: Embedding → Llama Decoder Blocks → RMSNorm → Dense without bias |

Known sensitivity: RT-DETR picks its queries from the top encoder scores. On untrained
or barely trained models, near-ties can pick different queries in ONNX Runtime than in
PyTorch, so the detection check can fail there. TorchScript matches exactly.
