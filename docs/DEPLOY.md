# Deploy

Every finished run becomes a **serving package**: a FastAPI server (`/predict`,
`/health`, `/metadata`), the trained model with its fitted preprocessing, a Dockerfile
and a README. Build one from **Deploy** on the Experiments page, a pipeline's **Export**
stage, a registered model's **Share** menu (desktop and web), or the command line:

```bash
aime deploy <run id> -o server                      # the package
aime deploy <run id> -o server --formats suggested  # + what the budget device runs
cd server && docker build -t my-model . && docker run -p 8000:8000 my-model
```

## Where a model can go

| Where | How | Sign-in |
| --- | --- | --- |
| Your own server or laptop | the package, `uvicorn app:app` or its Dockerfile | none |
| An edge device | the export formats below; each is checked against the trained model ([EXPORT.md](EXPORT.md)) | none |
| MLflow Model Registry | **Share ▸ Register in MLflow**, `aime models register <run id> <name> --also mlflow` | `MLFLOW_TRACKING_URI` |
| Weights & Biases | **Share ▸ Log to W&B**, `--also wandb` | `wandb login` |
| Hugging Face Hub | **Share ▸ Push model**, `aime hub push <run id> owner/name` | `hf auth login` |
| A Hugging Face Space | **Share ▸ Deploy as a Space**, `aime hub space <run id> owner/name`: the package as a Docker Space | `hf auth login` |

Publishing to the Hub is confirmed every time and defaults to a private repository. AI
Made Easy never asks for, stores or logs a token: each tool uses its own sign-in
([TOOLS.md](TOOLS.md)).

## Formats per task

What a design for each task serves and exports, using the task's first recipe. A design
that differs (another architecture, a custom block) is checked the same way when you
deploy it: unavailable formats are greyed out with the reason.

| Task | Example recipe | Trained with | `/predict` takes | Export formats |
| --- | --- | --- | --- | --- |
| Multi-class classification | MLP | PyTorch | arrays | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Binary classification | MLP | PyTorch | arrays | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Regression | MLP | PyTorch | arrays | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Multi-label classification | MLP | PyTorch | arrays | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Distribution matching | MLP | PyTorch | arrays | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Clustering | k-Means | scikit-learn | records | ONNX |
| Anomaly detection | Isolation Forest | scikit-learn | records | ONNX |
| Keyword spotting | Pretrained speech encoder — linear probe | PyTorch | speech | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Object detection | Faster R-CNN (MobileNetV3) | PyTorch | images | ONNX, ONNX (int8 quantized), TorchScript |
| Instance segmentation | Mask R-CNN | PyTorch | images | ONNX, ONNX (int8 quantized), TorchScript |
| Keypoint detection | Keypoint R-CNN | PyTorch | images | ONNX, ONNX (int8 quantized), TorchScript |
| Semantic segmentation | U-Net | PyTorch | images | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Time-series forecasting | Forecasting network | PyTorch | series | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Audio tagging | Audio tagger | PyTorch | speech | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin) |
| Speech recognition (CTC) | BiLSTM + CTC | PyTorch | speech | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin) |
| Variational autoencoder | Variational autoencoder | PyTorch | generation | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Image generation (GAN) | DCGAN | PyTorch | generation | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Image generation (diffusion) | Diffusion (DDPM) | PyTorch | generation | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Text-to-image fine-tuning (LoRA) | Stable Diffusion LoRA | PyTorch | text_to_image | — |
| Language modeling | Tiny GPT | PyTorch | prompt | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK), GGUF (llama.cpp, Ollama, LM Studio), GGUF Q8_0 (8-bit weights, about a quarter of the size), GGUF Q4_0 (4-bit weights, about an eighth of the size) |
| Sequence to sequence | Encoder-decoder transformer | PyTorch | prompt | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin) |
| Density estimation (normalizing flow) | RealNVP flow | PyTorch | density | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Node classification | GCN | PyTorch | graph | ONNX, ONNX (int8 quantized), TorchScript |
| Graph classification | GIN | PyTorch | graph | ONNX, ONNX (int8 quantized), TorchScript |
| Link prediction | GraphSAGE link predictor | PyTorch | graph | ONNX, ONNX (int8 quantized), TorchScript |
| Recommendation | Matrix factorization | PyTorch | ranking | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Reinforcement learning | PPO | PyTorch | observations | ONNX, ONNX (int8 quantized), TorchScript, Core ML, OpenVINO (IR: .xml + .bin), ExecuTorch (.pte), ExecuTorch int8 (.pte, XNNPACK) |
| Probabilistic inference | Bayesian network | pgmpy | evidence | — |
| Regime detection (HMM) | Hidden Markov model | pgmpy | sequences | — |
| Bayesian modeling | Hierarchical regression | PyMC | records | — |
| Gaussian-process regression / classification | Gaussian process | GPyTorch | records | — |
| State-space forecasting | Structural time series | statsmodels | series | — |
| Text generation (LLM) | Prompted LLM | exported script | prompt | — |
| LLM fine-tuning | LoRA fine-tuning | exported script | prompt | — |
| Retrieval-augmented answering (RAG) | RAG assistant | exported script | prompt | — |

Families without export formats are served from Python only: their package runs the
fitted model (a pgmpy network, a PyMC trace, a statsmodels result, a GPyTorch model)
behind the same `/predict`; text-to-image fine-tunes are served with diffusers. LLM
workflows are exported as scripts (`aime export`) rather than packaged.

*Generated by `scripts/gen_deploy.py`; CI checks it is current.*
