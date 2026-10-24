# Train anywhere

A design can train on this computer or somewhere else: an SSH host, a rented GPU
machine, Kaggle, Colab, Modal or a Docker container. Live curves, Stop, sweeps and
pipelines work the same wherever the run trains. The run record (metrics, checkpoints,
the environment) always ends up in your local run history.

Pick where a run trains with **Run on** in the Training panel (desktop and web), the
sweep and pipeline dialogs, or the command line:

```bash
aime run project.json --on ssh:gpu-box
```

`--on` takes `local`, `ssh:<host>`, `kaggle`, `colab`, `modal`, `container`, or the
name of a saved backend.

## Sign-in and secrets

**AI Made Easy never asks for, stores or logs a password, token or key.** Each tool
signs in its own way, and AI Made Easy uses that:

| Backend | Sign-in |
| --- | --- |
| SSH | `~/.ssh/config`, your keys and ssh-agent (it never prompts for a password) |
| Kaggle | `~/.kaggle/kaggle.json` (Kaggle ▸ Settings ▸ API ▸ Create New Token) |
| Modal | `modal token new` (writes `~/.modal.toml`) |
| Colab | Google Drive for desktop, or no sign-in at all (manual hand-off) |
| Docker | the local Docker daemon |

A saved backend (`$AIME_HOME/backends.json`) holds only its name and options, such as a
host alias or a GPU type. Unknown options are refused, so a secret cannot end up there
by mistake.

## What travels

A remote run is sent as a **bundle**:

- the generated training script
- the training worker
- `requirements.txt` (package names from the script, without pinned versions, so the
  remote keeps the versions it already has)
- every local file the design points at

Paths to local files are rewritten to `data/…` inside the bundle. Nothing else from
your computer is sent. On the far side, a small launcher (standard library only)
unpacks the bundle, trains, and writes events and checkpoints next to the run.

## Managing backends

```bash
aime backends list --options                   # every backend and every kind's options
aime backends add gpu-box --kind ssh -s host=gpu-box -s python=/opt/venv/bin/python
aime backends test gpu-box                     # connects; reports Python, GPU, budget device
aime backends remove gpu-box
```

The desktop app has the same in **Run ▸ Backends…**, and the web app under **Run on ▸
Manage backends…**.

**Test** reports the backend's accelerator and the device profile that budget checks
use for it:
- SSH: `nvidia-smi`
- Kaggle: T4, P100 or TPU
- Colab / Modal: the GPU you chose

## SSH (and GPU clouds)

Any host in `~/.ssh/config` works, using its alias, keys and jump hosts.

| Option | Default | Meaning |
| --- | --- | --- |
| `host` | (required) | a host alias from `~/.ssh/config`, or `user@hostname` |
| `python` | `python3` | the Python on the host (a virtualenv's python works) |
| `install` | `false` | pip-install the run's requirements on the host first |
| `remote_root` | `.aime-remote` | run folders on the host, relative to your home folder |
| `docker_image` | (none) | train inside this image on the host (with `gpus`, default true) |
| `retries` | `30` | reconnect attempts after a dropped connection |

The run keeps going if the connection drops: the training runs under `nohup` on the
host, and the app reconnects and continues from the last event it received. If the app
itself was closed, follow the run again with **Follow again** on the Experiments page,
or:

```bash
aime runs attach <run id>
```

### Rented GPU machines

Each of these hands you an SSH machine. Add it to `~/.ssh/config` and use its alias:

```
Host runpod
  HostName <ip shown by the provider>
  Port <port>
  User root
  IdentityFile ~/.ssh/id_ed25519
```

- **RunPod:** pick a PyTorch template; *Connect* shows the SSH host and port. Use
  `python=python3` (PyTorch is preinstalled).
- **Lambda:** instances run Lambda Stack (PyTorch preinstalled); the user is `ubuntu`.
- **Vast.ai:** choose a PyTorch image; *Connect* shows the ssh command (a proxy host and
  port). The user is `root`.
- **Paperspace (DigitalOcean):** machines with the ML-in-a-Box template; the user is
  `paperspace`.

On machines without the packages you need, set `install=true`, or `docker_image`
(for example `pytorch/pytorch:latest`).

A machine with several GPUs trains on them together when the Trainer's **GPUs** is more
than one (PyTorch, DistributedDataParallel; the batch size is per GPU). It uses as many
as the machine has, up to that number. **Compile** (`torch.compile`) and **Gradient
checkpointing** (less memory, about a third more compute) are on the Trainer too.

## Kaggle

```bash
pip install 'ai-made-easy[kaggle]'
aime backends add kaggle-gpu --kind kaggle -s accelerator=T4
```

The run is pushed as a **private** Kaggle notebook (a script kernel) and polled until it
finishes. Its outputs are then downloaded into the run folder. Kaggle has no live log
API, so **curves arrive when the kernel finishes**, not as they happen.

| Option | Default | Meaning |
| --- | --- | --- |
| `accelerator` | `T4` | `none`, `T4`, `P100` or `TPU` |
| `datasets` | `[]` | Kaggle datasets to attach, e.g. `["owner/name"]`, read from `/kaggle/input/<name>` |
| `internet` | `true` | needed to pip-install; Kaggle requires a phone-verified account for it |
| `poll_seconds` | `30` | how often to check the kernel |

The bundle travels inside the kernel's one file, so local data is limited to about 20 MB.
For bigger data, upload it as a Kaggle dataset, attach it with `datasets`, and point the
dataset block at `/kaggle/input/<name>/…`.

## Colab

Colab has no job API, so a Colab run is a notebook hand-off. Neither way back needs a
password or token in AI Made Easy.

**With Google Drive for desktop** (found automatically, or set `drive_folder`):
1. The bundle and a notebook go to `My Drive/AI Made Easy/`.
2. Open the notebook in Colab (*File ▸ Open notebook ▸ Google Drive*). On macOS, the
   run log also shows a direct link once Drive has synced it. Choose a GPU runtime and
   *Run all*.
3. The notebook writes events and checkpoints to the same Drive folder. Drive for desktop
   syncs them here, and the run record fills in as they arrive.

**Without Drive** (`manual=true`, or when Drive for desktop is not installed):
1. The notebook (with the bundle inside) is saved in the run folder.
2. Upload it at colab.research.google.com (*File ▸ Upload notebook*) and *Run all*.
3. The last cell downloads a results zip. Import it with **Import results…** on the
   Experiments page (desktop: **Run ▸ Import Run Results…**), or:

   ```bash
   aime runs import aime_run_<run id>.zip
   ```

| Option | Default | Meaning |
| --- | --- | --- |
| `gpu` | `T4` | the runtime the notebook asks for: `T4`, `L4`, `A100` or `none` |
| `drive_folder` | found | your local "My Drive" folder |
| `manual` | `false` | skip Drive: notebook plus results zip |
| `poll_seconds` | `10` | how often to look for new events in Drive |

## Modal

```bash
pip install 'ai-made-easy[modal]'
modal token new
aime run project.json --on modal
```

The run trains in a Modal sandbox on a cloud GPU, and logs and curves stream live. The
image is built from the run's requirements; the first run takes a few minutes while it
builds, and later runs reuse it. The bundle and the outputs pass through a Modal volume
named `aime-runs`.

| Option | Default | Meaning |
| --- | --- | --- |
| `gpu` | `T4` | `none`, `T4`, `L4`, `A10G`, `A100`, `A100-80GB` or `H100` |
| `timeout_minutes` | `240` | the longest a run may take |

## Docker

```bash
aime run project.json --on container
```

The run trains in a container on this computer, for a reproducible environment. Without
an `image` option, AI Made Easy builds one per set of requirements, like the deploy
package's Dockerfile: `python:3.11-slim` plus the run's packages, with CPU PyTorch
unless `gpus=true`. Docker caches the image for later runs. Podman works too, if
`docker` is not installed.

| Option | Default | Meaning |
| --- | --- | --- |
| `image` | built | a ready image to train in (its requirements are pip-installed first) |
| `gpus` | `false` | pass the GPUs through (`--gpus all`; needs the NVIDIA container toolkit) |

For a container on another machine, use an SSH backend with `docker_image`.

## Resuming a run

A run that stopped or failed on a backend resumes there (**Resume** in either app,
`aime runs resume <id>`, `POST /api/runs/{id}/resume`), as the same run. The training
script saves its state after each epoch; the backend brings that back with the outputs
when the run ends, and the resumed run carries it out again in its bundle. On Kaggle the
state comes back only with the kernel's outputs, so a kernel stopped from the app can't
resume. Kaggle and Colab runs carry at most 20 MB inside their one file, saved state
included.

## Sweeps, pipelines and agents

- **Sweeps:** trials train on the chosen backend (`aime sweep … --on gpu-box`, or the
  Run on selector in the sweep dialog).
- **Pipelines:** Train stages train on the chosen backend. The other stages (distill,
  prune, quantize, evaluate, export) run on this computer, from the checkpoints the
  Train stages brought back.
- **MCP:** `start_training` takes `on`, and there are `list_backends`, `test_backend`,
  `save_backend`, `attach_run` and `import_run_results` tools.
- **REST:** `GET/POST /api/backends`, `POST /api/backends/test`,
  `DELETE /api/backends/{name}`, `POST /api/runs/{id}/attach`, `POST /api/runs/import`
  (the zip as the body), and `on` in `POST /api/runs` and `POST /api/pipelines`.
