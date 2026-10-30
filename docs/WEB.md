# AI Made Easy on the web

`aime web` serves the designer in the browser together with a REST + WebSocket
API. It uses the same engine, run history, sweeps and model registry as the
desktop app, so runs started in either place show up in both.

## Run it locally

```bash
pip install "ai-made-easy[web,torch,vision,data,classic]"
aime web --open            # http://127.0.0.1:8765
```

What you get:

- **Design** — block library (drag onto the canvas or double-click), live
  validation with shapes on every wire, Quick Fixes, parameter inspector,
  model summary and generated code for every target; undo / redo, auto-arrange.
- **Training** — start a run and watch the log and loss curves stream live.
- **Data** — dataset profile, findings, column statistics, split preview and
  augmentation preview.
- **Experiments** — run history, comparison (metrics, differing parameters,
  overlaid curves, data fingerprints), sweeps, and serving packages for finished runs.
- **Models** — the model registry: stages and serving-package downloads.

Getting around (3.3):

- **File** menu: New project, Open, recent projects, Save (Ctrl/⌘ S), Download, Import
  model and Export canvas as PNG.
- **Command palette** (Ctrl/⌘ K): type to find any command; `@` lists the blocks on the
  canvas (also Ctrl/⌘ F) and Enter selects and centres one.
- **Shortcuts** (`?`): copy / paste / duplicate (Ctrl/⌘ C, V, D), group (Ctrl/⌘ G),
  select all (Ctrl/⌘ A), zoom (Ctrl/⌘ =, −, 0), undo / redo, pages (Alt+1…4).
- **Dockable panels** (3.12): drag the Block Library, the Inspector or the
  bottom console by its header to any screen edge or corner — edges glow while
  you drag. Every panel resizes from its splitters, collapses to a slim rail
  (click the rail to restore) and returns to the default arrangement with
  **Layout ▸ Reset layout**. The arrangement is remembered per browser. On
  narrow screens the panels become drawers over the canvas instead of squeezing
  it.
- Sticky notes and named groups on the canvas; drop a CSV or a zip of class folders on
  the canvas to start a project from it.
- A three-step tour on first start (the palette's "Show the tour again" brings it back).

Projects saved from the browser live in `$AIME_HOME/projects` (default
`~/.aime/projects`); *Download* exports the project JSON, which opens in the
desktop app too.

## Host it

The server is meant for one person or a trusted team. Before exposing it beyond
localhost, set an access token. Every API request must then carry it; the
browser asks for it once.

```bash
export AIME_WEB_TOKEN="$(openssl rand -hex 24)"
aime web --host 0.0.0.0 --port 8765
```

With Docker:

```bash
docker build -t ai-made-easy .
docker run -p 8765:8765 -e AIME_WEB_TOKEN=change-me -v aime-data:/data ai-made-easy
```

Behind a reverse proxy, forward WebSocket upgrades for `/api/runs/*/events`.
Training progress falls back to polling when WebSockets are blocked.
Terminate TLS at the proxy. The token is sent as a bearer header, so it must
only travel over HTTPS.

> Training runs arbitrary generated Python on the server, and dataset paths
> refer to the server's file system. Treat access to the server like shell
> access to the machine.

## API

Interactive documentation is at `/api/docs` (OpenAPI at `/api/openapi.json`).
The main groups:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/blocks`, `/api/samples` | block catalog (parameter schemas, ports), example projects |
| `POST /api/validate`, `/api/fix`, `/api/generate`, `/api/summary` | live validation, Quick Fixes, code, model summary |
| `GET/PUT/DELETE /api/projects/{name}` | saved projects |
| `POST /api/runs`, `GET /api/runs[/{id}]`, `WS /api/runs/{id}/events` | training and live events |
| `POST /api/sweeps`, `GET /api/sweeps/{id}/best` | hyperparameter sweeps |
| `GET /api/runs/{id}/deploy`, `/api/models/...` | serving packages (zip), model registry |
| `POST /api/import` | import ONNX / Keras / PyTorch models (multipart upload) |
| `POST /api/data/profile`, `/issues`, `/split`, `/augment` | data workspace |
| `POST /api/data/upload` | a dropped data file: stored, its facts detected for the wizard |
| `GET /api/runs/{id}/explain`, `POST /api/runs/{id}/diff` | feature importance; what changed since a run |
| `GET/POST /api/runs/{id}/try`, `POST /api/runs/{id}/counterfactuals` | try a finished run on your input; what would change a table row's prediction (`row`, optional `desired`, `target`, `locked`) |

## Develop the frontend

```bash
aime web --port 8765            # API
cd web && npm install && npm run dev   # Vite dev server with hot reload, proxies /api
npm run build                   # writes ai_made_easy/server/static (commit it)
npx playwright test             # browser smoke tests (starts its own server)
```

The frontend is React + TypeScript + Vite with React Flow for the canvas. The
built bundle is committed, so `pip install` works without Node.
