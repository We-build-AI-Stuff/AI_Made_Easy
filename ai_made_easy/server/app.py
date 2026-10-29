"""AI Made Easy web server: REST + WebSocket API over ``core.api`` and the browser UI.

``create_app()`` builds the FastAPI application; ``aime web`` serves it with
uvicorn. Every endpoint lives under ``/api``; the built frontend (``static/``)
is served at ``/``. Projects are stored as JSON under ``$AIME_HOME/projects``;
runs, sweeps and the model registry use the same stores as the desktop app.

Access control comes in two flavours:

- a shared token: when ``AIME_WEB_TOKEN`` (or ``create_app(token=...)``) is set,
  every request needs ``Authorization: Bearer <token>``. WebSockets and GET requests
  (download links the browser opens directly) may pass ``?token=<token>`` instead.
  Without it the server is meant for localhost only.
- user accounts (``aime web --auth`` or ``AIME_WEB_AUTH=1``): users sign in with a
  name and password (``server/users.py``); members work on their own projects,
  admins manage users and see every project. The shared token, when set, still
  works next to accounts as an admin key for scripts.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import logging
import os
import re
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from ai_made_easy import __version__
from ai_made_easy.core import api
from ai_made_easy.core.paths import aime_home, subdir
from ai_made_easy.core.runner.manager import max_upload_bytes
from ai_made_easy.server.users import (
    LoginThrottle,
    User,
    UserError,
    UserStore,
    issue_token,
    load_secret,
    read_token,
)

STATIC = Path(__file__).parent / "static"
_NAME = re.compile(r"^[\w][\w .-]{0,99}$")
_OWNERS = ".owners.json"    # project → user, beside the projects (not inside them)
_STALE_POLLS = 5  # ~10 s without events from a run no process is running
_CHUNK = 1 << 20
log = logging.getLogger(__name__)


def _too_large() -> HTTPException:
    return HTTPException(413, f"upload larger than {max_upload_bytes() >> 20} MB "
                              "(set AIME_MAX_UPLOAD_MB to change)")


async def _save_body(request: Request, target: Path) -> int:
    """Stream a raw request body to ``target`` without holding it in memory."""
    limit, size = max_upload_bytes(), 0
    with target.open("wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise _too_large()
            fh.write(chunk)
    return size


async def _save_upload(file: UploadFile, target: Path) -> int:
    limit, size = max_upload_bytes(), 0
    with target.open("wb") as fh:
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > limit:
                raise _too_large()
            fh.write(chunk)
    return size


def _graph_body(payload: dict) -> dict:
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        raise HTTPException(400, "body must contain a 'graph' object")
    return graph


def _project_file(root: Path, name: str) -> Path:
    if not _NAME.match(name) or name.endswith("."):
        raise HTTPException(400, "project names may use letters, digits, spaces, '_', '-' and '.'")
    return root / f"{name}.json"


def _zip_dir(folder: Path) -> Path:
    archive = shutil.make_archive(str(folder), "zip", root_dir=folder)
    return Path(archive)


def _data_url(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def create_app(*, token: str | None = None, projects_dir: str | Path | None = None,
               auth: bool | None = None, users_path: str | Path | None = None,
               secret_path: str | Path | None = None) -> FastAPI:
    token = (token if token is not None else os.environ.get("AIME_WEB_TOKEN")) or None
    if auth is None:
        auth = os.environ.get("AIME_WEB_AUTH", "").lower() in ("1", "true", "yes", "on")
    accounts = UserStore(users_path or aime_home() / "users.json") if auth else None
    secret = load_secret(secret_path or aime_home() / "web_secret.key") if auth else b""
    throttle = LoginThrottle() if auth else None
    projects = Path(projects_dir) if projects_dir else subdir("projects")
    save_lock = threading.Lock()
    projects.mkdir(parents=True, exist_ok=True)
    try:  # runs a crashed server or app left "running" would be followed forever
        api.manager().history.mark_interrupted()
    except Exception as exc:  # noqa: BLE001 — never block the server from starting
        log.warning("could not check for interrupted runs: %s", exc)

    def _owners() -> dict[str, str]:
        try:
            return json.loads((projects / _OWNERS).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _set_owner(name: str, owner: str | None) -> None:
        with save_lock:
            owners = _owners()
            if owner is None:
                owners.pop(name, None)
            else:
                owners[name] = owner
            tmp = projects / f".owners.{uuid.uuid4().hex[:8]}.tmp"
            tmp.write_text(json.dumps(owners), encoding="utf-8")
            tmp.replace(projects / _OWNERS)

    def _identity(request: Request, *, for_query: str = "") -> User | None:
        """Who is asking: a session token, the shared token (admin) or no one."""
        supplied = (request.headers.get("authorization", "") if request else ""
                    ).removeprefix("Bearer ").strip() or for_query
        if not supplied:
            return None
        if token and secrets.compare_digest(supplied, token):
            return User("token", "admin")   # the shared key acts as an admin
        if accounts is not None:
            return read_token(secret, supplied)
        return None

    def authorize(request: Request) -> None:
        if accounts is None:
            if token is None:
                return
            user = _identity(request, for_query=request.query_params.get("token", "")
                             if request.method == "GET" else "")
            if user is None:
                raise HTTPException(401, "missing or invalid access token")
            request.state.user = user
            return
        user = _identity(request, for_query=request.query_params.get("token", "")
                         if request.method == "GET" else "")
        if user is None:
            raise HTTPException(401, "sign in first")
        request.state.user = user

    def current_user(request: Request) -> User:
        """The signed-in user, or an anonymous admin when auth is off."""
        if accounts is None:
            return User("", "admin")
        user = getattr(request.state, "user", None) or _identity(request)
        if user is None:
            raise HTTPException(401, "sign in first")
        if user.name != "token" and accounts.get(user.name) is None:
            raise HTTPException(401, "this account is no longer available")
        return user

    def require_admin(request: Request) -> User:
        user = current_user(request)
        if user.role != "admin":
            raise HTTPException(403, "admins only")
        return user

    app = FastAPI(title="AI Made Easy", version=__version__, docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    guard = [Depends(authorize)]

    @app.exception_handler(api.ApiError)
    async def _api_error(_request: Request, exc: api.ApiError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def _not_found(_request: Request, exc: KeyError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc.args[0]) if exc.args else "not found"},
                            status_code=404)

    @app.exception_handler(ValueError)
    async def _bad_value(_request: Request, exc: ValueError):  # noqa: ANN202
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ------------------------------------------------------------------ info
    @app.get("/api/info")
    def info() -> dict:
        from ai_made_easy.core.registry import get_registry

        return {"version": __version__, "auth": token is not None,
                "auth_mode": "accounts" if accounts is not None else "token" if token else "none",
                "blocks": len(get_registry().all())}

    # ----------------------------------------------------------------- auth
    @app.get("/api/auth/status")
    def auth_status(request: Request) -> dict:
        """Public: which flavour of access control is on, and who is asking."""
        user = _identity(request, for_query=request.query_params.get("token", ""))
        if accounts is None:
            me = {"name": "", "role": "admin"} if (token is None or user is not None) else None
            return {"mode": "token" if token else "none", "user": me}
        return {"mode": "accounts",
                "user": ({"name": user.name, "role": user.role}
                         if user and (user.name == "token"
                                      or accounts.get(user.name)) else None)}

    @app.post("/api/auth/login")
    def auth_login(payload: dict = Body(...)) -> dict:  # noqa: B008
        if accounts is None:
            raise HTTPException(400, "this server does not use user accounts")
        name = str(payload.get("name", "")).strip()
        password = str(payload.get("password", ""))
        if not name or not password:
            raise HTTPException(400, "name and password are required")
        if (left := throttle.blocked(name)) > 0:
            raise HTTPException(429, f"too many failed sign-ins — try again in {int(left) + 1} s")
        user = accounts.verify(name, password)
        if user is None:
            throttle.record_failure(name)
            raise HTTPException(401, "wrong name or password")
        throttle.record_success(name)
        return {"token": issue_token(secret, user),
                "user": {"name": user.name, "role": user.role}}

    @app.get("/api/auth/me", dependencies=guard)
    def auth_me(request: Request) -> dict:
        user = current_user(request)
        return {"name": user.name, "role": user.role}

    # -------------------------------------------------------- user management
    def _user_or_400(payload: dict, key: str = "name") -> str:
        name = str(payload.get(key, "")).strip()
        if not name:
            raise HTTPException(400, "name is required")
        return name

    @app.get("/api/admin/users", dependencies=guard)
    def admin_users(request: Request) -> dict:
        require_admin(request)
        return {"users": accounts.list()}

    @app.post("/api/admin/users", dependencies=guard)
    def admin_add_user(request: Request, payload: dict = Body(...)) -> dict:  # noqa: B008
        require_admin(request)
        try:
            user = accounts.add(_user_or_400(payload), str(payload.get("password", "")),
                                str(payload.get("role", "member")))
        except UserError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"created": user.name, "role": user.role}

    @app.patch("/api/admin/users/{name}", dependencies=guard)
    def admin_patch_user(request: Request, name: str,
                         payload: dict = Body(...)) -> dict:  # noqa: B008
        admin = require_admin(request)
        try:
            if payload.get("password") is not None:
                accounts.set_password(name, str(payload["password"]))
            if payload.get("role") is not None:
                accounts.set_role(name, str(payload["role"]))
            if payload.get("disabled") is not None:
                if name == admin.name and payload["disabled"] is True:
                    raise UserError("you cannot disable yourself")
                accounts.set_disabled(name, bool(payload["disabled"]))
        except UserError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"updated": name}

    @app.delete("/api/admin/users/{name}", dependencies=guard)
    def admin_delete_user(request: Request, name: str) -> dict:
        admin = require_admin(request)
        if name == admin.name:
            raise HTTPException(400, "you cannot remove yourself")
        try:
            accounts.remove(name)
        except UserError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"removed": name}

    # ------------------------------------------------------------- catalog
    @app.get("/api/blocks", dependencies=guard)
    def blocks(category: str | None = None) -> dict:
        return api.list_blocks(category)

    @app.get("/api/families", dependencies=guard)
    def families() -> dict:
        return api.list_families()

    @app.get("/api/tasks", dependencies=guard)
    def tasks(family: str | None = None) -> dict:
        return api.list_tasks(family)

    @app.get("/api/devices", dependencies=guard)
    def devices() -> dict:
        return api.list_devices()

    @app.post("/api/budget", dependencies=guard)
    def budget(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            return api.estimate_budget(_graph_body(payload), payload.get("device"),
                                       payload.get("limits") or None)
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — incomplete designs are user-facing
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/describe", dependencies=guard)
    def describe(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.describe_design(_graph_body(payload))

    @app.post("/api/table_layout", dependencies=guard)
    def table_layout(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.table_layout(_graph_body(payload), str(payload.get("node", "")))

    @app.get("/api/samples", dependencies=guard)
    def samples() -> dict:
        return api.list_samples()

    @app.get("/api/samples/{name}", dependencies=guard)
    def sample(name: str) -> dict:
        from ai_made_easy.core.migrate import migrate

        return migrate(api.read_sample(name))

    # --------------------------------------------------------------- graph
    @app.post("/api/validate", dependencies=guard)
    def validate(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.validate(_graph_body(payload))

    @app.post("/api/fix", dependencies=guard)
    def fix(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.apply_fix(_graph_body(payload), int(payload.get("index", -1)))

    @app.get("/api/targets", dependencies=guard)
    def targets() -> dict:
        return api.targets()

    @app.post("/api/generate", dependencies=guard)
    def generate(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            code = api.generate(_graph_body(payload), str(payload.get("target", "pytorch_model")))
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — codegen errors are user-facing
            raise HTTPException(400, str(exc)) from exc
        return {"code": code}

    @app.post("/api/summary", dependencies=guard)
    def summary(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            return api.summarize(_graph_body(payload))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/expand", dependencies=guard)
    def expand(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.expand(_graph_body(payload), str(payload.get("node_id", "")))

    # ------------------------------------------------------------ projects
    def _may_write(user: User, name: str, owners: dict[str, str]) -> bool:
        """Admins (and the shared token) may touch anything; a member only their own
        and unowned projects — ones saved before accounts existed."""
        return user.role == "admin" or user.name == "token" or owners.get(name, "") in (
            "", user.name)

    @app.get("/api/projects", dependencies=guard)
    def list_projects(request: Request) -> dict:
        from ai_made_easy.core.migrate import kind_of

        user = current_user(request)
        owners = _owners() if accounts is not None else {}
        rows = []
        for p in sorted(projects.glob("*.json"), key=lambda p: -p.stat().st_mtime):
            if p.name.startswith("."):     # .owners.json bookkeeping, not a project
                continue
            if accounts is not None and not _may_write(user, p.stem, owners):
                continue
            row = {"name": p.stem, "modified": p.stat().st_mtime, "family": None,
                   "task": None, "owner": owners.get(p.stem)}
            try:
                row.update(kind_of(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001, S110 — an unreadable file still gets listed
                pass
            rows.append(row)
        return {"projects": rows}

    @app.get("/api/projects/{name}", dependencies=guard)
    def get_project(request: Request, name: str) -> dict:
        from ai_made_easy.core.migrate import migrate

        file = _project_file(projects, name)
        if not file.exists():
            raise HTTPException(404, f"no project named {name!r}")
        if accounts is not None and not _may_write(current_user(request), name, _owners()):
            raise HTTPException(404, f"no project named {name!r}")  # not theirs to know of
        return migrate(json.loads(file.read_text(encoding="utf-8")))

    @app.put("/api/projects/{name}", dependencies=guard)
    def put_project(request: Request, name: str, graph: dict = Body(...)) -> dict:  # noqa: B008
        from ai_made_easy.core.migrate import stamp

        user = current_user(request)
        if accounts is not None:
            owners = _owners()
            if not _may_write(user, name, owners):
                raise HTTPException(403, f"{name!r} belongs to "
                                         f"{owners.get(name, 'another user')}")
        file = _project_file(projects, name)
        try:
            graph = stamp({**graph, "name": name})
        except Exception as exc:  # noqa: BLE001 — refuse unreadable projects
            raise HTTPException(400, f"not a valid project: {exc}") from exc
        tmp = file.with_name(f"{file.stem}.{uuid.uuid4().hex[:8]}.tmp")  # one per save
        tmp.write_text(json.dumps(graph, indent=1), encoding="utf-8")
        with save_lock:     # Windows refuses to replace a file another save is replacing
            tmp.replace(file)
        if accounts is not None and user.name not in ("", "token"):
            _set_owner(name, user.name)
        return {"saved": name}

    @app.delete("/api/projects/{name}", dependencies=guard)
    def delete_project(request: Request, name: str) -> dict:
        user = current_user(request)
        if accounts is not None and not _may_write(user, name, _owners()):
            raise HTTPException(403, f"{name!r} belongs to another user")
        file = _project_file(projects, name)
        if not file.exists():
            raise HTTPException(404, f"no project named {name!r}")
        file.unlink()
        if accounts is not None:
            _set_owner(name, None)
        return {"deleted": name}

    # ---------------------------------------------------------------- runs
    @app.post("/api/runs", dependencies=guard)
    def start_run(payload: dict = Body(...)) -> dict:  # noqa: B008
        try:
            return api.start_training(_graph_body(payload), str(payload.get("framework", "auto")),
                                      str(payload.get("project", "")),
                                      tags=payload.get("tags"), on=payload.get("on") or None)
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — e.g. codegen errors
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/runs/import", dependencies=guard)
    async def import_results(request: Request) -> dict:
        """A Colab hand-off run's results zip, as the raw request body."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.zip"
            size = await _save_body(request, path)
            if not size:
                raise HTTPException(400, "send the results .zip as the request body")
            return await asyncio.to_thread(api.import_run_results, str(path))

    @app.post("/api/runs/{run_id}/attach", dependencies=guard)
    def attach(run_id: str) -> dict:
        return api.attach_run(run_id)

    # ---------------------------------------------------------------- backends
    @app.get("/api/backends", dependencies=guard)
    def list_backends() -> dict:
        return api.list_backends()

    @app.post("/api/backends", dependencies=guard)
    def save_backend(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.save_backend(str(payload.get("name", "")), str(payload.get("kind", "")),
                                payload.get("options") or {})

    @app.post("/api/backends/test", dependencies=guard)
    def test_backend(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.test_backend(str(payload.get("on") or "local"))

    @app.delete("/api/backends/{name}", dependencies=guard)
    def delete_backend(name: str) -> dict:
        return api.delete_backend(name)

    @app.get("/api/runs", dependencies=guard)
    def runs(project: str | None = None, parent: str | None = None) -> dict:
        return api.list_runs(project or None, parent or None)

    @app.post("/api/runs/compare", dependencies=guard)
    def compare(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.compare_runs([str(r) for r in payload.get("run_ids", [])])

    @app.get("/api/runs/{run_id}", dependencies=guard)
    def run(run_id: str) -> dict:
        return api.get_run(run_id)

    @app.get("/api/runs/{run_id}/samples", dependencies=guard)
    def run_samples(run_id: str) -> dict:
        return api.run_samples(run_id)

    @app.post("/api/runs/{run_id}/diff", dependencies=guard)
    def run_diff(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.diff_run(run_id, _graph_body(payload))

    @app.get("/api/runs/{run_id}/explain", dependencies=guard)
    def run_explain(run_id: str) -> dict:
        return api.run_explanation(run_id)

    @app.get("/api/runs/{run_id}/try", dependencies=guard)
    def try_describe(run_id: str) -> dict:
        """What a finished run takes (loads its predictor on this machine)."""
        return api.try_describe(run_id)

    @app.post("/api/runs/{run_id}/try", dependencies=guard)
    def try_run(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.try_run(run_id, payload.get("inputs"), explain=bool(payload.get("explain")))

    @app.post("/api/runs/{run_id}/counterfactuals", dependencies=guard)
    async def what_would_change(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        """What would change the prediction for one table row."""
        target = payload.get("target")
        try:
            goal = float(target) if target not in (None, "") else None
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "target must be a number") from exc
        return await asyncio.to_thread(
            api.what_would_change, run_id, payload.get("row") or {}, payload.get("desired"),
            goal, payload.get("locked") or [])

    @app.post("/api/runs/{run_id}/try/file", dependencies=guard)
    async def try_file(run_id: str, file: UploadFile = File(...)) -> Response:  # noqa: B008
        """Predictions for a whole file of inputs, as a CSV download."""
        work = Path(tempfile.mkdtemp(prefix="aime_try_"))
        try:
            target = work / "upload"
            await _save_upload(file, target)
            got = await asyncio.to_thread(api.try_file, run_id, file.filename or "inputs",
                                          target.read_bytes())
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return Response(got["csv"], media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="{got["filename"]}"',
            "X-Rows": str(got["rows"])})

    @app.get("/api/runs/{run_id}/metrics", dependencies=guard)
    def metrics(run_id: str) -> dict:
        return {"run_id": run_id, "epochs": api.manager().history.epochs(run_id)}

    @app.post("/api/runs/{run_id}/stop", dependencies=guard)
    def stop(run_id: str) -> dict:
        return api.stop_run(run_id)

    @app.patch("/api/runs/{run_id}", dependencies=guard)
    def update(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.update_run(run_id, tags=payload.get("tags"), note=payload.get("note"))

    @app.delete("/api/runs/{run_id}", dependencies=guard)
    def delete_run(run_id: str) -> dict:
        return api.delete_run(run_id)

    @app.websocket("/api/runs/{run_id}/events")
    async def run_events(ws: WebSocket, run_id: str, token_q: str = Query("", alias="token")):
        if accounts is not None:
            who = (User("token", "admin") if token and secrets.compare_digest(token_q, token)
                   else read_token(secret, token_q))
            if who is None or (who.name != "token" and accounts.get(who.name) is None):
                await ws.close(code=4401)
                return
        elif token is not None and not secrets.compare_digest(token_q, token):
            await ws.close(code=4401)
            return
        mgr = api.manager()
        try:
            mgr.history.get(run_id)
        except KeyError:
            await ws.close(code=4404)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def listener(rid: str, event: dict) -> None:
            if rid == run_id:
                loop.call_soon_threadsafe(queue.put_nowait, event)

        # a run in progress replays its log lines so far (a remote run's first lines say
        # where it went and what to do); a finished one replays epochs and its status
        live = mgr.get(run_id) if mgr.is_live(run_id) else None
        earlier = (list(live.logs[-300:]) if live is not None
                   and live.state not in ("finished", "failed", "stopped") else [])
        mgr.subscribe(listener)
        try:
            for line in earlier:
                await ws.send_json({"type": "log", "line": line})
            for epoch in mgr.history.epochs(run_id):  # replay what happened so far
                await ws.send_json({"type": "epoch", **epoch})
            if (samples := mgr.latest_samples(run_id)) is not None:
                await ws.send_json(samples)
            idle = 0
            while True:
                record = mgr.history.get(run_id)
                # nothing will change a run no process is running: report it and stop
                stale = idle >= _STALE_POLLS and not mgr.is_live(run_id)
                if (record.status != "running" or stale) and queue.empty():
                    await ws.send_json({"type": "status", "status": record.status,
                                        "final_metrics": record.final_metrics})
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=2.0)
                except TimeoutError:
                    if not mgr.is_live(run_id):
                        idle += 1
                        await asyncio.sleep(0.3)  # let the history catch up
                    continue
                idle = 0
                await ws.send_json(event)
                if event.get("type") == "done":
                    await asyncio.sleep(0.3)  # finalize writes the status after "done"
        except WebSocketDisconnect:
            pass
        finally:
            mgr.unsubscribe(listener)
            try:
                await ws.close()
            except RuntimeError:
                pass

    # -------------------------------------------------------------- wizard / AutoML
    @app.get("/api/wizard/tasks", dependencies=guard)
    def wizard_tasks() -> dict:
        return api.wizard_tasks()

    @app.post("/api/wizard/detect", dependencies=guard)
    def wizard_detect(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.detect_data(str(payload.get("path") or ""), str(payload.get("target") or ""),
                               str(payload.get("task") or ""))

    @app.post("/api/wizard/recommend", dependencies=guard)
    def wizard_recommend(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.recommend_recipes(
            str(payload.get("task") or ""), facts=payload.get("facts") or None,
            path=payload.get("path") or None, target=str(payload.get("target") or ""),
            budget=payload.get("budget") or None, modality=str(payload.get("modality") or ""),
            limit=payload.get("limit"))

    @app.post("/api/wizard/build", dependencies=guard)
    def wizard_build(payload: dict = Body(...)) -> dict:  # noqa: B008
        return {"graph": api.build_recipe(
            str(payload.get("recipe") or ""), str(payload.get("task") or ""),
            facts=payload.get("facts") or None, path=payload.get("path") or None,
            target=str(payload.get("target") or ""), knobs=payload.get("knobs") or None,
            budget=payload.get("budget") or None)}

    @app.get("/api/recipes", dependencies=guard)
    def recipes(task: str | None = None) -> dict:
        return api.list_recipes(task or None)

    @app.post("/api/explain", dependencies=guard)
    def explain(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.explain_design(_graph_body(payload))

    @app.post("/api/automl", dependencies=guard)
    def start_automl(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.start_automl(dict(payload.get("spec") or {}),
                                project=str(payload.get("project") or ""))

    @app.get("/api/automl/{sweep_id}", dependencies=guard)
    def automl_leaderboard(sweep_id: str) -> dict:
        try:
            return api.automl_leaderboard(sweep_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    # -------------------------------------------------------------- pipelines
    @app.post("/api/pipelines", dependencies=guard)
    def start_pipeline(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.start_pipeline(_graph_body(payload), project=str(payload.get("project") or ""),
                                  on=str(payload.get("on") or ""))

    @app.get("/api/pipelines", dependencies=guard)
    def pipelines(project: str | None = None) -> dict:
        return api.list_pipelines(project or None)

    @app.get("/api/pipelines/{pipeline_id}", dependencies=guard)
    def pipeline(pipeline_id: str) -> dict:
        try:
            return api.get_pipeline(pipeline_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/pipelines/{pipeline_id}/stop", dependencies=guard)
    def stop_pipeline(pipeline_id: str) -> dict:
        return api.stop_pipeline(pipeline_id)

    @app.post("/api/pipelines/{pipeline_id}/resume", dependencies=guard)
    def resume_pipeline(pipeline_id: str) -> dict:
        try:
            return api.resume_pipeline(pipeline_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    # -------------------------------------------------------------- sweeps
    @app.post("/api/sweeps/params", dependencies=guard)
    def sweep_params(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.sweepable_params(_graph_body(payload))

    @app.post("/api/sweeps", dependencies=guard)
    def start_sweep(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.start_sweep(_graph_body(payload), dict(payload.get("spec") or {}),
                               str(payload.get("project", "")))

    @app.get("/api/sweeps", dependencies=guard)
    def sweeps(project: str | None = None) -> dict:
        return api.list_sweeps(project or None)

    @app.get("/api/sweeps/{sweep_id}", dependencies=guard)
    def sweep(sweep_id: str) -> dict:
        data = api.get_sweep(sweep_id)
        data.pop("graph", None)
        return data

    @app.post("/api/sweeps/{sweep_id}/stop", dependencies=guard)
    def stop_sweep(sweep_id: str) -> dict:
        return api.stop_sweep(sweep_id)

    @app.get("/api/sweeps/{sweep_id}/analysis", dependencies=guard)
    def sweep_analysis(sweep_id: str) -> dict:
        return api.sweep_analysis(sweep_id)

    @app.get("/api/sweeps/{sweep_id}/again", dependencies=guard)
    def sweep_again(sweep_id: str) -> dict:
        return api.sweep_again(sweep_id)

    @app.get("/api/sweeps/{sweep_id}/best", dependencies=guard)
    def sweep_best(sweep_id: str) -> dict:
        return {"graph": api.sweep_best_graph(sweep_id)}

    # -------------------------------------------------------------- 3.8: tune and inspect
    @app.post("/api/lr-find", dependencies=guard)
    def find_lr(payload: dict = Body(...)) -> dict:  # noqa: B008
        """Start a learning-rate range test: {graph, start?, end?, steps?, project?,
        framework?}."""
        try:
            return api.find_learning_rate(
                _graph_body(payload), start=payload.get("start"), end=payload.get("end"),
                steps=payload.get("steps"), project=str(payload.get("project", "")),
                framework=str(payload.get("framework") or "auto"))
        except api.ApiError:
            raise
        except Exception as exc:  # noqa: BLE001 — e.g. codegen errors
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/lr-find/{run_id}", dependencies=guard)
    def lr_result(run_id: str) -> dict:
        return api.learning_rate_result(run_id)

    @app.post("/api/lr-find/apply", dependencies=guard)
    def lr_apply(payload: dict = Body(...)) -> dict:  # noqa: B008
        """{graph, lr} -> the design with that learning rate on its optimizer."""
        return {"graph": api.apply_learning_rate(_graph_body(payload), payload.get("lr"))}

    @app.post("/api/runs/{run_id}/more", dependencies=guard)
    def more(run_id: str, payload: dict = Body(default={})) -> dict:  # noqa: B008
        """Train more: {epochs, lr_scale?} for a network, {trees} for a boosted model ->
        the new run."""
        trees = payload.get("trees")
        return api.train_more(run_id, int(payload.get("epochs", 10)),
                              float(payload.get("lr_scale", 1.0)),
                              None if trees is None else int(trees))

    @app.get("/api/runs/{run_id}/resume", dependencies=guard)
    def resume_info(run_id: str) -> dict:
        """Whether the run can resume, and after which epoch."""
        return api.resume_info(run_id)

    @app.post("/api/runs/{run_id}/resume", dependencies=guard)
    def resume_run(run_id: str) -> dict:
        """Continue a stopped, failed or interrupted run as the same run."""
        return api.resume_run(run_id)

    @app.get("/api/runs/{run_id}/groups/columns", dependencies=guard)
    def group_columns(run_id: str) -> dict:
        return api.group_columns(run_id)

    @app.post("/api/runs/{run_id}/groups", dependencies=guard)
    def by_group(run_id: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        """{column} -> the scoring run; GET /api/groups/{that run} reads the result."""
        return api.results_by_group(run_id, str(payload.get("column", "")))

    @app.get("/api/groups/{run_id}", dependencies=guard)
    def group_result(run_id: str) -> dict:
        return api.group_result(run_id)

    # -------------------------------------------------------------- deploy
    @app.get("/api/runs/{run_id}/formats", dependencies=guard)
    def run_formats(run_id: str) -> dict:
        api.manager().history.get(run_id)
        return api.export_options(run_id)

    @app.get("/api/models/{name}/{version}/formats", dependencies=guard)
    def model_formats(name: str, version: str) -> dict:
        return api.export_options(model=name, version=version)

    @app.post("/api/export/formats", dependencies=guard)
    def design_formats(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.export_options(graph=_graph_body(payload),
                                  framework=payload.get("framework") or None)

    def _package_response(build) -> FileResponse:  # noqa: ANN001
        work = Path(tempfile.mkdtemp(prefix="aime_package_"))
        try:
            result = build(str(work / "package"))
            archive = _zip_dir(Path(result["path"]))
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        name = f"{result['metadata'].get('name', 'model')}-serving.zip"
        return FileResponse(archive, filename=name, media_type="application/zip",
                            background=BackgroundTask(shutil.rmtree, work, True))

    def _split(formats: str) -> list[str]:
        return [f.strip() for f in formats.split(",") if f.strip()]

    @app.get("/api/runs/{run_id}/deploy", dependencies=guard)
    def deploy_run(run_id: str, formats: str = "") -> FileResponse:
        return _package_response(lambda out: api.deploy_run(run_id, out, _split(formats)))

    @app.get("/api/models", dependencies=guard)
    def models() -> dict:
        return api.list_models()

    @app.post("/api/models", dependencies=guard)
    def register(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.register_model(str(payload.get("run_id", "")), str(payload.get("name", "")),
                                  str(payload.get("description", "")),
                                  also=list(payload.get("also") or []))

    @app.patch("/api/models/{name}/{version}", dependencies=guard)
    def set_stage(name: str, version: int, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.set_model_stage(name, version, str(payload.get("stage", "")),
                                   also=list(payload.get("also") or []))

    @app.post("/api/models/{name}/{version}/mirror", dependencies=guard)
    def mirror_model(name: str, version: str, payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.mirror_model(name, version, list(payload.get("to") or []))

    # ------------------------------------------- tracking, the Hub, Kaggle (3.2)
    @app.get("/api/tracking", dependencies=guard)
    def tracking_status() -> dict:
        return api.tracking_status()

    @app.post("/api/tracking", dependencies=guard)
    def tracking_for(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.tracking_status(payload.get("graph") or None)

    @app.get("/api/export/runs", dependencies=guard)
    def export_runs(format: str = "csv", epochs: bool = False,  # noqa: A002
                    project: str | None = None) -> FileResponse:
        if format not in ("csv", "parquet"):
            raise HTTPException(400, "format must be csv or parquet")
        work = Path(tempfile.mkdtemp(prefix="aime_runs_"))
        out = work / f"runs.{format}"
        try:
            api.export_runs(str(out), project=project, epochs=epochs)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return FileResponse(out, filename=f"{'epochs' if epochs else 'runs'}.{format}",
                            background=BackgroundTask(shutil.rmtree, work, True))

    @app.post("/api/data/upload", dependencies=guard)
    async def data_upload(file: UploadFile = File(...)) -> dict:  # noqa: B008
        """A data file dropped on the canvas: stored and detected for the wizard."""
        work = Path(tempfile.mkdtemp(prefix="aime_data_"))
        try:
            target = work / "upload"
            await _save_upload(file, target)
            return await asyncio.to_thread(api.store_upload, str(target),
                                           file.filename or "data")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @app.post("/api/data/fetch", dependencies=guard)
    async def data_fetch(payload: dict = Body(...)) -> dict:  # noqa: B008
        return await asyncio.to_thread(api.fetch_dataset, str(payload.get("source") or ""),
                                       force=bool(payload.get("force")))

    def _hub_source(payload: dict) -> dict:
        if payload.get("model"):
            return {"model": str(payload["model"]),
                    "version": payload.get("version") or "latest"}
        return {"run_id": str(payload.get("run_id") or "")}

    @app.post("/api/hub/push", dependencies=guard)
    async def hub_push(payload: dict = Body(...)) -> dict:  # noqa: B008
        return await asyncio.to_thread(lambda: api.push_to_hub(
            str(payload.get("repo_id") or ""), private=payload.get("private", True) is not False,
            **_hub_source(payload)))

    @app.post("/api/hub/pull", dependencies=guard)
    async def hub_pull(payload: dict = Body(...)) -> dict:  # noqa: B008
        return await asyncio.to_thread(api.pull_from_hub, str(payload.get("repo_id") or ""),
                                       str(payload.get("name") or ""))

    @app.post("/api/hub/space", dependencies=guard)
    async def hub_space(payload: dict = Body(...)) -> dict:  # noqa: B008
        return await asyncio.to_thread(lambda: api.deploy_to_space(
            str(payload.get("space_id") or ""), formats=list(payload.get("formats") or []),
            private=payload.get("private", True) is not False, **_hub_source(payload)))

    @app.post("/api/kaggle/submit", dependencies=guard)
    async def kaggle_submit(payload: dict = Body(...)) -> dict:  # noqa: B008
        return await asyncio.to_thread(lambda: api.submit_to_kaggle(
            str(payload.get("competition") or ""), str(payload.get("file") or ""),
            str(payload.get("message") or ""), confirm=payload.get("confirm") is True))

    @app.delete("/api/models/{name}/{version}", dependencies=guard)
    def delete_model(name: str, version: int) -> dict:
        return api.delete_model(name, version)

    @app.get("/api/models/{name}/{version}/deploy", dependencies=guard)
    def deploy_model(name: str, version: str, formats: str = "") -> FileResponse:
        return _package_response(lambda out: api.deploy_model(name, version, out,
                                                              _split(formats)))

    # -------------------------------------------------------------- import
    @app.post("/api/import", dependencies=guard)
    async def import_model(kind: str = Form(...), file: UploadFile = File(...),  # noqa: B008
                           attr: str = Form(""), input_shape: str = Form(""),
                           dtype: str = Form("float32"), kwargs: str = Form("{}")) -> dict:
        work = Path(tempfile.mkdtemp(prefix="aime_upload_"))
        try:
            target = work / Path(file.filename or "model").name
            await _save_upload(file, target)
            shape = [int(d) for d in re.split(r"[,x\s]+", input_shape.strip("()[] ")) if d] or None
            return await asyncio.to_thread(
                api.import_model, kind, str(target), attr, shape, dtype,
                json.loads(kwargs or "{}"))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @app.post("/api/import/mlflow", dependencies=guard)
    async def import_mlflow(payload: dict = Body(...)) -> dict:  # noqa: B008
        raw = str(payload.get("input_shape") or "")
        shape = [int(d) for d in re.split(r"[,x\s]+", raw.strip("()[] ")) if d] or None
        return await asyncio.to_thread(api.open_mlflow_model, str(payload.get("uri") or ""),
                                       shape)

    # ---------------------------------------------------------------- data
    @app.post("/api/data/profile", dependencies=guard)
    def data_profile(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.profile_data(_graph_body(payload), payload.get("node_id") or None)

    @app.post("/api/data/issues", dependencies=guard)
    def data_issues(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.data_issues(_graph_body(payload))

    @app.post("/api/data/split", dependencies=guard)
    def data_split(payload: dict = Body(...)) -> dict:  # noqa: B008
        return api.split_preview(_graph_body(payload))

    @app.post("/api/data/augment", dependencies=guard)
    def data_augment(payload: dict = Body(...)) -> dict:  # noqa: B008
        work = tempfile.mkdtemp(prefix="aime_augment_")
        try:
            result: dict[str, Any] = api.augmentation_preview(
                _graph_body(payload), work, images=int(payload.get("images", 4)),
                variants=int(payload.get("variants", 6)))
            for image in result["images"]:  # inline the PNGs: the folder is temporary
                image["original"] = _data_url(image["original"])
                image["eval"] = _data_url(image["eval"])
                image["train"] = [_data_url(p) for p in image["train"]]
                image["source"] = Path(image["source"]).parent.name + "/" + \
                    Path(image["source"]).name
            return result
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------ frontend
    @app.get("/api/{path:path}", include_in_schema=False)
    def api_404(path: str) -> None:
        raise HTTPException(404, f"no endpoint /api/{path}")

    index = STATIC / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):  # noqa: ANN202
        if index.exists():
            candidate = (STATIC / path).resolve()
            if path and candidate.is_file() and STATIC.resolve() in candidate.parents:
                # file names under assets/ carry a content hash: cache them for good
                immutable = candidate.parent.name == "assets"
                return FileResponse(candidate, headers={"Cache-Control": (
                    "public, max-age=31536000, immutable" if immutable else "no-cache")})
            # never cache the page itself, or an upgrade leaves it pointing at old assets
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>AI Made Easy API</h1><p>The web UI is not built. Run "
                            "<code>npm run build</code> in <code>web/</code>, or use the API "
                            "at <a href='/api/docs'>/api/docs</a>.</p>")

    return app


_TOKEN_Q = re.compile(r"([?&]token=)[^&\s\"]*")


class RedactToken(logging.Filter):
    """Keep ``?token=`` (download links, WebSockets) out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_TOKEN_Q.sub(r"\1***", a) if isinstance(a, str) else a
                                for a in record.args)
        record.msg = _TOKEN_Q.sub(r"\1***", str(record.msg))
        return True


def main(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False,
         auth: bool = False) -> None:
    import uvicorn

    for name in ("uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).addFilter(RedactToken())

    if auth:
        store = UserStore(aime_home() / "users.json")
        if store.count() == 0:    # first start: an admin to sign in with, printed once
            password = secrets.token_urlsafe(12)
            store.add("admin", password, "admin")
            print("accounts are on: created the first admin — sign in as "
                  f"'admin' with '{password}' (shown only here; "
                  "`aime users passwd admin` changes it)")
    elif host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("AIME_WEB_TOKEN"):
        print("warning: serving on a public interface without accounts (--auth) or "
              "AIME_WEB_TOKEN — anyone who can reach this port can train models and "
              "read your files")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(create_app(auth=auth), host=host, port=port, log_level="info")
