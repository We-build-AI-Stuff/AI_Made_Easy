"""Web user accounts: the store, signed sessions, login, roles and per-user projects.

Everything runs against ``create_app(auth=True, ...)`` with an isolated
``$AIME_HOME``, so nothing here touches a real ``~/.aime``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conftest import tiny_classifier_dict  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ai_made_easy.core import api  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402
from ai_made_easy.server.app import auth_enabled, create_app  # noqa: E402
from ai_made_easy.server.users import (  # noqa: E402
    LoginThrottle,
    User,
    UserError,
    UserStore,
    hash_password,
    issue_token,
    load_secret,
    read_token,
    verify_password,
)


@pytest.fixture()
def client(isolated_home, tmp_path):
    api.set_manager(RunManager(RunHistory()))
    app = create_app(auth=True, projects_dir=tmp_path / "projects")
    with TestClient(app) as c:
        yield c
    api.manager().stop_all()
    api.set_manager(None)


def _login(c, name: str, password: str) -> str:
    response = c.post("/api/auth/login", json={"name": name, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return token


def _add(store_path, name: str, password: str = "pw-long-enough", role: str = "member") -> None:
    UserStore(store_path).add(name, password, role)


# ==================================================================== the store

def test_store_hash_roundtrip_and_rules(isolated_home):
    path = isolated_home / "users.json"
    store = UserStore(path)
    store.add("Ada", "hunter2!", "admin")
    assert store.verify("Ada", "hunter2!") == User("Ada", "admin")
    assert store.verify("Ada", "wrong") is None
    assert store.verify("nobody", "x") is None
    assert verify_password("hunter2!", hash_password("hunter2!"))
    assert not verify_password("hunter2!", hash_password("hunter3!"))
    with pytest.raises(UserError):
        store.add("Ada", "again")            # duplicate
    with pytest.raises(UserError):
        store.add("bad name\nx", "pw")       # invalid characters
    with pytest.raises(UserError):
        store.add("ok", "short", "boss")     # unknown role
    # a fresh store over the same file sees the same accounts (restart)
    assert [u["name"] for u in UserStore(path).list()] == ["Ada"]


def test_last_admin_is_protected(isolated_home):
    store = UserStore(isolated_home / "users.json")
    store.add("root", "pw-root", "admin")
    for action in (lambda: store.set_role("root", "member"),
                   lambda: store.set_disabled("root", True),
                   lambda: store.remove("root")):
        with pytest.raises(UserError):
            action()
    store.add("helper", "pw-helper", "admin")
    store.set_disabled("root", True)         # fine: helper is still an enabled admin
    with pytest.raises(UserError):
        store.set_disabled("helper", True)   # would leave no enabled admin at all
    store.set_role("root", "member")         # fine: helper is the enabled admin


def test_tokens_are_signed_and_expire(isolated_home):
    secret = load_secret(isolated_home / "key")
    assert load_secret(isolated_home / "key") == secret       # stable across restarts
    token = issue_token(secret, User("ada", "member"), days=1)
    assert read_token(secret, token) == User("ada", "member")
    assert read_token(bytes(32), token) is None                 # signed with another key
    expired = issue_token(secret, User("ada", "member"), days=-1)
    assert read_token(secret, expired) is None
    assert read_token(secret, "v1.!!!.***") is None


# ==================================================================== logging in

def test_anonymous_requests_are_rejected(client):
    assert client.get("/api/blocks").status_code == 401
    assert client.get("/api/auth/status").json()["mode"] == "accounts"      # public
    assert client.get("/api/auth/status").json()["user"] is None
    assert client.post("/api/auth/login", json={}).status_code == 400


def test_login_and_role_endpoints(client, isolated_home):
    _add(isolated_home / "users.json", "ada", "pw-ada")
    _login(client, "ada", "pw-ada")
    assert client.get("/api/auth/me").json() == {"name": "ada", "role": "member"}
    assert client.get("/api/auth/status").json()["user"] == {"name": "ada", "role": "member"}
    assert client.get("/api/blocks").status_code == 200
    # a member may not manage users
    assert client.get("/api/admin/users").status_code == 403


def test_session_opens_the_run_websocket(client, isolated_home):
    from starlette.websockets import WebSocketDisconnect

    _add(isolated_home / "users.json", "ada", "pw-ada")
    token = _login(client, "ada", "pw-ada")
    with pytest.raises(WebSocketDisconnect):        # no token: rejected before the run is read
        with client.websocket_connect("/api/runs/whatever/events") as ws:
            ws.receive_json()
    with pytest.raises(WebSocketDisconnect):        # the session token itself is accepted,
        with client.websocket_connect(              # and the missing run closes the socket
                f"/api/runs/whatever/events?token={token}") as ws:
            ws.receive_json()


def test_bad_password_and_throttle(client, isolated_home):
    _add(isolated_home / "users.json", "ada", "pw-ada")
    for _ in range(5):
        assert client.post("/api/auth/login",
                           json={"name": "ada", "password": "nope"}).status_code == 401
    assert client.post("/api/auth/login",
                       json={"name": "ada", "password": "pw-ada"}).status_code == 429


def test_disabled_account_cannot_sign_in_or_stay_signed_in(client, isolated_home):
    path = isolated_home / "users.json"
    _add(path, "root", "pw-root", "admin")
    _add(path, "ada", "pw-ada")           # a member to disable
    _login(client, "ada", "pw-ada")
    assert client.get("/api/projects").status_code == 200
    UserStore(path).set_disabled("ada", True)
    assert client.post("/api/auth/login",
                       json={"name": "ada", "password": "pw-ada"}).status_code == 401
    assert client.get("/api/projects").status_code == 401   # the old session dies with it


# ==================================================================== user management

def test_admin_manages_users(client, isolated_home):
    _add(isolated_home / "users.json", "root", "pw-root", "admin")
    _login(client, "root", "pw-root")
    users = client.get("/api/admin/users").json()["users"]
    assert [u["name"] for u in users] == ["root"]
    created = client.post("/api/admin/users",
                          json={"name": "ada", "password": "pw-ada", "role": "member"})
    assert created.status_code == 200 and created.json()["role"] == "member"
    assert client.post("/api/admin/users",
                       json={"name": "x", "password": "pw"}).status_code == 200  # default role
    assert client.post("/api/admin/users", json={"name": "", "password": "pw"}).status_code == 400
    # change password and role
    assert client.patch("/api/admin/users/ada", json={"password": "pw-new"}).status_code == 200
    assert client.patch("/api/admin/users/ada", json={"role": "admin"}).status_code == 200
    assert client.post("/api/auth/login",
                       json={"name": "ada", "password": "pw-new"}).json()["user"]["role"] == "admin"
    # the last enabled admin (root is, ada now too — demote ada back) is protected
    assert client.patch("/api/admin/users/ada", json={"role": "member"}).status_code == 200
    assert client.patch("/api/admin/users/root", json={"disabled": True}).status_code == 400
    assert client.delete("/api/admin/users/root").status_code == 400
    # but an admin cannot remove or disable themselves even with others around
    _add(isolated_home / "users.json", "bee", "pw-bee", "admin")
    assert client.patch("/api/admin/users/root", json={"disabled": True}).status_code == 400
    assert client.delete("/api/admin/users/bee").status_code == 200


def test_shared_token_still_works_as_admin(client, isolated_home, tmp_path):
    _add(isolated_home / "users.json", "root", "pw-root", "admin")
    api.set_manager(RunManager(RunHistory()))
    app = create_app(auth=True, token="s3cret", projects_dir=tmp_path / "projects")
    with TestClient(app) as c:
        c.headers.update({"Authorization": "Bearer s3cret"})
        assert c.get("/api/blocks").status_code == 200
        assert c.get("/api/admin/users").status_code == 200
        assert c.put("/api/projects/any", json=tiny_classifier_dict()).status_code == 200
    api.manager().stop_all()


# ==================================================================== projects per user

def test_members_see_only_their_projects(client, isolated_home):
    path = isolated_home / "users.json"
    _add(path, "root", "pw-root", "admin")
    _add(path, "ada", "pw-ada")
    _add(path, "mal", "pw-mal")

    _login(client, "ada", "pw-ada")
    assert client.put("/api/projects/adas", json=tiny_classifier_dict()).status_code == 200
    assert [p["name"] for p in client.get("/api/projects").json()["projects"]] == ["adas"]
    assert client.get("/api/projects/adas").status_code == 200
    assert client.get("/api/projects/adas").json()["name"] == "adas"

    _login(client, "mal", "pw-mal")
    assert client.get("/api/projects").json()["projects"] == []          # hidden entirely
    assert client.get("/api/projects/adas").status_code == 404           # not theirs to know of
    assert client.delete("/api/projects/adas").status_code == 403
    assert client.put("/api/projects/adas", json=tiny_classifier_dict()).status_code == 403

    _login(client, "root", "pw-root")
    names = [p["name"] for p in client.get("/api/projects").json()["projects"]]
    assert names == ["adas"] and client.get("/api/projects").json()["projects"][0]["owner"] == "ada"
    assert client.put("/api/projects/adas", json=tiny_classifier_dict()).status_code == 200


def test_projects_without_an_owner_stay_shared(client, isolated_home, tmp_path):
    path = isolated_home / "users.json"
    _add(path, "root", "pw-root", "admin")
    _add(path, "ada", "pw-ada")
    projects = tmp_path / "projects"
    projects.mkdir(parents=True, exist_ok=True)   # a project from before accounts
    (projects / "legacy.json").write_text('{"schema_version": 3, "nodes": [], "edges": []}')
    _login(client, "ada", "pw-ada")
    listed = client.get("/api/projects").json()["projects"]
    assert [p["name"] for p in listed] == ["legacy"] and listed[0]["owner"] is None
    assert client.get("/api/projects/legacy").status_code == 200


def test_auth_off_changes_nothing(isolated_home, tmp_path):
    api.set_manager(RunManager(RunHistory()))
    app = create_app(auth=False, projects_dir=tmp_path / "projects")
    with TestClient(app) as c:
        assert c.get("/api/auth/status").json() == {"mode": "none",
                                                    "user": {"name": "", "role": "admin"}}
        assert c.post("/api/auth/login",
                      json={"name": "x", "password": "y"}).status_code == 400
        assert c.get("/api/blocks").status_code == 200
        assert c.put("/api/projects/open", json=tiny_classifier_dict()).status_code == 200
        assert [p["name"] for p in c.get("/api/projects").json()["projects"]] == ["open"]
    api.manager().stop_all()


# ==================================================== CodeRabbit review round

def test_demotion_takes_effect_at_once(client, isolated_home):
    """CR #1: the role comes from the store, not from the session token."""
    path = isolated_home / "users.json"
    _add(path, "root", "pw-root", "admin")
    _add(path, "bee", "pw-bee", "admin")
    _login(client, "bee", "pw-bee")
    assert client.get("/api/admin/users").status_code == 200        # still an admin
    UserStore(path).set_role("bee", "member")
    assert client.get("/api/auth/me").json()["role"] == "member"    # at once, same token
    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/api/projects").status_code == 200           # still signed in


def test_admin_save_does_not_steal_ownership(client, isolated_home):
    """CR #2: an admin (or the owner) saving a project leaves its owner alone."""
    path = isolated_home / "users.json"
    _add(path, "root", "pw-root", "admin")
    _add(path, "ada", "pw-ada")
    _login(client, "ada", "pw-ada")
    client.put("/api/projects/adas", json=tiny_classifier_dict())
    _login(client, "root", "pw-root")
    client.put("/api/projects/adas", json=tiny_classifier_dict())   # admin saves it
    owners = client.get("/api/projects").json()["projects"]
    assert next(p for p in owners if p["name"] == "adas")["owner"] == "ada"
    # and the member still sees and saves their project afterwards
    _login(client, "ada", "pw-ada")
    assert client.get("/api/projects/adas").status_code == 200
    assert client.put("/api/projects/adas", json=tiny_classifier_dict()).status_code == 200


def test_web_auth_env_var_is_honoured(isolated_home, monkeypatch):
    """CR #3: `aime web` without --auth must still see AIME_WEB_AUTH=1."""
    assert auth_enabled(False) is False           # an explicit flag always wins
    assert auth_enabled(True) is True
    monkeypatch.setenv("AIME_WEB_AUTH", "1")
    assert auth_enabled(None) is True             # what `aime web` resolves
    monkeypatch.setenv("AIME_WEB_AUTH", "nope")
    assert auth_enabled(None) is False


def test_the_name_token_is_reserved(isolated_home):
    """CR #4: 'token' is the shared key's identity; nobody may be called that."""
    store = UserStore(isolated_home / "users.json")
    with pytest.raises(UserError):
        store.add("token", "pw-token")
    with pytest.raises(UserError):
        store.add("Token", "pw-token")


def test_credential_files_are_owner_only(isolated_home):
    """CR #6: users.json and web_secret.key are created 0600 where the os allows."""
    import sys

    store = UserStore(isolated_home / "users.json")
    store.add("ada", "pw-ada")
    load_secret(isolated_home / "web_secret.key")
    for name in ("users.json", "web_secret.key"):
        mode = (isolated_home / name).stat().st_mode
        if sys.platform != "win32":          # Windows ignores the mode bits
            assert mode & 0o777 == 0o600, name


def test_throttle_memory_is_bounded(isolated_home):
    """CR #7: random names cannot grow the throttle without bound."""
    throttle = LoginThrottle(max_names=8)
    for i in range(50):
        throttle.record_failure(f"name-{i}")
    assert len(throttle._fails) <= 8          # noqa: SLF001 — the cap is the contract
    for _ in range(5):                        # the newest name is still tracked, and
        throttle.record_failure("name-49")    # still locks out after enough failures
    assert throttle.blocked("name-49") > 0
