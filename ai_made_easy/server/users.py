"""Web user accounts: a JSON-backed store with PBKDF2 password hashes and signed
session tokens — no external dependencies, and no passwords in code or logs.

Enable with ``aime web --auth`` (or ``AIME_WEB_AUTH=1``). Accounts live in
``$AIME_HOME/users.json``; the token-signing secret in ``$AIME_HOME/web_secret.key``
(created once, so sessions survive a restart). Roles are ``admin`` (manages users,
sees every project) and ``member`` (works on their own projects and ones without
an owner). ``AIME_WEB_TOKEN`` keeps working next to accounts as a shared admin
token, so scripts and CI need no login.

Sessions are stateless: a token is ``v1.<payload>.<hmac>`` where the payload names
the user, their role and an expiry, and the HMAC comes from the secret — logout
drops the token on the client; a stolen one works until it expires (14 days).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ITERATIONS = 240_000            # OWASP 2023 recommendation for PBKDF2-SHA256
TOKEN_DAYS = 14
NAME = re.compile(r"^[\w][\w .-]{0,31}$")
ROLES = ("admin", "member")


class UserError(ValueError):
    """A user-management mistake the message of which is shown to the user."""


@dataclass(frozen=True)
class User:
    name: str
    role: str
    disabled: bool = False


def hash_password(password: str, *, iterations: int = ITERATIONS) -> str:
    if not password or len(password) > 1024:
        raise UserError("password must be 1–1024 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


_DUMMY_HASH: list[str] = []    # computed on first use: import must stay cheap


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt),
                                  int(iterations))
        return hmac.compare_digest(got.hex(), digest)
    except (ValueError, TypeError):
        return False


class UserStore:
    """Accounts on disk (``users.json``), one writer at a time.

    The file is re-read whenever it changed on disk, so ``aime users add`` while
    ``aime web`` is running is picked up without a restart.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._users: dict[str, dict] = {}
        self._mtime = -1.0
        self._load()

    # ------------------------------------------------------------------ disk
    def _load(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            with self._lock:
                self._users, self._mtime = {}, -1.0
            return
        with self._lock:
            if mtime == self._mtime:
                return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = {}          # a corrupt file starts fresh rather than blocking login
        with self._lock:
            self._users = {n: u for n, u in data.items() if isinstance(u, dict)}
            self._mtime = mtime

    def _reload_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.stem}.{secrets.token_hex(4)}.tmp")
        _write_private(tmp, json.dumps(self._users, indent=1))
        tmp.replace(self.path)
        try:
            self._mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime = -1.0

    # ------------------------------------------------------------- accounts
    def _record(self, name: str) -> dict:
        user = self._users.get(name)
        if user is None:
            raise UserError(f"no user named {name!r}")
        return user

    def add(self, name: str, password: str, role: str = "member") -> User:
        self._reload_if_changed()
        name = name.strip()
        if not NAME.match(name) or name.lower() == "token":
            raise UserError("user names may use letters, digits, spaces, '_', '-' and '.' "
                            "(32 characters at most) — and not 'token', which is the "
                            "shared access key")
        if role not in ROLES:
            raise UserError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            if name in self._users:
                raise UserError(f"user {name!r} already exists")
            self._users[name] = {"role": role, "disabled": False,
                                 "created": time.time(), "hash": hash_password(password)}
            self._save()
        return User(name, role)

    def verify(self, name: str, password: str) -> User | None:
        """The user when the name exists, the password matches and it is enabled."""
        self._reload_if_changed()
        with self._lock:
            record = self._users.get(name.strip())
        if not record or record.get("disabled"):
            # hash anyway, so a missing or disabled name takes as long as a wrong
            # password: login timing must not reveal which names exist
            if not _DUMMY_HASH:
                _DUMMY_HASH.append(hash_password("timing-equalizer"))
            verify_password(password or "", record.get("hash", "") if record else _DUMMY_HASH[0])
            return None
        if not verify_password(password or "", record.get("hash", "")):
            return None
        return User(name.strip(), record["role"])

    def list(self) -> list[dict]:
        self._reload_if_changed()
        with self._lock:
            return [{"name": n, "role": u.get("role", "member"),
                     "disabled": bool(u.get("disabled")),
                     "created": u.get("created")} for n, u in sorted(self._users.items())]

    def count(self) -> int:
        self._reload_if_changed()
        with self._lock:
            return len(self._users)

    def get(self, name: str) -> User | None:
        """The user when they exist and are enabled (for re-checking a session)."""
        self._reload_if_changed()
        with self._lock:
            record = self._users.get(name.strip())
        if not record or record.get("disabled"):
            return None
        return User(name.strip(), record["role"])

    def enabled_admins(self) -> int:
        with self._lock:
            return sum(1 for u in self._users.values()
                       if u.get("role") == "admin" and not u.get("disabled"))

    def set_password(self, name: str, password: str) -> None:
        self._reload_if_changed()
        with self._lock:
            self._record(name)["hash"] = hash_password(password)
            self._save()

    def set_role(self, name: str, role: str) -> None:
        self._reload_if_changed()
        if role not in ROLES:
            raise UserError(f"role must be one of {', '.join(ROLES)}")
        with self._lock:
            record = self._record(name)
            demoting = (record["role"] == "admin" and role != "admin"
                        and not record.get("disabled"))   # a disabled admin counts for none
            if demoting and self._enabled_admins_locked() <= 1:
                raise UserError("cannot demote the only enabled admin")
            record["role"] = role
            self._save()

    def set_disabled(self, name: str, disabled: bool) -> None:
        self._reload_if_changed()
        with self._lock:
            record = self._record(name)
            if disabled and record["role"] == "admin" and self._enabled_admins_locked() <= 1:
                raise UserError("cannot disable the only enabled admin")
            record["disabled"] = disabled
            self._save()

    def remove(self, name: str) -> None:
        self._reload_if_changed()
        with self._lock:
            record = self._record(name)
            enabled_admin = record["role"] == "admin" and not record.get("disabled")
            if enabled_admin and self._enabled_admins_locked() <= 1:
                raise UserError("cannot remove the only enabled admin")
            del self._users[name]
            self._save()

    def _enabled_admins_locked(self) -> int:
        return sum(1 for u in self._users.values()
                   if u.get("role") == "admin" and not u.get("disabled"))


# ------------------------------------------------------------------- tokens

def _b64(raw: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _write_private(path: Path, text: str) -> None:
    """Create ``path`` owner-only (0600): credential files must not be world-readable."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def load_secret(path: str | Path) -> bytes:
    """The signing key at ``path``, created (once) when missing."""
    path = Path(path)
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    tmp = path.with_name(f"{path.stem}.{secrets.token_hex(4)}.tmp")
    _write_private(tmp, secret.hex())
    tmp.replace(path)
    return secret


def issue_token(secret: bytes, user: User, *, days: int = TOKEN_DAYS) -> str:
    payload = _b64(json.dumps({"u": user.name, "r": user.role,
                               "exp": time.time() + days * 86400}).encode())
    signature = _b64(hmac.new(secret, payload.encode(), hashlib.sha256).digest())
    return f"v1.{payload}.{signature}"


def read_token(secret: bytes, token: str) -> User | None:
    """The user a session token names, or None when it is bad or expired."""
    try:
        version, payload, signature = token.split(".")
        if version != "v1":
            return None
        want = hmac.new(secret, payload.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(want, _unb64(signature)):
            return None
        claims = json.loads(_unb64(payload))
        if float(claims["exp"]) < time.time():
            return None
        return User(str(claims["u"]), str(claims["r"]))
    except (ValueError, KeyError, TypeError, ArithmeticError):
        return None


# ------------------------------------------------------------- rate limiting

class LoginThrottle:
    """Locks a name out for ``lockout`` seconds after ``max_failures`` bad passwords."""

    def __init__(self, max_failures: int = 5, lockout: float = 60.0,
                 max_names: int = 512) -> None:
        self.max_failures = max_failures
        self.lockout = lockout
        self.max_names = max_names     # an attacker posting random names must not
        self._fails: dict[str, list[float]] = {}   # grow this without bound
        self._lock = threading.Lock()

    def _prune_all_locked(self, now: float) -> None:
        for name in list(self._fails):
            recent = [t for t in self._fails[name] if now - t < self.lockout]
            if recent:
                self._fails[name] = recent
            else:
                del self._fails[name]
        while len(self._fails) >= self.max_names:    # drop the least recently used
            oldest = min(self._fails, key=lambda n: self._fails[n][0])
            del self._fails[oldest]

    def _prune(self, name: str, now: float) -> list[float]:
        recent = [t for t in self._fails.get(name, []) if now - t < self.lockout]
        if recent:
            self._fails[name] = recent
        else:
            self._fails.pop(name, None)
        return recent

    def blocked(self, name: str) -> float:
        """Seconds left in the lockout, 0 when the name may try again."""
        with self._lock:
            recent = self._prune(name, time.time())
            if len(recent) >= self.max_failures:
                return self.lockout - (time.time() - recent[0])
        return 0.0

    def record_failure(self, name: str) -> None:
        with self._lock:
            now = time.time()
            self._prune_all_locked(now)
            recent = [t for t in self._fails.get(name, []) if now - t < self.lockout]
            recent.append(now)
            self._fails[name] = recent

    def record_success(self, name: str) -> None:
        with self._lock:
            self._fails.pop(name, None)
