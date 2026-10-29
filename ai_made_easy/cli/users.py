"""``aime users`` — manage the web app's accounts ($AIME_HOME/users.json).

    aime users list
    aime users add NAME [--role member|admin]     # asks for the password twice
    aime users passwd NAME
    aime users role NAME ROLE
    aime users enable NAME | disable NAME
    aime users remove NAME

Passwords are read with getpass (never an argument), hashed with PBKDF2. The
last enabled admin cannot be demoted, disabled or removed.
"""
from __future__ import annotations

import getpass
import json
import sys

from ai_made_easy.core.paths import aime_home
from ai_made_easy.server.users import UserError, UserStore


def _store() -> UserStore:
    return UserStore(aime_home() / "users.json")


def _password(prompt: str = "Password: ") -> str:
    if sys.stdin.isatty():
        first = getpass.getpass(prompt)
        second = getpass.getpass("Again: ")
    else:                       # piped (scripts, tests): two lines, no echo to hide
        first = sys.stdin.readline().rstrip("\n")
        second = sys.stdin.readline().rstrip("\n")
    if not first or first != second:
        raise ValueError("the passwords are empty or do not match")
    return first


def users_command(args) -> int:  # noqa: ANN001 — argparse Namespace
    store = _store()
    try:
        if args.users_cmd == "list":
            users = store.list()
            print(json.dumps(users, indent=2))
            print(f"{len(users)} user(s)")
        elif args.users_cmd == "add":
            password = _password()
            user = store.add(args.name, password, args.role)
            print(f"added {user.name!r} ({user.role})")
        elif args.users_cmd == "passwd":
            store.set_password(args.name, _password())
            print(f"new password for {args.name!r}")
        elif args.users_cmd == "role":
            store.set_role(args.name, args.role)
            print(f"{args.name!r} is now a {args.role}")
        elif args.users_cmd == "disable":
            store.set_disabled(args.name, True)
            print(f"disabled {args.name!r}")
        elif args.users_cmd == "enable":
            store.set_disabled(args.name, False)
            print(f"enabled {args.name!r}")
        elif args.users_cmd == "remove":
            store.remove(args.name)
            print(f"removed {args.name!r}")
        else:  # pragma: no cover — argparse chooses one of the above
            print(f"unknown users command {args.users_cmd!r}", file=sys.stderr)
            return 2
    except (UserError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
