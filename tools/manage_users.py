"""
User management CLI for maintenance-mode login.

There is no web-facing registration route on purpose -- accounts are
created, listed, and removed only through this script, which needs shell
access to the machine the app runs on. Passwords are always entered
interactively via getpass; they are never accepted as a command-line
argument, so they never end up in shell history or a process listing.

Usage:
    python -m QC_tools.manage_users add <username>
    python -m QC_tools.manage_users passwd <username>
    python -m QC_tools.manage_users list
    python -m QC_tools.manage_users remove <username>
    python -m QC_tools.manage_users logins [--limit N]
"""
from __future__ import annotations

import argparse
import getpass
import sys

from backend.core import auth, db

_MIN_PASSWORD_LEN = 8


def _prompt_new_password(prompt: str = "Password: ") -> str:
    """Prompt (with confirmation) for a new password, re-asking on mismatch or if too short."""
    while True:
        pw1 = getpass.getpass(prompt)
        if len(pw1) < _MIN_PASSWORD_LEN:
            print(f"Password must be at least {_MIN_PASSWORD_LEN} characters.", file=sys.stderr)
            continue
        pw2 = getpass.getpass("Confirm password: ")
        if pw1 != pw2:
            print("Passwords did not match, try again.", file=sys.stderr)
            continue
        return pw1


def cmd_add(args: argparse.Namespace) -> int:
    username = args.username.strip()
    if not username:
        print("Username cannot be empty.", file=sys.stderr)
        return 1
    if db.get_user(username):
        print(f"User '{username}' already exists.", file=sys.stderr)
        return 1

    password = _prompt_new_password()
    db.create_user(username, auth.hash_password(password))
    print(f"User '{username}' created.")
    return 0


def cmd_passwd(args: argparse.Namespace) -> int:
    username = args.username.strip()
    if not db.get_user(username):
        print(f"No such user: '{username}'", file=sys.stderr)
        return 1

    password = _prompt_new_password(f"New password for {username}: ")
    db.set_password(username, auth.hash_password(password))
    print(f"Password updated for '{username}'.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    users = db.list_users()
    if not users:
        print("No users.")
        return 0
    for u in users:
        print(f"{u['username']:<24} created {u['created_at']}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    username = args.username.strip()
    if not db.get_user(username):
        print(f"No such user: '{username}'", file=sys.stderr)
        return 1

    confirm = input(f"Type '{username}' again to confirm removal: ")
    if confirm != username:
        print("Confirmation did not match, aborted.", file=sys.stderr)
        return 1

    db.delete_user(username)
    print(f"User '{username}' removed.")
    return 0


def cmd_logins(args: argparse.Namespace) -> int:
    logins = db.get_recent_logins(limit=args.limit)
    if not logins:
        print("No login attempts recorded.")
        return 0
    for entry in logins:
        result = "OK    " if entry["success"] else "FAILED"
        ip = entry.get("ip_address") or "-"
        print(f"{entry['timestamp']}  {result}  {entry['username']:<24} {ip}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage maintenance-mode login users.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Create a new user")
    p_add.add_argument("username")
    p_add.set_defaults(func=cmd_add)

    p_passwd = sub.add_parser("passwd", help="Change a user's password")
    p_passwd.add_argument("username")
    p_passwd.set_defaults(func=cmd_passwd)

    p_list = sub.add_parser("list", help="List all users")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Delete a user")
    p_remove.add_argument("username")
    p_remove.set_defaults(func=cmd_remove)

    p_logins = sub.add_parser("logins", help="Show recent login attempts")
    p_logins.add_argument("--limit", type=int, default=20)
    p_logins.set_defaults(func=cmd_logins)

    return parser


def main() -> None:
    db.init_db()
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
