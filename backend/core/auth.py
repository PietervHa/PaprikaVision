"""
Password Hashing

Passwords are hashed with hashlib.scrypt -- stdlib, no extra dependency.
Each stored hash is self-describing (it embeds the cost parameters and the
salt it was created with), so the scrypt cost can be raised later without
invalidating hashes that already exist in the `users` table.

Stored format:  scrypt$n$r$p$salt_hex$hash_hex
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Optional, Sequence

_ALGORITHM = "scrypt"
_N = 2 ** 14  # CPU/memory cost factor
_R = 8        # block size
_P = 1        # parallelization factor
_KEY_LEN = 32
_SALT_LEN = 16


def hash_password(password: str) -> str:
    """Hash a plaintext password into a self-describing string, ready to store in `users.password_hash`."""
    salt = secrets.token_bytes(_SALT_LEN)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_KEY_LEN
    )
    return f"{_ALGORITHM}${_N}${_R}${_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Check a plaintext password against a hash produced by hash_password().
    Returns False on any malformed/unrecognized hash rather than raising,
    so a corrupt DB row fails closed instead of crashing the login route.
    """
    try:
        algorithm, n, r, p, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != _ALGORITHM:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False

    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
    )
    return hmac.compare_digest(derived, expected)


# A hash of a password nobody will ever type, computed once at import time.
# The login route runs verify_password() against this whenever the
# submitted username doesn't exist, so an unknown-username attempt takes
# the same amount of time as a wrong-password attempt for a real user,
# otherwise response timing alone would leak which usernames are registered.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


# --- Login lockout -----------------------------------------------------------
#
# Deliberately not a mutable "locked" flag anywhere, lockout state is derived
# on every request from the account's own failure history in login_log, so
# there's nothing to get out of sync and nothing extra to migrate. Only counts
# failures for usernames that exist: an unknown username can never lock a real
# account out, by construction (see get_recent_failures_since_last_success()).

MAX_FAILED_ATTEMPTS = 3      # consecutive failures (since the last success) that trigger a lockout
FAILURE_WINDOW = timedelta(minutes=30)   # those failures must fall within this span of each other
LOCKOUT_DURATION = timedelta(minutes=10)  # how long the account stays locked once triggered


def lockout_until(recent_failure_timestamps: Sequence[datetime], now: Optional[datetime] = None) -> Optional[datetime]:
    """
    Given a user's failed-login timestamps since their last successful login
    (newest first, at most MAX_FAILED_ATTEMPTS of them), return the datetime
    the account is locked out until, or None if it isn't currently locked.

    Locks only when MAX_FAILED_ATTEMPTS consecutive failures all fall within
    FAILURE_WINDOW of each other, a handful of mistyped passwords spread
    across a shift doesn't trigger it, a burst of guesses does. Each further
    attempt made while locked extends the lock, since it becomes the new
    "newest" failure.
    """
    if now is None:
        now = datetime.now()
    if len(recent_failure_timestamps) < MAX_FAILED_ATTEMPTS:
        return None

    newest = recent_failure_timestamps[0]
    oldest_of_streak = recent_failure_timestamps[MAX_FAILED_ATTEMPTS - 1]
    if newest - oldest_of_streak > FAILURE_WINDOW:
        return None  # spread out over time, not a rapid-fire attempt

    unlock_at = newest + LOCKOUT_DURATION
    if now >= unlock_at:
        return None
    return unlock_at
