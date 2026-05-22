"""Authentication helpers for Streamlit frontend."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, MutableMapping

AUTH_DB_PATH = Path("database/data/auth/users.db")
PASSWORD_HASH_ITERATIONS = 200_000
SESSION_IDLE_TIMEOUT = timedelta(minutes=30)
SESSION_MAX_AGE = timedelta(hours=12)
LOGIN_LOCKOUT_THRESHOLD = 5
LOGIN_LOCKOUT_MINUTES = 15
RESET_TOKEN_TTL = timedelta(minutes=15)


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def get_db_connection() -> sqlite3.Connection:
    """Return SQLite connection with row access by column name."""
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_users_columns(conn: sqlite3.Connection) -> None:
    """Apply additive schema migrations for auth table."""
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    required_columns = {
        "reset_token_hash": "TEXT",
        "reset_token_expiry_utc": "TEXT",
        "reset_requested_at_utc": "TEXT",
        "shoulder_width_cm": "REAL",
        "chest_circumference_cm": "REAL",
        "torso_length_cm": "REAL",
        "height_cm": "REAL",
        "gender": "TEXT",
        "date_of_birth": "TEXT",
        "phone_number": "TEXT",
        "profile_picture_path": "TEXT",
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}")


def init_auth_db() -> None:
    """Initialize auth storage if missing."""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                lockout_until_utc TEXT,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        _ensure_users_columns(conn)
        conn.commit()


def hash_password(password: str, salt: bytes) -> bytes:
    """Derive password hash using PBKDF2."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Verify password hash using constant-time comparison."""
    salt = bytes.fromhex(salt_hex)
    expected_hash = bytes.fromhex(hash_hex)
    candidate_hash = hash_password(password, salt)
    return hmac.compare_digest(candidate_hash, expected_hash)


def validate_registration(username: str, email: str, password: str) -> list[str]:
    """Validate registration input and return errors."""
    errors: list[str] = []
    clean_username = username.strip()
    clean_email = email.strip()

    if len(clean_username) < 3 or len(clean_username) > 32:
        errors.append("Username must be between 3 and 32 characters.")
    if not re.fullmatch(r"[A-Za-z0-9_]+", clean_username):
        errors.append("Username can only contain letters, digits, and underscore.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", clean_email):
        errors.append("Please provide a valid email address.")

    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must include at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        errors.append("Password must include at least one lowercase letter.")
    if not re.search(r"\d", password):
        errors.append("Password must include at least one number.")
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append("Password must include at least one special character.")
    return errors


def validate_new_password(password: str) -> list[str]:
    """Validate password complexity rules."""
    return validate_registration("temp_user", "temp@example.com", password)[2:]


def create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """Create user account securely."""
    clean_username = username.strip()
    clean_email = email.strip().lower()

    errors = validate_registration(clean_username, clean_email, password)
    if errors:
        return False, " ".join(errors)

    salt = secrets.token_bytes(16)
    password_digest = hash_password(password, salt)

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_salt, password_hash, created_at_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    clean_username,
                    clean_email,
                    salt.hex(),
                    password_digest.hex(),
                    utc_now().isoformat(),
                ),
            )
            conn.commit()
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def authenticate_user(login_id: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Authenticate a user with lockout protection."""
    clean_login_id = login_id.strip().lower()
    generic_error = "Invalid credentials."

    if not clean_login_id or not password:
        return False, generic_error, None

    with get_db_connection() as conn:
        user = conn.execute(
            """
            SELECT id, username, email, password_salt, password_hash,
                   failed_login_attempts, lockout_until_utc,
                   shoulder_width_cm, chest_circumference_cm, torso_length_cm, height_cm,
                   gender, date_of_birth, phone_number, profile_picture_path
            FROM users
            WHERE lower(email) = ? OR lower(username) = ?
            """,
            (clean_login_id, clean_login_id),
        ).fetchone()

        if user is None:
            return False, generic_error, None

        if user["lockout_until_utc"]:
            lockout_until = datetime.fromisoformat(user["lockout_until_utc"])
            if lockout_until > utc_now():
                return False, "Too many failed attempts. Try again later.", None

        if not verify_password(password, user["password_salt"], user["password_hash"]):
            failed_attempts = int(user["failed_login_attempts"]) + 1
            lockout_value = None
            if failed_attempts >= LOGIN_LOCKOUT_THRESHOLD:
                lockout_value = (utc_now() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)).isoformat()
                failed_attempts = 0

            conn.execute(
                """
                UPDATE users
                SET failed_login_attempts = ?, lockout_until_utc = ?
                WHERE id = ?
                """,
                (failed_attempts, lockout_value, user["id"]),
            )
            conn.commit()
            return False, generic_error, None

        conn.execute(
            """
            UPDATE users
            SET failed_login_attempts = 0, lockout_until_utc = NULL
            WHERE id = ?
            """,
            (user["id"],),
        )
        conn.commit()

    return True, "Login successful.", {
        "id": user["id"], 
        "username": user["username"], 
        "email": user["email"],
        "shoulder_width_cm": user["shoulder_width_cm"],
        "chest_circumference_cm": user["chest_circumference_cm"],
        "torso_length_cm": user["torso_length_cm"],
        "height_cm": user["height_cm"],
        "gender": user["gender"],
        "date_of_birth": user["date_of_birth"],
        "phone_number": user["phone_number"],
        "profile_picture_path": user["profile_picture_path"],
    }


def request_password_reset(login_id: str) -> tuple[bool, str, str | None]:
    """
    Generate password reset token.

    Returns a generic success message in all valid request paths to reduce account
    enumeration. Token is returned for local-development delivery.
    """
    clean_login_id = login_id.strip().lower()
    generic_msg = "If the account exists, a reset token has been generated."
    if not clean_login_id:
        return False, "Please enter email or username.", None

    with get_db_connection() as conn:
        user = conn.execute(
            """
            SELECT id
            FROM users
            WHERE lower(email) = ? OR lower(username) = ?
            """,
            (clean_login_id, clean_login_id),
        ).fetchone()

        if user is None:
            return True, generic_msg, None

        token = secrets.token_urlsafe(24)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expiry = (utc_now() + RESET_TOKEN_TTL).isoformat()

        conn.execute(
            """
            UPDATE users
            SET reset_token_hash = ?, reset_token_expiry_utc = ?, reset_requested_at_utc = ?
            WHERE id = ?
            """,
            (token_hash, expiry, utc_now().isoformat(), user["id"]),
        )
        conn.commit()

    return True, generic_msg, token


def reset_password_with_token(token: str, new_password: str) -> tuple[bool, str]:
    """Reset password when provided reset token is valid and unexpired."""
    clean_token = token.strip()
    if not clean_token:
        return False, "Reset token is required."

    password_errors = validate_new_password(new_password)
    if password_errors:
        return False, " ".join(password_errors)

    token_hash = hashlib.sha256(clean_token.encode("utf-8")).hexdigest()
    now = utc_now()
    with get_db_connection() as conn:
        user = conn.execute(
            """
            SELECT id, reset_token_expiry_utc
            FROM users
            WHERE reset_token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

        if user is None or not user["reset_token_expiry_utc"]:
            return False, "Invalid or expired reset token."

        expiry = datetime.fromisoformat(user["reset_token_expiry_utc"])
        if expiry <= now:
            return False, "Invalid or expired reset token."

        new_salt = secrets.token_bytes(16)
        new_hash = hash_password(new_password, new_salt)
        conn.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?,
                reset_token_hash = NULL, reset_token_expiry_utc = NULL,
                failed_login_attempts = 0, lockout_until_utc = NULL
            WHERE id = ?
            """,
            (new_salt.hex(), new_hash.hex(), user["id"]),
        )
        conn.commit()

    return True, "Password reset successful. Please login with your new password."


def update_user_profile(user_id: int, profile_data: dict[str, Any]) -> tuple[bool, str]:
    """Update user profile data including measurements and personal info."""
    allowed_fields = [
        "email", "gender", "date_of_birth", "phone_number", "profile_picture_path",
        "shoulder_width_cm", "chest_circumference_cm", "torso_length_cm", "height_cm"
    ]
    
    update_fields = {}
    for k, v in profile_data.items():
        if k in allowed_fields:
            update_fields[k] = v
            
    if not update_fields:
        return False, "No valid fields to update."
        
    set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
    values = list(update_fields.values())
    values.append(user_id)
    
    try:
        with get_db_connection() as conn:
            conn.execute(
                f"UPDATE users SET {set_clause} WHERE id = ?",
                values
            )
            conn.commit()
        return True, "Profile updated successfully."
    except Exception as e:
        return False, f"Error updating profile: {e}"


def initialize_auth_session(session_state: MutableMapping[str, Any]) -> None:
    """Initialize authentication keys in session state."""
    session_state.setdefault("authenticated", False)
    session_state.setdefault("auth_user", None)
    session_state.setdefault("auth_started_at", None)
    session_state.setdefault("auth_last_seen_at", None)


def login_session(session_state: MutableMapping[str, Any], user: dict[str, Any]) -> None:
    """Set auth session fields after login."""
    now = utc_now().isoformat()
    session_state["authenticated"] = True
    session_state["auth_user"] = user
    session_state["auth_started_at"] = now
    session_state["auth_last_seen_at"] = now


def logout_session(session_state: MutableMapping[str, Any]) -> None:
    """Clear authentication session."""
    session_state["authenticated"] = False
    session_state["auth_user"] = None
    session_state["auth_started_at"] = None
    session_state["auth_last_seen_at"] = None
    session_state.pop("result", None)
    session_state.pop("temp_path", None)


def is_session_valid(session_state: MutableMapping[str, Any]) -> bool:
    """Validate active user session with idle and max-age expiry."""
    if not session_state.get("authenticated"):
        return False

    started_raw = session_state.get("auth_started_at")
    last_seen_raw = session_state.get("auth_last_seen_at")
    if not started_raw or not last_seen_raw:
        logout_session(session_state)
        return False

    now = utc_now()
    started_at = datetime.fromisoformat(str(started_raw))
    last_seen = datetime.fromisoformat(str(last_seen_raw))

    if now - started_at > SESSION_MAX_AGE:
        logout_session(session_state)
        return False
    if now - last_seen > SESSION_IDLE_TIMEOUT:
        logout_session(session_state)
        return False

    session_state["auth_last_seen_at"] = now.isoformat()
    return True
