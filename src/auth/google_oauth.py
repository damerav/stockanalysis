"""Authentication for Streamlit — Local users (DB + bcrypt) + Google OAuth.

Supports two modes:
  1. Local users (default) — username/bcrypt-hashed password in SQLite
  2. Google OAuth — when google_client_id is configured

Users are stored in the `users` table of spy.db with bcrypt-hashed passwords.
Seed users from config.yaml are migrated on first run.
"""

import os
import json
import hmac
import hashlib
import logging
import sqlite3
import urllib.parse
from datetime import datetime
from typing import Optional

import bcrypt
import requests
import streamlit as st

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = "openid email profile"

# Default seed users (migrated to DB on first run)
_SEED_USERS = {
    "admin": {"password": "admin123", "role": "admin", "name": "Administrator"},
    "user": {"password": "user123", "role": "viewer", "name": "Default User"},
}


def _get_auth_config() -> dict:
    """Load auth config from config.yaml or environment."""
    import yaml

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    auth = cfg.get("auth", {})
    return {
        "mode": auth.get("mode", "local"),
        "users": auth.get("users", {}),  # only used for seed migration
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", auth.get("google_client_id", "")),
        "client_secret": os.environ.get(
            "GOOGLE_CLIENT_SECRET", auth.get("google_client_secret", "")
        ),
        "allowed_domains": auth.get("allowed_domains", []),
        "allowed_emails": auth.get("allowed_emails", []),
        "session_secret": os.environ.get(
            "SESSION_SECRET", auth.get("session_secret", "stockanalysis-default-secret")
        ),
    }


# ---------------------------------------------------------------------------
# Database-backed user management
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Resolve the spy.db path."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return cfg.get("database", {}).get("path", "./data/spy.db")


def _get_user_db() -> sqlite3.Connection:
    """Get a connection to the user database, ensuring the table exists."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            name TEXT,
            role TEXT DEFAULT 'viewer',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def _ensure_users_seeded():
    """Migrate seed users from config.yaml to DB on first run (one-time)."""
    conn = _get_user_db()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count > 0:
        conn.close()
        return  # already seeded

    # Seed from config.yaml users or defaults
    cfg = _get_auth_config()
    seed = cfg.get("users", {}) or _SEED_USERS
    now = datetime.now().isoformat()
    for username, data in seed.items():
        pw = data.get("password", "changeme")
        pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, name, role, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, pw_hash, data.get("name", username), data.get("role", "viewer"), now, now),
        )
    conn.commit()
    conn.close()
    logger.info(f"Seeded {len(seed)} users from config to database")


def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


def db_get_user(username: str) -> Optional[dict]:
    """Get a user from the database by username."""
    _ensure_users_seeded()
    conn = _get_user_db()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def db_list_users() -> list[dict]:
    """List all users from the database."""
    _ensure_users_seeded()
    conn = _get_user_db()
    rows = conn.execute("SELECT username, name, role, created_at, updated_at FROM users ORDER BY username").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_create_user(username: str, password: str, name: str = "", role: str = "viewer") -> bool:
    """Create a new user in the database. Returns True on success."""
    _ensure_users_seeded()
    conn = _get_user_db()
    now = datetime.now().isoformat()
    pw_hash = hash_password(password)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, name, role, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, pw_hash, name or username, role, now, now),
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False


def db_update_user(username: str, name: str = None, role: str = None, password: str = None) -> bool:
    """Update an existing user. Only non-None fields are updated."""
    conn = _get_user_db()
    now = datetime.now().isoformat()
    updates = []
    params = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if password:
        updates.append("password_hash = ?")
        params.append(hash_password(password))
    if not updates:
        conn.close()
        return False
    updates.append("updated_at = ?")
    params.append(now)
    params.append(username)
    conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE username = ?", params)
    conn.commit()
    changed = conn.total_changes > 0
    conn.close()
    return changed


def db_delete_user(username: str) -> bool:
    """Delete a user from the database."""
    conn = _get_user_db()
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    changed = conn.total_changes > 0
    conn.close()
    return changed


def db_user_count() -> int:
    """Return the total number of users."""
    _ensure_users_seeded()
    conn = _get_user_db()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Session token helpers
# ---------------------------------------------------------------------------

def create_session_token(user_info: dict) -> str:
    """Create an HMAC-signed session token encoding user info."""
    cfg = _get_auth_config()
    secret = cfg["session_secret"]
    payload = json.dumps(user_info, sort_keys=True)
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def verify_session_token(token: str) -> Optional[dict]:
    """Verify and decode a session token. Returns user dict or None."""
    if not token or "|" not in token:
        return None
    cfg = _get_auth_config()
    secret = cfg["session_secret"]
    try:
        payload, sig = token.rsplit("|", 1)
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return json.loads(payload)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Cookie helpers — persist auth across browser refreshes
# ---------------------------------------------------------------------------

_COOKIE_NAME = "sa_auth_token"
_COOKIE_MAX_AGE = 86400 * 7  # 7 days


def _set_cookie(token: str):
    """Set auth cookie in the browser via a tiny JS component."""
    import streamlit.components.v1 as components
    js = f"""
    <script>
    document.cookie = "{_COOKIE_NAME}={token}; path=/; max-age={_COOKIE_MAX_AGE}; SameSite=Lax";
    </script>
    """
    components.html(js, height=0, width=0)


def _clear_cookie():
    """Delete auth cookie from the browser."""
    import streamlit.components.v1 as components
    js = f"""
    <script>
    document.cookie = "{_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax";
    </script>
    """
    components.html(js, height=0, width=0)


def _read_cookie() -> Optional[str]:
    """Read the auth cookie value using st.context.cookies (Streamlit ≥1.37)."""
    try:
        cookies = st.context.cookies
        return cookies.get(_COOKIE_NAME)
    except Exception:
        return None


def _restore_session_from_cookie():
    """If session_state is empty but a valid cookie exists, restore the session."""
    if st.session_state.get("auth_user") is not None:
        return
    token = _read_cookie()
    if not token:
        return
    user = verify_session_token(token)
    if user:
        st.session_state["auth_user"] = user
        st.session_state["auth_token"] = token


# ---------------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------------

def is_authenticated() -> bool:
    """Check if the current Streamlit session has a valid user."""
    _restore_session_from_cookie()
    return st.session_state.get("auth_user") is not None


def get_user() -> Optional[dict]:
    """Return the current user dict or None."""
    return st.session_state.get("auth_user")


def get_session_token() -> Optional[str]:
    """Return the current session token (for Grafana proxy, etc.)."""
    return st.session_state.get("auth_token")


def logout():
    """Clear session state and cookie."""
    for key in ["auth_user", "auth_token"]:
        st.session_state.pop(key, None)
    _clear_cookie()


# ---------------------------------------------------------------------------
# Local authentication (DB-backed with bcrypt)
# ---------------------------------------------------------------------------

def _authenticate_local_user(username: str, password: str) -> Optional[dict]:
    """Validate username/password against the database. Returns user dict or None."""
    _ensure_users_seeded()
    user = db_get_user(username)
    if user and verify_password(password, user["password_hash"]):
        return {
            "email": f"{username}@local",
            "name": user.get("name", username),
            "role": user.get("role", "viewer"),
            "username": username,
        }
    return None


# ---------------------------------------------------------------------------
# Google OAuth helpers
# ---------------------------------------------------------------------------

def _get_redirect_uri() -> str:
    """Build the OAuth redirect URI from the current Streamlit URL."""
    base = os.environ.get("STREAMLIT_URL", "http://localhost:8501")
    return base.rstrip("/") + "/"


def handle_oauth_callback():
    """Exchange the authorization code for tokens and set session."""
    code = st.query_params.get("code")
    if not code:
        return

    cfg = _get_auth_config()
    if not cfg["client_id"] or not cfg["client_secret"]:
        logger.warning("OAuth callback received but no client credentials configured")
        return

    try:
        resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uri": _get_redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        tokens = resp.json()
        access_token = tokens.get("access_token")
        if not access_token:
            logger.error("No access_token in OAuth response: %s", tokens)
            return

        user_resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_info = user_resp.json()
        email = user_info.get("email", "")
        domain = email.split("@")[-1] if email else ""

        if cfg["allowed_domains"] and domain not in cfg["allowed_domains"]:
            st.error(f"Domain @{domain} is not allowed.")
            return
        if cfg["allowed_emails"] and email not in cfg["allowed_emails"]:
            st.error(f"Email {email} is not in the allow list.")
            return

        session_user = {
            "email": email,
            "name": user_info.get("name", email),
            "picture": user_info.get("picture", ""),
            "role": "admin",
        }
        st.session_state["auth_user"] = session_user
        st.session_state["auth_token"] = create_session_token(session_user)
        _set_cookie(st.session_state["auth_token"])
        st.query_params.clear()

    except Exception as e:
        logger.error("OAuth callback error: %s", e)
        st.error(f"Authentication failed: {e}")


# ---------------------------------------------------------------------------
# Login page renderer
# ---------------------------------------------------------------------------

def render_login_page() -> bool:
    """Render the login page. Returns True if user is now authenticated."""
    cfg = _get_auth_config()
    mode = cfg["mode"]
    has_google = bool(cfg["client_id"] and cfg["client_secret"])

    st.markdown(
        """<div style="text-align:center; padding:40px 0 20px 0;">
        <h1>📊 Stock Analysis</h1>
        <p style="color:#888;">Sign in to continue</p>
        </div>""",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 2, 1])

    with col:
        if mode == "local" or not has_google:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    user = _authenticate_local_user(username, password)
                    if user:
                        token = create_session_token(user)
                        st.session_state["auth_user"] = user
                        st.session_state["auth_token"] = token
                        _set_cookie(token)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            if has_google:
                st.divider()
                st.caption("Or sign in with Google")
                _render_google_button(cfg)

        elif mode == "google" and has_google:
            _render_google_button(cfg)
            st.divider()
            st.caption("Or use local account")
            with st.form("login_form_fallback"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    user = _authenticate_local_user(username, password)
                    if user:
                        token = create_session_token(user)
                        st.session_state["auth_user"] = user
                        st.session_state["auth_token"] = token
                        _set_cookie(token)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

    return is_authenticated()


def _render_google_button(cfg: dict):
    """Render the Google Sign-In button/link."""
    redirect_uri = _get_redirect_uri()
    params = urllib.parse.urlencode({
        "client_id": cfg["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    })
    auth_url = f"{GOOGLE_AUTH_URL}?{params}"
    st.link_button("🔐 Sign in with Google", auth_url, use_container_width=True)
