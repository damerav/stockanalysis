"""Authentication for Streamlit — Local users (DB + bcrypt) + Google OAuth.

Supports two modes:
  1. Local users (default) — username/bcrypt-hashed password in PostgreSQL
  2. Google OAuth — when google_client_id is configured

Users are stored in the `users` table in PostgreSQL with bcrypt-hashed passwords.
Seed users from config.yaml are migrated on first run.
"""

import os
import json
import hmac
import time
import hashlib
import logging
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

# Default seed users (passwords loaded from encrypted DB or env vars)
def _get_seed_users() -> dict:
    """Build seed user dict with passwords from encrypted storage."""
    try:
        from src.data.secrets_manager import get_secret
        admin_pw = get_secret("seed_admin_password", fallback=os.environ.get("SEED_ADMIN_PASSWORD"))
        user_pw = get_secret("seed_user_password", fallback=os.environ.get("SEED_USER_PASSWORD"))
    except Exception:
        admin_pw = os.environ.get("SEED_ADMIN_PASSWORD")
        user_pw = os.environ.get("SEED_USER_PASSWORD")
    if not admin_pw or not user_pw:
        logger.warning("Seed passwords not found in encrypted DB or env vars — seeding skipped")
        return {}
    return {
        "admin": {"password": admin_pw, "role": "admin", "name": "Administrator"},
        "user": {"password": user_pw, "role": "viewer", "name": "Default User"},
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

    # Load secrets from encrypted DB, fall back to env vars, then config
    try:
        from src.data.secrets_manager import get_secret
        session_secret = get_secret("session_secret") or os.environ.get(
            "SESSION_SECRET", auth.get("session_secret", ""))
        google_secret = get_secret("google_client_secret") or os.environ.get(
            "GOOGLE_CLIENT_SECRET", auth.get("google_client_secret", ""))
    except Exception:
        session_secret = os.environ.get("SESSION_SECRET", auth.get("session_secret", ""))
        google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", auth.get("google_client_secret", ""))

    # Warn if session secret is still a placeholder
    if not session_secret or session_secret in ("FROM_ENCRYPTED_DB", "change-me-to-random-secret"):
        session_secret = "stockanalysis-fallback-" + os.environ.get("HOSTNAME", "local")
        logger.warning("No session secret configured — using hostname-derived fallback")

    return {
        "mode": auth.get("mode", "local"),
        "users": auth.get("users", {}),  # only used for seed migration
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", auth.get("google_client_id", "")),
        "client_secret": google_secret,
        "allowed_domains": auth.get("allowed_domains", []),
        "allowed_emails": auth.get("allowed_emails", []),
        "session_secret": session_secret,
    }


# ---------------------------------------------------------------------------
# PostgreSQL connection (thread-safe fresh connection per query)
# ---------------------------------------------------------------------------

_pg_cfg = None
_pg_cfg_loaded = False


def _load_pg_config():
    """Load PostgreSQL config once from config.yaml, cache it."""
    global _pg_cfg, _pg_cfg_loaded
    if _pg_cfg_loaded:
        return _pg_cfg
    import yaml
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        pg = cfg.get("database", {}).get("postgres")
        if pg and pg.get("dbname") and pg.get("user"):
            _pg_cfg = pg
    except Exception:
        pass
    _pg_cfg_loaded = True
    return _pg_cfg


def _pg_connect():
    """Create a fresh PostgreSQL connection (thread-safe)."""
    pg = _load_pg_config()
    if not pg:
        return None
    try:
        import psycopg2
        password = os.environ.get("STOCKAPP_DB_PASSWORD", pg.get("password", ""))
        conn = psycopg2.connect(
            host=pg.get("host", "localhost"),
            port=pg.get("port", 5432),
            dbname=pg["dbname"],
            user=pg["user"],
            password=password,
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error("PostgreSQL connection failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Database-backed user management (PostgreSQL)
# ---------------------------------------------------------------------------

def _ensure_users_seeded():
    """Seed default users into PostgreSQL if the table is empty (one-time)."""
    conn = _pg_connect()
    if not conn:
        logger.error("Cannot seed users — PostgreSQL unavailable")
        return
    try:
        cur = conn.cursor()
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                name TEXT,
                role TEXT DEFAULT 'viewer',
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        if count > 0:
            cur.close()
            conn.close()
            return

        cfg = _get_auth_config()
        seed = cfg.get("users", {}) or _get_seed_users()
        now = datetime.now().isoformat()
        for username, data in seed.items():
            pw = data.get("password", "changeme")
            pw_hash = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
            cur.execute(
                "INSERT INTO users (username, password_hash, name, role, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                (username, pw_hash, data.get("name", username), data.get("role", "viewer"), now, now),
            )
        cur.close()
        conn.close()
        logger.info("Seeded %d users into PostgreSQL", len(seed))
    except Exception as e:
        logger.error("Error seeding users: %s", e)
        conn.close()


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
    """Get a user from PostgreSQL by username."""
    _ensure_users_seeded()
    conn = _pg_connect()
    if not conn:
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT username, password_hash, name, role, created_at, updated_at "
            "FROM users WHERE username = %s", (username,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "username": row[0], "password_hash": row[1], "name": row[2],
                "role": row[3], "created_at": row[4], "updated_at": row[5],
            }
        return None
    except Exception as e:
        logger.error("db_get_user error: %s", e)
        conn.close()
        return None


def db_list_users() -> list[dict]:
    """List all users from PostgreSQL."""
    _ensure_users_seeded()
    conn = _pg_connect()
    if not conn:
        return []
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, name, role, created_at, updated_at FROM users ORDER BY username")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"username": r[0], "name": r[1], "role": r[2], "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error("db_list_users error: %s", e)
        conn.close()
        return []


def db_create_user(username: str, password: str, name: str = "", role: str = "viewer") -> bool:
    """Create a new user in PostgreSQL. Returns True on success."""
    _ensure_users_seeded()
    conn = _pg_connect()
    if not conn:
        return False
    now = datetime.now().isoformat()
    pw_hash = hash_password(password)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, name, role, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
            (username, pw_hash, name or username, role, now, now),
        )
        inserted = cur.rowcount > 0
        cur.close()
        conn.close()
        return inserted
    except Exception as e:
        logger.error("db_create_user error: %s", e)
        conn.close()
        return False


def db_update_user(username: str, name: str = None, role: str = None, password: str = None) -> bool:
    """Update an existing user. Only non-None fields are updated."""
    conn = _pg_connect()
    if not conn:
        return False
    now = datetime.now().isoformat()
    updates = []
    params = []
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    if role is not None:
        updates.append("role = %s")
        params.append(role)
    if password:
        updates.append("password_hash = %s")
        params.append(hash_password(password))
    if not updates:
        conn.close()
        return False
    updates.append("updated_at = %s")
    params.append(now)
    params.append(username)
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE username = %s", params)
        changed = cur.rowcount > 0
        cur.close()
        conn.close()
        return changed
    except Exception as e:
        logger.error("db_update_user error: %s", e)
        conn.close()
        return False


def db_delete_user(username: str) -> bool:
    """Delete a user from PostgreSQL."""
    conn = _pg_connect()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE username = %s", (username,))
        changed = cur.rowcount > 0
        cur.close()
        conn.close()
        return changed
    except Exception as e:
        logger.error("db_delete_user error: %s", e)
        conn.close()
        return False


def db_user_count() -> int:
    """Return the total number of users."""
    _ensure_users_seeded()
    conn = _pg_connect()
    if not conn:
        return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception as e:
        logger.error("db_user_count error: %s", e)
        conn.close()
        return 0


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
# Session persistence — server-side files + browser cookie + query param
# ---------------------------------------------------------------------------

_COOKIE_NAME = "sa_session_id"
_COOKIE_MAX_AGE = 86400 * 7  # 7 days
_SESSION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", ".sessions",
)


def _ensure_session_dir():
    """Create the server-side session directory if needed."""
    os.makedirs(_SESSION_DIR, exist_ok=True)


def _generate_session_id() -> str:
    """Generate a random session ID."""
    import secrets
    return secrets.token_urlsafe(32)


def _save_server_session(session_id: str, token: str):
    """Save auth token to a server-side file keyed by session_id."""
    _ensure_session_dir()
    path = os.path.join(_SESSION_DIR, f"{session_id}.json")
    data = {"token": token, "created": datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(data, f)


def _load_server_session(session_id: str) -> Optional[str]:
    """Load auth token from server-side session file. Returns token or None."""
    if not session_id:
        return None
    path = os.path.join(_SESSION_DIR, f"{session_id}.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("token")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _delete_server_session(session_id: str):
    """Delete a server-side session file."""
    if not session_id:
        return
    path = os.path.join(_SESSION_DIR, f"{session_id}.json")
    try:
        os.remove(path)
    except OSError:
        pass


def _cleanup_stale_sessions(max_age_days: int = 7):
    """Remove session files older than max_age_days. Called sparingly."""
    _ensure_session_dir()
    try:
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        for fname in os.listdir(_SESSION_DIR):
            fpath = os.path.join(_SESSION_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
    except OSError:
        pass


def _set_cookie(session_id: str):
    """Set session ID cookie in the browser via JS injection."""
    import streamlit.components.v1 as components
    js = f"""
    <script>
    document.cookie = "{_COOKIE_NAME}={session_id}; path=/; max-age={_COOKIE_MAX_AGE}; SameSite=Lax";
    </script>
    """
    components.html(js, height=0, width=0)


def _clear_cookie():
    """Delete session cookie from the browser."""
    import streamlit.components.v1 as components
    js = f"""
    <script>
    document.cookie = "{_COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax";
    </script>
    """
    components.html(js, height=0, width=0)


def _read_cookie() -> Optional[str]:
    """Read the session ID cookie using st.context.cookies (Streamlit ≥1.37)."""
    try:
        cookies = st.context.cookies
        return cookies.get(_COOKIE_NAME)
    except Exception:
        return None


def _persist_session(token: str):
    """Persist session using server-side file + cookie + query param (triple redundancy)."""
    session_id = _generate_session_id()
    _save_server_session(session_id, token)
    _set_cookie(session_id)
    # Also store in query params as fallback (survives refresh even if cookie JS fails)
    st.query_params["sid"] = session_id
    # Keep in session_state for current run
    st.session_state["_session_id"] = session_id


def _restore_session_from_cookie():
    """If session_state is empty but a valid session exists, restore it.

    Checks three sources in order:
      1. st.query_params['sid'] (most reliable — embedded in URL)
      2. Browser cookie (set via JS)
      3. session_state._session_id (same tab, no refresh)
    """
    if st.session_state.get("auth_user") is not None:
        return

    # Try query param first (survives refresh reliably)
    session_id = st.query_params.get("sid")

    # Fallback to cookie
    if not session_id:
        session_id = _read_cookie()

    # Fallback to session_state (shouldn't be needed but just in case)
    if not session_id:
        session_id = st.session_state.get("_session_id")

    if not session_id:
        return

    # Look up the server-side session file
    token = _load_server_session(session_id)
    if not token:
        return

    user = verify_session_token(token)
    if user:
        st.session_state["auth_user"] = user
        st.session_state["auth_token"] = token
        st.session_state["_session_id"] = session_id
        # Re-set query param in case it was restored from cookie
        if "sid" not in st.query_params:
            st.query_params["sid"] = session_id


# ---------------------------------------------------------------------------
# Public API used by app.py
# ---------------------------------------------------------------------------

def is_authenticated() -> bool:
    """Check if the current Streamlit session has a valid user."""
    # Cleanup stale sessions once per app lifecycle
    if not st.session_state.get("_sessions_cleaned"):
        _cleanup_stale_sessions()
        st.session_state["_sessions_cleaned"] = True
    _restore_session_from_cookie()
    return st.session_state.get("auth_user") is not None


def get_user() -> Optional[dict]:
    """Return the current user dict or None."""
    return st.session_state.get("auth_user")


def get_session_token() -> Optional[str]:
    """Return the current session token (for Grafana proxy, etc.)."""
    return st.session_state.get("auth_token")


def logout():
    """Clear session state, server-side session file, and cookie."""
    session_id = st.session_state.get("_session_id")
    _delete_server_session(session_id)
    for key in ["auth_user", "auth_token", "_session_id"]:
        st.session_state.pop(key, None)
    _clear_cookie()
    # Clear the sid query param
    if "sid" in st.query_params:
        st.query_params.clear()


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
        _persist_session(st.session_state["auth_token"])
        # Don't clear all query_params — _persist_session sets 'sid'
        for k in list(st.query_params.keys()):
            if k != "sid":
                del st.query_params[k]

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
                        _persist_session(token)
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
                        _persist_session(token)
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
