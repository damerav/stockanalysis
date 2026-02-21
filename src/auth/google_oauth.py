"""Authentication for Streamlit — Local users + Google OAuth.

Supports two modes:
  1. Local users (default) — username/password from config.yaml
  2. Google OAuth — when google_client_id is configured

Config in config.yaml:
  auth:
    mode: "local"                # "local" or "google"
    session_secret: "random-secret-key"
    users:
      admin:
        password: "admin123"
        role: "admin"
        name: "Administrator"
      user:
        password: "user123"
        role: "viewer"
        name: "Default User"
    google_client_id: ""
    google_client_secret: ""
    allowed_domains: []
    allowed_emails: []
"""

import os
import json
import time
import hmac
import hashlib
import logging
import urllib.parse
from typing import Optional

import requests
import streamlit as st

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = "openid email profile"

# Default local users (used if not defined in config.yaml)
DEFAULT_USERS = {
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
        "users": auth.get("users", DEFAULT_USERS),
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
        return  # already authenticated
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
# Local authentication
# ---------------------------------------------------------------------------

def _authenticate_local_user(username: str, password: str) -> Optional[dict]:
    """Validate username/password against config. Returns user dict or None."""
    cfg = _get_auth_config()
    users = cfg.get("users", DEFAULT_USERS)
    entry = users.get(username)
    if entry and entry.get("password") == password:
        return {
            "email": f"{username}@local",
            "name": entry.get("name", username),
            "role": entry.get("role", "viewer"),
            "username": username,
        }
    return None


# ---------------------------------------------------------------------------
# Google OAuth helpers
# ---------------------------------------------------------------------------

def _get_redirect_uri() -> str:
    """Build the OAuth redirect URI from the current Streamlit URL."""
    # In production, set STREAMLIT_URL env var
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
        # Exchange code for tokens
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

        # Fetch user info
        user_resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_info = user_resp.json()
        email = user_info.get("email", "")
        domain = email.split("@")[-1] if email else ""

        # Check allowed domains / emails
        if cfg["allowed_domains"] and domain not in cfg["allowed_domains"]:
            st.error(f"Domain @{domain} is not allowed.")
            return
        if cfg["allowed_emails"] and email not in cfg["allowed_emails"]:
            st.error(f"Email {email} is not in the allow list.")
            return

        # Set session
        session_user = {
            "email": email,
            "name": user_info.get("name", email),
            "picture": user_info.get("picture", ""),
            "role": "admin",  # Google users get admin by default
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

    # Center the login form
    _, col, _ = st.columns([1, 2, 1])

    with col:
        # --- Local login form ---
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

            # Show Google option if credentials exist even in local mode
            if has_google:
                st.divider()
                st.caption("Or sign in with Google")
                _render_google_button(cfg)

        # --- Google OAuth primary ---
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
