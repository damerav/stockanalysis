"""Secrets Manager — Encrypted secret storage in PostgreSQL.

Provides Fernet-based symmetric encryption for storing sensitive values
(API keys, passwords, session secrets) in the `app_secrets` table.

The master encryption key is derived from the STOCKAPP_MASTER_KEY environment
variable. If not set, falls back to a machine-specific key derived from
hostname + username (acceptable for single-machine deployments, but env var
is strongly recommended for production).

Usage:
    from src.data.secrets_manager import get_secret, set_secret

    # Store a secret
    set_secret("fred_api_key", "your-api-key-here")

    # Retrieve a secret (returns None if not found)
    key = get_secret("fred_api_key")

    # Retrieve with fallback
    key = get_secret("fred_api_key", fallback="default-value")
"""

import os
import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Master key derivation
# ---------------------------------------------------------------------------

def _derive_master_key() -> bytes:
    """Derive a Fernet-compatible key from STOCKAPP_MASTER_KEY env var.

    Falls back to a deterministic machine-specific key if env var is not set.
    """
    raw = os.environ.get("STOCKAPP_MASTER_KEY", "")
    if not raw:
        # Machine-specific fallback (not ideal, but works for single-machine setups)
        import socket, getpass
        raw = f"stockapp-{socket.gethostname()}-{getpass.getuser()}-fallback-key"
        logger.warning("STOCKAPP_MASTER_KEY not set — using machine-derived key")
    # Fernet requires a 32-byte url-safe base64-encoded key
    digest = hashlib.sha256(raw.encode()).digest()
    return base64.urlsafe_b64encode(digest)


_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_derive_master_key())
    return _fernet


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value. Returns base64-encoded ciphertext."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a base64-encoded ciphertext back to plaintext."""
    return _get_fernet().decrypt(ciphertext.encode()).decode()


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _pg_connect():
    """Get a fresh PostgreSQL connection for secrets operations."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        pg = cfg.get("database", {}).get("postgres", {})
    except Exception:
        pg = {}

    # Allow env var override for DB password (bootstrap scenario)
    host = os.environ.get("STOCKAPP_DB_HOST", pg.get("host", "localhost"))
    port = int(os.environ.get("STOCKAPP_DB_PORT", pg.get("port", 5432)))
    dbname = os.environ.get("STOCKAPP_DB_NAME", pg.get("dbname", "stockanalysis"))
    user = os.environ.get("STOCKAPP_DB_USER", pg.get("user", "stockapp"))
    password = os.environ.get("STOCKAPP_DB_PASSWORD", pg.get("password", ""))

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error("secrets_manager: PostgreSQL connection failed: %s", e)
        return None


def _ensure_table(conn):
    """Create the app_secrets table if it doesn't exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_secrets (
            key TEXT PRIMARY KEY,
            encrypted_value TEXT NOT NULL,
            description TEXT,
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.close()


def get_secret(key: str, fallback: str = None) -> Optional[str]:
    """Retrieve and decrypt a secret by key. Returns fallback if not found."""
    conn = _pg_connect()
    if not conn:
        logger.warning("get_secret: no DB connection, returning fallback for '%s'", key)
        return fallback
    try:
        _ensure_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT encrypted_value FROM app_secrets WHERE key = %s", (key,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return decrypt_value(row[0])
        return fallback
    except Exception as e:
        logger.error("get_secret error for '%s': %s", key, e)
        try:
            conn.close()
        except Exception:
            pass
        return fallback


def set_secret(key: str, value: str, description: str = "") -> bool:
    """Encrypt and store a secret. Returns True on success."""
    conn = _pg_connect()
    if not conn:
        logger.error("set_secret: no DB connection")
        return False
    try:
        _ensure_table(conn)
        encrypted = encrypt_value(value)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO app_secrets (key, encrypted_value, description, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET
                encrypted_value = EXCLUDED.encrypted_value,
                description = EXCLUDED.description,
                updated_at = NOW()
        """, (key, encrypted, description))
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error("set_secret error for '%s': %s", key, e)
        try:
            conn.close()
        except Exception:
            pass
        return False


def delete_secret(key: str) -> bool:
    """Delete a secret by key. Returns True if deleted."""
    conn = _pg_connect()
    if not conn:
        return False
    try:
        _ensure_table(conn)
        cur = conn.cursor()
        cur.execute("DELETE FROM app_secrets WHERE key = %s", (key,))
        deleted = cur.rowcount > 0
        cur.close()
        conn.close()
        return deleted
    except Exception as e:
        logger.error("delete_secret error for '%s': %s", key, e)
        try:
            conn.close()
        except Exception:
            pass
        return False


def list_secrets() -> list[dict]:
    """List all secret keys (without values) and their descriptions."""
    conn = _pg_connect()
    if not conn:
        return []
    try:
        _ensure_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT key, description, updated_at FROM app_secrets ORDER BY key")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"key": r[0], "description": r[1], "updated_at": str(r[2])} for r in rows]
    except Exception as e:
        logger.error("list_secrets error: %s", e)
        try:
            conn.close()
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Bulk migration helper
# ---------------------------------------------------------------------------

def migrate_secrets_from_config(config: dict) -> int:
    """Migrate plaintext secrets from config dict to encrypted DB storage.

    Returns the number of secrets migrated.
    """
    count = 0
    mappings = [
        ("fred_api_key", config.get("fred", {}).get("api_key", ""), "FRED API key"),
        ("finnhub_api_key", config.get("finnhub", {}).get("api_key", ""), "Finnhub API key"),
        ("polygon_api_key", config.get("polygon", {}).get("api_key", ""), "Polygon API key"),
        ("db_password", config.get("database", {}).get("postgres", {}).get("password", ""), "PostgreSQL password"),
        ("session_secret", config.get("auth", {}).get("session_secret", ""), "Session signing secret"),
        ("telegram_token", config.get("alerts", {}).get("telegram_token", ""), "Telegram bot token"),
        ("google_client_secret", config.get("auth", {}).get("google_client_secret", ""), "Google OAuth client secret"),
        ("sync_api_key", config.get("sync", {}).get("api_key", ""), "Relay sync API key"),
    ]
    for key, value, desc in mappings:
        if value and value not in ("", "YOUR_POLYGON_KEY", "YOUR_RELAY_API_KEY"):
            if set_secret(key, value, desc):
                count += 1
                logger.info("Migrated secret: %s", key)
    return count
