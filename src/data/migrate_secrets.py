"""One-time migration: move plaintext secrets from config.yaml to encrypted DB.

Usage (on DGX):
    cd ~/stockanalysis
    source .venv/bin/activate
    python -m src.data.migrate_secrets

This script:
1. Reads current config.yaml
2. Encrypts and stores all secrets in the app_secrets PostgreSQL table
3. Rewrites config.yaml with placeholder values (secrets removed)
4. Prints a summary of migrated secrets

After running, set STOCKAPP_MASTER_KEY env var to a strong random string
for production use. The same key must be used on every startup.
"""

import os
import sys
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.yaml")


def main():
    # Load current config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f) or {}

    # Migrate secrets to encrypted DB
    from src.data.secrets_manager import migrate_secrets_from_config, set_secret

    count = migrate_secrets_from_config(config)
    logger.info("Migrated %d secrets to encrypted DB", count)

    # Also store seed user passwords (so source code doesn't need them)
    admin_pw = os.environ.get("SEED_ADMIN_PASSWORD", "admin123")
    user_pw = os.environ.get("SEED_USER_PASSWORD", "user123")
    set_secret("seed_admin_password", admin_pw, "Default admin seed password")
    set_secret("seed_user_password", user_pw, "Default user seed password")
    count += 2

    # Rewrite config.yaml with placeholders
    # API keys
    if config.get("fred", {}).get("api_key"):
        config["fred"]["api_key"] = "FROM_ENCRYPTED_DB"
    if config.get("finnhub", {}).get("api_key"):
        config["finnhub"]["api_key"] = "FROM_ENCRYPTED_DB"
    if config.get("polygon", {}).get("api_key"):
        config["polygon"]["api_key"] = "FROM_ENCRYPTED_DB"

    # DB password — keep in config.yaml as bootstrap credential
    # In production, use STOCKAPP_DB_PASSWORD env var instead
    # pg = config.get("database", {}).get("postgres", {})
    # (intentionally NOT replacing — needed for DB connection bootstrap)

    # Auth secrets
    auth = config.get("auth", {})
    if auth.get("session_secret"):
        auth["session_secret"] = "FROM_ENCRYPTED_DB"
    if auth.get("google_client_secret"):
        auth["google_client_secret"] = "FROM_ENCRYPTED_DB"

    # Alert tokens
    alerts = config.get("alerts", {})
    if alerts.get("telegram_token"):
        alerts["telegram_token"] = "FROM_ENCRYPTED_DB"

    # Sync key
    sync = config.get("sync", {})
    if sync.get("api_key"):
        sync["api_key"] = "FROM_ENCRYPTED_DB"

    # Write updated config
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Updated config.yaml — plaintext secrets replaced with 'FROM_ENCRYPTED_DB'")
    logger.info("Total secrets migrated: %d", count)
    logger.info("")
    logger.info("IMPORTANT: Set STOCKAPP_MASTER_KEY environment variable for production:")
    logger.info("  export STOCKAPP_MASTER_KEY='your-strong-random-string-here'")


if __name__ == "__main__":
    main()
