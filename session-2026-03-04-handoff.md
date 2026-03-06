# Session Handoff — March 4, 2026 (Evening)

## Version: v2.7.2

## Commits This Session

| Hash | Description |
|------|-------------|
| `87db20b` | feat: encrypted secrets management — remove hardcoded passwords from source |
| `d7e2a2d` | feat: add market breadth & index fundamentals features |
| `33ae611` | docs: update project context for v2.7.2 |
| `05387f1` | fix: deprecate migrate_to_duckdb (broken import of removed _get_duckdb_path) |

## What Was Done

### 1. Encrypted Secrets Manager (commit `87db20b`)
- Created `src/data/secrets_manager.py` — Fernet-encrypted `app_secrets` table
- Created `src/data/migrate_secrets.py` — one-time migration script (already run on DGX)
- 6 secrets migrated: fred_api_key, finnhub_api_key, db_password, session_secret, seed_admin_password, seed_user_password
- 8 source files updated to resolve secrets from encrypted DB with env var fallbacks
- `config.yaml` API keys replaced with `FROM_ENCRYPTED_DB` placeholders
- DB password kept in config.yaml as bootstrap credential (needed to connect before secrets manager can read)
- Added `cryptography>=42.0` to requirements.txt
