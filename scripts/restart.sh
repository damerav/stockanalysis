#!/bin/bash
# ============================================================
# Stock Analysis Platform — Restart Script
# Run on DGX Spark: ./scripts/restart.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Stopping all components..."
bash "$SCRIPT_DIR/stop.sh" "$@"

sleep 2

echo "Starting all components..."
bash "$SCRIPT_DIR/start.sh"
