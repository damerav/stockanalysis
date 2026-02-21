#!/bin/bash
# ============================================================
# Stock Analysis Platform — Shutdown Script
# Run on DGX Spark: ./scripts/stop.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.pids"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[STOP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

stop_process() {
    local name="$1"
    local pidfile="$PID_DIR/${name}.pid"

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            # Wait up to 10 seconds for graceful shutdown
            for i in $(seq 1 10); do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null
                log "$name force-killed (PID $pid)"
            else
                log "$name stopped (PID $pid)"
            fi
        else
            log "$name was not running (stale PID $pid)"
        fi
        rm -f "$pidfile"
    else
        warn "$name — no PID file found"
    fi
}

echo "============================================================"
echo "  Shutting down Stock Analysis Platform"
echo "============================================================"
echo ""

# Stop in reverse order
stop_process "scheduler"
stop_process "es_strategy"
stop_process "dashboard"

# Ollama — optional, ask before stopping
if [ -f "$PID_DIR/ollama.pid" ]; then
    OLLAMA_PID=$(cat "$PID_DIR/ollama.pid")
    if kill -0 "$OLLAMA_PID" 2>/dev/null; then
        if [ "$1" = "--all" ]; then
            stop_process "ollama"
        else
            warn "Ollama still running (PID $OLLAMA_PID) — use --all to stop it too"
        fi
    else
        rm -f "$PID_DIR/ollama.pid"
    fi
fi

# Also kill any orphaned streamlit processes on port 8501
STREAMLIT_PID=$(fuser 8501/tcp 2>/dev/null)
if [ -n "$STREAMLIT_PID" ]; then
    kill $STREAMLIT_PID 2>/dev/null
    log "Killed orphaned process on port 8501 (PID $STREAMLIT_PID)"
fi

echo ""
log "All components stopped"
echo ""
