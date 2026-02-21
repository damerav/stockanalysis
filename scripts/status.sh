#!/bin/bash
# ============================================================
# Stock Analysis Platform — Status Check
# Run on DGX Spark: ./scripts/status.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }

echo "============================================================"
echo "  Stock Analysis Platform — Status"
echo "============================================================"
echo ""

# --- Process Status ---
echo -e "${CYAN}Processes:${NC}"

check_process() {
    local name="$1"
    local pidfile="$PID_DIR/${name}.pid"

    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            ok "$name running (PID $pid)"
        else
            fail "$name dead (stale PID $pid)"
        fi
    else
        fail "$name not started"
    fi
}

check_process "dashboard"
check_process "es_strategy"
check_process "confidence_api"
check_process "metrics_exporter"
check_process "scheduler"

# Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(m['name'] for m in d.get('models',[])))" 2>/dev/null)
    ok "Ollama running — models: ${MODELS:-none}"
else
    fail "Ollama offline"
fi

echo ""

# --- Port Status ---
echo -e "${CYAN}Ports:${NC}"

check_port() {
    local port="$1"
    local name="$2"
    if fuser "$port/tcp" > /dev/null 2>&1; then
        ok "Port $port ($name) — listening"
    else
        fail "Port $port ($name) — not listening"
    fi
}

check_port 8501 "Dashboard"
check_port 8100 "Confidence API"
check_port 9190 "Metrics Exporter"
check_port 11434 "Ollama"

echo ""

# --- Database ---
echo -e "${CYAN}Database:${NC}"
DB="$DATA_DIR/spy.db"
if [ -f "$DB" ]; then
    SIZE=$(du -h "$DB" | cut -f1)
    ok "spy.db exists ($SIZE)"

    # Row counts for key tables
    for TABLE in prices predictions performance; do
        COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM $TABLE;" 2>/dev/null || echo "0")
        LATEST=$(sqlite3 "$DB" "SELECT MAX(date) FROM $TABLE;" 2>/dev/null || echo "—")
        info "$TABLE: $COUNT rows (latest: ${LATEST:-—})"
    done
else
    fail "spy.db not found"
fi

echo ""

# --- Models ---
echo -e "${CYAN}Models:${NC}"
MODEL_DIR="$PROJECT_DIR/models"
if [ -d "$MODEL_DIR" ]; then
    LATEST=$(ls -t "$MODEL_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        MSIZE=$(du -h "$LATEST" | cut -f1)
        MNAME=$(basename "$LATEST")
        ok "Latest: $MNAME ($MSIZE)"
        COUNT=$(ls "$MODEL_DIR"/*.json 2>/dev/null | wc -l)
        info "$COUNT model file(s) total"
    else
        fail "No trained models"
    fi
else
    fail "Models directory missing"
fi

echo ""

# --- Disk ---
echo -e "${CYAN}Disk:${NC}"
DISK_USAGE=$(du -sh "$PROJECT_DIR" 2>/dev/null | cut -f1)
info "Project size: $DISK_USAGE"
DISK_FREE=$(df -h "$PROJECT_DIR" | tail -1 | awk '{print $4}')
info "Disk free: $DISK_FREE"

echo ""

# --- Recent Logs ---
echo -e "${CYAN}Recent Log Activity:${NC}"
for LOGFILE in dashboard scheduler es_strategy confidence_api metrics_exporter; do
    LOGPATH="$LOG_DIR/${LOGFILE}.log"
    if [ -f "$LOGPATH" ]; then
        LAST_LINE=$(tail -1 "$LOGPATH" 2>/dev/null)
        info "$LOGFILE: $(echo "$LAST_LINE" | cut -c1-80)"
    fi
done

echo ""
echo "============================================================"
