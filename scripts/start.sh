#!/bin/bash
# ============================================================
# Stock Analysis Platform — Startup Script
# Run on DGX Spark: ./scripts/start.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[START]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

cd "$PROJECT_DIR"

# --- Check if already running ---
if [ -f "$PID_DIR/dashboard.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/dashboard.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "System appears to be running (dashboard PID $OLD_PID)"
        echo "  Run ./scripts/stop.sh first, or ./scripts/status.sh to check"
        exit 1
    fi
fi

# --- Activate venv ---
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    log "Virtual environment activated"
else
    err "Virtual environment not found at $VENV"
    exit 1
fi

# --- Step 1: Start Ollama (if not running) ---
log "Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "Ollama already running"
else
    log "Starting Ollama..."
    nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    echo $! > "$PID_DIR/ollama.pid"
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "Ollama started (PID $(cat $PID_DIR/ollama.pid))"
    else
        warn "Ollama failed to start — continuing without LLM"
    fi
fi

# --- Step 2: LLM Health Check ---
log "Running LLM health check..."
python -m src.launcher --check-llm --config config.yaml 2>&1 | tail -5
log "LLM check complete"

# --- Step 3: Start Dashboard ---
log "Starting unified dashboard on port 8501..."
nohup streamlit run src/dashboard/app.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 \
    > "$LOG_DIR/dashboard.log" 2>&1 &
echo $! > "$PID_DIR/dashboard.pid"
sleep 3

if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
    log "Dashboard running (PID $(cat $PID_DIR/dashboard.pid))"
else
    # Streamlit may take a few more seconds
    sleep 5
    log "Dashboard started (PID $(cat $PID_DIR/dashboard.pid)) — may take a moment to respond"
fi

# --- Step 4: Start ES Strategy Runner (paper mode) ---
log "Starting ES strategy runner (paper mode)..."
nohup python -m src.es_strategy.runner \
    --mode paper --config config.yaml \
    > "$LOG_DIR/es_strategy.log" 2>&1 &
echo $! > "$PID_DIR/es_strategy.pid"
log "ES strategy runner started (PID $(cat $PID_DIR/es_strategy.pid))"

# --- Step 4b: Start AI Confidence API (optional) ---
if [ "${START_API:-true}" = "true" ]; then
    log "Starting AI Confidence API on port 8100..."
    nohup python -m uvicorn src.api.confidence_server:app \
        --host 0.0.0.0 --port 8100 \
        > "$LOG_DIR/confidence_api.log" 2>&1 &
    echo $! > "$PID_DIR/confidence_api.pid"
    log "Confidence API started (PID $(cat $PID_DIR/confidence_api.pid))"
fi

# --- Step 5: Start Prometheus Metrics Exporter ---
log "Starting Prometheus metrics exporter on port 9190..."
nohup python -m src.api.metrics_exporter \
    > "$LOG_DIR/metrics_exporter.log" 2>&1 &
echo $! > "$PID_DIR/metrics_exporter.pid"
log "Metrics exporter started (PID $(cat $PID_DIR/metrics_exporter.pid))"

# --- Step 6: Start Pipeline Scheduler ---
log "Starting pipeline scheduler..."
nohup python -c "
import time, yaml, logging, sys
from datetime import datetime
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('scheduler')

with open('config.yaml') as f:
    config = yaml.safe_load(f) or {}

HOUR, MINUTE = 16, 30
last_run = ''
logger.info('Scheduler started — pipeline at %d:%02d ET (Mon-Fri)', HOUR, MINUTE)

while True:
    try:
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        if now.weekday() < 5 and now.hour == HOUR and now.minute >= MINUTE and now.minute < MINUTE + 2 and today != last_run:
            last_run = today
            logger.info('Triggering daily pipeline')
            from src.pipeline.daily_run import DailyPipeline
            pipeline = DailyPipeline(config)
            results = pipeline.run()
            logger.info('Pipeline complete: %s', results.get('total_elapsed', 0))
    except Exception as e:
        logger.error('Scheduler error: %s', e)
    time.sleep(60)
" > "$LOG_DIR/scheduler.log" 2>&1 &
echo $! > "$PID_DIR/scheduler.pid"
log "Scheduler started (PID $(cat $PID_DIR/scheduler.pid))"

# --- Summary ---
echo ""
echo "============================================================"
log "Stock Analysis Platform is running"
echo "============================================================"
echo ""
echo "  Dashboard:  http://$(hostname -I | awk '{print $1}'):8501"
echo "  Grafana:    http://$(hostname -I | awk '{print $1}'):3000  (if Docker stack running)"
echo "  API:        http://$(hostname -I | awk '{print $1}'):8100/health"
echo "  Metrics:    http://$(hostname -I | awk '{print $1}'):9190/metrics"
echo "  ES Runner:  paper mode (PID $(cat $PID_DIR/es_strategy.pid))"
echo "  Scheduler:  daily pipeline at 4:30 PM ET"
echo ""
echo "  Logs:       $LOG_DIR/"
echo "  PIDs:       $PID_DIR/"
echo ""
echo "  Stop:       ./scripts/stop.sh"
echo "  Status:     ./scripts/status.sh"
echo ""
echo "  To start Grafana + Prometheus:"
echo "    cd cloud && docker compose up -d grafana prometheus"
echo "============================================================"
