#!/bin/bash
# shutdown.sh — Kill scheduler, dashboard, and clear caches
# Usage: ssh abidamera@192.168.1.211 "cd ~/stockanalysis && bash scripts/shutdown.sh"

set -e
echo "=== Shutting down stockanalysis ==="

# Kill scheduler
SCHED_PIDS=$(pgrep -f "src.launcher" 2>/dev/null || true)
if [ -n "$SCHED_PIDS" ]; then
    echo "Killing scheduler PIDs: $SCHED_PIDS"
    echo "$SCHED_PIDS" | xargs kill -9 2>/dev/null || true
else
    echo "No scheduler running"
fi

# Kill streamlit
ST_PIDS=$(pgrep -f "streamlit" 2>/dev/null || true)
if [ -n "$ST_PIDS" ]; then
    echo "Killing streamlit PIDs: $ST_PIDS"
    echo "$ST_PIDS" | xargs kill -9 2>/dev/null || true
else
    echo "No streamlit running"
fi

# Free ports
fuser -k 8501/tcp 2>/dev/null || true
fuser -k 8100/tcp 2>/dev/null || true

# Clear Python caches
echo "Clearing __pycache__..."
find ~/stockanalysis -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Clear Streamlit cache
echo "Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null || true
rm -rf /tmp/streamlit-* 2>/dev/null || true

echo "=== Shutdown complete ==="
