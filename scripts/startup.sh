#!/bin/bash
# startup.sh — Start scheduler + dashboard (clean, no cache)
# Usage: ssh abidamera@192.168.1.211 "cd ~/stockanalysis && bash scripts/startup.sh"

set -e
echo "=== Starting stockanalysis ==="

cd ~/stockanalysis
source .venv/bin/activate

# Verify nothing is running
RUNNING=$(pgrep -f "src.launcher|streamlit" 2>/dev/null || true)
if [ -n "$RUNNING" ]; then
    echo "WARNING: Processes still running: $RUNNING"
    echo "Run scripts/shutdown.sh first"
    exit 1
fi

# Start scheduler (manages dashboard + pipeline)
echo "Starting scheduler..."
nohup python -m src.launcher --spy --config config.yaml > logs/scheduler.log 2>&1 &
SCHED_PID=$!
echo "Scheduler started: PID $SCHED_PID"

# Wait for dashboard to come up
echo "Waiting for dashboard..."
for i in $(seq 1 30); do
    sleep 2
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
        echo "Dashboard is UP on port 8501"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "WARNING: Dashboard not responding after 60s — check logs/scheduler.log"
    fi
done

# Show running processes
echo ""
echo "=== Running processes ==="
pgrep -af "src.launcher|streamlit" 2>/dev/null || echo "None found"
echo ""
echo "Dashboard: http://192.168.1.211:8501"
echo "=== Startup complete ==="
