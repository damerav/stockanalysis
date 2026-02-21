#!/bin/bash
# Start relay server + both dashboards
uvicorn src.sync.relay_server:app --host 0.0.0.0 --port 8000 &
sleep 2
streamlit run src/dashboard/realtime_app.py --server.port 8501 --server.headless true &
streamlit run src/dashboard/es_dashboard.py --server.port 8502 --server.headless true &
wait
