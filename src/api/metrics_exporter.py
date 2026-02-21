"""Prometheus metrics exporter for the Stock Analysis Platform.

Exposes all system metrics at /metrics for Prometheus scraping.
Also serves historical time-series data from SQLite via JSON API
endpoints for Grafana panels (Moving Averages, Bollinger Bands, etc.).

Runs as a standalone HTTP server on port 9190.

Usage:
    python -m src.api.metrics_exporter
"""

import os
import json
import time
import sqlite3
import logging
import threading
import urllib.parse
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

import yaml
import requests

logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.yaml")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")

# Cache metrics to avoid hammering DB/files on every scrape
_cache = {"metrics": "", "updated": 0}
CACHE_TTL = 10  # seconds


def _load_config():
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def collect_spy_metrics():
    """Collect SPY predictor metrics from DB (primary) and state files (fallback)."""
    lines = []
    db_path = os.path.join(DATA_DIR, "spy.db")

    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path, timeout=5)

            # Latest price
            row = conn.execute(
                "SELECT close FROM prices ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if row:
                lines.append(f'spy_last_close {_safe_float(row[0])}')

            # Latest technicals (columns: sma_20, sma_50, rsi_14, macd, macd_signal, bb_upper, bb_lower, atr_14)
            row = conn.execute(
                "SELECT sma_20, sma_50, rsi_14, macd, macd_signal, bb_upper, bb_lower, atr_14 "
                "FROM technicals ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if row:
                lines.append(f'spy_sma20 {_safe_float(row[0])}')
                lines.append(f'spy_sma50 {_safe_float(row[1])}')
                lines.append(f'spy_rsi {_safe_float(row[2])}')
                lines.append(f'spy_macd {_safe_float(row[3])}')
                lines.append(f'spy_macd_signal {_safe_float(row[4])}')
                lines.append(f'spy_bb_upper {_safe_float(row[5])}')
                lines.append(f'spy_bb_lower {_safe_float(row[6])}')
                lines.append(f'spy_atr {_safe_float(row[7])}')

            # Latest VIX from macro
            row = conn.execute(
                "SELECT vix FROM macro ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if row:
                lines.append(f'spy_vix {_safe_float(row[0])}')

            # Latest prediction
            row = conn.execute(
                "SELECT direction, confidence FROM predictions ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if row:
                direction_val = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}.get(
                    str(row[0]).upper(), 0
                )
                lines.append(f'spy_prediction_direction {direction_val}')
                lines.append(f'spy_prediction_confidence {_safe_float(row[1])}')

            # Table row counts
            tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                       "predictions", "intraday_bars", "options_chain",
                       "options_analytics", "intraday_features", "performance"]
            for t in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    lines.append(f'spy_table_rows{{table="{t}"}} {count}')
                except Exception:
                    pass
            conn.close()
        except Exception as e:
            logger.debug(f"SPY DB read error: {e}")

    return lines



def collect_es_metrics():
    """Collect ES strategy metrics from state file."""
    lines = []
    state_path = os.path.join(DATA_DIR, "es_state.json")
    try:
        with open(state_path) as f:
            state = json.load(f)
        lines.append(f'es_daily_pnl {_safe_float(state.get("daily_pnl"))}')
        lines.append(f'es_total_pnl {_safe_float(state.get("total_pnl"))}')
        lines.append(f'es_open_lots {_safe_float(state.get("open_lots"))}')
        lines.append(f'es_trades_today {_safe_float(state.get("trades_today"))}')
        lines.append(f'es_win_rate {_safe_float(state.get("win_rate"))}')
        lines.append(f'es_max_drawdown {_safe_float(state.get("max_drawdown"))}')
        lines.append(f'es_sharpe_ratio {_safe_float(state.get("sharpe_ratio"))}')
        lines.append(f'es_current_price {_safe_float(state.get("current_price"))}')
        lines.append(f'es_kc_mid {_safe_float(state.get("kc_mid"))}')
        lines.append(f'es_kc_upper {_safe_float(state.get("kc_upper"))}')
        lines.append(f'es_kc_lower {_safe_float(state.get("kc_lower"))}')
        lines.append(f'es_vwap {_safe_float(state.get("vwap"))}')
        lines.append(f'es_atr {_safe_float(state.get("atr"))}')
        lines.append(f'es_rsi {_safe_float(state.get("rsi"))}')

        # Position info
        pos = state.get("position", {})
        if isinstance(pos, dict):
            lines.append(f'es_position_lots {_safe_float(pos.get("lots"))}')
            lines.append(f'es_position_entry_price {_safe_float(pos.get("entry_price"))}')
            lines.append(f'es_position_unrealized_pnl {_safe_float(pos.get("unrealized_pnl"))}')

        # Regime
        regime_map = {"Low": 0, "Med": 1, "High": 2}
        regime_str = state.get("vol_regime", "Med")
        lines.append(f'es_vol_regime {regime_map.get(regime_str, 1)}')

        # Signal history count
        signals = state.get("signals", [])
        lines.append(f'es_signal_count {len(signals) if isinstance(signals, list) else 0}')
    except Exception as e:
        logger.debug(f"ES state read error: {e}")

    return lines


def collect_system_health():
    """Collect system health metrics: DB, Ollama, model, API."""
    lines = []

    # Database health
    db_path = os.path.join(DATA_DIR, "spy.db")
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        lines.append(f"system_db_size_mb {size_mb:.2f}")
        lines.append("system_db_online 1")
    else:
        lines.append("system_db_online 0")

    # Ollama health
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            lines.append("system_ollama_online 1")
            lines.append(f"system_ollama_model_count {len(models)}")
            config = _load_config()
            target = config.get("llm", {}).get("model", "deepseek-r1:70b")
            has_target = 1 if any(target in m.get("name", "") for m in models) else 0
            lines.append(f"system_ollama_target_loaded {has_target}")
        else:
            lines.append("system_ollama_online 0")
    except Exception:
        lines.append("system_ollama_online 0")

    # XGBoost model
    if os.path.exists(MODELS_DIR):
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".json")], reverse=True)
        lines.append(f"system_model_count {len(model_files)}")
        if model_files:
            latest = os.path.join(MODELS_DIR, model_files[0])
            size_kb = os.path.getsize(latest) / 1024
            lines.append(f"system_model_latest_size_kb {size_kb:.1f}")
            lines.append("system_model_loaded 1")
        else:
            lines.append("system_model_loaded 0")
    else:
        lines.append("system_model_loaded 0")

    # Confidence API health
    try:
        resp = requests.get("http://localhost:8100/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            lines.append("system_confidence_api_online 1")
            lines.append(f'system_confidence_entry_gate_loaded {1 if data.get("entry_gate_loaded") else 0}')
            lines.append(f'system_confidence_exit_ctrl_loaded {1 if data.get("exit_ctrl_loaded") else 0}')
            lines.append(f'system_confidence_uptime_seconds {_safe_float(data.get("uptime_seconds"))}')
        else:
            lines.append("system_confidence_api_online 0")
    except Exception:
        lines.append("system_confidence_api_online 0")

    # Streamlit dashboard health
    try:
        resp = requests.get("http://localhost:8501", timeout=3)
        lines.append(f"system_dashboard_online {1 if resp.status_code == 200 else 0}")
    except Exception:
        lines.append("system_dashboard_online 0")

    return lines


def collect_pipeline_metrics():
    """Collect pipeline and scheduler metrics."""
    lines = []

    # Check scheduler PID
    pid_file = os.path.join(PROJECT_DIR, ".pids", "scheduler.pid")
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file).read().strip())
            # Check if process is running (Linux)
            os.kill(pid, 0)
            lines.append("pipeline_scheduler_running 1")
        except (ProcessLookupError, ValueError, PermissionError):
            lines.append("pipeline_scheduler_running 0")
    else:
        lines.append("pipeline_scheduler_running 0")

    # ES runner PID
    pid_file = os.path.join(PROJECT_DIR, ".pids", "es_strategy.pid")
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file).read().strip())
            os.kill(pid, 0)
            lines.append("pipeline_es_runner_running 1")
        except (ProcessLookupError, ValueError, PermissionError):
            lines.append("pipeline_es_runner_running 0")
    else:
        lines.append("pipeline_es_runner_running 0")

    # Audit log stats (GAP 15)
    audit_path = os.path.join(LOGS_DIR, "trade_audit.jsonl")
    if os.path.exists(audit_path):
        try:
            size_kb = os.path.getsize(audit_path) / 1024
            lines.append(f"pipeline_audit_log_size_kb {size_kb:.1f}")
            with open(audit_path) as f:
                line_count = sum(1 for _ in f)
            lines.append(f"pipeline_audit_log_entries {line_count}")
        except Exception:
            pass

    return lines


def collect_confidence_api_metrics():
    """Collect detailed confidence API metrics from audit log."""
    lines = []
    audit_path = os.path.join(LOGS_DIR, "trade_audit.jsonl")
    if not os.path.exists(audit_path):
        return lines

    try:
        # Read last 100 entries for recent stats
        entries = []
        with open(audit_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        recent = entries[-100:] if len(entries) > 100 else entries

        if recent:
            # Average latency
            latencies = [e.get("latency_ms", 0) for e in recent if e.get("latency_ms")]
            if latencies:
                lines.append(f"confidence_avg_latency_ms {sum(latencies)/len(latencies):.2f}")
                lines.append(f"confidence_max_latency_ms {max(latencies):.2f}")
                lines.append(f"confidence_p99_latency_ms {sorted(latencies)[int(len(latencies)*0.99)]:.2f}")

            # Allow/block ratio
            allows = sum(1 for e in recent if e.get("advice") == "allow")
            blocks = sum(1 for e in recent if e.get("advice") == "block")
            lines.append(f"confidence_recent_allows {allows}")
            lines.append(f"confidence_recent_blocks {blocks}")

            # Total requests
            lines.append(f"confidence_total_requests {len(entries)}")
    except Exception as e:
        logger.debug(f"Audit log parse error: {e}")

    return lines


def generate_metrics():
    """Generate all Prometheus metrics."""
    now = time.time()
    if now - _cache["updated"] < CACHE_TTL and _cache["metrics"]:
        return _cache["metrics"]

    all_lines = []
    all_lines.append("# HELP spy_last_close SPY last closing price")
    all_lines.append("# TYPE spy_last_close gauge")
    all_lines.extend(collect_spy_metrics())

    all_lines.append("")
    all_lines.append("# HELP es_daily_pnl ES strategy daily P&L")
    all_lines.append("# TYPE es_daily_pnl gauge")
    all_lines.extend(collect_es_metrics())

    all_lines.append("")
    all_lines.append("# HELP system_db_online Database availability")
    all_lines.append("# TYPE system_db_online gauge")
    all_lines.extend(collect_system_health())

    all_lines.append("")
    all_lines.append("# HELP pipeline_scheduler_running Scheduler process status")
    all_lines.append("# TYPE pipeline_scheduler_running gauge")
    all_lines.extend(collect_pipeline_metrics())

    all_lines.append("")
    all_lines.append("# HELP confidence_avg_latency_ms Average confidence API latency")
    all_lines.append("# TYPE confidence_avg_latency_ms gauge")
    all_lines.extend(collect_confidence_api_metrics())

    all_lines.append("")
    result = "\n".join(all_lines) + "\n"
    _cache["metrics"] = result
    _cache["updated"] = now
    return result


def query_historical(series, days=365):
    """Query historical time series data from SQLite.

    Returns list of {"time": "YYYY-MM-DD", "value": float} dicts.

    Supported series:
      spy_close, sma_20, sma_50, rsi_14, macd, macd_signal,
      bb_upper, bb_lower, bb_mid, atr_14, vix,
      prediction_direction, prediction_confidence
    """
    db_path = os.path.join(DATA_DIR, "spy.db")
    if not os.path.exists(db_path):
        return []

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Map series name to table.column
    series_map = {
        "spy_close": ("prices", "close"),
        "sma_20": ("technicals", "sma_20"),
        "sma_50": ("technicals", "sma_50"),
        "rsi_14": ("technicals", "rsi_14"),
        "macd": ("technicals", "macd"),
        "macd_signal": ("technicals", "macd_signal"),
        "bb_upper": ("technicals", "bb_upper"),
        "bb_lower": ("technicals", "bb_lower"),
        "bb_mid": ("technicals", "bb_mid"),
        "atr_14": ("technicals", "atr_14"),
        "vix": ("macro", "vix"),
        "prediction_direction": ("predictions", "direction"),
        "prediction_confidence": ("predictions", "confidence"),
    }

    if series not in series_map:
        return []

    table, column = series_map[series]

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        rows = conn.execute(
            f"SELECT date, {column} FROM {table} WHERE date >= ? ORDER BY date ASC",
            (cutoff,),
        ).fetchall()
        conn.close()

        results = []
        for date_str, val in rows:
            if val is None:
                continue
            if series == "prediction_direction":
                val = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}.get(
                    str(val).upper(), 0
                )
            results.append({"time": date_str + "T00:00:00Z", "value": float(val)})
        return results
    except Exception as e:
        logger.debug(f"Historical query error ({series}): {e}")
        return []


def build_grafana_dataframe(series_list, days=365):
    """Build a Grafana-compatible JSON response with multiple series.

    Returns a flat array of objects with 'time' and one value column per series.
    This format works directly with the JSON API plugin's JSONPath: $[*].time, $[*].value
    """
    # For single series, return simple array
    if len(series_list) == 1:
        return query_historical(series_list[0], days)

    # For multiple series, merge on date
    all_data = {}
    for s in series_list:
        for point in query_historical(s, days):
            t = point["time"]
            if t not in all_data:
                all_data[t] = {"time": t}
            all_data[t][s] = point["value"]

    return sorted(all_data.values(), key=lambda x: x["time"])


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics, /api/history, and /grafana-proxy endpoints."""

    # Grafana proxy config
    GRAFANA_URL = os.environ.get("GRAFANA_INTERNAL_URL", "http://localhost:3001")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token")
        self.end_headers()
        self.wfile.write(body)

    def _get_session_secret(self):
        """Get session secret for token verification."""
        config = _load_config()
        return os.environ.get(
            "SESSION_SECRET",
            config.get("auth", {}).get("session_secret", "stockanalysis-default-secret"),
        )

    def _verify_token(self):
        """Verify auth token from query param or header. Returns email or None."""
        # Check query param first (for iframe URLs)
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        token = ""
        if "auth_token" in params:
            token = params["auth_token"][0]
        elif self.headers.get("X-Auth-Token"):
            token = self.headers["X-Auth-Token"]

        if not token:
            return None

        secret = self._get_session_secret()
        # Inline token verification (avoid importing from src.auth in this context)
        import base64
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None
            payload_b64, sig = parts
            import hmac as _hmac
            import hashlib as _hashlib
            expected = _hmac.new(
                secret.encode(), payload_b64.encode(), _hashlib.sha256
            ).hexdigest()[:32]
            if not _hmac.compare_digest(sig, expected):
                return None
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            if payload.get("exp", 0) < time.time():
                return None
            return payload.get("email")
        except Exception:
            return None

    def _proxy_to_grafana(self, method="GET", body=None):
        """Proxy request to Grafana with X-WEBAUTH-USER header."""
        email = self._verify_token()
        if not email:
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return

        # Strip /grafana-proxy prefix and auth_token param from URL
        parsed = urllib.parse.urlparse(self.path)
        grafana_path = parsed.path.replace("/grafana-proxy", "", 1) or "/"
        # Remove auth_token from query params before forwarding
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        params.pop("auth_token", None)
        query = urllib.parse.urlencode(params, doseq=True)
        target_url = f"{self.GRAFANA_URL}{grafana_path}"
        if query:
            target_url += f"?{query}"

        # Forward headers, add auth
        headers = {
            "X-WEBAUTH-USER": email,
        }
        # Copy select headers from original request
        for h in ["Accept", "Content-Type", "Accept-Encoding"]:
            if self.headers.get(h):
                headers[h] = self.headers[h]

        try:
            resp = requests.request(
                method, target_url, headers=headers, data=body,
                timeout=30, allow_redirects=False, stream=True,
            )

            self.send_response(resp.status_code)
            # Forward response headers
            skip_headers = {"transfer-encoding", "connection", "content-encoding"}
            for key, val in resp.headers.items():
                if key.lower() not in skip_headers:
                    self.send_header(key, val)
            # Allow iframe embedding
            self.send_header("X-Frame-Options", "ALLOWALL")
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # Grafana proxy
        if path.startswith("/grafana-proxy"):
            self._proxy_to_grafana("GET")
            return

        if path == "/metrics" or path == "/":
            metrics = generate_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(metrics.encode("utf-8"))

        elif path == "/health":
            self._send_json({"status": "ok"})

        elif path == "/api/history":
            series_str = params.get("series", ["spy_close"])[0]
            days = int(params.get("days", ["365"])[0])
            series_list = [s.strip() for s in series_str.split(",") if s.strip()]
            frames = build_grafana_dataframe(series_list, days)
            self._send_json(frames)

        elif path == "/api/series":
            available = [
                "spy_close", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal",
                "bb_upper", "bb_lower", "bb_mid", "atr_14", "vix",
                "prediction_direction", "prediction_confidence",
            ]
            self._send_json({"series": available})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST for Grafana JSON API datasource and proxy."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        if path.startswith("/grafana-proxy"):
            self._proxy_to_grafana("POST", body)
            return

        if path == "/api/history":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            series_list = payload.get("series", ["spy_close"])
            days = payload.get("days", 365)
            if isinstance(series_list, str):
                series_list = [s.strip() for s in series_list.split(",")]
            frames = build_grafana_dataframe(series_list, days)
            self._send_json(frames)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server(host="0.0.0.0", port=9190):
    """Start the metrics exporter HTTP server."""
    server = HTTPServer((host, port), MetricsHandler)
    logger.info(f"Prometheus metrics exporter running on http://{host}:{port}/metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_server()
