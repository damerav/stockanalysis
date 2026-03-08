"""7C. Launcher & Scheduler — Single command to start all components.

Usage:
    python -m src.launcher --all                    # Start everything
    python -m src.launcher --spy                    # SPY predictor only
    python -m src.launcher --es                     # ES strategy only
    python -m src.launcher --dashboards-only        # Dashboards only
    python -m src.launcher --check-llm              # LLM health check only
    python -m src.launcher --pipeline               # Run pipeline once now
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import yaml

from src.realtime.streamer import PolygonStreamer
from src.realtime.dashboard_bridge import read_state
from src.data.db_router import DbRouter

logger = logging.getLogger(__name__)

# Pipeline schedule: 4:30 PM ET (16:30), Mon-Fri
PIPELINE_HOUR = 16
PIPELINE_MINUTE = 30
SCHEDULE_CHECK_INTERVAL = 60  # seconds

# P2: Intraday prediction update times (ET)
INTRADAY_UPDATES = [
    (8, 30),   # Pre-market: after overnight news + 8:30 AM economic releases
    (12, 0),   # Mid-day: after morning session price action
    (13, 30),  # Early afternoon: post-lunch reversal window
    (15, 0),   # Late session: final hour positioning
]


class ProcessManager:
    """Manages child processes for all system components."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self, name: str, cmd: list[str], env: dict = None):
        """Start a named subprocess."""
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        try:
            proc = subprocess.Popen(
                cmd, env=merged_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            with self._lock:
                self.processes[name] = proc
            logger.info(f"Started {name} (PID {proc.pid}): {' '.join(cmd)}")
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")

    def stop(self, name: str):
        """Stop a named subprocess."""
        with self._lock:
            proc = self.processes.pop(name, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            logger.info(f"Stopped {name}")

    def stop_all(self):
        """Stop all managed processes."""
        names = list(self.processes.keys())
        for name in names:
            self.stop(name)

    def health_check(self) -> dict:
        """Check health of all processes. Returns status dict."""
        status = {}
        with self._lock:
            for name, proc in list(self.processes.items()):
                rc = proc.poll()
                if rc is None:
                    status[name] = "running"
                else:
                    status[name] = f"exited ({rc})"
                    logger.warning(f"{name} has exited with code {rc}")
        return status

    def restart_crashed(self):
        """Restart any processes that have crashed."""
        with self._lock:
            crashed = [(n, p) for n, p in self.processes.items() if p.poll() is not None]
        for name, proc in crashed:
            logger.warning(f"Restarting crashed process: {name}")
            cmd = proc.args
            self.stop(name)
            self.start(name, cmd)


class Scheduler:
    """Background scheduler that triggers the daily pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._thread: threading.Thread = None
        self._last_run_date: str = ""
        self._last_intraday_runs: dict = {}  # P2: track intraday updates
        self._last_vigilance_check: float = 0  # P3: event-driven monitoring
        self._vigilance: object = None  # lazy-loaded
        self._last_kb_rebuild_week: str = ""  # weekly knowledge base rebuild

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()
        logger.info(f"Scheduler started — pipeline at {PIPELINE_HOUR}:{PIPELINE_MINUTE:02d} ET (Mon-Fri)")
        logger.info(f"Intraday updates at: {', '.join(f'{h}:{m:02d}' for h, m in INTRADAY_UPDATES)}")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                weekday = now.weekday()  # 0=Mon, 6=Sun

                # Only Mon-Fri
                if weekday < 5:
                    # Full pipeline at 4:30 PM
                    if (now.hour == PIPELINE_HOUR and
                            now.minute >= PIPELINE_MINUTE and
                            now.minute < PIPELINE_MINUTE + 2 and
                            today != self._last_run_date):
                        self._last_run_date = today
                        self._run_pipeline()

                        # Weekly knowledge base rebuild (Mondays after pipeline)
                        iso_week = now.strftime("%G-W%V")
                        if weekday == 0 and iso_week != self._last_kb_rebuild_week:
                            self._last_kb_rebuild_week = iso_week
                            self._rebuild_knowledge_base()

                    # P2: Intraday prediction updates
                    for hour, minute in INTRADAY_UPDATES:
                        key = f"{today}_{hour}:{minute:02d}"
                        if (now.hour == hour and
                                now.minute >= minute and
                                now.minute < minute + 2 and
                                key not in self._last_intraday_runs):
                            self._last_intraday_runs[key] = True
                            self._run_intraday_update(hour, minute)

                # P3: Event-driven vigilance monitoring (every 5 min, market hours)
                self._run_vigilance_check()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            time.sleep(SCHEDULE_CHECK_INTERVAL)

    def _run_pipeline(self):
        """Execute the daily pipeline in a subprocess."""
        logger.info("Scheduler triggering daily pipeline")
        try:
            cmd = [sys.executable, "-m", "src.pipeline.daily_run",
                   "--config", "config.yaml"]
            proc = subprocess.Popen(cmd)
            proc.wait(timeout=7200)  # 2 hour max
            logger.info(f"Pipeline completed with exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.error("Pipeline timed out after 2 hours")
            proc.kill()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")

    def _rebuild_knowledge_base(self):
        """Weekly: re-embed source files into knowledge_vectors for RAG chatbot."""
        logger.info("Weekly knowledge base rebuild starting")
        try:
            cmd = [sys.executable, "-m", "src.data.build_knowledge_base"]
            proc = subprocess.Popen(cmd)
            proc.wait(timeout=300)  # 5 min max
            logger.info(f"Knowledge base rebuild completed (exit {proc.returncode})")
        except subprocess.TimeoutExpired:
            logger.error("Knowledge base rebuild timed out")
            proc.kill()
        except Exception as e:
            logger.error(f"Knowledge base rebuild failed: {e}")

    def _run_intraday_update(self, hour: int, minute: int):
        """P2: Run intraday prediction refresh (steps 3-5, 8-9, 11 only)."""
        label = f"{hour}:{minute:02d}"
        logger.info(f"Intraday prediction update triggered ({label})")
        try:
            cmd = [sys.executable, "-c",
                   "from src.pipeline.daily_run import DailyPipeline; "
                   "import yaml; "
                   "config = yaml.safe_load(open('config.yaml')) or {}; "
                   "p = DailyPipeline(config); "
                   "p._init_components(); "
                   "p._step3_news(); "
                   "p._step4_sentiment(); "
                   "p._step5_macro(); "
                   "p._step8_technicals(); "
                   "p._step9_intraday(); "
                   "p._step11_predict(); "
                   f"print('Intraday update {label} complete')"]
            proc = subprocess.Popen(cmd)
            proc.wait(timeout=600)  # 10 min max
            logger.info(f"Intraday update ({label}) completed: exit {proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.error(f"Intraday update ({label}) timed out")
            proc.kill()
        except Exception as e:
            logger.error(f"Intraday update ({label}) failed: {e}")

    def _run_vigilance_check(self):
        """P3: Event-driven market condition monitoring."""
        now = time.time()
        interval = self.config.get("vigilance", {}).get("check_interval_sec", 300)
        if now - self._last_vigilance_check < interval:
            return

        self._last_vigilance_check = now
        try:
            if self._vigilance is None:
                from src.pipeline.vigilance import VigilanceMonitor
                self._vigilance = VigilanceMonitor(self.config)

            alerts = self._vigilance.check()
            if alerts:
                logger.info(f"Vigilance: {len(alerts)} alert(s) triggered")
                # Fire alerts through existing alert infrastructure
                for alert in alerts:
                    self._send_vigilance_alert(alert)
        except Exception as e:
            logger.debug(f"Vigilance check error: {e}")

    def _send_vigilance_alert(self, alert: dict):
        """Send a vigilance alert through configured channels."""
        try:
            from src.pipeline.alerts import send_alerts
            alert_data = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "direction": f"⚡ {alert['type'].upper()}",
                "confidence": 0,
                "report": alert["message"],
            }
            send_alerts(self.config, alert_data)
        except Exception as e:
            logger.debug(f"Vigilance alert send failed: {e}")


class SystemLauncher:
    """Main launcher that orchestrates all system components."""

    def __init__(self, config: dict):
        self.config = config
        self.pm = ProcessManager()
        self.scheduler = Scheduler(config)
        self.streamer: PolygonStreamer | None = None
        self._running = False

    def start_all(self):
        """Start all components: LLM check, backends, dashboards, scheduler."""
        self._running = True
        logger.info("Starting full system...")

        # Phase 0: LLM health check
        self._check_llm()

        # Start ES strategy runner (paper mode by default)
        ai_flag = ["--ai"] if self.config.get("es_strategy", {}).get("ai_enabled") else []
        self.pm.start("es_strategy", [
            sys.executable, "-m", "src.es_strategy.runner",
            "--mode", "paper", "--config", "config.yaml",
        ] + ai_flag)

        # Start dashboards
        self._start_dashboards()

        # Start scheduler
        self.scheduler.start()

        # Health monitoring loop
        self._monitor()

    def start_spy_only(self):
        """Start SPY predictor components only."""
        self._running = True
        self._check_llm()
        self._start_dashboards(spy_only=True)
        self.scheduler.start()
        self._monitor()

    def start_es_only(self):
        """Start ES strategy components only."""
        self._running = True
        ai_flag = ["--ai"] if self.config.get("es_strategy", {}).get("ai_enabled") else []
        self.pm.start("es_strategy", [
            sys.executable, "-m", "src.es_strategy.runner",
            "--mode", "paper", "--config", "config.yaml",
        ] + ai_flag)
        self._start_dashboards(es_only=True)
        self._monitor()

    def start_dashboards_only(self):
        """Start dashboards without backends."""
        self._running = True
        self._start_dashboards()
        self._monitor()

    def run_pipeline_now(self):
        """Run the daily pipeline immediately (no scheduling)."""
        from src.pipeline.daily_run import DailyPipeline
        pipeline = DailyPipeline(self.config)
        results = pipeline.run()
        return results

    def _check_llm(self):
        """Run LLM health check."""
        from src.llm.analyzer import LLMAnalyzer
        llm = LLMAnalyzer(self.config)
        ok = llm.check_health()
        if ok:
            logger.info("LLM ready")
        else:
            logger.warning("LLM unavailable — continuing without it")

    def _start_dashboards(self, spy_only: bool = False, es_only: bool = False):
        """Start the unified Streamlit dashboard, Confidence API, and Streamer."""
        self._start_streamer()

        self.pm.start("dashboard", [
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.address", "0.0.0.0",
        ])

        # Confidence API (port 8100) — used by ES Strategy AI layer
        self.pm.start("confidence_api", [
            sys.executable, "-m", "src.api.confidence_server",
        ])

    def _start_streamer(self):
        """Start the Polygon.io WebSocket streamer if API key is available."""
        api_key = self.config.get("polygon", {}).get("api_key", "")
        if not api_key or api_key in ("YOUR_POLYGON_KEY", "FROM_ENCRYPTED_DB"):
            try:
                from src.data.secrets_manager import get_secret
                api_key = get_secret("polygon_api_key", fallback="")
            except Exception:
                api_key = ""

        if not api_key:
            logger.warning("Polygon API key not found — WebSocket streamer disabled")
            return

        try:
            self.streamer = PolygonStreamer(api_key, self.config)
            db_router = DbRouter(self.config)

            def on_bar_handler(bar: dict):
                """Write 5-second bars to the database."""
                try:
                    db_router.execute(
                        "INSERT INTO intraday_bars "
                        "(timestamp, ticker, open, high, low, close, volume, vwap) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT (timestamp, ticker) DO UPDATE SET "
                        "close = EXCLUDED.close, "
                        "high = GREATEST(intraday_bars.high, EXCLUDED.high), "
                        "low = LEAST(intraday_bars.low, EXCLUDED.low), "
                        "volume = intraday_bars.volume + EXCLUDED.volume, "
                        "vwap = EXCLUDED.vwap",
                        (bar["timestamp"], "SPY", bar["open"], bar["high"],
                         bar["low"], bar["close"], bar["volume"], bar.get("vwap", 0))
                    )
                except Exception as e:
                    logger.error(f"on_bar DB write failed: {e}")

            def on_flow_alert_handler(alert: dict):
                """Update spy_state.json with new flow alerts."""
                try:
                    import json
                    state_path = "./data/spy_state.json"
                    current = read_state("spy_state.json")
                    alerts = current.get("flow_alerts", [])
                    alerts.append(alert)
                    current["flow_alerts"] = alerts[-20:]
                    current["updated_at"] = datetime.now().isoformat()
                    # Atomic write
                    import tempfile
                    fd, tmp = tempfile.mkstemp(dir="./data", suffix=".tmp")
                    with os.fdopen(fd, "w") as f:
                        json.dump(current, f, default=str)
                    os.replace(tmp, state_path)
                except Exception as e:
                    logger.error(f"on_flow_alert failed: {e}")

            self.streamer.set_callbacks(on_bar=on_bar_handler,
                                        on_flow_alert=on_flow_alert_handler)
            self.streamer.start()
            logger.info("Polygon WebSocket streamer started")
        except Exception as e:
            logger.warning(f"Streamer failed to start: {e}")
            self.streamer = None

    def _update_streamer_state(self):
        """Write streamer health to a JSON file for dashboard consumption."""
        if not self.streamer:
            return
        import json
        state = {
            "updated_at": datetime.now().isoformat(),
            "is_stocks_alive": self.streamer.is_stocks_alive,
            "is_options_alive": self.streamer.is_options_alive,
        }
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir="./data", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, default=str)
            os.replace(tmp, "./data/streamer_state.json")
        except Exception as e:
            logger.debug(f"Failed to write streamer state: {e}")

    def _monitor(self):
        """Health monitoring loop — watches processes, restarts crashes."""
        logger.info("System running. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(30)
                self._update_streamer_state()
                status = self.pm.health_check()
                crashed = [n for n, s in status.items() if "exited" in s]
                if crashed:
                    logger.warning(f"Crashed processes: {crashed}")
                    self.pm.restart_crashed()
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown of all components."""
        self._running = False
        self.scheduler.stop()
        if self.streamer:
            try:
                self.streamer.stop()
                logger.info("Polygon streamer stopped")
            except Exception as e:
                logger.debug(f"Streamer stop error: {e}")
        self.pm.stop_all()
        logger.info("System shutdown complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stock Analysis System Launcher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Start all components")
    group.add_argument("--spy", action="store_true", help="SPY predictor only")
    group.add_argument("--es", action="store_true", help="ES strategy only")
    group.add_argument("--dashboards-only", action="store_true", help="Dashboards only")
    group.add_argument("--check-llm", action="store_true", help="LLM health check only")
    group.add_argument("--pipeline", action="store_true", help="Run pipeline once now")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        logger.warning(f"Could not load {args.config}, using defaults")
        config = {}

    launcher = SystemLauncher(config)

    if args.check_llm:
        launcher._check_llm()
    elif args.pipeline:
        results = launcher.run_pipeline_now()
        print(f"\nPipeline complete. Steps: {len(results)}")
    elif args.all:
        launcher.start_all()
    elif args.spy:
        launcher.start_spy_only()
    elif args.es:
        launcher.start_es_only()
    elif args.dashboards_only:
        launcher.start_dashboards_only()


if __name__ == "__main__":
    main()
