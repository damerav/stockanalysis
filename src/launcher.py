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

logger = logging.getLogger(__name__)

# Pipeline schedule: 4:30 PM ET (16:30), Mon-Fri
PIPELINE_HOUR = 16
PIPELINE_MINUTE = 30
SCHEDULE_CHECK_INTERVAL = 60  # seconds


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

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="scheduler")
        self._thread.start()
        logger.info(f"Scheduler started — pipeline at {PIPELINE_HOUR}:{PIPELINE_MINUTE:02d} ET (Mon-Fri)")

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
                    if (now.hour == PIPELINE_HOUR and
                            now.minute >= PIPELINE_MINUTE and
                            now.minute < PIPELINE_MINUTE + 2 and
                            today != self._last_run_date):
                        self._last_run_date = today
                        self._run_pipeline()
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


class SystemLauncher:
    """Main launcher that orchestrates all system components."""

    def __init__(self, config: dict):
        self.config = config
        self.pm = ProcessManager()
        self.scheduler = Scheduler(config)
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
        """Start the unified Streamlit dashboard."""
        self.pm.start("dashboard", [
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.address", "0.0.0.0",
        ])

    def _monitor(self):
        """Health monitoring loop — watches processes, restarts crashes."""
        logger.info("System running. Press Ctrl+C to stop.")
        try:
            while self._running:
                time.sleep(30)
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
