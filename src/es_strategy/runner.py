"""6D. ES Strategy Runner — Live, Backtest, and Paper trading modes.

Usage:
    python -m src.es_strategy.runner --mode backtest --data es_1min.csv
    python -m src.es_strategy.runner --mode paper
    python -m src.es_strategy.runner --mode live
    python -m src.es_strategy.runner --mode live --ai
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.es_strategy.engine import ESStrategyEngine
from src.es_strategy.indicators import compute_bar_indicators
from src.es_strategy.ai_models import ESEntryGate, ESExitController, DriftMonitor
from src.realtime.dashboard_bridge import write_es_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config, return empty dict on failure."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.warning(f"Could not load {path}, using defaults")
        return {}


def parse_timestamp(ts_raw) -> str:
    """Normalise various timestamp formats to ISO string."""
    if isinstance(ts_raw, (int, float)):
        # Polygon-style epoch millis or seconds
        if ts_raw > 1e12:
            ts_raw = ts_raw / 1000
        return datetime.fromtimestamp(ts_raw, tz=timezone.utc).isoformat()
    return str(ts_raw)


# ---------------------------------------------------------------------------
# CSV / DataFrame bar loading
# ---------------------------------------------------------------------------

def load_csv_bars(filepath: str) -> pd.DataFrame:
    """Load 1-min bars from CSV.

    Expected columns: timestamp (or datetime/date), open, high, low, close, volume
    """
    df = pd.read_csv(filepath)

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect timestamp column
    ts_col = None
    for candidate in ("timestamp", "datetime", "date", "time"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {filepath}. "
                         f"Columns: {list(df.columns)}")
    if ts_col != "timestamp":
        df.rename(columns={ts_col: "timestamp"}, inplace=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {filepath}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df["volume"] = df["volume"].fillna(0)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# AI wrapper
# ---------------------------------------------------------------------------

class AILayer:
    """Optional AI entry gate + exit controller + drift monitor."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.entry_gate: Optional[ESEntryGate] = None
        self.exit_ctrl: Optional[ESExitController] = None
        self.drift_mon: Optional[DriftMonitor] = None
        self._bar_window: list = []
        self._window_size = 20

        if enabled:
            self._init_models()

    def _init_models(self):
        self.entry_gate = ESEntryGate()
        if not self.entry_gate.load():
            logger.warning("ES entry gate model not found — AI entry disabled")
            self.entry_gate = None

        self.exit_ctrl = ESExitController()
        if not self.exit_ctrl.load():
            logger.info("ES exit CNN not found — using default trail multipliers")
            self.exit_ctrl = None

        self.drift_mon = DriftMonitor()

    def should_enter(self, features: np.ndarray, regime: str) -> dict:
        """Ask AI gate whether to enter. Returns prediction dict."""
        if not self.enabled or self.entry_gate is None:
            return {"should_enter": True, "quantity": 3, "p_enter": 0.0, "ai_used": False}
        result = self.entry_gate.predict(features, regime)
        result["ai_used"] = True
        return result

    def get_trail_multipliers(self, regime: str) -> dict:
        """Ask CNN for trail multipliers. Falls back to defaults."""
        if not self.enabled or self.exit_ctrl is None:
            return self.exit_ctrl.predict(np.zeros((20, 19)), regime) if self.exit_ctrl else {
                "runner_trail": 2.0, "tp2_trail": 1.25, "p_cont_5": 0.5,
            }
        window = np.array(self._bar_window[-self._window_size:])
        if len(window) < self._window_size:
            return {"runner_trail": 2.0, "tp2_trail": 1.25, "p_cont_5": 0.5}
        return self.exit_ctrl.predict(window, regime)

    def push_bar(self, bar_features: np.ndarray):
        """Accumulate bar features for the CNN window."""
        self._bar_window.append(bar_features)
        if len(self._bar_window) > self._window_size * 2:
            self._bar_window = self._bar_window[-self._window_size:]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ESRunner:
    """Runs the ES strategy in live, backtest, or paper mode."""

    def __init__(self, config: dict, mode: str = "backtest", ai_enabled: bool = False):
        self.config = config
        self.mode = mode
        self.engine = ESStrategyEngine(config)
        self.ai = AILayer(enabled=ai_enabled)
        self.results: list[dict] = []
        self._indicator_df = pd.DataFrame()

    # ---- public API ----

    def run_backtest(self, filepath: str) -> dict:
        """Run strategy over a CSV of 1-min bars. Returns summary dict."""
        logger.info(f"Backtest starting: {filepath}")
        df = load_csv_bars(filepath)
        df = compute_bar_indicators(df)

        atr_history = df["atr_14"].dropna()
        total_bars = len(df)

        for idx in range(34, total_bars):  # skip warm-up for indicators
            row = df.iloc[idx]
            bar = {
                "timestamp": str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            indicators = {
                "kc_upper": float(row.get("kc_upper", 0)),
                "kc_lower": float(row.get("kc_lower", 0)),
                "kc_mid": float(row.get("kc_mid", 0)),
                "atr_14": float(row.get("atr_14", 1)),
                "rsi_14": float(row.get("rsi_14", 50)),
                "roc_3": float(row.get("roc_3", 0)),
                "vwap": float(row.get("vwap", row["close"])),
                "ema_9": float(row.get("ema_9", row["close"])),
                "atr_history": atr_history.iloc[:idx],
            }

            signals = self._process_bar(bar, indicators)
            if signals:
                for s in signals:
                    self.results.append({
                        "bar_idx": idx,
                        "timestamp": bar["timestamp"],
                        "price": bar["close"],
                        **s.to_dict(),
                    })

        summary = self._build_summary(total_bars)
        logger.info(f"Backtest complete: {summary['trades']} trades, "
                    f"P&L=${summary['total_pnl']:+,.0f}")
        return summary

    def run_paper(self, config_path: str = "config.yaml"):
        """Paper mode: live data feed, signals logged only (no execution)."""
        logger.info("Paper trading mode — signals will be logged, not executed")
        self._run_live_loop(paper=True)

    def run_live(self, config_path: str = "config.yaml"):
        """Live mode: Polygon feed, signals written to dashboard state."""
        logger.info("Live mode — signals will update dashboard state")
        self._run_live_loop(paper=False)

    # ---- internal ----

    def _process_bar(self, bar: dict, indicators: dict) -> list:
        """Process one bar through engine, optionally with AI layer."""
        # AI entry gate check
        if self.ai.enabled and self.engine.position.is_flat:
            features = self._build_ai_features(bar, indicators)
            self.ai.push_bar(features)
            gate = self.ai.should_enter(features, self.engine.regime)
            if gate.get("ai_used") and not gate.get("should_enter"):
                from src.es_strategy.engine import Signal
                return [Signal("AI_REJECT", f"p={gate['p_enter']:.3f}", bar.get("timestamp", ""))]

        signals = self.engine.process_bar(bar, indicators)
        return signals

    def _build_ai_features(self, bar: dict, indicators: dict) -> np.ndarray:
        """Build the 17-feature vector for the AI entry gate."""
        price = bar["close"]
        kc_mid = indicators.get("kc_mid", price)
        atr_val = indicators.get("atr_14", 1)
        vwap_val = indicators.get("vwap", price)

        features = np.array([
            (price - kc_mid) / atr_val if atr_val else 0,       # price_vs_kc_mid
            (price - vwap_val) / atr_val if atr_val else 0,     # price_vs_vwap
            indicators.get("rsi_14", 50),                        # rsi
            indicators.get("roc_3", 0),                          # roc_3
            0.5,                                                  # atr_regime_pct (placeholder)
            bar.get("volume", 0) / 1000,                         # volume_ratio (normalised)
            (indicators.get("kc_upper", 0) - indicators.get("kc_lower", 0)) / atr_val if atr_val else 0,  # kc_width
            0.0,                                                  # ema9_slope (needs history)
            0.0,                                                  # macd_hist (placeholder)
            0.0,                                                  # bb_width (placeholder)
            0.0,                                                  # momentum_3bar
            0.0,                                                  # momentum_5bar
            0.0,                                                  # bars_since_trade
            self.engine.position.daily_pnl / 1000,               # daily_pnl (normalised)
            0.0,                                                  # time_sin
            0.0,                                                  # time_cos
            0.0,                                                  # spread_vs_atr
        ], dtype=np.float64)
        return features

    def _run_live_loop(self, paper: bool = False):
        """Main loop for live / paper modes — polls Polygon or reads streamer."""
        from src.realtime.dashboard_bridge import write_es_state

        poll_interval = 1.0  # seconds between bar checks
        bar_buffer: list[dict] = []
        indicator_warmup = 35

        logger.info(f"{'Paper' if paper else 'Live'} loop starting. "
                    f"Press Ctrl+C to stop.")

        try:
            while True:
                bar = self._fetch_next_bar(paper)
                if bar is None:
                    time.sleep(poll_interval)
                    continue

                bar_buffer.append(bar)

                # Build DataFrame for indicator computation
                df = pd.DataFrame(bar_buffer)
                if len(df) < indicator_warmup:
                    time.sleep(poll_interval)
                    continue

                df = compute_bar_indicators(df)
                latest = df.iloc[-1]
                atr_history = df["atr_14"].dropna()

                indicators = {
                    "kc_upper": float(latest.get("kc_upper", 0)),
                    "kc_lower": float(latest.get("kc_lower", 0)),
                    "kc_mid": float(latest.get("kc_mid", 0)),
                    "atr_14": float(latest.get("atr_14", 1)),
                    "rsi_14": float(latest.get("rsi_14", 50)),
                    "roc_3": float(latest.get("roc_3", 0)),
                    "vwap": float(latest.get("vwap", bar["close"])),
                    "ema_9": float(latest.get("ema_9", bar["close"])),
                    "atr_history": atr_history,
                }

                signals = self._process_bar(bar, indicators)

                # Log signals
                for s in signals:
                    tag = "[PAPER]" if paper else "[LIVE]"
                    logger.info(f"{tag} {s.type} {s.detail}")
                    self.results.append({
                        "timestamp": bar.get("timestamp", ""),
                        "price": bar["close"],
                        **s.to_dict(),
                    })

                # Update dashboard state
                state = self.engine.get_state()
                chart_bars = bar_buffer[-200:]  # last 200 bars for chart
                write_es_state(
                    position=state["position"],
                    signals=[s.to_dict() for s in signals],
                    regime=state["regime"],
                    pnl=state["pnl"],
                    chart_data={
                        "bars": chart_bars[-60:],
                        "kc_upper": float(latest.get("kc_upper", 0)),
                        "kc_lower": float(latest.get("kc_lower", 0)),
                        "kc_mid": float(latest.get("kc_mid", 0)),
                        "vwap": float(latest.get("vwap", 0)),
                    },
                )

                # Cap buffer to avoid memory growth
                if len(bar_buffer) > 5000:
                    bar_buffer = bar_buffer[-2000:]

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("Runner stopped by user")
        finally:
            # Flatten any open position on shutdown
            if not self.engine.position.is_flat and not paper:
                last_price = bar_buffer[-1]["close"] if bar_buffer else 0
                self.engine.position.close_all(last_price, "SHUTDOWN")
                logger.info("Position flattened on shutdown")

    def _fetch_next_bar(self, paper: bool) -> Optional[dict]:
        """Fetch the next bar from the data source.

        In live/paper mode, reads from the realtime streamer's latest bar file
        or Polygon WebSocket aggregated bars.
        """
        bar_file = os.path.join("data", "es_latest_bar.json")
        try:
            if os.path.exists(bar_file):
                with open(bar_file, "r") as f:
                    bar = json.load(f)
                # Only process if timestamp is new
                ts = bar.get("timestamp", "")
                if hasattr(self, "_last_ts") and ts == self._last_ts:
                    return None
                self._last_ts = ts
                return bar
        except (json.JSONDecodeError, IOError):
            pass
        return None

    def _build_summary(self, total_bars: int) -> dict:
        """Build backtest summary from collected results."""
        trades = [r for r in self.results if r["type"].startswith("ENTRY_")]
        exits = [r for r in self.results if "EXIT" in r["type"] or "STOP" in r["type"]
                 or "FLATTEN" in r["type"] or "CIRCUIT" in r["type"]
                 or "JUMP" in r["type"]]

        total_pnl = self.engine.position.daily_pnl
        # Also sum any closed-trade P&L from signals
        for r in self.results:
            detail = r.get("detail", "")
            if "$" in detail:
                try:
                    pnl_str = detail.split("$")[1].split(" ")[0].replace(",", "").replace("+", "")
                    total_pnl += float(pnl_str)
                except (IndexError, ValueError):
                    pass

        return {
            "mode": "backtest",
            "total_bars": total_bars,
            "trades": len(trades),
            "exits": len(exits),
            "total_pnl": round(total_pnl, 2),
            "signals": self.results,
            "final_state": self.engine.get_state(),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ES Futures Strategy Runner")
    parser.add_argument("--mode", choices=["live", "backtest", "paper"],
                        default="backtest", help="Running mode")
    parser.add_argument("--data", type=str, default=None,
                        help="CSV file path for backtest mode")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--ai", action="store_true",
                        help="Enable AI entry gate + CNN exit controller")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    runner = ESRunner(config, mode=args.mode, ai_enabled=args.ai)

    if args.mode == "backtest":
        if not args.data:
            logger.error("Backtest mode requires --data <csv_file>")
            sys.exit(1)
        if not os.path.exists(args.data):
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
        summary = runner.run_backtest(args.data)
        # Print summary
        print(f"\n{'='*50}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*50}")
        print(f"Bars processed: {summary['total_bars']}")
        print(f"Trades:         {summary['trades']}")
        print(f"Total P&L:      ${summary['total_pnl']:+,.2f}")
        print(f"{'='*50}\n")

        # Save results to JSON
        out_path = args.data.replace(".csv", "_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to {out_path}")

    elif args.mode == "paper":
        runner.run_paper(args.config)

    elif args.mode == "live":
        runner.run_live(args.config)


if __name__ == "__main__":
    main()
