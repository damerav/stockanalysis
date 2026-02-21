"""1A. Polygon WebSocket Streamer — Real-time SPY trades + SPX options."""

import json
import time
import asyncio
import logging
import threading
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Market hours (ET): 9:30 AM - 4:00 PM
MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MIN = 16, 0
HEARTBEAT_TIMEOUT = 60  # seconds
MAX_RECONNECT_DELAY = 30


class BarAggregator:
    """Aggregates raw ticks into 5-second OHLCV bars."""

    def __init__(self, interval_sec: int = 5):
        self.interval = interval_sec
        self._current_bar = None
        self._bar_start = None

    def _bar_key(self, ts: float) -> float:
        """Round timestamp down to bar boundary."""
        return ts - (ts % self.interval)

    def add_tick(self, price: float, size: int, timestamp_ms: int) -> Optional[dict]:
        """Add a tick. Returns completed bar if interval elapsed, else None."""
        ts = timestamp_ms / 1000.0
        key = self._bar_key(ts)

        if self._bar_start is None or key != self._bar_start:
            completed = self._current_bar
            self._bar_start = key
            self._current_bar = {
                "timestamp": datetime.fromtimestamp(key, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "open": price, "high": price, "low": price, "close": price,
                "volume": size, "vwap_num": price * size, "vwap_den": size,
            }
            return completed

        bar = self._current_bar
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += size
        bar["vwap_num"] += price * size
        bar["vwap_den"] += size
        return None

    def finalize(self) -> Optional[dict]:
        """Return current incomplete bar."""
        return self._current_bar


class OptionsFlowTracker:
    """Detects sweeps and block trades in SPX options flow."""

    def __init__(self, sweep_threshold: float = 50_000, block_threshold: float = 100_000):
        self.sweep_threshold = sweep_threshold
        self.block_threshold = block_threshold
        self._recent_trades: dict[str, list] = defaultdict(list)
        self.alerts: list[dict] = []

    def add_trade(self, symbol: str, price: float, size: int, exchange: str,
                  timestamp_ms: int) -> Optional[dict]:
        """Track an options trade. Returns alert dict if sweep/block detected."""
        notional = price * size * 100  # options are per 100 shares
        ts = timestamp_ms / 1000.0

        # Block trade detection: single trade > threshold
        if notional >= self.block_threshold:
            alert = {
                "type": "BLOCK", "symbol": symbol, "notional": notional,
                "size": size, "price": price, "exchange": exchange,
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "direction": "PUT" if "P" in symbol.upper() else "CALL",
            }
            self.alerts.append(alert)
            logger.info(f"BLOCK: {symbol} ${notional:,.0f} x{size} @ {price}")
            return alert

        # Sweep detection: same option, multiple exchanges, <2 sec, >threshold total
        key = symbol
        self._recent_trades[key].append({
            "price": price, "size": size, "exchange": exchange,
            "ts": ts, "notional": notional,
        })
        # Clean old trades (>2 seconds)
        self._recent_trades[key] = [
            t for t in self._recent_trades[key] if ts - t["ts"] < 2.0
        ]
        trades = self._recent_trades[key]
        exchanges = set(t["exchange"] for t in trades)
        total_notional = sum(t["notional"] for t in trades)

        if len(exchanges) >= 2 and total_notional >= self.sweep_threshold:
            alert = {
                "type": "SWEEP", "symbol": symbol, "notional": total_notional,
                "legs": len(trades), "exchanges": len(exchanges),
                "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "direction": "PUT" if "P" in symbol.upper() else "CALL",
            }
            self.alerts.append(alert)
            self._recent_trades[key] = []  # reset after detection
            logger.info(f"SWEEP: {symbol} ${total_notional:,.0f} ({len(trades)}x across {len(exchanges)} exchanges)")
            return alert

        return None

    def get_put_call_ratio(self) -> Optional[float]:
        """Compute live put/call ratio from recent alerts."""
        if not self.alerts:
            return None
        puts = sum(1 for a in self.alerts if a["direction"] == "PUT")
        calls = sum(1 for a in self.alerts if a["direction"] == "CALL")
        return puts / calls if calls > 0 else None


class PolygonStreamer:
    """Manages WebSocket connections to Polygon for stocks and options."""

    def __init__(self, api_key: str, config: dict = None):
        self.api_key = api_key
        config = config or {}
        self.ws_stocks_url = config.get("ws_stocks_url", "wss://socket.polygon.io/stocks")
        self.ws_options_url = config.get("ws_options_url", "wss://socket.polygon.io/options")
        self.sweep_threshold = config.get("flow_sweep_threshold", 50_000)
        self.block_threshold = config.get("flow_block_threshold", 100_000)

        self.bar_aggregator = BarAggregator(interval_sec=5)
        self.flow_tracker = OptionsFlowTracker(self.sweep_threshold, self.block_threshold)

        self._running = False
        self._stocks_thread: Optional[threading.Thread] = None
        self._options_thread: Optional[threading.Thread] = None
        self._on_bar: Optional[Callable] = None
        self._on_flow_alert: Optional[Callable] = None
        self._last_stock_msg = time.time()
        self._last_option_msg = time.time()

    def set_callbacks(self, on_bar: Callable = None, on_flow_alert: Callable = None):
        """Set callbacks for completed bars and flow alerts."""
        self._on_bar = on_bar
        self._on_flow_alert = on_flow_alert

    def start(self):
        """Start both WebSocket threads."""
        self._running = True
        self._stocks_thread = threading.Thread(target=self._run_stocks, daemon=True, name="ws-stocks")
        self._options_thread = threading.Thread(target=self._run_options, daemon=True, name="ws-options")
        self._stocks_thread.start()
        self._options_thread.start()
        logger.info("WebSocket streamer started (stocks + options threads)")

    def stop(self):
        """Stop both WebSocket threads."""
        self._running = False
        logger.info("WebSocket streamer stopping")

    def _run_stocks(self):
        """Stocks WebSocket thread with auto-reconnect."""
        delay = 1
        while self._running:
            try:
                asyncio.run(self._connect_stocks())
            except Exception as e:
                logger.error(f"Stocks WS error: {e}")
            if self._running:
                logger.info(f"Stocks WS reconnecting in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    def _run_options(self):
        """Options WebSocket thread with auto-reconnect."""
        delay = 1
        while self._running:
            try:
                asyncio.run(self._connect_options())
            except Exception as e:
                logger.error(f"Options WS error: {e}")
            if self._running:
                logger.info(f"Options WS reconnecting in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_DELAY)

    async def _connect_stocks(self):
        """Connect to stocks WebSocket and process SPY trades."""
        import websockets
        async with websockets.connect(self.ws_stocks_url) as ws:
            # Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            auth_resp = await ws.recv()
            logger.info(f"Stocks WS auth: {auth_resp[:100]}")

            # Subscribe to SPY trades
            await ws.send(json.dumps({"action": "subscribe", "params": "T.SPY"}))
            logger.info("Subscribed to T.SPY")

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_TIMEOUT)
                    self._last_stock_msg = time.time()
                    data = json.loads(msg)
                    if isinstance(data, list):
                        for tick in data:
                            if tick.get("ev") == "T":
                                bar = self.bar_aggregator.add_tick(
                                    price=tick["p"], size=tick["s"], timestamp_ms=tick["t"]
                                )
                                if bar and self._on_bar:
                                    bar["vwap"] = bar["vwap_num"] / bar["vwap_den"] if bar["vwap_den"] else 0
                                    self._on_bar(bar)
                except asyncio.TimeoutError:
                    logger.warning("Stocks WS heartbeat timeout, reconnecting")
                    return

    async def _connect_options(self):
        """Connect to options WebSocket and track SPX options flow."""
        import websockets
        async with websockets.connect(self.ws_options_url) as ws:
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            auth_resp = await ws.recv()
            logger.info(f"Options WS auth: {auth_resp[:100]}")

            await ws.send(json.dumps({"action": "subscribe", "params": "T.O:SPX*"}))
            logger.info("Subscribed to T.O:SPX*")

            while self._running:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=HEARTBEAT_TIMEOUT)
                    self._last_option_msg = time.time()
                    data = json.loads(msg)
                    if isinstance(data, list):
                        for tick in data:
                            if tick.get("ev") == "T":
                                alert = self.flow_tracker.add_trade(
                                    symbol=tick.get("sym", ""),
                                    price=tick.get("p", 0),
                                    size=tick.get("s", 0),
                                    exchange=str(tick.get("x", "")),
                                    timestamp_ms=tick.get("t", 0),
                                )
                                if alert and self._on_flow_alert:
                                    self._on_flow_alert(alert)
                except asyncio.TimeoutError:
                    logger.warning("Options WS heartbeat timeout, reconnecting")
                    return

    @property
    def is_stocks_alive(self) -> bool:
        return time.time() - self._last_stock_msg < HEARTBEAT_TIMEOUT * 2

    @property
    def is_options_alive(self) -> bool:
        return time.time() - self._last_option_msg < HEARTBEAT_TIMEOUT * 2
