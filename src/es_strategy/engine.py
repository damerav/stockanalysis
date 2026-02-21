"""6C. ES Strategy Engine — Entry/exit logic, session guards, circuit breaker."""

import logging
from datetime import datetime, time as dtime
from typing import Optional

from src.es_strategy.position import Position, Direction, LotStatus
from src.es_strategy.indicators import RegimeDetector

logger = logging.getLogger(__name__)


class Signal:
    """Represents a strategy signal."""
    def __init__(self, sig_type: str, detail: str = "", timestamp: str = ""):
        self.type = sig_type
        self.detail = detail
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {"type": self.type, "detail": self.detail, "timestamp": self.timestamp}


class ESStrategyEngine:
    """ES Futures strategy: Keltner Channel entries, 3-lot tiered exits."""

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("es_strategy", {})
        self.C = cfg.get("credit_C", 10.0)
        self.K = cfg.get("strike_K", 6000.0)
        self.max_lots = cfg.get("max_lots", 3)
        self.jump_exit_pts = cfg.get("jump_exit_points", 5.0)
        self.emergency_stop_pct = cfg.get("emergency_stop_pct", 0.20)
        self.circuit_breaker_usd = cfg.get("circuit_breaker_usd", -2000.0)
        self.session_close_ct = cfg.get("session_close_ct", "15:55")
        self.session_reset_ct = cfg.get("session_reset_ct", "17:00")
        self.ai_enabled = cfg.get("ai_enabled", False)

        self.position = Position()
        self.regime_detector = RegimeDetector(
            lookback=cfg.get("regime_lookback", 10080),
            pct_low=cfg.get("regime_pct_low", 33),
            pct_high=cfg.get("regime_pct_high", 66),
        )

        self.signals: list[Signal] = []
        self.circuit_breaker_active = False
        self._bars_since_entry = 0
        self._phase2_enabled = True

        # TP multipliers by regime
        self._tp_multipliers = {
            "Low":  {"tp1": 1.0, "tp2": 1.5, "runner_trail": 2.0},
            "Med":  {"tp1": 1.2, "tp2": 1.8, "runner_trail": 2.5},
            "High": {"tp1": 1.5, "tp2": 2.2, "runner_trail": 3.0},
        }

    @property
    def regime(self) -> str:
        return self.regime_detector.regime

    def _emit(self, sig_type: str, detail: str = "", timestamp: str = ""):
        sig = Signal(sig_type, detail, timestamp)
        self.signals.append(sig)
        logger.info(f"SIGNAL: {sig_type} {detail}")
        return sig

    # --- Entry Logic ---

    def _check_entry(self, bar: dict, indicators: dict) -> Optional[Signal]:
        """Phase 1: Pure-edge entry at K ± C."""
        if not self.position.is_flat:
            return None
        if self.circuit_breaker_active:
            return None

        price = bar["close"]
        kc_upper = indicators.get("kc_upper", 0)
        kc_lower = indicators.get("kc_lower", 0)
        atr_val = indicators.get("atr_14", 1)
        ts = bar.get("timestamp", "")

        # Anti-chase gate: price must be within 0.5 ATR of KC band
        direction = None
        if price <= kc_lower and abs(price - kc_lower) < 0.5 * atr_val:
            direction = Direction.LONG
        elif price >= kc_upper and abs(price - kc_upper) < 0.5 * atr_val:
            direction = Direction.SHORT

        if direction is None:
            return None

        # Emergency stop at 20% × C
        stop_dist = self.C * self.emergency_stop_pct
        if direction == Direction.LONG:
            stop = price - stop_dist
        else:
            stop = price + stop_dist

        num_lots = self.max_lots
        self.position.enter(direction, price, num_lots, stop, ts)
        self._bars_since_entry = 0

        return self._emit(
            f"ENTRY_{direction.value}",
            f"{num_lots} lots @ {price:.2f} stop={stop:.2f}",
            ts,
        )

    def _check_phase2_entry(self, bar: dict, indicators: dict) -> Optional[Signal]:
        """Phase 2: Confluence reload — K±C + 2/3 filters."""
        if not self.position.is_flat or not self._phase2_enabled:
            return None
        if self.circuit_breaker_active:
            return None

        price = bar["close"]
        kc_upper = indicators.get("kc_upper", 0)
        kc_lower = indicators.get("kc_lower", 0)
        roc_3 = indicators.get("roc_3", 0)
        atr_val = indicators.get("atr_14", 1)
        vwap_val = indicators.get("vwap", price)
        ts = bar.get("timestamp", "")

        # Must be at KC band
        at_lower = price <= kc_lower
        at_upper = price >= kc_upper

        if not (at_lower or at_upper):
            return None

        # Confluence filters (need 2 of 3)
        filters_passed = 0

        # Filter 1: ROC confirms direction
        if at_lower and roc_3 < -0.5:
            filters_passed += 1
        elif at_upper and roc_3 > 0.5:
            filters_passed += 1

        # Filter 2: ATR regime not extreme
        if self.regime != "High":
            filters_passed += 1

        # Filter 3: VWAP alignment
        if at_lower and price < vwap_val:
            filters_passed += 1
        elif at_upper and price > vwap_val:
            filters_passed += 1

        if filters_passed < 2:
            return None

        direction = Direction.LONG if at_lower else Direction.SHORT
        stop_dist = self.C * self.emergency_stop_pct
        stop = price - stop_dist if direction == Direction.LONG else price + stop_dist

        self.position.enter(direction, price, self.max_lots, stop, ts)
        self._bars_since_entry = 0

        return self._emit(
            f"ENTRY_{direction.value}",
            f"Phase2 {self.max_lots} lots @ {price:.2f} (filters={filters_passed}/3)",
            ts,
        )

    # --- Exit Logic ---

    def _check_exits(self, bar: dict, indicators: dict) -> list[Signal]:
        """Check all exit conditions for active position."""
        if self.position.is_flat:
            return []

        signals = []
        price = bar["close"]
        atr_val = indicators.get("atr_14", 1)
        ts = bar.get("timestamp", "")
        regime = self.regime
        mults = self._tp_multipliers.get(regime, self._tp_multipliers["Med"])

        self.position.update_unrealized(price)
        self._bars_since_entry += 1

        # Emergency stop check
        if self.position.check_stop(price):
            pnl = self.position.close_all(price, "STOP_HIT")
            signals.append(self._emit("STOP_HIT", f"P&L=${pnl:+,.0f}", ts))
            return signals

        # Jump exit: 5 pts adverse during first minute (1 bar at 1-min)
        if self._bars_since_entry <= 1:
            adverse = self._calc_adverse(price)
            if adverse >= self.jump_exit_pts:
                pnl = self.position.close_all(price, "JUMP_EXIT")
                signals.append(self._emit("JUMP_EXIT", f"${pnl:+,.0f} ({adverse:.1f}pts)", ts))
                return signals

        # TP1: First lot target
        tp1_target = self._calc_tp(mults["tp1"], atr_val)
        for lot in self.position.lots:
            if lot.id == 0 and lot.status == LotStatus.ACTIVE:
                if self._hit_target(price, tp1_target):
                    pnl = self.position.close_lot(0, price, "TP1")
                    if pnl is not None:
                        signals.append(self._emit("EXIT_TP1", f"Lot 0 ${pnl:+,.0f}", ts))
                        # Ratchet stop to breakeven
                        self.position.update_stop(self.position.entry_price)
                        signals.append(self._emit("STOP_UPDATE", "Ratchet to breakeven", ts))

        # TP2: Second lot target
        tp2_target = self._calc_tp(mults["tp2"], atr_val)
        for lot in self.position.lots:
            if lot.id == 1 and lot.status == LotStatus.ACTIVE:
                if self._hit_target(price, tp2_target):
                    pnl = self.position.close_lot(1, price, "TP2")
                    if pnl is not None:
                        signals.append(self._emit("EXIT_TP2", f"Lot 1 ${pnl:+,.0f}", ts))
                        # Ratchet stop to TP1 level
                        self.position.update_stop(tp1_target if self.position.direction == Direction.LONG
                                                  else self.position.entry_price - (tp1_target - self.position.entry_price))

        # Runner: trailing stop
        runner_trail = mults["runner_trail"] * atr_val
        for lot in self.position.lots:
            if lot.id == 2 and lot.status == LotStatus.ACTIVE:
                if self.position.direction == Direction.LONG:
                    new_stop = price - runner_trail
                else:
                    new_stop = price + runner_trail
                self.position.update_stop(new_stop)

        return signals

    def _calc_tp(self, multiplier: float, atr_val: float) -> float:
        """Calculate target price based on entry + multiplier × ATR."""
        if self.position.direction == Direction.LONG:
            return self.position.entry_price + multiplier * atr_val
        else:
            return self.position.entry_price - multiplier * atr_val

    def _hit_target(self, price: float, target: float) -> bool:
        if self.position.direction == Direction.LONG:
            return price >= target
        else:
            return price <= target

    def _calc_adverse(self, price: float) -> float:
        """Calculate adverse movement in points."""
        if self.position.direction == Direction.LONG:
            return max(0, self.position.entry_price - price)
        else:
            return max(0, price - self.position.entry_price)

    # --- Session Guards ---

    def _check_session_guards(self, bar: dict) -> Optional[Signal]:
        """Flatten before session close, flatten before events."""
        ts = bar.get("timestamp", "")
        if not ts:
            return None

        try:
            bar_time = datetime.fromisoformat(ts) if "T" in ts else datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            current_time = bar_time.time()
        except (ValueError, TypeError):
            return None

        # Parse session close time
        close_parts = self.session_close_ct.split(":")
        close_time = dtime(int(close_parts[0]), int(close_parts[1]))

        # Flatten before session close
        if current_time >= close_time and not self.position.is_flat:
            pnl = self.position.close_all(bar["close"], "SESSION_FLATTEN")
            return self._emit("SESSION_FLATTEN", f"P&L=${pnl:+,.0f}", ts)

        # Reset at session reset time
        reset_parts = self.session_reset_ct.split(":")
        reset_time = dtime(int(reset_parts[0]), int(reset_parts[1]))
        if current_time >= reset_time:
            self.circuit_breaker_active = False
            self.position.reset_daily()

        return None

    def _check_circuit_breaker(self, bar: dict) -> Optional[Signal]:
        """Circuit breaker: daily P&L ≤ threshold → flatten + disable."""
        if self.circuit_breaker_active:
            return None

        if self.position.daily_pnl <= self.circuit_breaker_usd:
            ts = bar.get("timestamp", "")
            if not self.position.is_flat:
                pnl = self.position.close_all(bar["close"], "CIRCUIT_BREAKER")
            self.circuit_breaker_active = True
            return self._emit(
                "CIRCUIT_BREAKER",
                f"Daily P&L ${self.position.daily_pnl:+,.0f} hit limit",
                ts,
            )
        return None

    # --- Main Processing ---

    def process_bar(self, bar: dict, indicators: dict) -> list[Signal]:
        """Process a single bar through the full strategy logic.

        Args:
            bar: dict with timestamp, open, high, low, close, volume
            indicators: dict with kc_upper, kc_lower, kc_mid, atr_14, rsi_14, roc_3, vwap

        Returns:
            List of signals generated
        """
        signals = []

        # Update regime
        import pandas as pd
        atr_val = indicators.get("atr_14", 0)
        atr_hist = indicators.get("atr_history", pd.Series(dtype=float))
        self.regime_detector.update(atr_val, atr_hist)

        # Session guards
        guard = self._check_session_guards(bar)
        if guard:
            signals.append(guard)
            return signals

        # Circuit breaker
        cb = self._check_circuit_breaker(bar)
        if cb:
            signals.append(cb)
            return signals

        # Exit checks (before entry)
        exit_sigs = self._check_exits(bar, indicators)
        signals.extend(exit_sigs)

        # Entry checks
        entry = self._check_entry(bar, indicators)
        if entry:
            signals.append(entry)
        elif self.position.is_flat:
            entry2 = self._check_phase2_entry(bar, indicators)
            if entry2:
                signals.append(entry2)

        return signals

    def get_state(self) -> dict:
        """Get full engine state for dashboard."""
        return {
            "position": self.position.to_dict(),
            "regime": self.regime,
            "pnl": {
                "daily": round(self.position.daily_pnl, 2),
                "unrealized": round(self.position.unrealized_pnl, 2),
            },
            "circuit_breaker": "ACTIVE" if self.circuit_breaker_active else "OK",
            "trade_count": self.position.trade_count,
            "signals": [s.to_dict() for s in self.signals[-50:]],
        }
