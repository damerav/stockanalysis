"""6C. ES Strategy Engine — Entry/exit logic, session guards, circuit breaker."""

import logging
from datetime import datetime, time as dtime
from typing import Optional

from src.es_strategy.position import Position, Direction, LotStatus
from src.es_strategy.indicators import RegimeDetector
from src.es_strategy.rl_trail import RLTrailingAgent

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
        from src.strategy import rules_store as rs
        self.C = rs.get_rule("spread", "credit_C", 10.0)
        self.K = rs.get_rule("spread", "strike_K", 6000.0)
        self.max_lots = rs.get_rule("sizing", "max_lots", 3)
        self.jump_exit_pts = rs.get_rule("risk", "jump_exit_points", 5.0)
        self.emergency_stop_pct = rs.get_rule("risk", "emergency_stop_pct", 0.20)
        self.circuit_breaker_usd = rs.get_rule("risk", "circuit_breaker_usd", -2000.0)
        self.session_close_ct = rs.get_rule("session", "session_close_ct", "15:55")
        self.session_reset_ct = rs.get_rule("session", "session_reset_ct", "17:00")
        self.ai_enabled = rs.get_rule("ai", "ai_enabled", True)

        self.position = Position()
        self.regime_detector = RegimeDetector(
            lookback=rs.get_rule("regime", "lookback_minutes", 10080),
            pct_low=rs.get_rule("regime", "pct_low", 33),
            pct_high=rs.get_rule("regime", "pct_high", 66),
        )

        self.signals: list[Signal] = []
        self.circuit_breaker_active = False
        self._bars_since_entry = 0
        self._phase2_enabled = rs.get_rule("entry", "phase2_enabled", True)
        self._phase2_min_filters = rs.get_rule("entry", "phase2_min_filters", 2)
        self._phase2_roc_threshold = rs.get_rule("entry", "phase2_roc_threshold", 0.5)
        self._anti_chase_atr_pct = rs.get_rule("entry", "anti_chase_atr_pct", 0.5)

        # AI exit confidence settings
        self._trail_ai_enabled = rs.get_rule("ai", "trail_ai_enabled", True)
        self._ai_trail_mults = None  # set externally by runner when AI provides them

        # TP multipliers by regime
        self._tp_multipliers = {
            "Low": {
                "tp1": rs.get_rule("tp_low", "tp1_mult", 1.0),
                "tp2": rs.get_rule("tp_low", "tp2_mult", 1.5),
                "runner_trail": rs.get_rule("tp_low", "runner_trail_mult", 2.0),
            },
            "Med": {
                "tp1": rs.get_rule("tp_med", "tp1_mult", 1.2),
                "tp2": rs.get_rule("tp_med", "tp2_mult", 1.8),
                "runner_trail": rs.get_rule("tp_med", "runner_trail_mult", 2.5),
            },
            "High": {
                "tp1": rs.get_rule("tp_high", "tp1_mult", 1.5),
                "tp2": rs.get_rule("tp_high", "tp2_mult", 2.2),
                "runner_trail": rs.get_rule("tp_high", "runner_trail_mult", 3.0),
            },
        }

        # GAP 9: RL trailing stop agent
        self.rl_trail = RLTrailingAgent(
            alpha=rs.get_rule("rl", "rl_alpha", 0.1),
            gamma=rs.get_rule("rl", "rl_gamma", 0.95),
            epsilon=rs.get_rule("rl", "rl_epsilon", 0.1),
            lambda_dd=rs.get_rule("rl", "rl_lambda_dd", 0.5),
        )
        self.rl_trail.load()

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

        # Anti-chase gate: price must be within configurable ATR fraction of KC band
        direction = None
        if price <= kc_lower and abs(price - kc_lower) < self._anti_chase_atr_pct * atr_val:
            direction = Direction.LONG
        elif price >= kc_upper and abs(price - kc_upper) < self._anti_chase_atr_pct * atr_val:
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
        if at_lower and roc_3 < -self._phase2_roc_threshold:
            filters_passed += 1
        elif at_upper and roc_3 > self._phase2_roc_threshold:
            filters_passed += 1

        # Filter 2: ATR regime not extreme
        if self.regime != "High":
            filters_passed += 1

        # Filter 3: VWAP alignment
        if at_lower and price < vwap_val:
            filters_passed += 1
        elif at_upper and price > vwap_val:
            filters_passed += 1

        if filters_passed < self._phase2_min_filters:
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

        # AI dynamic trailing: override TP2/runner multipliers if CNN provided them
        if self._trail_ai_enabled and self._ai_trail_mults:
            ai_m = self._ai_trail_mults
            mults = dict(mults)  # copy to avoid mutating defaults
            if "tp2_trail" in ai_m:
                mults["tp2"] = ai_m["tp2_trail"]
            if "runner_trail" in ai_m:
                mults["runner_trail"] = ai_m["runner_trail"]

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

        # Runner: trailing stop (GAP 9: RL-adjusted)
        runner_trail = mults["runner_trail"] * atr_val
        for lot in self.position.lots:
            if lot.id == 2 and lot.status == LotStatus.ACTIVE:
                # RL agent adjusts the base trail
                rl_adj = self.rl_trail.get_trail_adjustment(
                    regime=regime,
                    atr_pct=indicators.get("atr_regime_pct", 0.5) if isinstance(indicators, dict) else 0.5,
                    unrealized_pnl=self.position.unrealized_pnl,
                    bars_held=self._bars_since_entry,
                    rsi=indicators.get("rsi_14", 50) if isinstance(indicators, dict) else 50,
                    roc=indicators.get("roc_3", 0) if isinstance(indicators, dict) else 0,
                    atr_val=atr_val,
                )
                adjusted_trail = max(runner_trail + rl_adj, 0.5 * atr_val)  # floor at 0.5×ATR
                if self.position.direction == Direction.LONG:
                    new_stop = price - adjusted_trail
                else:
                    new_stop = price + adjusted_trail
                self.position.update_stop(new_stop)

                # Update RL agent with current equity
                equity = self.position.daily_pnl + self.position.unrealized_pnl
                self.rl_trail.update(
                    new_equity=equity, regime=regime,
                    atr_pct=indicators.get("atr_regime_pct", 0.5) if isinstance(indicators, dict) else 0.5,
                    unrealized_pnl=self.position.unrealized_pnl,
                    bars_held=self._bars_since_entry,
                    rsi=indicators.get("rsi_14", 50) if isinstance(indicators, dict) else 50,
                    roc=indicators.get("roc_3", 0) if isinstance(indicators, dict) else 0,
                )

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
            "spread": {"strike_K": self.K, "credit_C": self.C},
            "pnl": {
                "daily": round(self.position.daily_pnl, 2),
                "unrealized": round(self.position.unrealized_pnl, 2),
            },
            "circuit_breaker": "ACTIVE" if self.circuit_breaker_active else "OK",
            "trade_count": self.position.trade_count,
            "signals": [s.to_dict() for s in self.signals[-50:]],
            "ai_enabled": self.ai_enabled,
            "trail_ai_enabled": self._trail_ai_enabled,
            "ai_trail_mults": self._ai_trail_mults,
        }

    def update_spread(self, strike_K: float, credit_C: float):
        """GAP 19: Update spread inputs dynamically (from broker or manual entry).

        Only updates if position is flat to avoid mid-trade parameter changes.
        """
        if not self.position.is_flat:
            logger.warning("Cannot update spread while position is open")
            return False
        self.K = strike_K
        self.C = credit_C
        logger.info(f"Spread updated: K={strike_K}, C={credit_C}")
        return True
    # Mapping from (rule_group, rule_key) → engine attribute path
    _RULE_ATTR_MAP = {
        ("spread", "credit_C"): "C",
        ("spread", "strike_K"): "K",
        ("sizing", "max_lots"): "max_lots",
        ("entry", "anti_chase_atr_pct"): "_anti_chase_atr_pct",
        ("entry", "phase2_enabled"): "_phase2_enabled",
        ("entry", "phase2_min_filters"): "_phase2_min_filters",
        ("entry", "phase2_roc_threshold"): "_phase2_roc_threshold",
        ("risk", "jump_exit_points"): "jump_exit_pts",
        ("risk", "emergency_stop_pct"): "emergency_stop_pct",
        ("risk", "circuit_breaker_usd"): "circuit_breaker_usd",
        ("session", "session_close_ct"): "session_close_ct",
        ("session", "session_reset_ct"): "session_reset_ct",
        ("ai", "ai_enabled"): "ai_enabled",
        ("ai", "trail_ai_enabled"): "_trail_ai_enabled",
    }

    def apply_overrides(self, overrides: dict):
        """Apply rule overrides in-memory without touching the database.

        Args:
            overrides: dict keyed by (group, key) tuples or flat "group.key" strings.
                       Values are the desired parameter values.

        Example:
            engine.apply_overrides({
                ("spread", "credit_C"): 12.0,
                ("tp_high", "tp1_mult"): 1.8,
            })
        """
        for raw_key, value in overrides.items():
            # Normalize key to (group, key) tuple
            if isinstance(raw_key, str) and "." in raw_key:
                group, key = raw_key.split(".", 1)
            elif isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
                group, key = raw_key
            else:
                continue

            # Direct attribute mapping
            attr = self._RULE_ATTR_MAP.get((group, key))
            if attr:
                setattr(self, attr, value)
                continue

            # TP multiplier groups
            if group in ("tp_low", "tp_med", "tp_high"):
                regime_label = {"tp_low": "Low", "tp_med": "Med", "tp_high": "High"}[group]
                if regime_label in self._tp_multipliers:
                    tp_key_map = {"tp1_mult": "tp1", "tp2_mult": "tp2",
                                  "runner_trail_mult": "runner_trail"}
                    mapped = tp_key_map.get(key)
                    if mapped:
                        self._tp_multipliers[regime_label][mapped] = value
                continue

            # Regime detector params
            if group == "regime":
                if key == "lookback_minutes":
                    self.regime_detector.lookback = int(value)
                elif key == "pct_low":
                    self.regime_detector.pct_low = int(value)
                elif key == "pct_high":
                    self.regime_detector.pct_high = int(value)
                continue

            # RL agent params (rebuild agent with new params)
            if group == "rl":
                rl_map = {"rl_alpha": "alpha", "rl_gamma": "gamma",
                          "rl_epsilon": "epsilon", "rl_lambda_dd": "lambda_dd"}
                rl_attr = rl_map.get(key)
                if rl_attr:
                    setattr(self.rl_trail, rl_attr, value)
                continue

            # AI thresholds
            if group == "ai":
                if key == "entry_conf_threshold":
                    # Stored externally in AILayer, not engine — skip for backtest
                    pass
                elif key == "ai_fail_closed":
                    pass  # AILayer config
                continue



    def save_rl_agent(self):
        """Save RL trailing agent Q-table."""
        self.rl_trail.save()
