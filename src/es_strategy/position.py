"""6B. Position Manager — 3-lot tiered position tracking."""

import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class LotStatus(Enum):
    ACTIVE = "ACTIVE"
    TP1_FILLED = "TP1_FILLED"
    TP2_FILLED = "TP2_FILLED"
    STOPPED = "STOPPED"
    EXITED = "EXITED"


@dataclass
class Lot:
    id: int
    role: str  # "TP1", "TP2", "Runner"
    entry_price: float = 0.0
    status: LotStatus = LotStatus.ACTIVE
    exit_price: float = 0.0
    pnl: float = 0.0
    stop_price: float = 0.0

    def close(self, price: float, direction: Direction):
        self.exit_price = price
        mult = 1.0 if direction == Direction.LONG else -1.0
        self.pnl = (price - self.entry_price) * mult * 50  # ES = $50/point
        self.status = LotStatus.EXITED


@dataclass
class Position:
    """Manages a 3-lot tiered ES futures position."""

    direction: Direction = Direction.FLAT
    lots: list = field(default_factory=list)
    entry_price: float = 0.0
    entry_time: str = ""
    initial_stop: float = 0.0
    current_stop: float = 0.0
    daily_pnl: float = 0.0
    trade_count: int = 0

    @property
    def is_flat(self) -> bool:
        return self.direction == Direction.FLAT

    @property
    def active_lots(self) -> list[Lot]:
        return [l for l in self.lots if l.status == LotStatus.ACTIVE]

    @property
    def active_count(self) -> int:
        return len(self.active_lots)

    @property
    def unrealized_pnl(self) -> float:
        return sum(l.pnl for l in self.lots)

    def enter(self, direction: Direction, price: float, num_lots: int,
              stop: float, timestamp: str = ""):
        """Open a new position. Must be flat first (no pyramiding)."""
        if not self.is_flat:
            logger.warning("Cannot enter: position already open (no pyramiding)")
            return False

        self.direction = direction
        self.entry_price = price
        self.entry_time = timestamp
        self.initial_stop = stop
        self.current_stop = stop
        self.trade_count += 1

        roles = ["TP1", "TP2", "Runner"]
        self.lots = []
        for i in range(min(num_lots, 3)):
            lot = Lot(id=i, role=roles[i], entry_price=price, stop_price=stop)
            self.lots.append(lot)

        logger.info(f"ENTRY {direction.value} {num_lots} lots @ {price}, stop={stop}")
        return True

    def close_lot(self, lot_id: int, price: float, reason: str = "") -> Optional[float]:
        """Close a specific lot. Returns P&L or None if lot not found."""
        for lot in self.lots:
            if lot.id == lot_id and lot.status == LotStatus.ACTIVE:
                lot.close(price, self.direction)
                logger.info(f"EXIT Lot {lot_id} ({lot.role}) @ {price} "
                           f"P&L=${lot.pnl:+,.0f} [{reason}]")

                # Check if all lots closed
                if not self.active_lots:
                    total_pnl = sum(l.pnl for l in self.lots)
                    self.daily_pnl += total_pnl
                    logger.info(f"Position fully closed. Trade P&L=${total_pnl:+,.0f}")
                    self._reset()

                return lot.pnl
        return None

    def close_all(self, price: float, reason: str = "") -> float:
        """Flatten entire position. Returns total P&L."""
        total = 0.0
        for lot in self.active_lots:
            lot.close(price, self.direction)
            total += lot.pnl
        self.daily_pnl += total
        logger.info(f"FLATTEN ALL @ {price} P&L=${total:+,.0f} [{reason}]")
        self._reset()
        return total

    def update_stop(self, new_stop: float):
        """Ratchet stop (only moves in favorable direction)."""
        if self.direction == Direction.LONG:
            if new_stop > self.current_stop:
                self.current_stop = new_stop
                for lot in self.active_lots:
                    lot.stop_price = new_stop
        elif self.direction == Direction.SHORT:
            if new_stop < self.current_stop:
                self.current_stop = new_stop
                for lot in self.active_lots:
                    lot.stop_price = new_stop

    def update_unrealized(self, current_price: float):
        """Update unrealized P&L for all active lots."""
        mult = 1.0 if self.direction == Direction.LONG else -1.0
        for lot in self.active_lots:
            lot.pnl = (current_price - lot.entry_price) * mult * 50

    def check_stop(self, current_price: float) -> bool:
        """Check if stop is hit. Returns True if stopped out."""
        if self.is_flat:
            return False
        if self.direction == Direction.LONG and current_price <= self.current_stop:
            return True
        if self.direction == Direction.SHORT and current_price >= self.current_stop:
            return True
        return False

    def _reset(self):
        """Reset to flat."""
        self.direction = Direction.FLAT
        self.lots = []
        self.entry_price = 0.0
        self.entry_time = ""
        self.initial_stop = 0.0
        self.current_stop = 0.0

    def reset_daily(self):
        """Reset daily counters (called at session reset)."""
        self.daily_pnl = 0.0
        self.trade_count = 0

    def to_dict(self) -> dict:
        """Serialize for dashboard/JSON."""
        return {
            "status": self.direction.value,
            "lots": self.active_count,
            "entry_price": self.entry_price,
            "stop": self.current_stop,
            "lots_detail": [
                {"id": l.id, "role": l.role, "status": l.status.value,
                 "pnl": round(l.pnl, 2)}
                for l in self.lots
            ],
        }
