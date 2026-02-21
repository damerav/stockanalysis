"""GAP 9: Q-Learning Trailing Stop Agent.

Action: ±0.1×ATR adjustment to trailing stop.
Reward: ΔEquity − λ×Drawdown.
State: [regime_idx, atr_pct, unrealized_pnl_norm, bars_held, rsi_norm, roc_norm].

This is a tabular Q-learning agent with discretized state space.
"""

import logging
import os
import json
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Actions: tighten, hold, widen
ACTIONS = [-0.1, 0.0, 0.1]  # multiplier of ATR to adjust trail
N_ACTIONS = len(ACTIONS)

# State discretization bins
REGIME_MAP = {"Low": 0, "Med": 1, "High": 2}
ATR_BINS = 5
PNL_BINS = 5
BARS_BINS = 5
RSI_BINS = 5
ROC_BINS = 5

STATE_DIMS = (3, ATR_BINS, PNL_BINS, BARS_BINS, RSI_BINS, ROC_BINS)
TOTAL_STATES = 3 * ATR_BINS * PNL_BINS * BARS_BINS * RSI_BINS * ROC_BINS


class RLTrailingAgent:
    """Q-learning agent for adaptive trailing stop adjustment."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.1, lambda_dd: float = 0.5):
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate
        self.lambda_dd = lambda_dd  # drawdown penalty weight

        # Q-table: states × actions
        self.q_table = np.zeros((TOTAL_STATES, N_ACTIONS))
        self._prev_state = None
        self._prev_action = None
        self._prev_equity = 0.0
        self._peak_equity = 0.0
        self._model_path = "./models/rl_trail_qtable.json"

    def discretize_state(self, regime: str, atr_pct: float,
                         unrealized_pnl: float, bars_held: int,
                         rsi: float, roc: float) -> int:
        """Convert continuous state to discrete index."""
        r = REGIME_MAP.get(regime, 1)
        a = min(int(np.clip(atr_pct * ATR_BINS, 0, ATR_BINS - 1)), ATR_BINS - 1)
        p = min(int(np.clip((unrealized_pnl + 500) / 200, 0, PNL_BINS - 1)), PNL_BINS - 1)
        b = min(int(bars_held / 10), BARS_BINS - 1)
        rs = min(int(rsi / 20), RSI_BINS - 1)
        rc = min(int(np.clip((roc + 2) / 0.8, 0, ROC_BINS - 1)), ROC_BINS - 1)

        idx = (r * ATR_BINS * PNL_BINS * BARS_BINS * RSI_BINS * ROC_BINS +
               a * PNL_BINS * BARS_BINS * RSI_BINS * ROC_BINS +
               p * BARS_BINS * RSI_BINS * ROC_BINS +
               b * RSI_BINS * ROC_BINS +
               rs * ROC_BINS +
               rc)
        return min(idx, TOTAL_STATES - 1)

    def select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_table[state_idx]))

    def get_trail_adjustment(self, regime: str, atr_pct: float,
                             unrealized_pnl: float, bars_held: int,
                             rsi: float, roc: float, atr_val: float) -> float:
        """Get trailing stop adjustment in points.

        Returns: adjustment in points (positive = widen, negative = tighten).
        """
        state_idx = self.discretize_state(regime, atr_pct, unrealized_pnl,
                                           bars_held, rsi, roc)
        action_idx = self.select_action(state_idx)
        adjustment = ACTIONS[action_idx] * atr_val

        self._prev_state = state_idx
        self._prev_action = action_idx
        return adjustment

    def update(self, new_equity: float, regime: str, atr_pct: float,
               unrealized_pnl: float, bars_held: int,
               rsi: float, roc: float):
        """Update Q-table with reward from last action.

        Reward = ΔEquity − λ × Drawdown
        """
        if self._prev_state is None:
            self._prev_equity = new_equity
            self._peak_equity = new_equity
            return

        # Compute reward
        delta_equity = new_equity - self._prev_equity
        self._peak_equity = max(self._peak_equity, new_equity)
        drawdown = self._peak_equity - new_equity
        reward = delta_equity - self.lambda_dd * drawdown

        # New state
        new_state = self.discretize_state(regime, atr_pct, unrealized_pnl,
                                           bars_held, rsi, roc)

        # Q-learning update
        old_q = self.q_table[self._prev_state, self._prev_action]
        best_next = np.max(self.q_table[new_state])
        new_q = old_q + self.alpha * (reward + self.gamma * best_next - old_q)
        self.q_table[self._prev_state, self._prev_action] = new_q

        self._prev_equity = new_equity

    def save(self, path: str = None):
        """Save Q-table to JSON."""
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table": self.q_table.tolist(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "lambda_dd": self.lambda_dd,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"RL trail agent saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load Q-table from JSON."""
        path = path or self._model_path
        try:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                self.q_table = np.array(data["q_table"])
                self.alpha = data.get("alpha", self.alpha)
                self.gamma = data.get("gamma", self.gamma)
                self.epsilon = data.get("epsilon", self.epsilon)
                self.lambda_dd = data.get("lambda_dd", self.lambda_dd)
                logger.info(f"RL trail agent loaded from {path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")
        return False
