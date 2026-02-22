"""8A. What-If Engine — Parameter sweeps, scenario injection, Monte Carlo.

Runs on DGX Spark (has GPU + data + models). Reuses the ES backtest runner
and XGBoost predictor for all computations.
"""

import copy
import logging
import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from src.data.init_db import get_connection, load_config
from src.data.features import build_feature_vector, get_feature_columns, get_target
from src.model.trainer import SPYPredictor
from src.whatif.presets import STRESS_SCENARIOS
from src.whatif.narrator import WhatIfNarrator

logger = logging.getLogger(__name__)


class WhatIfEngine:
    """Core compute module for What-If analysis across both subsystems."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        wi_cfg = self.config.get("whatif", {})
        self.max_sims = wi_cfg.get("max_sims", 1000)
        self.es_lookback = wi_cfg.get("es_default_lookback_days", 20)
        self.spy_noise_pct = wi_cfg.get("spy_default_noise_pct", 2.0)
        self.narrator = WhatIfNarrator(self.config)
        self._predictor: Optional[SPYPredictor] = None
        self._feature_df: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = get_feature_columns()

    def _get_predictor(self) -> SPYPredictor:
        """Lazy-load the SPY predictor model."""
        if self._predictor is None:
            self._predictor = SPYPredictor(self.config)
            self._predictor.load_latest_model()
        return self._predictor

    def _get_features(self) -> pd.DataFrame:
        """Lazy-load the full feature DataFrame."""
        if self._feature_df is None:
            conn = get_connection(self.config)
            self._feature_df = build_feature_vector(conn, config=self.config)
            conn.close()
        return self._feature_df

    # ------------------------------------------------------------------
    # 8B: ES Strategy What-If
    # ------------------------------------------------------------------

    def es_parameter_sweep(self, params_grid: dict) -> dict:
        """Sweep ES strategy parameters over backtest data.

        Args:
            params_grid: Dict of param_name → list of values to try.
                e.g. {"credit_C": [8, 10, 12], "strike_K": [5950, 6000, 6050]}

        Returns:
            Dict with grid results: {(param_combo): backtest_summary}
        """
        from src.es_strategy.runner import ESRunner

        base_config = copy.deepcopy(self.config)
        results = {}
        keys = list(params_grid.keys())
        combos = self._cartesian(params_grid)

        logger.info(f"ES parameter sweep: {len(combos)} combinations")

        for combo in combos:
            cfg = copy.deepcopy(base_config)
            for key, val in zip(keys, combo):
                cfg.setdefault("es_strategy", {})[key] = val

            label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
            try:
                summary = self._run_es_backtest(cfg)
                results[label] = {
                    "params": dict(zip(keys, combo)),
                    "total_pnl": summary.get("total_pnl", 0),
                    "trades": summary.get("trades", 0),
                }
            except Exception as e:
                logger.warning(f"Sweep failed for {label}: {e}")
                results[label] = {"params": dict(zip(keys, combo)), "error": str(e)}

        return {"type": "es_parameter_sweep", "results": results}

    def es_compare_scenarios(self, scenario_list: list[dict]) -> dict:
        """Side-by-side backtest comparison of ES config variants.

        Args:
            scenario_list: List of dicts, each with "label" and "overrides" keys.
                overrides are applied to es_strategy config section.
        """
        from src.es_strategy.runner import ESRunner

        results = []
        for scenario in scenario_list:
            label = scenario.get("label", "unnamed")
            overrides = scenario.get("overrides", {})
            cfg = copy.deepcopy(self.config)
            cfg.setdefault("es_strategy", {}).update(overrides)

            try:
                summary = self._run_es_backtest(cfg)
                results.append({
                    "label": label,
                    "total_pnl": summary.get("total_pnl", 0),
                    "trades": summary.get("trades", 0),
                    "overrides": overrides,
                })
            except Exception as e:
                results.append({"label": label, "error": str(e)})

        return {"type": "es_compare", "results": results}

    def _run_es_backtest(self, cfg: dict) -> dict:
        """Run an ES backtest with the given config. Uses synthetic data if no CSV."""
        from src.es_strategy.runner import ESRunner

        # Look for a CSV file in data/
        csv_path = None
        for candidate in ("data/es_1min.csv", "data/es_backtest.csv"):
            if os.path.exists(candidate):
                csv_path = candidate
                break

        if csv_path is None:
            # Generate synthetic 1-min bars from daily prices for backtesting
            csv_path = self._generate_synthetic_bars()

        if csv_path is None:
            return {"total_pnl": 0, "trades": 0, "error": "no backtest data"}

        runner = ESRunner(cfg, mode="backtest", ai_enabled=False)
        return runner.run_backtest(csv_path)

    def _generate_synthetic_bars(self) -> Optional[str]:
        """Generate synthetic 1-min bars from daily price data for backtesting."""
        conn = get_connection(self.config)
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices "
            "ORDER BY date DESC LIMIT ?",
            conn, params=(self.es_lookback,),
        )
        conn.close()

        if df.empty or len(df) < 5:
            return None

        df = df.sort_values("date").reset_index(drop=True)
        bars = []
        for _, day in df.iterrows():
            # Generate 390 1-min bars per day (6.5 hours)
            o, h, l, c = day["open"], day["high"], day["low"], day["close"]
            vol = day["volume"] or 1_000_000
            n = 390
            # Random walk from open to close, bounded by high/low
            rng = np.random.default_rng(hash(day["date"]) & 0xFFFFFFFF)
            path = np.linspace(o, c, n) + rng.normal(0, (h - l) / 20, n)
            path = np.clip(path, l, h)

            for i in range(n):
                ts = f"{day['date']} {9 + i // 60:02d}:{30 + i % 60:02d}:00"
                bar_h = path[i] + rng.uniform(0, (h - l) / 50)
                bar_l = path[i] - rng.uniform(0, (h - l) / 50)
                bars.append({
                    "timestamp": ts,
                    "open": round(path[max(0, i - 1)], 2),
                    "high": round(bar_h, 2),
                    "low": round(bar_l, 2),
                    "close": round(path[i], 2),
                    "volume": int(vol / n),
                })

        synth_df = pd.DataFrame(bars)
        tmp = os.path.join(tempfile.gettempdir(), "es_whatif_synth.csv")
        synth_df.to_csv(tmp, index=False)
        return tmp

    # ------------------------------------------------------------------
    # 8C: SPY Predictor What-If
    # ------------------------------------------------------------------

    def spy_scenario_inject(self, overrides: dict) -> dict:
        """Override features and re-run XGBoost inference.

        Args:
            overrides: Dict of feature_name → new_value.
                e.g. {"vix": 35.0, "sentiment_score": -0.8}

        Returns:
            Dict with original and modified predictions.
        """
        predictor = self._get_predictor()
        fv = self._get_features()
        if fv is None or fv.empty:
            return {"error": "no feature data"}

        available = [c for c in self._feature_cols if c in fv.columns]
        latest = fv[available].iloc[-1].copy()

        # Original prediction
        original = predictor.predict(latest.values.astype(np.float64))

        # Modified prediction
        modified_features = latest.copy()
        for key, val in overrides.items():
            if key in modified_features.index:
                modified_features[key] = val

        modified = predictor.predict(modified_features.values.astype(np.float64))

        return {
            "type": "spy_scenario_inject",
            "overrides": overrides,
            "original": original,
            "modified": modified,
        }

    def spy_feature_ablation(self, drop_list: list[str]) -> dict:
        """Drop features and measure accuracy impact.

        Args:
            drop_list: List of feature names to zero out.

        Returns:
            Dict with accuracy comparison.
        """
        predictor = self._get_predictor()
        fv = self._get_features()
        if fv is None or fv.empty:
            return {"error": "no feature data"}

        available = [c for c in self._feature_cols if c in fv.columns]
        X = fv[available].copy()
        y = get_target(fv)

        # Baseline accuracy on last 50 rows
        tail = min(50, len(X) - 1)
        X_test = X.iloc[-tail:]
        y_test = y.iloc[-tail:]

        baseline_preds = []
        ablated_preds = []
        for i in range(len(X_test)):
            row = X_test.iloc[i].values.astype(np.float64).copy()
            baseline_preds.append(predictor.predict(row)["predicted_class"])

            ablated = row.copy()
            for feat in drop_list:
                if feat in X_test.columns:
                    idx = list(X_test.columns).index(feat)
                    ablated[idx] = 0.0
            ablated_preds.append(predictor.predict(ablated)["predicted_class"])

        y_vals = y_test.values
        valid = ~np.isnan(y_vals.astype(float))
        baseline_acc = np.mean(np.array(baseline_preds)[valid] == y_vals[valid]) if valid.any() else 0
        ablated_acc = np.mean(np.array(ablated_preds)[valid] == y_vals[valid]) if valid.any() else 0

        return {
            "type": "spy_feature_ablation",
            "dropped": drop_list,
            "baseline_accuracy": round(float(baseline_acc), 4),
            "ablated_accuracy": round(float(ablated_acc), 4),
            "accuracy_impact": round(float(baseline_acc - ablated_acc), 4),
            "samples": tail,
        }

    def spy_monte_carlo(self, n_sims: int = 500, noise_pct: float = None) -> dict:
        """Random perturbation of features → prediction distribution.

        Args:
            n_sims: Number of simulations (capped at max_sims).
            noise_pct: Gaussian noise as % of feature std dev.
        """
        n_sims = min(n_sims, self.max_sims)
        noise_pct = noise_pct or self.spy_noise_pct

        predictor = self._get_predictor()
        fv = self._get_features()
        if fv is None or fv.empty:
            return {"error": "no feature data"}

        available = [c for c in self._feature_cols if c in fv.columns]
        X = fv[available]
        latest = X.iloc[-1].values.astype(np.float64).copy()
        stds = X.std().values.astype(np.float64)
        stds = np.nan_to_num(stds, nan=1.0)

        rng = np.random.default_rng(42)
        counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        confidences = []

        for _ in range(n_sims):
            noise = rng.normal(0, stds * noise_pct / 100)
            perturbed = latest + noise
            pred = predictor.predict(perturbed)
            counts[pred["direction"]] = counts.get(pred["direction"], 0) + 1
            confidences.append(pred["confidence"])

        total = sum(counts.values()) or 1
        return {
            "type": "spy_monte_carlo",
            "n_sims": n_sims,
            "noise_pct": noise_pct,
            "distribution": {k: round(v / total * 100, 1) for k, v in counts.items()},
            "avg_confidence": round(float(np.mean(confidences)), 1),
            "std_confidence": round(float(np.std(confidences)), 1),
        }

    def market_stress_test(self, scenario_name: str) -> dict:
        """Run a pre-built stress test scenario.

        Args:
            scenario_name: Key from STRESS_SCENARIOS dict.
        """
        if scenario_name not in STRESS_SCENARIOS:
            available = list(STRESS_SCENARIOS.keys())
            return {"error": f"Unknown scenario. Available: {available}"}

        scenario = STRESS_SCENARIOS[scenario_name]
        result = self.spy_scenario_inject(scenario["overrides"])
        result["scenario"] = scenario["label"]
        result["description"] = scenario["description"]
        return result

    # ------------------------------------------------------------------
    # LLM Explain
    # ------------------------------------------------------------------

    def llm_explain(self, result: dict, llm_available: bool = True) -> str:
        """Ask DeepSeek to narrate what-if findings."""
        rtype = result.get("type", "")
        if rtype.startswith("es_"):
            return self.narrator.narrate_es_result(result, llm_available)
        else:
            return self.narrator.narrate_spy_result(result, llm_available)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cartesian(grid: dict) -> list[tuple]:
        """Compute cartesian product of parameter grid values."""
        import itertools
        keys = list(grid.keys())
        return list(itertools.product(*[grid[k] for k in keys]))

    @staticmethod
    def list_stress_scenarios() -> list[dict]:
        """Return available stress test scenarios."""
        return [
            {"name": k, "label": v["label"], "description": v["description"]}
            for k, v in STRESS_SCENARIOS.items()
        ]

