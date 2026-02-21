"""Feature Drift Monitor — PSI + KS test for detecting distribution shifts.

Alerts when feature distributions shift materially after macro regime changes,
which can degrade model performance without triggering accuracy alerts.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from scipy import stats

logger = logging.getLogger(__name__)

# Thresholds
PSI_WARN = 0.1   # moderate shift
PSI_ALERT = 0.2  # significant shift
KS_ALPHA = 0.05  # KS test significance level


def compute_psi(expected: np.ndarray, actual: np.ndarray,
                n_bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1  → no significant shift
    PSI 0.1-0.2 → moderate shift, monitor
    PSI > 0.2  → significant shift, retrain recommended

    Args:
        expected: Reference (training) distribution
        actual: Current (production) distribution
        n_bins: Number of bins for discretisation

    Returns:
        PSI value (float)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # Use expected distribution to define bin edges
    _, bin_edges = np.histogram(expected, bins=n_bins)
    # Clip actual to expected range
    bin_edges[0] = min(bin_edges[0], actual.min())
    bin_edges[-1] = max(bin_edges[-1], actual.max())

    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Convert to proportions with smoothing to avoid log(0)
    expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
    actual_pct = (actual_counts + 1) / (len(actual) + n_bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_ks(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Kolmogorov-Smirnov test between two distributions.

    Returns dict with statistic, p_value, and whether shift is significant.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p_val = stats.ks_2samp(expected, actual)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 6),
        "significant": p_val < KS_ALPHA,
    }


def monitor_features(train_df: pd.DataFrame, current_df: pd.DataFrame,
                     feature_cols: list[str]) -> dict:
    """Run PSI + KS drift check on all features.

    Args:
        train_df: Training data (reference distribution)
        current_df: Recent production data (last 30-60 days)
        feature_cols: List of feature column names to check

    Returns:
        Dict with per-feature results and overall summary.
    """
    results = {}
    alerts = []
    warnings = []

    for col in feature_cols:
        if col not in train_df.columns or col not in current_df.columns:
            continue

        expected = train_df[col].values.astype(float)
        actual = current_df[col].values.astype(float)

        psi = compute_psi(expected, actual)
        ks = compute_ks(expected, actual)

        status = "ok"
        if psi > PSI_ALERT:
            status = "alert"
            alerts.append(col)
        elif psi > PSI_WARN:
            status = "warning"
            warnings.append(col)

        results[col] = {
            "psi": round(psi, 4),
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "ks_significant": ks["significant"],
            "status": status,
        }

    summary = {
        "total_features": len(results),
        "alerts": len(alerts),
        "warnings": len(warnings),
        "alert_features": alerts,
        "warning_features": warnings,
        "details": results,
    }

    if alerts:
        logger.warning(f"DRIFT ALERT: {len(alerts)} features with PSI > {PSI_ALERT}: "
                       f"{', '.join(alerts)}")
    if warnings:
        logger.info(f"Drift warning: {len(warnings)} features with PSI > {PSI_WARN}: "
                    f"{', '.join(warnings)}")

    return summary
