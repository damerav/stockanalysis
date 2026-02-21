"""8C. Pre-built stress test scenarios for SPY What-If analysis."""

# Each preset overrides specific features in the feature vector.
# Keys must match get_feature_columns() names from src/data/features.py.

STRESS_SCENARIOS = {
    "vix_spike_40": {
        "label": "VIX Spike to 40",
        "description": "Sudden volatility spike — fear gauge at extreme levels",
        "overrides": {
            "vix": 40.0,
            "vix_change": 15.0,
            "sentiment_score": -0.6,
            "rsi_14": 28.0,
            "momentum_5d": -0.04,
            "momentum_10d": -0.06,
        },
    },
    "gap_down_3pct": {
        "label": "Gap Down 3%",
        "description": "Market opens 3% lower — overnight shock event",
        "overrides": {
            "vix": 30.0,
            "vix_change": 10.0,
            "sentiment_score": -0.8,
            "rsi_14": 22.0,
            "momentum_5d": -0.05,
            "momentum_10d": -0.07,
            "price_vs_sma20": -0.04,
            "price_vs_sma50": -0.06,
            "volume_trend": 2.5,
        },
    },
    "march_2020_crash": {
        "label": "2020 March Crash",
        "description": "Pandemic-style crash — extreme fear, high vol, bearish everything",
        "overrides": {
            "vix": 65.0,
            "vix_change": 20.0,
            "sentiment_score": -0.95,
            "rsi_14": 18.0,
            "momentum_5d": -0.10,
            "momentum_10d": -0.18,
            "price_vs_sma20": -0.12,
            "price_vs_sma50": -0.18,
            "volume_trend": 3.5,
            "put_call_ratio": 1.8,
        },
    },
    "fed_rate_cut": {
        "label": "Surprise Fed Rate Cut",
        "description": "Emergency rate cut — initially bullish, uncertainty high",
        "overrides": {
            "fed_funds": 3.5,
            "vix": 22.0,
            "vix_change": -3.0,
            "sentiment_score": 0.4,
            "us10y_yield": 3.2,
        },
    },
    "melt_up": {
        "label": "Melt-Up Rally",
        "description": "Euphoric rally — low vol, strong momentum, bullish sentiment",
        "overrides": {
            "vix": 11.0,
            "vix_change": -2.0,
            "sentiment_score": 0.85,
            "rsi_14": 78.0,
            "momentum_5d": 0.04,
            "momentum_10d": 0.07,
            "price_vs_sma20": 0.03,
            "price_vs_sma50": 0.06,
            "volume_trend": 1.4,
            "put_call_ratio": 0.5,
        },
    },
}
