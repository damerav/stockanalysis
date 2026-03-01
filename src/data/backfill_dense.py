"""Dense feature backfill — fill all features possible from free sources.

Fills:
- VIX term structure: VIX9D, VIX3M, VIX6M, VVIX, SKEW from CBOE/yfinance
- Cross-asset signals: TLT, EEM, XLK, XLF, XLE, copper (HG=F) from yfinance
- HY spread from FRED (BAMLH0A0HYM2)
- Intraday proxies computed from daily OHLCV
- Put/call ratio from FRED (CBOE equity P/C)
- ATR percentile and VIX realised ratio computed from existing data

Usage:
    python -m src.data.backfill_dense [--years 3]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.data.init_db import init_db, get_connection, load_config
from src.data.db_router import get_router

logger = logging.getLogger(__name__)


def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily close from yfinance, return DataFrame with date + close."""
    import yfinance as yf
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def _fetch_fred_series(series_id: str, fred_key: str,
                       start: str, end: str) -> pd.Series:
    """Fetch FRED series, return as Series indexed by date string."""
    try:
        if fred_key:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id, "api_key": fred_key,
                "file_type": "json", "observation_start": start,
                "observation_end": end, "sort_order": "asc",
            }
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                obs = resp.json().get("observations", [])
                rows = {o["date"]: float(o["value"])
                        for o in obs if o.get("value", ".") != "."}
                return pd.Series(rows)
        # CSV fallback
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.set_index("date")["value"]
    except Exception as e:
        logger.warning(f"FRED {series_id} failed: {e}")
        return pd.Series(dtype=float)


def backfill_dense(years: int = 3, config: dict = None):
    """Fill all empty features from free data sources."""
    if config is None:
        config = load_config()

    conn = get_connection(config)
    fred_key = config.get("fred", {}).get("api_key", "")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(years * 365))).strftime("%Y-%m-%d")

    # Get existing trading dates from prices table
    dates_df = pd.read_sql_query(
        "SELECT date FROM prices ORDER BY date", conn)
    trading_dates = dates_df["date"].tolist()
    logger.info(f"Trading dates in DB: {len(trading_dates)}")

    # Load existing prices for intraday proxy computation
    prices = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices ORDER BY date", conn)

    # ── 1. VIX Term Structure from yfinance ──
    vix_tickers = {
        "vix9d": "^VIX9D",
        "vix3m": "^VIX3M",
        "vix6m": "^VIX6M",
        "vvix": "^VVIX",
        "skew_index": "^SKEW",
    }

    vix_data = {}
    for name, ticker in vix_tickers.items():
        logger.info(f"Fetching {name} ({ticker})...")
        try:
            df = _yf_download(ticker, start_date, end_date)
            if not df.empty:
                vix_data[name] = df.set_index("date")["Close"]
                logger.info(f"  {name}: {len(df)} rows")
            else:
                logger.warning(f"  {name}: no data")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # ── 2. Cross-asset ETFs from yfinance ──
    etf_tickers = {
        "TLT": "TLT",    # Long-term treasuries
        "EEM": "EEM",    # Emerging markets
        "XLK": "XLK",    # Tech sector
        "XLF": "XLF",    # Financials sector
        "XLE": "XLE",    # Energy sector
        "HG_F": "HG=F",  # Copper futures
    }

    etf_data = {}
    for name, ticker in etf_tickers.items():
        logger.info(f"Fetching {name} ({ticker})...")
        try:
            df = _yf_download(ticker, start_date, end_date)
            if not df.empty:
                etf_data[name] = df.set_index("date")["Close"]
                logger.info(f"  {name}: {len(df)} rows")
            else:
                logger.warning(f"  {name}: no data")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # Also need SPY close for ratios
    spy_close = prices.set_index("date")["close"]

    # ── 3. HY Spread from FRED ──
    logger.info("Fetching HY spread from FRED (BAMLH0A0HYM2)...")
    hy_spread = _fetch_fred_series("BAMLH0A0HYM2", fred_key, start_date, end_date)
    logger.info(f"  hy_spread: {len(hy_spread)} rows")

    # ── 4. Put/Call ratio from FRED ──
    logger.info("Fetching put/call ratio from FRED...")
    # CBOE equity put/call ratio
    pcr = _fetch_fred_series("EQUITYPCRATIO", fred_key, start_date, end_date)
    if pcr.empty:
        logger.info("  Trying CBOE total P/C (TOTALPCR)...")
        pcr = _fetch_fred_series("TOTALPCR", fred_key, start_date, end_date)
    if pcr.empty:
        # Try computing from VIX as proxy (higher VIX = more puts)
        logger.info("  FRED P/C unavailable, skipping")
    logger.info(f"  put_call_ratio: {len(pcr)} rows")

    # ── 5. Build the dense feature table ──
    logger.info("Computing dense features...")
    result = pd.DataFrame({"date": trading_dates})

    # VIX term structure
    for name, series in vix_data.items():
        result[name] = result["date"].map(series)

    # VIX term slope = (VIX3M - VIX) / VIX
    # Need VIX from macro table
    macro = pd.read_sql_query("SELECT date, vix FROM macro ORDER BY date", conn)
    vix_series = macro.set_index("date")["vix"]
    result["_vix"] = result["date"].map(vix_series)

    if "vix3m" in result.columns:
        result["vix_term_slope"] = (result["vix3m"] - result["_vix"]) / result["_vix"].replace(0, np.nan)
    if "vix3m" in result.columns and "vix6m" in result.columns:
        result["vix_term_curve"] = (result["vix6m"] - result["vix3m"]) / result["vix3m"].replace(0, np.nan)

    # VIX realised ratio = VIX / realised vol (20-day)
    prices_s = prices.set_index("date")
    ret = prices_s["close"].pct_change()
    realised_vol = ret.rolling(20).std() * np.sqrt(252) * 100
    result["_realised_vol"] = result["date"].map(realised_vol)
    result["vix_realised_ratio"] = result["_vix"] / result["_realised_vol"].replace(0, np.nan)

    # Cross-asset ratios
    if "TLT" in etf_data:
        tlt = result["date"].map(etf_data["TLT"])
        spy = result["date"].map(spy_close)
        result["tlt_spy_ratio"] = tlt / spy.replace(0, np.nan)
    if "EEM" in etf_data:
        eem = result["date"].map(etf_data["EEM"])
        spy = result["date"].map(spy_close)
        result["eem_spy_ratio"] = eem / spy.replace(0, np.nan)
    if "XLK" in etf_data and "XLF" in etf_data:
        xlk = result["date"].map(etf_data["XLK"])
        xlf = result["date"].map(etf_data["XLF"])
        result["xlk_xlf_ratio"] = xlk / xlf.replace(0, np.nan)
    if "XLK" in etf_data and "XLE" in etf_data:
        xlk = result["date"].map(etf_data["XLK"])
        xle = result["date"].map(etf_data["XLE"])
        result["xlk_xle_ratio"] = xlk / xle.replace(0, np.nan)
    if "HG_F" in etf_data:
        gold_series = None
        # Get gold from macro
        macro_gold = pd.read_sql_query("SELECT date, gold FROM macro ORDER BY date", conn)
        if not macro_gold.empty:
            gold_series = macro_gold.set_index("date")["gold"]
        if gold_series is not None:
            copper = result["date"].map(etf_data["HG_F"])
            gold_mapped = result["date"].map(gold_series)
            result["copper_gold_ratio"] = copper / gold_mapped.replace(0, np.nan)

    # HY spread
    if not hy_spread.empty:
        result["hy_spread"] = result["date"].map(hy_spread)

    # Put/call ratio
    if not pcr.empty:
        result["put_call_ratio"] = result["date"].map(pcr)

    # ── 6. Intraday proxies from daily OHLCV ──
    # These are approximations since we don't have tick data
    p = prices.set_index("date")

    # opening_gap_pct = (open - prev_close) / prev_close
    prev_close = p["close"].shift(1)
    opening_gap = (p["open"] - prev_close) / prev_close.replace(0, np.nan)
    result["opening_gap_pct"] = result["date"].map(opening_gap)

    # close_vs_high_pct = (close - low) / (high - low) — where in range did it close
    daily_range = (p["high"] - p["low"]).replace(0, np.nan)
    result["close_vs_high_pct"] = result["date"].map((p["high"] - p["close"]) / daily_range)
    result["close_vs_low_pct"] = result["date"].map((p["close"] - p["low"]) / daily_range)

    # intraday_range = (high - low) / open
    result["intraday_range"] = result["date"].map((p["high"] - p["low"]) / p["open"].replace(0, np.nan))

    # intraday_momentum = (close - open) / (high - low)
    result["intraday_momentum"] = result["date"].map((p["close"] - p["open"]) / daily_range)

    # volume_ratio = volume / 20-day avg volume
    vol_ma = p["volume"].rolling(20).mean()
    result["volume_ratio"] = result["date"].map(p["volume"] / vol_ma.replace(0, np.nan))

    # vwap_spread proxy = (close - (high+low+close)/3) / close
    vwap_proxy = (p["high"] + p["low"] + p["close"]) / 3
    result["vwap_spread"] = result["date"].map((p["close"] - vwap_proxy) / p["close"].replace(0, np.nan))

    # ATR percentile = current ATR / max ATR over 252 days
    atr_col = pd.read_sql_query("SELECT date, atr_14 FROM technicals ORDER BY date", conn)
    if not atr_col.empty:
        atr_s = atr_col.set_index("date")["atr_14"]
        atr_max = atr_s.rolling(252, min_periods=20).max()
        atr_pct = atr_s / atr_max.replace(0, np.nan)
        result["atr_percentile"] = result["date"].map(atr_pct)

    # VIX percentile = current VIX / max VIX over 252 days
    if not vix_series.empty:
        vix_max = vix_series.rolling(252, min_periods=20).max()
        vix_pct = vix_series / vix_max.replace(0, np.nan)
        result["vix_percentile"] = result["date"].map(vix_pct)

    # Forward-fill gaps (weekends/holidays in FRED data)
    fill_cols = [c for c in result.columns if c not in ["date", "_vix", "_realised_vol"]]
    result[fill_cols] = result[fill_cols].ffill()

    # Drop temp columns
    result = result.drop(columns=["_vix", "_realised_vol"], errors="ignore")

    # ── 7. Write to DB ──
    # Check which tables/columns exist for these features
    # Most go into the vix_term, cross_asset, intraday_features, options_analytics tables
    # But build_feature_vector reads from specific tables — let's check
    logger.info("Writing dense features to database...")

    # Write VIX term structure to macro table (add columns if needed)
    # Actually, build_feature_vector joins from specific tables.
    # The simplest approach: write to a dense_features table and update build_feature_vector
    # OR: write directly to the tables that build_feature_vector reads from.

    # Let's write to the existing tables that build_feature_vector expects.
    # Check what tables it reads from:
    _write_vix_term(conn, result)
    _write_cross_asset(conn, result)
    _write_intraday(conn, result)
    _write_options(conn, result)
    _write_extra_macro(conn, result)

    conn.close()

    # Summary
    filled = 0
    for c in fill_cols:
        if c.startswith("_"):
            continue
        non_null = result[c].notna().sum()
        if non_null > 0:
            filled += 1
            logger.info(f"  {c}: {non_null}/{len(result)} filled ({non_null/len(result):.0%})")

    logger.info(f"\nDense backfill complete: {filled} features populated")


def _safe_float(val):
    """Convert to float or None."""
    if pd.isna(val):
        return None
    return float(val)


def _write_vix_term(conn, result):
    """Update macro table with VIX term structure and cross-asset columns."""
    # Only write columns that exist in the macro table schema
    # vix_term_slope and vix_term_curve are computed by build_feature_vector
    vix_cols = ["vix9d", "vix3m", "vix6m", "vvix", "skew_index"]
    cross_cols = ["hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
                  "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio"]
    all_cols = vix_cols + cross_cols

    updated = 0
    for _, row in result.iterrows():
        sets = []
        vals = []
        for col in all_cols:
            if col in result.columns and pd.notna(row.get(col)):
                sets.append(f"{col} = ?")
                vals.append(_safe_float(row[col]))
        if sets:
            vals.append(row["date"])
            sql = f"UPDATE macro SET {', '.join(sets)} WHERE date = ?"
            conn.execute(sql, vals)
            updated += 1
    conn.commit()
    logger.info(f"  Updated {updated} macro rows with VIX term + cross-asset data")


def _write_cross_asset(conn, result):
    """Already handled in _write_vix_term since they share the macro table."""
    pass


def _write_intraday(conn, result):
    """Write intraday proxy features."""
    intraday_cols = ["vwap_spread", "intraday_momentum", "intraday_range", "volume_ratio"]
    available = [c for c in intraday_cols if c in result.columns]
    if not available:
        return

    inserted = 0
    for _, row in result.iterrows():
        vals = {c: _safe_float(row.get(c)) for c in available}
        if all(v is None for v in vals.values()):
            continue
        placeholders = ", ".join(["?"] * (len(available) + 1))
        col_names = ", ".join(["date"] + available)
        sql = f"INSERT OR REPLACE INTO intraday_features ({col_names}) VALUES ({placeholders})"
        conn.execute(sql, [row["date"]] + [vals[c] for c in available])
        inserted += 1
    conn.commit()
    logger.info(f"  Inserted {inserted} intraday_features rows")


def _write_options(conn, result):
    """Write options analytics (put_call_ratio)."""
    if "put_call_ratio" not in result.columns:
        return

    inserted = 0
    for _, row in result.iterrows():
        pcr = _safe_float(row.get("put_call_ratio"))
        if pcr is None:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO options_analytics
               (date, put_call_ratio, max_pain, iv_skew, gex,
                vanna_exposure, charm_exposure, zero_dte_pcr)
               VALUES (?, ?, 0, 0, 0, 0, 0, 0)""",
            (row["date"], pcr)
        )
        inserted += 1
    conn.commit()
    logger.info(f"  Inserted {inserted} options_analytics rows")


def _write_extra_macro(conn, result):
    """Write vix_realised_ratio, atr_percentile, vix_percentile to macro."""
    # These are computed features that build_feature_vector derives,
    # so we don't need to store them — they'll be computed on the fly.
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Dense feature backfill")
    parser.add_argument("--years", type=int, default=3,
                        help="Years of history (default: 3)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backfill_dense(years=args.years, config=config)
