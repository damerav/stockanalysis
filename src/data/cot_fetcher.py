"""CFTC Commitment of Traders (COT) Fetcher — institutional futures positioning.

Fetches weekly COT reports from the CFTC for S&P 500 E-mini futures.
Uses the Traders in Financial Futures (TFF) report which covers equity index futures.

Data source: CFTC.gov TFF Futures-Only reports (free, no API key).
Published every Friday for positions as of Tuesday.

Derived features:
  - cot_commercial_net: Dealer/Intermediary net position (long - short)
  - cot_leveraged_net: Leveraged Funds net position (long - short)
  - cot_asset_mgr_net: Asset Manager net position (long - short)
  - cot_commercial_change: Week-over-week change in dealer net
  - cot_leveraged_change: Week-over-week change in leveraged net
  - cot_spec_ratio: Leveraged long / (long + short) — bullish when high

Weekly data is forward-filled to daily for model consumption.
"""

import io
import logging
import zipfile
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# TFF (Traders in Financial Futures) report — includes equity index futures
# Historical yearly archives (zipped, WITH headers)
_CFTC_TFF_HISTORY_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

# CFTC Market Code for CME E-mini S&P 500
_SP500_CFTC_CODE = "13874A"


def _download_year(year: int) -> Optional[pd.DataFrame]:
    """Download a single year's TFF report (zipped, has headers)."""
    url = _CFTC_TFF_HISTORY_URL.format(year=year)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            txt_name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            with zf.open(txt_name) as f:
                df = pd.read_csv(f, low_memory=False)
                df.columns = [c.strip().strip('"') for c in df.columns]
        logger.info(f"COT TFF {year}: {len(df)} rows downloaded")
        return df
    except Exception as e:
        logger.debug(f"COT TFF {year} download failed: {e}")
        return None


def _resolve_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Find the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _filter_sp500(df: pd.DataFrame) -> pd.DataFrame:
    """Filter TFF data to E-mini S&P 500 futures only (code 13874A)."""
    code_col = _resolve_col(df, "CFTC_Contract_Market_Code")
    if code_col:
        mask = df[code_col].astype(str).str.strip() == _SP500_CFTC_CODE
        filtered = df[mask]
        if not filtered.empty:
            return filtered

    # Fallback: match by market name
    name_col = _resolve_col(df, "Market_and_Exchange_Names")
    if name_col is None:
        return pd.DataFrame()
    mask = df[name_col].astype(str).str.upper().str.contains("E-MINI S&P 500", na=False)
    # Exclude sector indices (E-MINI S&P ENERGY, etc.)
    mask = mask & ~df[name_col].astype(str).str.upper().str.contains(
        "ENERGY|FINANCIAL|HEALTH|INDUSTRIAL|MATERIAL|TECHNOLOGY|UTILITIES|COMMUNICATION|REAL ESTATE|STAPLES",
        na=False
    )
    return df[mask]


def _extract_features(sp500_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Extract COT features from filtered E-mini S&P 500 TFF data."""
    if sp500_df.empty:
        return None

    date_col = _resolve_col(sp500_df, "Report_Date_as_YYYY-MM-DD",
                            "As_of_Date_In_Form_YYMMDD")
    if date_col is None:
        logger.warning("COT: Cannot find date column")
        return None

    # TFF report column names
    dealer_long = _resolve_col(sp500_df, "Dealer_Positions_Long_All")
    dealer_short = _resolve_col(sp500_df, "Dealer_Positions_Short_All")
    asset_long = _resolve_col(sp500_df, "Asset_Mgr_Positions_Long_All")
    asset_short = _resolve_col(sp500_df, "Asset_Mgr_Positions_Short_All")
    lev_long = _resolve_col(sp500_df, "Lev_Money_Positions_Long_All")
    lev_short = _resolve_col(sp500_df, "Lev_Money_Positions_Short_All")

    if not all([dealer_long, dealer_short, lev_long, lev_short]):
        logger.warning(f"COT: Missing required position columns. "
                       f"Available: {list(sp500_df.columns[:20])}")
        return None

    rows = []
    for _, row in sp500_df.iterrows():
        try:
            date_str = str(row[date_col]).strip()
            dt = None
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%y%m%d"):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                continue

            d_long = float(row.get(dealer_long, 0) or 0)
            d_short = float(row.get(dealer_short, 0) or 0)
            a_long = float(row.get(asset_long, 0) or 0) if asset_long else 0
            a_short = float(row.get(asset_short, 0) or 0) if asset_short else 0
            l_long = float(row.get(lev_long, 0) or 0)
            l_short = float(row.get(lev_short, 0) or 0)

            commercial_net = d_long - d_short
            leveraged_net = l_long - l_short
            asset_mgr_net = a_long - a_short
            lev_total = l_long + l_short
            spec_ratio = (l_long / lev_total) if lev_total > 0 else 0.5

            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "cot_commercial_net": commercial_net,
                "cot_leveraged_net": leveraged_net,
                "cot_asset_mgr_net": asset_mgr_net,
                "cot_spec_ratio": round(spec_ratio, 4),
            })
        except Exception as e:
            logger.debug(f"COT row parse error: {e}")
            continue

    if not rows:
        return None

    result = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # Compute week-over-week changes
    result["cot_commercial_change"] = result["cot_commercial_net"].diff().fillna(0)
    result["cot_leveraged_change"] = result["cot_leveraged_net"].diff().fillna(0)

    logger.info(f"COT features extracted: {len(result)} weeks, "
                f"{result['date'].iloc[0]} to {result['date'].iloc[-1]}")
    return result


def fetch_cot_data(years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch COT data for S&P 500 E-mini futures from TFF report.

    Args:
        years: Number of years of history to fetch (default 2).

    Returns:
        DataFrame with weekly COT features, or None on failure.
    """
    all_dfs = []
    current_year = datetime.now().year

    for year in range(current_year, current_year - years - 1, -1):
        df = _download_year(year)
        if df is not None:
            sp500 = _filter_sp500(df)
            if not sp500.empty:
                all_dfs.append(sp500)
                logger.info(f"COT TFF {year}: {len(sp500)} E-mini S&P 500 rows")

    if not all_dfs:
        logger.warning("COT: No S&P 500 data from any year")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    features = _extract_features(combined)
    return features


def store_cot_data(router, cot_df: pd.DataFrame):
    """Store COT data to the cot_data table."""
    if cot_df is None or cot_df.empty:
        return

    cols = ["cot_commercial_net", "cot_leveraged_net", "cot_asset_mgr_net",
            "cot_spec_ratio", "cot_commercial_change", "cot_leveraged_change"]
    col_str = ", ".join(cols)
    ph_str = ", ".join(["?"] * len(cols))

    inserted = 0
    for _, row in cot_df.iterrows():
        vals = [row["date"]]
        for c in cols:
            v = row.get(c)
            vals.append(float(v) if pd.notna(v) else None)
        try:
            router.execute(
                f"INSERT OR REPLACE INTO cot_data (date, {col_str}) VALUES (?, {ph_str})",
                tuple(vals)
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"COT insert failed for {row['date']}: {e}")

    logger.info(f"Stored {inserted} COT data rows")


def backfill_cot_data(router, years: int = 5):
    """Backfill COT data for historical training."""
    logger.info(f"Backfilling {years} years of COT data from CFTC.gov TFF report...")
    cot_df = fetch_cot_data(years=years)
    if cot_df is not None:
        store_cot_data(router, cot_df)
        logger.info(f"COT backfill complete: {len(cot_df)} weekly rows")
    else:
        logger.warning("COT backfill returned no data")
