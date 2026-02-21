# Phase 1: Data Ingestion from Polygon.io

## Requirement
Build the data ingestion layer that connects to Polygon.io for real-time and historical market data, with fallback sources, and stores everything in a local SQLite database.

## Sub-components
1. **1A** - Polygon WebSocket Streamer: Real-time SPY trades + SPX options via WebSocket
2. **1B** - Polygon REST Fetcher: Daily bars, intraday bars, options chain with Greeks
3. **1C** - Fallback Data Sources: yfinance, Finnhub, RSS feeds, FRED for macro data
4. **1D** - Gap Detection & Backfill: Find missing dates and auto-fill from available sources
5. **1E** - Initial Bulk Load: First-time 252-day historical backfill
6. **1F** - SQLite Database Schema: 10 tables covering prices, technicals, sentiment, options, etc.

## Acceptance Criteria
- WebSocket connects and receives SPY trades within 5 seconds
- 5-second bars are produced with correct OHLCV aggregation
- Options sweeps and block trades are detected and logged
- REST API fetches daily bars, options chain, and 5s bars successfully
- Gap detection finds and backfills missing dates automatically
- All 10 SQLite tables are created and populated with 1 year of history
- System continues operating if options WebSocket fails (stocks-only mode)
- yfinance fallback works when Polygon REST is unavailable

## Target Environment
- NVIDIA DGX Spark (Python 3.12.3, SQLite 3.45.1)
- Code written locally on Windows, synced via Mutagen, runs on DGX via SSH

## Dependencies
- Polygon.io Advanced subscription (API key placeholder for now)
- Python packages: websockets, aiohttp, yfinance, feedparser, requests, pandas, numpy
