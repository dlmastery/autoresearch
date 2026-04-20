"""Data download module for FX pairs and macro signals.

Downloads OHLCV data via yfinance, caches locally as parquet files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker dictionaries
# ---------------------------------------------------------------------------

PAIRS: Dict[str, str] = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "JPY=X":    "USD/JPY",
    "CHF=X":    "USD/CHF",
    "EURGBP=X": "EUR/GBP",
    "EURJPY=X": "EUR/JPY",
}

MACRO_TICKERS: Dict[str, str] = {
    "^TNX":     "US 10Y Treasury Yield",
    "^FVX":     "US 5Y Treasury Yield",
    "^IRX":     "US 13W Treasury Yield",
    "^VIX":     "VIX Volatility Index",
    "^GSPC":    "S&P 500",
    "^N225":    "Nikkei 225",
    "GC=F":     "Gold Futures",
    "CL=F":     "Crude Oil Futures",
    "DX-Y.NYB": "US Dollar Index (DXY)",
}

# Default date range
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2026-04-01"

# Default cache directory — avoids re-downloading on every run
DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".data_cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, start: str, end: str, cache_dir: str) -> Path:
    """Build a deterministic parquet filename for caching."""
    safe_name = ticker.replace("=", "_").replace("^", "_").replace(".", "_").replace("-", "_")
    fname = f"{safe_name}_{start}_{end}.parquet"
    return Path(cache_dir) / fname


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the second level of a MultiIndex column (yfinance quirk)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_pair(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download a single ticker via yfinance, with parquet caching.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start, end : str
        Date strings in YYYY-MM-DD format.
    cache_dir : str or None
        Directory to cache parquet files. If None, no caching.

    Returns
    -------
    pd.DataFrame
        OHLCV data indexed by Date.
    """
    # Try cache first
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(ticker, start, end, cache_dir)
        if cp.exists():
            logger.info("Cache hit: %s", cp)
            return pd.read_parquet(cp)

    # Download from yfinance
    logger.info("Downloading %s  [%s -> %s]", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False)

    # Handle MultiIndex columns produced by recent yfinance versions
    df = _flatten_columns(df)

    # Write cache
    if cache_dir is not None:
        cp = _cache_path(ticker, start, end, cache_dir)
        df.to_parquet(cp)
        logger.info("Cached to %s", cp)

    return df


def download_all_pairs(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """Download all 6 FX pairs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to OHLCV DataFrame.
    """
    result: Dict[str, pd.DataFrame] = {}
    for ticker in PAIRS:
        df = download_pair(ticker, start, end, cache_dir)
        result[ticker] = df
    return result


def download_macro_signals(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """Download all 9 macro tickers, tolerant of individual failures.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to OHLCV DataFrame.  Tickers that fail to
        download are silently skipped (logged as warnings).
    """
    result: Dict[str, pd.DataFrame] = {}
    for ticker in MACRO_TICKERS:
        try:
            df = download_pair(ticker, start, end, cache_dir)
            if df is not None and not df.empty:
                result[ticker] = df
        except Exception:
            logger.warning("Failed to download macro ticker %s", ticker, exc_info=True)
    return result
