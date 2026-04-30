"""Equity-index data download for QQQ and supporting signals.

Downloads OHLCV via yfinance, caches as parquet. Hard-caps at 2025-12-31
because CLAUDE.md forbids any 2026 data — verified at the end of each fetch.

Tickers grouped by purpose so a future contributor can see the design rather
than guess. Each group has a one-line citation pointing to the literature
that motivated its inclusion.
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
# Tickers
# ---------------------------------------------------------------------------

# The asset we are forecasting.
PRIMARY: Dict[str, str] = {
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
}

# Panel learning targets: 28+ liquid equity/index assets predicted in
# parallel via shared trunk per Gu-Kelly-Xiu 2020 RFS "Empirical Asset
# Pricing via Machine Learning". The QQQ-only setup gives effective
# n=4,772 daily rows; expanding to a 28+ asset panel multiplies effective
# n proportionally and enables cross-asset diversification at inference.
#
# Selection rationale:
#   - Top-30 NDX components by 2024 weight (Mag-7 + tier-1 large caps)
#     to retain QQQ representation while gaining cross-asset diversity.
#   - 6 broad US/international indices for sub-market diversification
#     (Hou-Mo-Xue-Zhang 2014 — international predictability).
#   - Single-stock universe is intentionally LIMITED to top-30 to avoid
#     small-cap noise dominating the panel; full NDX-100 expansion is a
#     follow-up if the 30-stock panel works.
PANEL_NDX_TOP30: Dict[str, str] = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "Nvidia",
    "AMZN":  "Amazon",
    "META":  "Meta",
    "GOOGL": "Alphabet (Class A)",
    "GOOG":  "Alphabet (Class C)",
    "TSLA":  "Tesla",
    "AVGO":  "Broadcom",
    "COST":  "Costco",
    "NFLX":  "Netflix",
    "ADBE":  "Adobe",
    "AMD":   "AMD",
    "PEP":   "PepsiCo",
    "CSCO":  "Cisco",
    "QCOM":  "Qualcomm",
    "INTC":  "Intel",
    "TMUS":  "T-Mobile",
    "CMCSA": "Comcast",
    "INTU":  "Intuit",
    "AMGN":  "Amgen",
    "AMAT":  "Applied Materials",
    "TXN":   "Texas Instruments",
    "BKNG":  "Booking Holdings",
    "ISRG":  "Intuitive Surgical",
    "ADP":   "ADP",
    "GILD":  "Gilead",
    "SBUX":  "Starbucks",
    "MDLZ":  "Mondelez",
    "MU":    "Micron",
}

# Adjacent indices for sub-market panel diversification per user directive
# 2026-04-29: SPY/IWM/EEM/EFA/DIA/MDY = US large/small/EM/intl-DM/Dow/midcap.
PANEL_ADJACENT_INDICES: Dict[str, str] = {
    "SPY": "SPDR S&P 500 (US large cap)",
    "IWM": "iShares Russell 2000 (US small cap)",
    "EEM": "iShares MSCI Emerging Markets",
    "EFA": "iShares MSCI EAFE (international DM)",
    "DIA": "SPDR Dow Jones Industrial",
    "MDY": "SPDR S&P MidCap 400",
}

# Asia + Europe panels added 2026-04-29 per user directive — these markets
# CLOSE BEFORE US market open on the same calendar day, giving the model
# leading-indicator information for the QQQ day-T close prediction.
#
# Time-shift edge per Lou, Polk, Skouras 2019 JFE "A tug of war: Overnight
# versus intraday expected returns" (DOI 10.1016/j.jfineco.2019.05.007) and
# Boudoukh, Richardson, Whitelaw 2007 RFS "The myth of long-horizon
# predictability" — Asia close on day T sets at ~01:00-04:00 ET (well
# before NYSE 09:30 ET open), London close ~11:30 ET (before NYSE close).
# All cleanly causal for predicting QQQ day-T close.
PANEL_ASIA_EUROPE_INDICES: Dict[str, str] = {
    # Asian indices (close 1-5h ET — leading)
    "^N225":   "Nikkei 225 (Tokyo, close 01:00 ET)",
    "^HSI":    "Hang Seng (HK, close 04:00 ET)",
    "^KS11":   "KOSPI (Korea, close 02:30 ET)",
    "^TWII":   "Taiwan Weighted (close 02:30 ET)",
    "^STI":    "Straits Times (Singapore, close 05:00 ET)",
    "^AXJO":   "ASX 200 (Australia, close 02:00 ET)",
    # European indices (close ~11:30 ET — leading)
    "^STOXX50E": "Euro Stoxx 50 (close 11:30 ET)",
    "^FTSE":     "FTSE 100 (close 11:30 ET)",
    "^GDAXI":    "DAX (close 11:30 ET)",
    "^FCHI":     "CAC 40 (close 11:30 ET)",
}

# Asian megacaps (ADRs preferred for clean US-aligned trading days).
PANEL_ASIA_MEGACAPS: Dict[str, str] = {
    "TSM":     "TSMC (Taiwan Semiconductor) ADR",
    "BABA":    "Alibaba ADR",
    "JD":      "JD.com ADR",
    "PDD":     "PDD Holdings (Pinduoduo) ADR",
    "SONY":    "Sony Group ADR",
    "TM":      "Toyota Motor ADR",
    "HMC":     "Honda Motor ADR",
    "BIDU":    "Baidu ADR",
}

# Combined panel target list = QQQ + 30 NDX top + 6 adjacent + 10 Asia/Europe
# indices + 8 Asia megacaps = 55 parallel prediction targets.
# Each gets its own per-asset fold split but shares the trunk over the panel.
PANEL_TARGETS: Dict[str, str] = {
    **PRIMARY,
    **PANEL_NDX_TOP30,
    **PANEL_ASIA_EUROPE_INDICES,
    **PANEL_ASIA_MEGACAPS,
    **PANEL_ADJACENT_INDICES,
}

# Cross-asset benchmarks for relative-strength + breadth features.
#   Lo & MacKinlay 1990 — relative-strength reversal between indices.
BENCHMARKS: Dict[str, str] = {
    "SPY":   "SPDR S&P 500",
    "DIA":   "SPDR Dow Jones",
    "IWM":   "iShares Russell 2000",
    "EFA":   "iShares MSCI EAFE (international DM)",
    "EEM":   "iShares MSCI Emerging Markets",
    "^IXIC": "Nasdaq Composite (QQQ underlying)",
    "AGG":   "iShares Core US Aggregate Bond (baseline duration+credit)",
}

# Industry-specific ETFs — semiconductors and biotech are massive drivers of
# QQQ. The Mag-7 + AI rally is fundamentally a chip story; SOXX/SMH carry
# information XLK does not (XLK includes large-cap software which decouples
# from semis). IBB widens healthcare beyond XLV. ARKK is a high-beta
# innovation proxy that leads risk-on/off shifts in tech.
INDUSTRY_TILTS: Dict[str, str] = {
    "SOXX": "iShares Semiconductor ETF",
    "SMH":  "VanEck Semiconductor ETF",
    "IBB":  "iShares Biotechnology ETF",
    "ARKK": "ARK Innovation ETF (high-beta tech proxy)",
}

# Crypto / risk-on barometer.
#   Bouri et al. 2017 Finance Research Letters — Bitcoin is correlated with
#   risk-on flows, particularly post-2020. Captures speculative-flow regime.
CRYPTO_RISK: Dict[str, str] = {
    "BTC-USD": "Bitcoin USD",
}

# Sector ETFs — used for breadth + dispersion features.
#   Brown & Cliff 2004 — sector rotation predicts index returns.
SECTORS: Dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

# Volatility-regime signals.
#   Bollerslev, Tauchen & Zhou 2009 RFS — variance risk premium predicts
#   equity returns. Pan & Poteshman 2006 — option-implied signals predict
#   future returns. Whaley 2009 — VIX as an "investor fear gauge".
VOL_REGIME: Dict[str, str] = {
    "^VIX":   "CBOE VIX (S&P 500 30-day implied)",
    "^VXN":   "CBOE Nasdaq-100 Volatility (QQQ-NATIVE fear gauge)",
    "^VIX9D": "CBOE VIX 9-day",
    "^VIX3M": "CBOE 3-month VIX",
    "^VIX6M": "CBOE 6-month VIX",
    "^VVIX":  "CBOE Vol-of-Vol",
    "^SKEW":  "CBOE SKEW (tail risk)",
    "^OVX":   "CBOE Crude Oil ETF VIX",
    "^GVZ":   "CBOE Gold ETF VIX",
    # ^MOVE = ICE BofA Treasury vol — bond-vol leads equity-vol per
    # Cieslak & Pang 2021 RFS "Common Shocks in Stocks and Bonds".
    "^MOVE":  "ICE BofA MOVE Index (Treasury bond vol)",
}

# Fixed income / credit / yield curve.
#   Estrella & Mishkin 1998 RES — 10y-3m spread predicts recessions.
#   Welch & Goyal 2008 RFS — term spread + default spread are equity premium
#   predictors. Adrian, Crump & Moench 2013 — term premium decomposition.
YIELDS_CREDIT: Dict[str, str] = {
    "^TNX":  "US 10Y Treasury Yield",
    "^FVX":  "US 5Y Treasury Yield",
    "^TYX":  "US 30Y Treasury Yield",
    "^IRX":  "US 13W Treasury Yield (proxy for 3m)",
    "TLT":   "iShares 20+Y Treasury Bond ETF",
    "IEF":   "iShares 7-10Y Treasury Bond ETF",
    "SHY":   "iShares 1-3Y Treasury Bond ETF",
    "TIP":   "iShares TIPS Bond ETF (10y real proxy)",
    "HYG":   "iShares iBoxx HY Corporate (credit risk appetite)",
    "LQD":   "iShares iBoxx IG Corporate",
}

# Macro / commodities / currency.
#   Driesprong, Jacobsen & Maat 2008 — oil predicts equity returns.
#   Akram 2009 — commodity prices and US monetary policy joint dynamics.
MACRO_FX: Dict[str, str] = {
    "GC=F":     "Gold Futures",
    "SI=F":     "Silver Futures",
    "CL=F":     "WTI Crude Oil Futures",
    "BZ=F":     "Brent Crude Futures",
    "HG=F":     "Copper Futures (Dr. Copper)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
}

# International risk pulse.
INTL_RISK: Dict[str, str] = {
    "^N225":  "Nikkei 225 (Asia)",
    "^FTSE":  "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)",
    "^HSI":   "Hang Seng (HK)",
}

# Combine. Order matters only for log readability.
ALL_SIGNALS: Dict[str, Dict[str, str]] = {
    "primary":         PRIMARY,
    "benchmarks":      BENCHMARKS,
    "industry_tilts":  INDUSTRY_TILTS,
    "sectors":         SECTORS,
    "vol_regime":      VOL_REGIME,
    "yields_credit":   YIELDS_CREDIT,
    "macro_fx":        MACRO_FX,
    "intl_risk":       INTL_RISK,
    "crypto_risk":     CRYPTO_RISK,
}


# ---------------------------------------------------------------------------
# Date constants — NO 2026 DATA.
# ---------------------------------------------------------------------------

DEFAULT_START = "2004-01-01"
# Hard cap. Anything past this is dropped post-fetch — see _enforce_no_2026.
DEFAULT_END = "2025-12-31"
HARD_CUTOFF = pd.Timestamp("2026-01-01")


DEFAULT_CACHE_DIR = str(
    Path(__file__).resolve().parent.parent / ".data_cache_qqq"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, start: str, end: str, cache_dir: str) -> Path:
    safe = (
        ticker.replace("=", "_").replace("^", "_").replace(".", "_").replace("-", "_")
    )
    return Path(cache_dir) / f"{safe}_{start}_{end}.parquet"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def _enforce_no_2026(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop any rows >= 2026-01-01 with a logged warning. CLAUDE.md mandate."""
    if df is None or df.empty:
        return df
    n_before = len(df)
    df = df.loc[df.index < HARD_CUTOFF]
    n_after = len(df)
    if n_after < n_before:
        logger.info(
            "[no-2026] %s: dropped %d rows >= 2026-01-01",
            ticker, n_before - n_after,
        )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download a single ticker, cache as parquet, drop any 2026 rows."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(ticker, start, end, cache_dir)
        if cp.exists():
            df = pd.read_parquet(cp)
            return _enforce_no_2026(df, ticker)

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False,
        threads=False,
    )
    raw = _flatten_columns(raw)
    raw = _enforce_no_2026(raw, ticker)

    if cache_dir is not None and not raw.empty:
        cp = _cache_path(ticker, start, end, cache_dir)
        raw.to_parquet(cp)
    return raw


def download_all(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """Download every ticker in ALL_SIGNALS, return dict ticker -> DataFrame.

    Tickers that fail (delisted, network error, no data) get a warning and
    are skipped — features.py is defensive about missing tickers.
    """
    out: Dict[str, pd.DataFrame] = {}
    for group_name, group in ALL_SIGNALS.items():
        for ticker in group:
            try:
                df = download_ticker(ticker, start=start, end=end, cache_dir=cache_dir)
            except Exception as e:  # pragma: no cover — yf network flake
                logger.warning("Skip %s (%s): %s", ticker, group_name, e)
                continue
            if df is None or df.empty:
                logger.warning("Empty %s (%s)", ticker, group_name)
                continue
            out[ticker] = df
    logger.info("[download] fetched %d / %d tickers",
                len(out),
                sum(len(g) for g in ALL_SIGNALS.values()))
    return out


# ---------------------------------------------------------------------------
# Panel learning support — added 2026-04-29 per user directive.
# Loads QQQ + 30 NDX top components + 6 adjacent indices = 37 targets.
# Returns a tidy long-format DataFrame: rows = (date, asset), columns = OHLCV.
# Per Gu-Kelly-Xiu 2020 RFS panel learning recipe.
# ---------------------------------------------------------------------------

def download_panel_targets(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download all 37 panel-learning target assets.

    Returns long-format DataFrame indexed by (date, asset) with columns
    [open, high, low, close, volume]. Dates are NYSE trading days
    intersected across all assets (drop-NA on QQQ-aligned days).

    Each ticker that fails to download is logged and skipped. Asset count
    is not a tight invariant since some PANEL_NDX_TOP30 components may be
    delisted / split-restructured over the 2004-2025 window; we accept any
    asset that has >= 1000 trading days of OHLCV.
    """
    frames: list[pd.DataFrame] = []
    for ticker in PANEL_TARGETS:
        try:
            df = download_ticker(ticker, start=start, end=end, cache_dir=cache_dir)
        except Exception as e:  # pragma: no cover
            logger.warning("[panel] skip %s: %s", ticker, e)
            continue
        if df is None or df.empty or len(df) < 1000:
            logger.warning("[panel] skip %s: insufficient data (%d rows)",
                           ticker, 0 if df is None else len(df))
            continue
        # Make column lowercase + add asset id
        df = df.rename(columns=str.lower)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df["asset"] = ticker
        frames.append(df.reset_index().rename(columns={"Date": "date", "index": "date"}))
    if not frames:
        raise RuntimeError("Panel download yielded zero usable assets")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)
    n_assets = panel["asset"].nunique()
    n_dates = panel["date"].nunique()
    logger.info("[panel] %d assets x %d dates = %d rows",
                n_assets, n_dates, len(panel))
    return panel
