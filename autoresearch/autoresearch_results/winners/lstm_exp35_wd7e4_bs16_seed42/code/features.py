"""Feature engineering: ~55+ backward-looking features from 6 FX pairs + 9 macro signals.

All features are strictly backward-looking -- no future data leakage.

Feature groups
--------------
1. Per-pair technical (13 per pair): log returns, rolling vol, RSI, MACD, microstructure
2. Cross-pair (5):  rolling 21d correlation of primary pair vs each secondary pair
3. Macro (21):  returns & levels for 9 tickers, yield-curve slope, VIX chg, DXY vol
4. Targets (separate): forward 1d and 5d returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP_PERIOD: int = 63  # rows to drop at start (longest lookback window)

# Fallback PAIRS dict -- used if data.download is not yet available.
_PAIRS_FALLBACK: dict[str, str] = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "JPY=X":    "USD/JPY",
    "CHF=X":    "USD/CHF",
    "EURGBP=X": "EUR/GBP",
    "EURJPY=X": "EUR/JPY",
}

MACRO_TICKERS: list[str] = [
    "^VIX", "^TNX", "^IRX", "DX-Y.NYB", "GC=F",
    "CL=F", "^GSPC", "TLT", "HYG",
]

# Friendly short names for macro tickers (used in column naming)
_MACRO_SHORT: dict[str, str] = {
    "^VIX": "VIX",
    "^TNX": "TNX",
    "^IRX": "IRX",
    "DX-Y.NYB": "DXY",
    "GC=F": "GOLD",
    "CL=F": "OIL",
    "^GSPC": "SPX",
    "TLT": "TLT",
    "HYG": "HYG",
}


def _get_pairs() -> dict[str, str]:
    """Import PAIRS from data.download, falling back to built-in dict."""
    try:
        from .download import PAIRS  # type: ignore[import-untyped]
        return PAIRS
    except (ImportError, ModuleNotFoundError):
        return _PAIRS_FALLBACK


def _sanitise_prefix(name: str) -> str:
    """Turn 'EUR/USD' -> 'EURUSD' for clean column names."""
    return name.replace("/", "")


# ====================================================================
# 1.  Per-Pair Technical Features
# ====================================================================

def compute_single_pair_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute 13 technical features for one FX pair.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close (and optionally Volume).
    prefix : short pair name used to prefix every column, e.g. "EURUSD".

    Returns
    -------
    DataFrame with same index as *df*, 13 columns named ``{prefix}_*``.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    feats: dict[str, pd.Series] = {}

    # --- Log returns ---
    log_close = np.log(close)
    feats["log_ret_1d"] = log_close.diff(1)
    feats["log_ret_5d"] = log_close.diff(5)
    feats["log_ret_21d"] = log_close.diff(21)

    # --- Rolling volatility (annualised, 252 trading days) ---
    daily_log_ret = log_close.diff(1)
    feats["rvol_5d"] = daily_log_ret.rolling(5).std() * np.sqrt(252)
    feats["rvol_21d"] = daily_log_ret.rolling(21).std() * np.sqrt(252)
    feats["rvol_63d"] = daily_log_ret.rolling(63).std() * np.sqrt(252)

    # --- RSI (14-day, Wilder smoothing) ---
    feats["rsi_14"] = _rsi(close, period=14)

    # --- MACD (12 / 26 / 9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_line"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_line - macd_signal

    # --- Microstructure ---
    feats["ohlc_range"] = (high - low) / close
    feats["overnight_gap"] = open_ / close.shift(1) - 1
    feats["norm_true_range"] = _normalised_true_range(high, low, close)

    # Prefix all column names
    out = pd.DataFrame(
        {f"{prefix}_{name}": series for name, series in feats.items()},
        index=df.index,
    )
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI -- strictly backward-looking."""
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _normalised_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """ATR-style true range normalised by close, single-day."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr / close


# ====================================================================
# 2.  Cross-Pair Features
# ====================================================================

def _compute_cross_pair_features(
    all_pairs: dict[str, pd.DataFrame],
    primary: str = "EURUSD=X",
) -> pd.DataFrame:
    """Rolling 21d correlation between primary pair returns and each other pair."""
    pairs_map = _get_pairs()
    primary_name = _sanitise_prefix(
        pairs_map.get(primary, primary.replace("=X", ""))
    )
    primary_ret = np.log(all_pairs[primary]["Close"]).diff(1)

    feats: dict[str, pd.Series] = {}
    for ticker, df in all_pairs.items():
        if ticker == primary:
            continue
        name = _sanitise_prefix(
            pairs_map.get(ticker, ticker.replace("=X", ""))
        )
        other_ret = np.log(df["Close"]).diff(1)
        # Align on common index
        combined = pd.concat(
            [primary_ret.rename("primary"), other_ret.rename("other")], axis=1
        ).dropna()
        corr = combined["primary"].rolling(21).corr(combined["other"])
        feats[f"{primary_name}_{name}_corr_21d"] = corr

    return pd.DataFrame(feats)


# ====================================================================
# 3.  Macro Features
# ====================================================================

def _compute_macro_features(
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return macro features: per-ticker returns+levels, slope, VIX chg, DXY vol."""
    feats: dict[str, pd.Series] = {}

    for ticker, df in macro_data.items():
        short = _MACRO_SHORT.get(ticker, ticker)
        close = df["Close"]
        feats[f"{short}_level"] = close
        feats[f"{short}_ret_1d"] = close.pct_change(1)

    # --- Derived macro features ---
    # Yield-curve slope: 10Y minus 3M
    if "^TNX" in macro_data and "^IRX" in macro_data:
        feats["yield_curve_slope"] = (
            macro_data["^TNX"]["Close"] - macro_data["^IRX"]["Close"]
        )

    # VIX 5-day change
    if "^VIX" in macro_data:
        feats["VIX_5d_chg"] = macro_data["^VIX"]["Close"].diff(5)

    # DXY 21-day rolling volatility (annualised)
    if "DX-Y.NYB" in macro_data:
        dxy_ret = np.log(macro_data["DX-Y.NYB"]["Close"]).diff(1)
        feats["DXY_rvol_21d"] = dxy_ret.rolling(21).std() * np.sqrt(252)

    return pd.DataFrame(feats)


# ====================================================================
# 4.  Combine Everything
# ====================================================================

def compute_all_features(
    all_pairs: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame],
    primary: str = "EURUSD=X",
) -> pd.DataFrame:
    """Build the full feature matrix.

    Parameters
    ----------
    all_pairs : {ticker: OHLCV DataFrame} for 6 FX pairs.
    macro_data : {ticker: OHLCV DataFrame} for 9 macro tickers.
    primary : ticker of the primary FX pair (default ``"EURUSD=X"``).

    Returns
    -------
    DataFrame aligned on common dates, warmup rows dropped, NaN rows dropped.
    """
    pairs_map = _get_pairs()

    # 1) Per-pair features
    pair_frames: list[pd.DataFrame] = []
    for ticker, df in all_pairs.items():
        name = _sanitise_prefix(
            pairs_map.get(ticker, ticker.replace("=X", ""))
        )
        pair_frames.append(compute_single_pair_features(df, prefix=name))

    # 2) Cross-pair features
    cross = _compute_cross_pair_features(all_pairs, primary=primary)

    # 3) Macro features
    macro = _compute_macro_features(macro_data)

    # Concatenate on common dates
    combined = pd.concat(pair_frames + [cross, macro], axis=1, join="inner")

    # Drop warmup rows
    combined = combined.iloc[WARMUP_PERIOD:]

    # Drop any remaining NaN rows
    combined = combined.dropna()

    return combined


# ====================================================================
# 5.  Targets (forward-looking -- kept separate from features)
# ====================================================================

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward 1-day and 5-day returns.

    Parameters
    ----------
    df : OHLCV DataFrame with a Close column.

    Returns
    -------
    DataFrame with columns ``fwd_ret_1d`` and ``fwd_ret_5d``, NaN rows dropped.
    """
    close = df["Close"]
    targets = pd.DataFrame(
        {
            "fwd_ret_1d": close.pct_change(1).shift(-1),
            "fwd_ret_5d": close.pct_change(5).shift(-5),
        },
        index=df.index,
    )
    return targets.dropna()
