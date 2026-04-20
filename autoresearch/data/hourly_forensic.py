"""Download and cache hourly data for DXY vs EUR/USD forensic analysis.

yfinance limits hourly data to ~730 days per request.
We download in 60-day chunks with rate limiting and cache each chunk.
"""

import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".data_cache" / "hourly"


def download_hourly_chunked(
    ticker: str,
    start: str = "2024-01-01",
    end: str = "2026-04-14",
    chunk_days: int = 59,
    sleep_sec: float = 1.5,
) -> pd.DataFrame:
    """Download hourly data in chunks, caching each to disk.

    Args:
        ticker: yfinance ticker (e.g., 'EURUSD=X', 'DX-Y.NYB')
        start: start date string
        end: end date string
        chunk_days: days per request (yfinance limit ~730, use 59 for safety)
        sleep_sec: seconds between requests to avoid rate limiting

    Returns:
        Combined DataFrame of all chunks, sorted by datetime index.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("=", "_").replace(".", "_").replace("^", "_")

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    all_chunks = []
    current = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        cache_file = CACHE_DIR / f"{safe_ticker}_{current.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}_1h.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            print(f"  Cache hit: {cache_file.name} ({len(df)} rows)")
        else:
            print(f"  Downloading {ticker} {current.date()} to {chunk_end.date()}...", end="", flush=True)
            try:
                df = yf.download(
                    ticker,
                    start=current.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval="1h",
                    progress=False,
                )
                # Flatten multi-level columns if present
                if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)

                if len(df) > 0:
                    df.to_parquet(cache_file)
                    print(f" {len(df)} rows cached")
                else:
                    print(f" empty")
            except Exception as e:
                print(f" ERROR: {e}")
                df = pd.DataFrame()
            time.sleep(sleep_sec)

        if len(df) > 0:
            all_chunks.append(df)

        current = chunk_end

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    return combined


def run_forensic_analysis():
    """Download hourly EUR/USD and DXY, then compute lag-correlation structure."""
    from scipy.stats import spearmanr
    import numpy as np

    print("=" * 70)
    print("FORENSIC: Downloading hourly EUR/USD")
    print("=" * 70)
    eur_h = download_hourly_chunked("EURUSD=X", start="2024-01-01", end="2026-04-14")

    print()
    print("=" * 70)
    print("FORENSIC: Downloading hourly DXY")
    print("=" * 70)
    dxy_h = download_hourly_chunked("DX-Y.NYB", start="2024-01-01", end="2026-04-14")

    if len(eur_h) == 0 or len(dxy_h) == 0:
        print("ERROR: No data downloaded")
        return

    eur_close = eur_h["Close"]
    dxy_close = dxy_h["Close"]

    # Align on common timestamps
    common = eur_close.index.intersection(dxy_close.index)
    print(f"\nCommon hourly timestamps: {len(common)}")
    print(f"Date range: {common[0]} to {common[-1]}")

    eur_c = eur_close.loc[common]
    dxy_c = dxy_close.loc[common]

    # Compute hourly returns
    eur_ret = eur_c.pct_change(1)
    dxy_ret = dxy_c.pct_change(1)

    # Compute IC at each lag from -48h to +48h
    # IC(DXY_ret(t), EUR_ret(t + lag))
    print("\n" + "=" * 70)
    print("LAG-CORRELATION STRUCTURE (hourly)")
    print("IC(DXY_ret(t), EUR_ret(t + lag))")
    print("=" * 70)
    print(f"{'Lag (hours)':>12s} {'IC':>8s} {'p-value':>10s} {'Interpretation':>30s}")
    print("-" * 65)

    for lag in range(-6, 49):
        eur_shifted = eur_ret.shift(-lag)  # EUR at t+lag
        valid = ~(dxy_ret.isna() | eur_shifted.isna())
        if valid.sum() < 100:
            continue
        ic, pval = spearmanr(dxy_ret[valid], eur_shifted[valid])

        # Interpretation
        if lag == 0:
            interp = "Same hour (contemporaneous)"
        elif lag < 0:
            interp = f"EUR {-lag}h BEFORE DXY"
        elif lag <= 6:
            interp = f"EUR {lag}h after DXY (same day)"
        elif lag <= 24:
            interp = f"EUR {lag}h after (~next day)"
        else:
            interp = f"EUR {lag}h after (2 days out)"

        marker = ""
        if abs(ic) > 0.3:
            marker = " *** HIGH"
        elif abs(ic) > 0.1:
            marker = " ** moderate"

        print(f"{lag:>12d} {ic:>+8.4f} {pval:>10.2e} {interp:>30s}{marker}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("If IC peaks at lag=0 and decays: CONTEMPORANEOUS (same price moves)")
    print("If IC peaks at lag=6:  DXY close to EUR close gap (date label artifact)")
    print("If IC peaks at lag=24: next-day momentum (genuine or leakage)")
    print("If IC is flat across lags: spurious / mechanical")


if __name__ == "__main__":
    run_forensic_analysis()
