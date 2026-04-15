"""Tests for data.download module."""

import os
import tempfile

import pandas as pd
import pytest

from autoresearch.data.download import (
    MACRO_TICKERS,
    PAIRS,
    download_all_pairs,
    download_macro_signals,
    download_pair,
)


# ---------------------------------------------------------------------------
# Dict sanity checks
# ---------------------------------------------------------------------------

def test_pairs_has_six_entries():
    assert len(PAIRS) == 6


def test_macro_tickers_has_nine_entries():
    assert len(MACRO_TICKERS) == 9


# ---------------------------------------------------------------------------
# Single pair download
# ---------------------------------------------------------------------------

def test_download_pair_returns_dataframe():
    """download_pair should return a DataFrame with OHLCV columns."""
    with tempfile.TemporaryDirectory() as tmp:
        df = download_pair("EURUSD=X", "2024-01-01", "2024-02-01", cache_dir=tmp)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns, f"Missing column: {col}"


def test_download_pair_no_multiindex():
    """Returned DataFrame should NOT have MultiIndex columns."""
    with tempfile.TemporaryDirectory() as tmp:
        df = download_pair("EURUSD=X", "2024-01-01", "2024-02-01", cache_dir=tmp)
        assert not isinstance(df.columns, pd.MultiIndex)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_caching_creates_parquet_file():
    """First call should create a parquet file in cache_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        download_pair("EURUSD=X", "2024-01-01", "2024-02-01", cache_dir=tmp)
        parquet_files = [f for f in os.listdir(tmp) if f.endswith(".parquet")]
        assert len(parquet_files) >= 1


def test_caching_second_call_uses_cache():
    """Second call should read from cache (parquet file already exists)."""
    with tempfile.TemporaryDirectory() as tmp:
        df1 = download_pair("EURUSD=X", "2024-01-01", "2024-02-01", cache_dir=tmp)
        # Second call — should hit cache, not network
        df2 = download_pair("EURUSD=X", "2024-01-01", "2024-02-01", cache_dir=tmp)
        # Parquet may round-trip datetime index at different resolution (s vs ms),
        # so we compare values only, not exact dtypes.
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, check_index_type=False)


# ---------------------------------------------------------------------------
# Bulk downloads
# ---------------------------------------------------------------------------

def test_download_all_pairs_returns_six():
    """download_all_pairs should return a dict with 6 DataFrames."""
    with tempfile.TemporaryDirectory() as tmp:
        result = download_all_pairs("2024-01-01", "2024-02-01", cache_dir=tmp)
        assert isinstance(result, dict)
        assert len(result) == 6
        for key, df in result.items():
            assert isinstance(df, pd.DataFrame)


def test_download_macro_signals_returns_at_least_six():
    """download_macro_signals should return at least 6 DataFrames (tolerant of failures)."""
    with tempfile.TemporaryDirectory() as tmp:
        result = download_macro_signals("2024-01-01", "2024-02-01", cache_dir=tmp)
        assert isinstance(result, dict)
        assert len(result) >= 6, f"Expected >=6 macro signals, got {len(result)}"
