"""Tests for data.features – ~55 backward-looking features from FX + macro data."""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, freq="B")
    close = 1.10 + np.cumsum(rng.randn(n) * 0.002)
    high = close + rng.uniform(0.001, 0.005, n)
    low = close - rng.uniform(0.001, 0.005, n)
    # Ensure open is between low and high
    open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)
    volume = rng.randint(1_000, 100_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_multi_pair_data(n: int = 300, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return dict mapping 6 FX pair tickers to OHLCV DataFrames.

    Uses the same tickers as data.download.PAIRS.
    """
    tickers = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X", "EURGBP=X", "EURJPY=X"]
    data = {}
    for i, ticker in enumerate(tickers):
        data[ticker] = _make_ohlcv(n=n, seed=seed + i)
    return data


def _make_macro_data(n: int = 300, seed: int = 100) -> dict[str, pd.DataFrame]:
    """Return dict mapping 9 macro tickers to OHLCV-like DataFrames."""
    tickers = [
        "^VIX", "^TNX", "^IRX", "DX-Y.NYB", "GC=F",
        "CL=F", "^GSPC", "TLT", "HYG",
    ]
    data = {}
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n, freq="B")
    for i, t in enumerate(tickers):
        close = 50.0 + np.cumsum(rng.randn(n) * 0.5)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        open_ = low + rng.uniform(0.0, 1.0, n) * (high - low)
        volume = rng.randint(1000, 100000, n).astype(float)
        data[t] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSinglePairFeatures:
    """Tests for compute_single_pair_features."""

    def test_shape_and_columns(self):
        from autoresearch.data.features import compute_single_pair_features
        df = _make_ohlcv(300)
        result = compute_single_pair_features(df, prefix="EURUSD")
        # Should have same number of rows as input
        assert len(result) == len(df)
        # Check key columns are present and prefixed
        expected_cols = [
            "EURUSD_log_ret_1d",
            "EURUSD_log_ret_5d",
            "EURUSD_log_ret_21d",
            "EURUSD_rvol_5d",
            "EURUSD_rvol_21d",
            "EURUSD_rvol_63d",
            "EURUSD_rsi_14",
            "EURUSD_macd_line",
            "EURUSD_macd_signal",
            "EURUSD_macd_hist",
            "EURUSD_ohlc_range",
            "EURUSD_overnight_gap",
            "EURUSD_norm_true_range",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_after_warmup(self):
        from autoresearch.data.features import compute_single_pair_features, WARMUP_PERIOD
        df = _make_ohlcv(300)
        result = compute_single_pair_features(df, prefix="EURUSD")
        trimmed = result.iloc[WARMUP_PERIOD:]
        assert not trimmed.isna().any().any(), (
            f"NaN found after warmup:\n{trimmed.isna().sum()[trimmed.isna().sum() > 0]}"
        )

    def test_column_prefixes(self):
        from autoresearch.data.features import compute_single_pair_features
        df = _make_ohlcv(300)
        result = compute_single_pair_features(df, prefix="GBPUSD")
        for col in result.columns:
            assert col.startswith("GBPUSD_"), f"Column not prefixed: {col}"


class TestCrossPairFeatures:
    """Tests for cross-pair correlation features."""

    def test_cross_pair_columns_exist(self):
        from autoresearch.data.features import compute_all_features
        pairs = _make_multi_pair_data(300)
        macro = _make_macro_data(300)
        result = compute_all_features(pairs, macro, primary="EURUSD=X")
        # 5 cross-pair correlations (each non-primary pair with primary)
        cross_cols = [c for c in result.columns if "corr_21d" in c]
        assert len(cross_cols) == 5, f"Expected 5 cross-pair corr cols, got {len(cross_cols)}: {cross_cols}"


class TestMacroFeatures:
    """Tests for macro features."""

    def test_macro_columns_exist(self):
        from autoresearch.data.features import compute_all_features
        pairs = _make_multi_pair_data(300)
        macro = _make_macro_data(300)
        result = compute_all_features(pairs, macro, primary="EURUSD=X")
        # Check yield curve slope
        assert "yield_curve_slope" in result.columns
        # Check VIX 5d change
        assert "VIX_5d_chg" in result.columns
        # Check DXY 21d vol
        assert "DXY_rvol_21d" in result.columns


class TestAllFeaturesCombined:
    """Tests for the full compute_all_features pipeline."""

    def test_min_columns(self):
        from autoresearch.data.features import compute_all_features
        pairs = _make_multi_pair_data(300)
        macro = _make_macro_data(300)
        result = compute_all_features(pairs, macro, primary="EURUSD=X")
        assert result.shape[1] >= 40, f"Expected >= 40 columns, got {result.shape[1]}"

    def test_no_nan(self):
        from autoresearch.data.features import compute_all_features
        pairs = _make_multi_pair_data(300)
        macro = _make_macro_data(300)
        result = compute_all_features(pairs, macro, primary="EURUSD=X")
        assert not result.isna().any().any(), (
            f"NaN in combined features:\n{result.isna().sum()[result.isna().sum() > 0]}"
        )

    def test_warmup_rows_dropped(self):
        from autoresearch.data.features import compute_all_features, WARMUP_PERIOD
        pairs = _make_multi_pair_data(300)
        macro = _make_macro_data(300)
        result = compute_all_features(pairs, macro, primary="EURUSD=X")
        # After dropping warmup, should have fewer rows than input
        assert len(result) <= 300 - WARMUP_PERIOD


class TestBackwardOnly:
    """The most critical test: features must be strictly backward-looking.

    Features computed on data[0:200] at time t must equal features computed
    on data[0:150] at the same time t, for all t in the overlap.
    """

    def test_single_pair_backward_only(self):
        from autoresearch.data.features import compute_single_pair_features, WARMUP_PERIOD
        full_df = _make_ohlcv(250)
        trunc_df = full_df.iloc[:180].copy()

        full_feats = compute_single_pair_features(full_df, prefix="TEST")
        trunc_feats = compute_single_pair_features(trunc_df, prefix="TEST")

        # Overlap: rows present in both, after warmup
        overlap_idx = trunc_feats.index[WARMUP_PERIOD:]
        full_overlap = full_feats.loc[overlap_idx]
        trunc_overlap = trunc_feats.loc[overlap_idx]

        pd.testing.assert_frame_equal(
            full_overlap, trunc_overlap, check_exact=False, atol=1e-12,
            obj="Backward-only check (single pair)",
        )

    def test_all_features_backward_only(self):
        from autoresearch.data.features import compute_all_features, WARMUP_PERIOD
        full_pairs = _make_multi_pair_data(250)
        full_macro = _make_macro_data(250)

        trunc_pairs = {k: v.iloc[:180].copy() for k, v in full_pairs.items()}
        trunc_macro = {k: v.iloc[:180].copy() for k, v in full_macro.items()}

        full_feats = compute_all_features(full_pairs, full_macro, primary="EURUSD=X")
        trunc_feats = compute_all_features(trunc_pairs, trunc_macro, primary="EURUSD=X")

        overlap_idx = full_feats.index.intersection(trunc_feats.index)
        assert len(overlap_idx) > 50, f"Overlap too small: {len(overlap_idx)}"

        full_overlap = full_feats.loc[overlap_idx]
        trunc_overlap = trunc_feats.loc[overlap_idx]

        pd.testing.assert_frame_equal(
            full_overlap, trunc_overlap, check_exact=False, atol=1e-12,
            obj="Backward-only check (all features)",
        )


class TestTargets:
    """Tests for compute_targets."""

    def test_target_shape(self):
        from autoresearch.data.features import compute_targets
        df = _make_ohlcv(100)
        targets = compute_targets(df)
        # Should have 2 columns: fwd_ret_1d, fwd_ret_5d
        assert targets.shape[1] == 2
        assert "fwd_ret_1d" in targets.columns
        assert "fwd_ret_5d" in targets.columns

    def test_target_no_nan(self):
        from autoresearch.data.features import compute_targets
        df = _make_ohlcv(100)
        targets = compute_targets(df)
        assert not targets.isna().any().any()

    def test_target_values_correct(self):
        """Verify forward returns numerically."""
        from autoresearch.data.features import compute_targets
        df = _make_ohlcv(100)
        targets = compute_targets(df)
        close = df["Close"]

        # Forward 1-day return: pct_change(1).shift(-1)
        expected_1d = close.pct_change(1).shift(-1)
        # Forward 5-day return: pct_change(5).shift(-5)
        expected_5d = close.pct_change(5).shift(-5)

        # After dropna, verify values match
        common_idx = targets.index
        np.testing.assert_array_almost_equal(
            targets["fwd_ret_1d"].values,
            expected_1d.loc[common_idx].values,
            decimal=12,
        )
        np.testing.assert_array_almost_equal(
            targets["fwd_ret_5d"].values,
            expected_5d.loc[common_idx].values,
            decimal=12,
        )

    def test_targets_fewer_rows_than_input(self):
        """Targets drop NaN rows, so fewer rows than input."""
        from autoresearch.data.features import compute_targets
        df = _make_ohlcv(100)
        targets = compute_targets(df)
        # At minimum, first row (no prev for pct_change) and last 5 (shift) are gone
        assert len(targets) < len(df)
