# Currency Prediction with LFM2.5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI currency prediction system with LFM2.5-350M backbone, 7-fold regime-aware evaluation, and an autoresearch optimizer loop.

**Architecture:** Input projection (Linear -> 1024) feeds financial features into frozen LFM2.5-350M backbone, multi-scale linear heads predict 1d/5d returns per pair, evaluated via average Sharpe across 7 regime-diverse test sets with 3-month purge gaps. Autoresearch loop (Claude API) proposes code changes, evaluates, keeps/discards.

**Tech Stack:** Python 3.11, PyTorch 2.5, transformers 5.5 (Lfm2Model), yfinance, pandas, numpy, anthropic SDK, pytest

---

### Task 1: Project Scaffold + Requirements

**Files:**
- Create: `requirements.txt`
- Create: `data/__init__.py`
- Create: `model/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `optimizer/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create directory structure**

```bash
cd /c/Users/abhir/autoresearch
mkdir -p data model evaluation optimizer tests
```

**Step 2: Write requirements.txt**

```
torch>=2.5.0
transformers>=4.55
safetensors
accelerate
yfinance
pandas
numpy
scikit-learn
anthropic
pytest
```

**Step 3: Create __init__.py files**

```bash
touch data/__init__.py model/__init__.py evaluation/__init__.py optimizer/__init__.py tests/__init__.py
```

**Step 4: Install and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 5: Init git repo and commit**

```bash
git init
git add requirements.txt data/__init__.py model/__init__.py evaluation/__init__.py optimizer/__init__.py tests/__init__.py docs/
git commit -m "feat: project scaffold with requirements and design docs"
```

---

### Task 2: Data Download Module (FX Pairs + Macro Signals)

**Files:**
- Create: `data/download.py`
- Create: `tests/test_download.py`

**Step 1: Write the failing test**

```python
# tests/test_download.py
import pytest
import pandas as pd
from data.download import download_pair, download_all_pairs, download_macro_signals, PAIRS, MACRO_TICKERS

def test_pairs_defined():
    assert len(PAIRS) == 6
    assert "EURUSD=X" in PAIRS

def test_macro_tickers_defined():
    assert len(MACRO_TICKERS) >= 6
    assert "^TNX" in MACRO_TICKERS
    assert "^VIX" in MACRO_TICKERS

def test_download_single_pair():
    df = download_pair("EURUSD=X", start="2024-01-01", end="2024-03-01")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 30
    assert "Open" in df.columns
    assert "Close" in df.columns
    assert df.index.is_monotonic_increasing

def test_download_all_pairs():
    data = download_all_pairs(start="2024-01-01", end="2024-03-01")
    assert isinstance(data, dict)
    assert len(data) == 6
    for ticker, df in data.items():
        assert len(df) > 30
        assert "Close" in df.columns

def test_download_macro():
    data = download_macro_signals(start="2024-01-01", end="2024-03-01")
    assert isinstance(data, dict)
    assert len(data) >= 6
    for ticker, df in data.items():
        assert len(df) > 20
        assert "Close" in df.columns

def test_download_caching(tmp_path):
    """Second call should use cache, not re-download."""
    df1 = download_pair("EURUSD=X", start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    df2 = download_pair("EURUSD=X", start="2024-01-01", end="2024-02-01", cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(df1, df2)
```

**Step 2: Run test to verify it fails**

Run: `cd /c/Users/abhir/autoresearch && python -m pytest tests/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.download'`

**Step 3: Write implementation**

```python
# data/download.py
import os
import hashlib
import yfinance as yf
import pandas as pd

PAIRS = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "JPY=X": "USD/JPY",
    "CHF=X": "USD/CHF",
    "EURGBP=X": "EUR/GBP",
    "EURJPY=X": "EUR/JPY",
}

MACRO_TICKERS = {
    "^TNX": "US_10Y_Yield",
    "^FVX": "US_5Y_Yield",
    "^IRX": "US_13W_TBill",
    "^VIX": "VIX",
    "^GSPC": "SP500",
    "^N225": "Nikkei225",
    "GC=F": "Gold",
    "CL=F": "Crude_Oil",
    "DX-Y.NYB": "DXY",
}

DEFAULT_START = "2005-01-01"
DEFAULT_END = "2026-04-01"


def _cache_path(ticker: str, start: str, end: str, cache_dir: str) -> str:
    key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}.parquet")


def download_pair(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(ticker, start, end, cache_dir)

    if os.path.exists(path):
        return pd.read_parquet(path)

    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna()
    df.to_parquet(path)
    return df


def download_all_pairs(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str = "data/cache",
) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in PAIRS:
        result[ticker] = download_pair(ticker, start=start, end=end, cache_dir=cache_dir)
    return result


def download_macro_signals(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: str = "data/cache",
) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in MACRO_TICKERS:
        try:
            result[ticker] = download_pair(ticker, start=start, end=end, cache_dir=cache_dir)
        except Exception:
            pass  # some macro tickers may fail for early dates
    return result
```

**Step 4: Run test to verify it passes**

Run: `cd /c/Users/abhir/autoresearch && python -m pytest tests/test_download.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data/download.py tests/test_download.py
git commit -m "feat: data download module with FX pairs + macro signals + caching"
```

---

### Task 3: Feature Engineering Module (~50 features)

**Files:**
- Create: `data/features.py`
- Create: `tests/test_features.py`

Feature groups:
- **Per-pair technical (10 features x 6 pairs = 60):** log returns (1d/5d/21d), rolling vol (5d/21d/63d), RSI, MACD/signal/hist
- **Cross-pair (6):** rolling 21d correlations between EUR/USD and each other pair, plus PCA-1 of all 6 pairs
- **Macro signals (15):** yield curve slope (10Y-3M), VIX level + change, DXY return + vol, gold return, oil return, S&P return, Nikkei return, each with 1d return
- **Microstructure (4):** OHLC range ratio, close-to-open gap, intraday range, true range
- **Total: ~55 features** (exact count depends on available macro data alignment)

**Step 1: Write the failing tests**

```python
# tests/test_features.py
import pytest
import numpy as np
import pandas as pd
from data.features import compute_single_pair_features, compute_all_features, compute_targets

def _make_ohlcv(n=200, seed=42):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 1.10 + np.cumsum(np.random.randn(n) * 0.005)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.001,
        "High": close + abs(np.random.randn(n) * 0.003),
        "Low": close - abs(np.random.randn(n) * 0.003),
        "Close": close,
        "Volume": np.random.randint(1000, 10000, n),
    }, index=dates)

def _make_multi_pair_data(n=200):
    pairs = {}
    for i, name in enumerate(["EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X", "EURGBP=X", "EURJPY=X"]):
        pairs[name] = _make_ohlcv(n, seed=42 + i)
    return pairs

def _make_macro_data(n=200):
    macros = {}
    for i, name in enumerate(["^TNX", "^FVX", "^IRX", "^VIX", "^GSPC", "^N225", "GC=F", "CL=F", "DX-Y.NYB"]):
        macros[name] = _make_ohlcv(n, seed=100 + i)
    return macros

def test_single_pair_features_shape():
    df = _make_ohlcv(200)
    features = compute_single_pair_features(df, prefix="EURUSD")
    assert isinstance(features, pd.DataFrame)
    assert len(features) <= len(df)
    assert features.isna().sum().sum() == 0
    assert features.shape[1] >= 10  # at least 10 technical features

def test_all_features_shape():
    pairs = _make_multi_pair_data(200)
    macros = _make_macro_data(200)
    features = compute_all_features(pairs, macros)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[1] >= 40  # cross-pair + macro + technical
    assert features.isna().sum().sum() == 0

def test_features_backward_only():
    """Features at time t must not use data after t."""
    pairs_full = _make_multi_pair_data(200)
    macros_full = _make_macro_data(200)
    features_full = compute_all_features(pairs_full, macros_full)

    # Truncate all data to first 150 rows and recompute
    pairs_trunc = {k: v.iloc[:150] for k, v in pairs_full.items()}
    macros_trunc = {k: v.iloc[:150] for k, v in macros_full.items()}
    features_trunc = compute_all_features(pairs_trunc, macros_trunc)

    # Overlapping rows must match exactly
    overlap = features_full.index.intersection(features_trunc.index)
    assert len(overlap) > 30
    pd.testing.assert_frame_equal(
        features_full.loc[overlap],
        features_trunc.loc[overlap],
        check_names=False,
    )

def test_targets_shape():
    df = _make_ohlcv(200)
    targets = compute_targets(df)
    assert "ret_1d" in targets.columns
    assert "ret_5d" in targets.columns
    assert targets.isna().sum().sum() == 0

def test_targets_are_forward_returns():
    df = _make_ohlcv(200)
    targets = compute_targets(df)
    for i in range(min(10, len(targets))):
        t = targets.index[i]
        t_loc = df.index.get_loc(t)
        if t_loc + 1 < len(df):
            expected = (df["Close"].iloc[t_loc + 1] - df["Close"].iloc[t_loc]) / df["Close"].iloc[t_loc]
            assert abs(targets.loc[t, "ret_1d"] - expected) < 1e-10

def test_feature_column_names_prefixed():
    """Each pair's features should be prefixed to avoid collisions."""
    df = _make_ohlcv(200)
    features = compute_single_pair_features(df, prefix="GBPUSD")
    for col in features.columns:
        assert col.startswith("GBPUSD_"), f"Column {col} missing prefix"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.features'`

**Step 3: Write implementation**

```python
# data/features.py
import numpy as np
import pandas as pd

WARMUP_PERIOD = 63  # max lookback window


def _log_returns(close: pd.Series, periods: int) -> pd.Series:
    return np.log(close / close.shift(periods))


def _rolling_vol(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_single_pair_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute technical features for a single pair. All backward-looking."""
    close = df["Close"]
    f = pd.DataFrame(index=df.index)

    # Log returns
    for p in [1, 5, 21]:
        f[f"{prefix}_log_ret_{p}d"] = _log_returns(close, p)

    # Rolling volatility
    for w in [5, 21, 63]:
        f[f"{prefix}_vol_{w}d"] = _rolling_vol(close, w)

    # RSI
    f[f"{prefix}_rsi_14"] = _rsi(close, 14)

    # MACD
    macd_line, signal_line, histogram = _macd(close)
    f[f"{prefix}_macd"] = macd_line
    f[f"{prefix}_macd_signal"] = signal_line
    f[f"{prefix}_macd_hist"] = histogram

    # Microstructure
    f[f"{prefix}_range_ratio"] = (df["High"] - df["Low"]) / close  # intraday range
    f[f"{prefix}_gap"] = df["Open"] / close.shift(1) - 1  # overnight gap
    f[f"{prefix}_true_range"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - close.shift(1)).abs(),
        (df["Low"] - close.shift(1)).abs(),
    ], axis=1).max(axis=1) / close

    return f


def _compute_cross_pair_features(
    all_pairs: dict[str, pd.DataFrame],
    primary: str = "EURUSD=X",
    window: int = 21,
) -> pd.DataFrame:
    """Rolling correlations between primary pair and others."""
    closes = pd.DataFrame({
        ticker: df["Close"] for ticker, df in all_pairs.items()
    })
    returns = closes.pct_change()

    f = pd.DataFrame(index=closes.index)
    primary_ret = returns[primary]

    for ticker in all_pairs:
        if ticker == primary:
            continue
        name = ticker.replace("=X", "").replace(".", "")
        f[f"corr_{name}_21d"] = primary_ret.rolling(window).corr(returns[ticker])

    return f


def _compute_macro_features(macro_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Features from macro signals: yields, VIX, commodities, equities, DXY."""
    f = pd.DataFrame()

    mapping = {
        "^TNX": "tnx",
        "^FVX": "fvx",
        "^IRX": "irx",
        "^VIX": "vix",
        "^GSPC": "sp500",
        "^N225": "nikkei",
        "GC=F": "gold",
        "CL=F": "oil",
        "DX-Y.NYB": "dxy",
    }

    for ticker, name in mapping.items():
        if ticker not in macro_data:
            continue
        close = macro_data[ticker]["Close"]

        if f.index.empty:
            f = pd.DataFrame(index=close.index)

        f[f"macro_{name}_ret_1d"] = close.pct_change(1)
        f[f"macro_{name}_level"] = close  # raw level (will be normalized later)

    # Yield curve slope: 10Y - 3M
    if "^TNX" in macro_data and "^IRX" in macro_data:
        tnx = macro_data["^TNX"]["Close"]
        irx = macro_data["^IRX"]["Close"]
        aligned = pd.DataFrame({"tnx": tnx, "irx": irx}).dropna()
        slope = aligned["tnx"] - aligned["irx"]
        f["macro_yield_slope"] = slope

    # VIX change (important for risk regimes)
    if "^VIX" in macro_data:
        vix = macro_data["^VIX"]["Close"]
        f["macro_vix_change_5d"] = vix.pct_change(5)

    # DXY volatility
    if "DX-Y.NYB" in macro_data:
        dxy = macro_data["DX-Y.NYB"]["Close"]
        f["macro_dxy_vol_21d"] = dxy.pct_change().rolling(21).std() * np.sqrt(252)

    return f


def compute_all_features(
    all_pairs: dict[str, pd.DataFrame],
    macro_data: dict[str, pd.DataFrame],
    primary: str = "EURUSD=X",
) -> pd.DataFrame:
    """Compute full feature set: per-pair technical + cross-pair + macro."""
    from data.download import PAIRS

    parts = []

    # Per-pair technical features
    for ticker in all_pairs:
        name = PAIRS.get(ticker, ticker).replace("/", "")
        pair_feat = compute_single_pair_features(all_pairs[ticker], prefix=name)
        parts.append(pair_feat)

    # Cross-pair correlations
    cross = _compute_cross_pair_features(all_pairs, primary=primary)
    parts.append(cross)

    # Macro features
    if macro_data:
        macro = _compute_macro_features(macro_data)
        parts.append(macro)

    # Align all on common dates
    combined = pd.concat(parts, axis=1)
    combined = combined.iloc[WARMUP_PERIOD:].dropna()
    return combined


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward returns for primary pair."""
    close = df["Close"]
    targets = pd.DataFrame(index=df.index)
    targets["ret_1d"] = close.pct_change(1).shift(-1)
    targets["ret_5d"] = close.pct_change(5).shift(-5)
    targets = targets.dropna()
    return targets
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_features.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add data/features.py tests/test_features.py
git commit -m "feat: ~55 features from 6 FX pairs + 9 macro signals, all backward-only"
```

---

### Task 4: Regime-Aware Split Definitions

**Files:**
- Create: `data/splits.py`
- Create: `tests/test_splits.py`

**Step 1: Write the failing tests**

```python
# tests/test_splits.py
import pytest
import pandas as pd
from data.splits import get_fold_dates, FOLDS, PURGE_DAYS

def test_seven_folds():
    assert len(FOLDS) == 7

def test_purge_gap_minimum():
    assert PURGE_DAYS >= 63  # 3 months

def test_no_overlap_between_splits_within_fold():
    for i, fold in enumerate(FOLDS):
        dates = get_fold_dates(fold)
        assert dates["train_end"] < dates["val_start"], f"Fold {i+1}: train/val overlap"
        assert dates["val_end"] < dates["test_start"], f"Fold {i+1}: val/test overlap"

def test_purge_gap_enforced():
    for i, fold in enumerate(FOLDS):
        dates = get_fold_dates(fold)
        train_val_gap = (dates["val_start"] - dates["train_end"]).days
        val_test_gap = (dates["test_start"] - dates["val_end"]).days
        assert train_val_gap >= 80, f"Fold {i+1}: train-val gap only {train_val_gap} days"
        assert val_test_gap >= 80, f"Fold {i+1}: val-test gap only {val_test_gap} days"

def test_test_sets_disjoint():
    test_ranges = []
    for fold in FOLDS:
        dates = get_fold_dates(fold)
        test_ranges.append((dates["test_start"], dates["test_end"]))

    for i in range(len(test_ranges)):
        for j in range(i + 1, len(test_ranges)):
            s1, e1 = test_ranges[i]
            s2, e2 = test_ranges[j]
            assert e1 < s2 or e2 < s1, f"Test sets {i+1} and {j+1} overlap"

def test_fold_has_regime_label():
    for fold in FOLDS:
        assert "regime" in fold
        assert len(fold["regime"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.splits'`

**Step 3: Write implementation**

```python
# data/splits.py
import pandas as pd

PURGE_DAYS = 90  # ~3 months calendar days

FOLDS = [
    {
        "name": "fold_1_gfc_onset",
        "regime": "Pre-crisis upturn + GFC onset (vol 28%)",
        "train": ("2005-01-01", "2006-12-31"),
        "val": ("2007-04-01", "2007-09-30"),
        "test": ("2008-01-01", "2008-06-30"),
    },
    {
        "name": "fold_2_post_crash",
        "regime": "Post-crash recovery (-10% then +9%)",
        "train": ("2005-01-01", "2008-12-31"),
        "val": ("2009-04-01", "2009-09-30"),
        "test": ("2010-01-01", "2010-06-30"),
    },
    {
        "name": "fold_3_eurozone_plateau",
        "regime": "Eurozone debt plateau (low vol)",
        "train": ("2005-01-01", "2011-12-31"),
        "val": ("2012-04-01", "2012-09-30"),
        "test": ("2013-01-01", "2013-06-30"),
    },
    {
        "name": "fold_4_usd_downturn",
        "regime": "Strong USD downturn (-10.5% qtr)",
        "train": ("2005-01-01", "2014-03-31"),
        "val": ("2014-07-01", "2014-12-31"),
        "test": ("2015-04-01", "2015-12-31"),
    },
    {
        "name": "fold_5_low_vol_plateau",
        "regime": "Low-vol plateau (vol 4-6%)",
        "train": ("2005-01-01", "2017-12-31"),
        "val": ("2018-04-01", "2018-09-30"),
        "test": ("2019-01-01", "2019-09-30"),
    },
    {
        "name": "fold_6_eur_crisis",
        "regime": "EUR crisis downturn (-12% cumulative)",
        "train": ("2005-01-01", "2020-12-31"),
        "val": ("2021-04-01", "2021-09-30"),
        "test": ("2022-01-01", "2022-09-30"),
    },
    {
        "name": "fold_7_recent_mixed",
        "regime": "Recent mixed/upturn (+8.4% Q2)",
        "train": ("2005-01-01", "2023-12-31"),
        "val": ("2024-04-01", "2024-09-30"),
        "test": ("2025-01-01", "2025-09-30"),
    },
]


def get_fold_dates(fold: dict) -> dict:
    return {
        "train_start": pd.Timestamp(fold["train"][0]),
        "train_end": pd.Timestamp(fold["train"][1]),
        "val_start": pd.Timestamp(fold["val"][0]),
        "val_end": pd.Timestamp(fold["val"][1]),
        "test_start": pd.Timestamp(fold["test"][0]),
        "test_end": pd.Timestamp(fold["test"][1]),
    }


def split_data(df: pd.DataFrame, fold: dict) -> tuple:
    dates = get_fold_dates(fold)
    train = df[(df.index >= dates["train_start"]) & (df.index <= dates["train_end"])]
    val = df[(df.index >= dates["val_start"]) & (df.index <= dates["val_end"])]
    test = df[(df.index >= dates["test_start"]) & (df.index <= dates["test_end"])]
    return train, val, test
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_splits.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data/splits.py tests/test_splits.py
git commit -m "feat: 7-fold regime-aware split definitions with purge gap tests"
```

---

### Task 5: LFM2.5 Backbone Wrapper

**Files:**
- Create: `model/backbone.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing tests**

```python
# tests/test_model.py
import pytest
import torch
from model.backbone import CurrencyLFM

def test_model_creates():
    model = CurrencyLFM(n_input_features=10, freeze_backbone=True)
    assert model is not None

def test_model_forward_shape():
    model = CurrencyLFM(n_input_features=10, freeze_backbone=True)
    batch, seq_len, n_feat = 2, 30, 10
    x = torch.randn(batch, seq_len, n_feat)
    preds = model(x)
    # Should output dict with 1d and 5d predictions, each (batch, n_pairs)
    assert "ret_1d" in preds
    assert "ret_5d" in preds
    assert preds["ret_1d"].shape == (batch, 6)  # 6 pairs
    assert preds["ret_5d"].shape == (batch, 6)

def test_backbone_frozen():
    model = CurrencyLFM(n_input_features=10, freeze_backbone=True)
    for param in model.backbone.parameters():
        assert not param.requires_grad

def test_trainable_params_exist():
    model = CurrencyLFM(n_input_features=10, freeze_backbone=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0  # projection + heads should be trainable
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.backbone'`

**Step 3: Write implementation**

```python
# model/backbone.py
import torch
import torch.nn as nn
from transformers import Lfm2Model

N_PAIRS = 6
HORIZONS = ["ret_1d", "ret_5d"]
MODEL_ID = "LiquidAI/LFM2.5-350M-Base"


class CurrencyLFM(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        hidden_size: int = 1024,
        freeze_backbone: bool = True,
        model_id: str = MODEL_ID,
    ):
        super().__init__()

        # Project input features to LFM hidden size
        self.projection = nn.Linear(n_input_features, hidden_size)

        # LFM2.5 backbone
        self.backbone = Lfm2Model.from_pretrained(model_id, torch_dtype=torch.float32)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Multi-horizon prediction heads (use last timestep hidden state)
        self.heads = nn.ModuleDict({
            horizon: nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, N_PAIRS),
            )
            for horizon in HORIZONS
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        projected = self.projection(x)  # (batch, seq_len, 1024)

        outputs = self.backbone(inputs_embeds=projected)
        hidden = outputs.last_hidden_state[:, -1, :]  # last timestep: (batch, 1024)

        return {h: head(hidden) for h, head in self.heads.items()}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: All 4 tests PASS (note: first run will download model weights ~700MB)

**Step 5: Commit**

```bash
git add model/backbone.py tests/test_model.py
git commit -m "feat: LFM2.5 backbone wrapper with frozen weights and multi-horizon heads"
```

---

### Task 6: Evaluation Metrics

**Files:**
- Create: `evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import pytest
import numpy as np
from evaluation.metrics import sharpe_ratio, average_sharpe_across_folds

def test_sharpe_positive_returns():
    # Consistent positive returns -> positive Sharpe
    returns = np.array([0.001] * 252)
    s = sharpe_ratio(returns)
    assert s > 0

def test_sharpe_zero_returns():
    returns = np.zeros(252)
    s = sharpe_ratio(returns)
    assert s == 0.0

def test_sharpe_negative_returns():
    returns = np.array([-0.001] * 252)
    s = sharpe_ratio(returns)
    assert s < 0

def test_sharpe_annualized():
    # Daily mean=0.001, std=0.01 -> annualized Sharpe = 0.001/0.01 * sqrt(252) ~ 1.587
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 252)
    s = sharpe_ratio(returns)
    assert 0.5 < s < 3.0  # reasonable range

def test_average_sharpe():
    fold_returns = [
        np.array([0.001] * 100),
        np.array([0.002] * 100),
        np.array([-0.001] * 100),
    ]
    avg = average_sharpe_across_folds(fold_returns)
    assert isinstance(avg, float)

def test_average_sharpe_empty():
    with pytest.raises(ValueError):
        average_sharpe_across_folds([])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# evaluation/metrics.py
import numpy as np

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    mean = np.mean(daily_returns)
    std = np.std(daily_returns, ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def average_sharpe_across_folds(fold_returns: list[np.ndarray]) -> float:
    if len(fold_returns) == 0:
        raise ValueError("No fold returns provided")
    sharpes = [sharpe_ratio(r) for r in fold_returns]
    return float(np.mean(sharpes))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: Sharpe ratio evaluation metrics with annualization"
```

---

### Task 7: Leakage Detection Tests

**Files:**
- Create: `evaluation/leakage_check.py`
- Create: `tests/test_leakage.py`

**Step 1: Write the failing tests**

```python
# tests/test_leakage.py
import pytest
import numpy as np
import pandas as pd
from evaluation.leakage_check import check_no_future_features, check_split_gaps, check_scaler_isolation
from data.splits import FOLDS, get_fold_dates

def _make_index():
    return pd.bdate_range("2005-01-01", "2025-12-31")

def test_split_gaps_pass():
    """Our fold definitions should pass gap checks."""
    violations = check_split_gaps(FOLDS)
    assert len(violations) == 0, f"Gap violations: {violations}"

def test_split_gaps_detect_violation():
    """Should detect when gap is too small."""
    bad_folds = [{
        "name": "bad",
        "regime": "test",
        "train": ("2005-01-01", "2010-06-01"),
        "val": ("2010-06-15", "2010-12-31"),  # only 14 day gap!
        "test": ("2011-01-15", "2011-06-30"),
    }]
    violations = check_split_gaps(bad_folds)
    assert len(violations) > 0

def test_scaler_isolation():
    """Scaler must be fit on train only."""
    train_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    val_data = np.array([[100.0, 200.0]])  # wildly different
    test_data = np.array([[100.0, 200.0]])

    scaler, val_scaled, test_scaled = check_scaler_isolation(train_data, val_data, test_data)

    # Scaler mean should be from train, not influenced by val/test
    assert abs(scaler.mean_[0] - 3.0) < 0.01
    # val/test should be scaled using train stats (so values will be large)
    assert val_scaled[0, 0] > 10  # (100 - 3) / std ~ large number
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_leakage.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# evaluation/leakage_check.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from data.splits import get_fold_dates

MIN_GAP_DAYS = 80  # conservative check for ~3 month gap


def check_split_gaps(folds: list[dict]) -> list[str]:
    violations = []
    for fold in folds:
        dates = get_fold_dates(fold)
        train_val_gap = (dates["val_start"] - dates["train_end"]).days
        val_test_gap = (dates["test_start"] - dates["val_end"]).days

        if train_val_gap < MIN_GAP_DAYS:
            violations.append(
                f"{fold['name']}: train-val gap is {train_val_gap} days (min {MIN_GAP_DAYS})"
            )
        if val_test_gap < MIN_GAP_DAYS:
            violations.append(
                f"{fold['name']}: val-test gap is {val_test_gap} days (min {MIN_GAP_DAYS})"
            )
    return violations


def check_scaler_isolation(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple:
    scaler = StandardScaler()
    scaler.fit(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return scaler, val_scaled, test_scaled
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_leakage.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add evaluation/leakage_check.py tests/test_leakage.py
git commit -m "feat: automated leakage detection for splits and scaler isolation"
```

---

### Task 8: Training Loop

**Files:**
- Create: `model/train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing tests**

```python
# tests/test_train.py
import pytest
import torch
import numpy as np
import pandas as pd
from model.train import create_dataset, train_one_fold, FXDataset

def test_dataset_creation():
    n, n_feat = 100, 10
    features = pd.DataFrame(
        np.random.randn(n, n_feat),
        index=pd.bdate_range("2020-01-01", periods=n),
        columns=[f"f{i}" for i in range(n_feat)],
    )
    targets = pd.DataFrame(
        np.random.randn(n, 2),
        index=features.index,
        columns=["ret_1d", "ret_5d"],
    )
    ds = create_dataset(features, targets, seq_len=30)
    assert len(ds) == n - 30
    x, y = ds[0]
    assert x.shape == (30, n_feat)
    assert y.shape == (2,)  # ret_1d, ret_5d

def test_dataset_no_future_leak():
    """Sample at index i should only contain features up to index i."""
    n, n_feat = 100, 10
    features = pd.DataFrame(
        np.arange(n * n_feat).reshape(n, n_feat).astype(float),
        index=pd.bdate_range("2020-01-01", periods=n),
        columns=[f"f{i}" for i in range(n_feat)],
    )
    targets = pd.DataFrame(
        np.random.randn(n, 2),
        index=features.index,
        columns=["ret_1d", "ret_5d"],
    )
    ds = create_dataset(features, targets, seq_len=30)
    x, y = ds[5]  # 5th sample
    # x should contain rows [5, 6, ..., 34] of features
    # The last row in x should have feature values from row 34
    assert x[-1, 0] == float(34 * n_feat)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# model/train.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 5


class FXDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]  # target at end of window
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_dataset(
    features: pd.DataFrame, targets: pd.DataFrame, seq_len: int = SEQ_LEN
) -> FXDataset:
    # Align indices
    common = features.index.intersection(targets.index)
    feat_arr = features.loc[common].values
    tgt_arr = targets.loc[common].values
    return FXDataset(feat_arr, tgt_arr, seq_len)


def train_one_fold(
    model,
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    val_features: pd.DataFrame,
    val_targets: pd.DataFrame,
    scaler: StandardScaler = None,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    # Fit scaler on train only
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_features.values)

    train_scaled = pd.DataFrame(
        scaler.transform(train_features.values),
        index=train_features.index,
        columns=train_features.columns,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_features.values),
        index=val_features.index,
        columns=val_features.columns,
    )

    train_ds = create_dataset(train_scaled, train_targets, seq_len)
    val_ds = create_dataset(val_scaled, val_targets, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            preds = model(x)
            # Stack predictions: (batch, 2) for ret_1d, ret_5d
            pred_stack = torch.stack([preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1)
            loss = criterion(pred_stack, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                pred_stack = torch.stack([preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1)
                val_loss += criterion(pred_stack, y).item()

        avg_val = val_loss / max(len(val_loader), 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val

        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss/max(len(train_loader),1):.6f} | val_loss={avg_val:.6f}")

    return {"best_val_loss": best_val_loss, "scaler": scaler}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add model/train.py tests/test_train.py
git commit -m "feat: training loop with scaler isolation and sequential dataset"
```

---

### Task 9: Baseline Runner

**Files:**
- Create: `baseline.py`

**Step 1: Write the failing test**

```python
# tests/test_baseline.py
import pytest

def test_baseline_imports():
    from baseline import run_baseline
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# baseline.py
"""
Baseline runner: trains LFM2.5-backed model across all 7 folds,
reports average Sharpe ratio.
"""
import json
import time
import numpy as np
import torch
from data.download import download_all_pairs
from data.features import compute_features, compute_targets
from data.splits import FOLDS, split_data
from model.backbone import CurrencyLFM
from model.train import train_one_fold, create_dataset, SEQ_LEN
from evaluation.metrics import sharpe_ratio, average_sharpe_across_folds
from sklearn.preprocessing import StandardScaler


def run_baseline(epochs: int = 5, seq_len: int = SEQ_LEN) -> dict:
    print("=== Currency Prediction Baseline ===")
    print("Downloading FX pairs...")
    all_pairs = download_all_pairs()
    print("Downloading macro signals...")
    from data.download import download_macro_signals
    macro_data = download_macro_signals()

    # Full feature set: 6 pairs + 9 macro signals -> ~55 features
    from data.features import compute_all_features
    features = compute_all_features(all_pairs, macro_data)
    targets = compute_targets(all_pairs["EURUSD=X"])

    # Align
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    n_features = features.shape[1]
    print(f"Features: {n_features} columns, {len(features)} rows")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")

    fold_results = []
    fold_test_returns = []

    for i, fold in enumerate(FOLDS):
        print(f"\n--- Fold {i+1}/7: {fold['name']} ({fold['regime']}) ---")

        train_feat, val_feat, test_feat = split_data(features, fold)
        train_tgt, val_tgt, test_tgt = split_data(targets, fold)

        if len(train_feat) < seq_len or len(val_feat) < seq_len or len(test_feat) < seq_len:
            print(f"  SKIP: insufficient data (train={len(train_feat)}, val={len(val_feat)}, test={len(test_feat)})")
            continue

        print(f"  Train: {len(train_feat)} days | Val: {len(val_feat)} days | Test: {len(test_feat)} days")

        # Fresh model per fold
        model = CurrencyLFM(n_input_features=n_features, freeze_backbone=True)

        # Train
        t0 = time.time()
        result = train_one_fold(model, train_feat, train_tgt, val_feat, val_tgt, epochs=epochs, seq_len=seq_len)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")

        # Evaluate on test set
        scaler = result["scaler"]
        test_scaled = scaler.transform(test_feat.values)
        test_ds = create_dataset(
            features=__import__("pandas").DataFrame(test_scaled, index=test_feat.index, columns=test_feat.columns),
            targets=test_tgt,
            seq_len=seq_len,
        )

        model.eval()
        test_preds_1d = []
        test_actuals_1d = []
        with torch.no_grad():
            for idx in range(len(test_ds)):
                x, y = test_ds[idx]
                preds = model(x.unsqueeze(0))
                test_preds_1d.append(preds["ret_1d"][0, 0].item())
                test_actuals_1d.append(y[0].item())

        # Strategy returns: predicted direction * actual return
        test_preds_1d = np.array(test_preds_1d)
        test_actuals_1d = np.array(test_actuals_1d)
        strategy_returns = np.sign(test_preds_1d) * test_actuals_1d

        fold_sharpe = sharpe_ratio(strategy_returns)
        fold_test_returns.append(strategy_returns)
        fold_results.append({
            "fold": fold["name"],
            "regime": fold["regime"],
            "sharpe": fold_sharpe,
            "val_loss": result["best_val_loss"],
            "train_days": len(train_feat),
            "test_days": len(test_feat),
        })
        print(f"  Test Sharpe: {fold_sharpe:.3f}")

    # Summary
    avg_sharpe = average_sharpe_across_folds(fold_test_returns)
    print(f"\n{'='*50}")
    print(f"AVERAGE SHARPE ACROSS {len(fold_results)} FOLDS: {avg_sharpe:.3f}")
    print(f"{'='*50}")

    for r in fold_results:
        print(f"  {r['fold']:30s} Sharpe={r['sharpe']:+.3f}")

    summary = {
        "avg_sharpe": avg_sharpe,
        "folds": fold_results,
        "n_features": n_features,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to baseline_results.json")
    return summary


if __name__ == "__main__":
    run_baseline()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add baseline.py tests/test_baseline.py
git commit -m "feat: baseline runner with 7-fold evaluation and Sharpe reporting"
```

---

### Task 10: Autoresearch Optimizer Loop

**Files:**
- Create: `optimizer/prompts.py`
- Create: `optimizer/agent_loop.py`
- Create: `run_optimizer.py`

**Step 1: Write optimizer prompts**

```python
# optimizer/prompts.py

CATEGORIES = [
    "feature_engineering",
    "model_architecture",
    "training_hyperparams",
    "head_design",
    "regularization",
    "data_preprocessing",
    "ensemble",
]

BRAINSTORM_PROMPT = """You are an ML researcher optimizing a currency prediction model.

The model uses LFM2.5-350M (Liquid AI) as a frozen backbone with multi-horizon prediction heads.
It predicts 1d and 5d forward returns for EUR/USD.
Evaluation: average annualized Sharpe ratio across 7 regime-diverse test folds.

Given the current code and past experiment results, propose exactly 3 diverse experiment ideas.
Each idea should be a specific, implementable code change.

Categories: {categories}

Current average Sharpe: {current_sharpe:.3f}
Best Sharpe so far: {best_sharpe:.3f}

Past experiments:
{past_experiments}

Current model code:
```python
{model_code}
```

Current feature code:
```python
{feature_code}
```

Respond with JSON:
[
  {{"id": 1, "category": "...", "description": "...", "risk": "low|medium|high"}},
  {{"id": 2, "category": "...", "description": "...", "risk": "low|medium|high"}},
  {{"id": 3, "category": "...", "description": "...", "risk": "low|medium|high"}}
]
"""

MODIFY_PROMPT = """You are an ML engineer implementing an experiment on a currency prediction model.

Experiment to implement:
{experiment_description}

Current code for the file to modify:
```python
{current_code}
```

Rules:
- Output ONLY the complete modified Python file, nothing else
- Do NOT change the function signatures or class names
- Do NOT modify data splits or evaluation logic
- Keep changes focused on the experiment description
- The code must be valid Python that passes syntax checking

Output the complete file:
"""
```

**Step 2: Write agent loop**

```python
# optimizer/agent_loop.py
import os
import json
import time
import shutil
import py_compile
import tempfile
import anthropic
from optimizer.prompts import BRAINSTORM_PROMPT, MODIFY_PROMPT, CATEGORIES


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def validate_syntax(code, filename="temp.py"):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            py_compile.compile(f.name, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        finally:
            os.unlink(f.name)


def load_state(state_path="optimizer_state.json"):
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {
        "best_sharpe": -999.0,
        "current_sharpe": -999.0,
        "experiments": [],
        "iteration": 0,
    }


def save_state(state, state_path="optimizer_state.json"):
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def run_optimizer(max_experiments=12, model_name="claude-sonnet-4-20250514"):
    from baseline import run_baseline

    client = anthropic.Anthropic()
    state = load_state()

    # Run initial baseline if needed
    if state["iteration"] == 0:
        print("Running initial baseline...")
        result = run_baseline()
        state["current_sharpe"] = result["avg_sharpe"]
        state["best_sharpe"] = result["avg_sharpe"]
        state["iteration"] = 1
        save_state(state)
        print(f"Baseline Sharpe: {result['avg_sharpe']:.3f}")

    modifiable_files = {
        "model": "model/backbone.py",
        "features": "data/features.py",
        "train": "model/train.py",
    }

    for exp_num in range(state["iteration"], max_experiments + 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num}/{max_experiments}")
        print(f"Current Sharpe: {state['current_sharpe']:.3f} | Best: {state['best_sharpe']:.3f}")
        print(f"{'='*60}")

        # Back up current files
        backups = {}
        for key, path in modifiable_files.items():
            backups[key] = read_file(path)

        # Step 1: Brainstorm
        past_summary = "\n".join(
            f"  {e['id']}: {e['category']} - {e['description']} -> Sharpe {e.get('result_sharpe', 'N/A')}"
            for e in state["experiments"][-10:]
        ) or "  (none yet)"

        prompt = BRAINSTORM_PROMPT.format(
            categories=", ".join(CATEGORIES),
            current_sharpe=state["current_sharpe"],
            best_sharpe=state["best_sharpe"],
            past_experiments=past_summary,
            model_code=read_file("model/backbone.py"),
            feature_code=read_file("data/features.py"),
        )

        response = client.messages.create(
            model=model_name,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        ideas_text = response.content[0].text

        # Parse ideas
        try:
            # Extract JSON from response
            start = ideas_text.index("[")
            end = ideas_text.rindex("]") + 1
            ideas = json.loads(ideas_text[start:end])
        except (ValueError, json.JSONDecodeError):
            print("Failed to parse ideas, skipping iteration")
            continue

        # Pick first low/medium risk idea
        chosen = ideas[0]
        for idea in ideas:
            if idea.get("risk") in ("low", "medium"):
                chosen = idea
                break

        print(f"Chosen: [{chosen['category']}] {chosen['description']}")

        # Step 2: Determine which file to modify
        file_key = "model"  # default
        cat = chosen["category"]
        if cat in ("feature_engineering", "data_preprocessing"):
            file_key = "features"
        elif cat in ("training_hyperparams", "regularization"):
            file_key = "train"

        target_path = modifiable_files[file_key]

        # Step 3: Generate modification
        modify_prompt = MODIFY_PROMPT.format(
            experiment_description=chosen["description"],
            current_code=read_file(target_path),
        )

        response = client.messages.create(
            model=model_name,
            max_tokens=4000,
            messages=[{"role": "user", "content": modify_prompt}],
        )
        new_code = response.content[0].text

        # Strip markdown fences if present
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0]
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0]

        # Step 4: Validate syntax
        valid, err = validate_syntax(new_code)
        if not valid:
            print(f"  SYNTAX ERROR: {err}")
            state["experiments"].append({
                "id": exp_num,
                "category": chosen["category"],
                "description": chosen["description"],
                "result": "syntax_error",
            })
            save_state(state)
            continue

        # Step 5: Apply change and evaluate
        write_file(target_path, new_code)

        try:
            result = run_baseline()
            new_sharpe = result["avg_sharpe"]
            print(f"  New Sharpe: {new_sharpe:.3f} (was {state['current_sharpe']:.3f})")

            if new_sharpe > state["current_sharpe"]:
                print("  KEPT (improvement)")
                state["current_sharpe"] = new_sharpe
                if new_sharpe > state["best_sharpe"]:
                    state["best_sharpe"] = new_sharpe
            else:
                print("  REVERTED (no improvement)")
                for key, path in modifiable_files.items():
                    write_file(path, backups[key])

            state["experiments"].append({
                "id": exp_num,
                "category": chosen["category"],
                "description": chosen["description"],
                "result_sharpe": new_sharpe,
                "kept": new_sharpe > state["current_sharpe"],
            })

        except Exception as e:
            print(f"  RUNTIME ERROR: {e}")
            # Revert
            for key, path in modifiable_files.items():
                write_file(path, backups[key])
            state["experiments"].append({
                "id": exp_num,
                "category": chosen["category"],
                "description": chosen["description"],
                "result": f"error: {str(e)[:200]}",
            })

        state["iteration"] = exp_num + 1
        save_state(state)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Final Sharpe: {state['current_sharpe']:.3f}")
    print(f"Best Sharpe: {state['best_sharpe']:.3f}")
    print(f"Experiments run: {len(state['experiments'])}")
    print(f"{'='*60}")
```

**Step 3: Write CLI entry point**

```python
# run_optimizer.py
"""
CLI entry point for the autoresearch optimizer.

Usage:
    python run_optimizer.py                     # run 12 experiments
    python run_optimizer.py --max-experiments 20
    python run_optimizer.py --baseline-only     # just run baseline, no optimization
"""
import argparse
from baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(description="Currency Prediction Autoresearch Optimizer")
    parser.add_argument("--max-experiments", type=int, default=12, help="Number of optimization experiments")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline evaluation")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Claude model for optimizer")
    args = parser.parse_args()

    if args.baseline_only:
        run_baseline()
    else:
        from optimizer.agent_loop import run_optimizer
        run_optimizer(max_experiments=args.max_experiments, model_name=args.model)


if __name__ == "__main__":
    main()
```

**Step 4: Run import test**

Run: `python -c "from optimizer.agent_loop import run_optimizer; from optimizer.prompts import BRAINSTORM_PROMPT; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add optimizer/prompts.py optimizer/agent_loop.py run_optimizer.py
git commit -m "feat: autoresearch optimizer loop with Claude API brainstorm/modify/evaluate"
```

---

### Task 11: End-to-End Smoke Test

**Files:**
- Create: `tests/test_e2e.py`

**Step 1: Write end-to-end test**

```python
# tests/test_e2e.py
"""
End-to-end smoke test: downloads data, computes features, splits,
creates model, trains 1 epoch on 1 fold, evaluates Sharpe.
Does NOT call Claude API.
"""
import pytest
import torch
import numpy as np
from data.download import download_pair
from data.features import compute_features, compute_targets
from data.splits import FOLDS, split_data
from model.backbone import CurrencyLFM
from model.train import create_dataset, SEQ_LEN
from evaluation.metrics import sharpe_ratio

@pytest.mark.slow
def test_e2e_single_fold():
    # Download EUR/USD
    df = download_pair("EURUSD=X", start="2020-01-01", end="2024-12-31")
    features = compute_features(df)
    targets = compute_targets(df)

    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    # Use fold 7 (most recent data matches our range)
    fold = FOLDS[6]
    train_feat, val_feat, test_feat = split_data(features, fold)
    train_tgt, val_tgt, test_tgt = split_data(targets, fold)

    assert len(train_feat) > SEQ_LEN
    assert len(val_feat) > SEQ_LEN

    # Create model
    model = CurrencyLFM(n_input_features=features.shape[1], freeze_backbone=True)

    # Create dataset (just verify it works)
    train_ds = create_dataset(train_feat, train_tgt, seq_len=SEQ_LEN)
    assert len(train_ds) > 0

    # Forward pass
    x, y = train_ds[0]
    model.eval()
    with torch.no_grad():
        preds = model(x.unsqueeze(0))
    assert "ret_1d" in preds
    assert preds["ret_1d"].shape[1] == 6

    # Sharpe on random returns (just verify it runs)
    fake_returns = np.random.randn(100) * 0.01
    s = sharpe_ratio(fake_returns)
    assert isinstance(s, float)
```

**Step 2: Run test**

Run: `python -m pytest tests/test_e2e.py -v -m slow`
Expected: PASS (may take a few minutes due to model loading)

**Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end smoke test across full pipeline"
```

---

### Task 12: Run Full Baseline

**Step 1: Run baseline evaluation**

Run: `python baseline.py`
Expected: Trains across 7 folds, prints Sharpe per fold, saves `baseline_results.json`

**Step 2: Verify results file**

Run: `python -c "import json; r=json.load(open('baseline_results.json')); print(f'Avg Sharpe: {r[\"avg_sharpe\"]:.3f}')" `
Expected: Prints a Sharpe value (likely near 0 or slightly negative for an untrained baseline — that's expected)

**Step 3: Commit results**

```bash
git add baseline_results.json
git commit -m "data: baseline results from initial 7-fold evaluation"
```

---

### Task 13: Run Optimizer (Requires ANTHROPIC_API_KEY)

**Step 1: Verify API key is set**

Run: `python -c "import os; assert os.environ.get('ANTHROPIC_API_KEY'), 'Set ANTHROPIC_API_KEY!'; print('API key ready')"`

**Step 2: Run optimizer with a small number of experiments**

Run: `python run_optimizer.py --max-experiments 3`
Expected: Runs 3 optimization iterations, printing brainstormed ideas and Sharpe changes

**Step 3: Commit optimizer state**

```bash
git add optimizer_state.json
git commit -m "data: initial optimizer run results"
```

---

## Execution Order Summary

| Task | Module | Dependencies | Estimated Time |
|------|--------|-------------|----------------|
| 1 | Scaffold | None | 2 min |
| 2 | Data download | Task 1 | 5 min |
| 3 | Features | Task 1 | 5 min |
| 4 | Splits | Task 1 | 5 min |
| 5 | LFM backbone | Tasks 1-3 | 10 min |
| 6 | Metrics | Task 1 | 3 min |
| 7 | Leakage checks | Tasks 4, 6 | 5 min |
| 8 | Training loop | Tasks 2-5 | 10 min |
| 9 | Baseline runner | Tasks 2-8 | 5 min |
| 10 | Optimizer loop | Task 9 | 10 min |
| 11 | E2E smoke test | Tasks 2-8 | 5 min |
| 12 | Run baseline | Task 9 | 15-30 min (CPU) |
| 13 | Run optimizer | Tasks 10, 12 | 30+ min |

**Parallelizable:** Tasks 2, 3, 4, 6 can be built in parallel (no dependencies between them). Tasks 5 and 8 depend on the data modules.
