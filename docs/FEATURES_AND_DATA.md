# Features and Data Sources — QQQ AutoResearch Project

Comprehensive reference for every input the QQQ models see: every ticker downloaded, every cross-asset relationship constructed, every engineered feature, every prediction target, and every academic citation that motivated each design choice. Authoritative sources: `data/download.py`, `data/features.py`, `data/splits.py`.

---

## 1. Prediction targets (what we forecast)

The asset under prediction is **QQQ** (Invesco QQQ Trust, ticker `QQQ`) — the Nasdaq-100 ETF. All targets are computed from QQQ adjusted-close (`auto_adjust=True` from yfinance — splits + dividends already absorbed).

| Variant | Field | Formula | Use |
|---------|-------|---------|-----|
| A | `fwd_ret_1d` | `log(close[t+1]) - log(close[t])` | **PRIMARY** optimisation target |
| B | `fwd_ret_5d` | `log(close[t+5]) - log(close[t])` | Auxiliary head (multi-horizon supervision) |
| C | `fwd_sign_concordance` | `1 if sign(r1) == sign(r5) else 0` | Side-channel metric |
| D | `fwd_voladj_ret_1d` | `r1 / sqrt(rolling_var_20d(r1))` | Vol-adjusted target — orthogonalises trend bias from skill |

The heteroscedastic loss (Kendall & Gal 2017 NeurIPS arXiv:1612.01474) trains heads A and B jointly via:
`L = exp(-s) * huber(mu, y) + 0.5 * s` with `s = log(var)` predicted per sample.

Source: `data/features.py:746-770` (`compute_qqq_targets`).

---

## 2. Data universe — every ticker downloaded

All data sourced from Yahoo Finance via `yfinance`. Default window: **2004-01-01 → 2025-12-31** (hard cap; CLAUDE.md mandates no 2026 leakage). Cached to `.data_cache_qqq/<safe_ticker>_<start>_<end>.parquet`.

The download is grouped by economic role; each group cites the seminal paper that motivated its inclusion (`data/download.py`).

### 2.1 Primary asset (1 ticker)

| Ticker | Description |
|--------|-------------|
| **QQQ** | Invesco QQQ Trust (Nasdaq-100) — the asset we forecast |

### 2.2 Benchmarks for cross-asset relative strength (7 tickers)

Citation: Lo & MacKinlay 1990 RFS — relative-strength reversal between indices.

| Ticker | Description |
|--------|-------------|
| SPY | SPDR S&P 500 |
| DIA | SPDR Dow Jones Industrial |
| IWM | iShares Russell 2000 (US small cap) |
| EFA | iShares MSCI EAFE (international DM) |
| EEM | iShares MSCI Emerging Markets |
| ^IXIC | Nasdaq Composite (QQQ underlying — ~3500 names vs QQQ's top-100) |
| AGG | iShares Core US Aggregate Bond (baseline duration + credit) |

### 2.3 Industry tilts (4 tickers)

Mag-7 + AI rally is a chip story; XLK alone misses pure-play semis.

| Ticker | Description |
|--------|-------------|
| SOXX | iShares Semiconductor ETF |
| SMH | VanEck Semiconductor ETF |
| IBB | iShares Biotechnology ETF (widens healthcare beyond XLV) |
| ARKK | ARK Innovation ETF (high-beta innovation proxy; leads risk-on/off shifts) |

### 2.4 Sectors — 11 SPDR sector ETFs

Citation: Brown & Cliff 2004 — sector rotation predicts index returns.

| Ticker | Sector |
|--------|--------|
| XLK | Technology |
| XLF | Financials |
| XLV | Health Care |
| XLY | Consumer Discretionary |
| XLP | Consumer Staples |
| XLE | Energy |
| XLI | Industrials |
| XLU | Utilities |
| XLB | Materials |
| XLRE | Real Estate (starts Oct 2015 → dropped by late-column filter) |
| XLC | Communication Services (starts Jun 2018 → dropped by late-column filter) |

### 2.5 Volatility-regime panel (10 tickers)

Citations: Bollerslev, Tauchen & Zhou 2009 RFS (variance risk premium); Pan & Poteshman 2006 (option-implied signals); Whaley 2009 (VIX as fear gauge); Cieslak & Pang 2021 RFS (bond-vol leads equity-vol).

| Ticker | Description |
|--------|-------------|
| ^VIX | CBOE VIX (S&P 500 30-day implied) |
| **^VXN** | **CBOE Nasdaq-100 Volatility — QQQ-NATIVE fear gauge** |
| ^VIX9D | CBOE VIX 9-day (~2011 start, dropped) |
| ^VIX3M | CBOE 3-month VIX |
| ^VIX6M | CBOE 6-month VIX |
| ^VVIX | CBOE Vol-of-Vol (~2007 borderline) |
| ^SKEW | CBOE SKEW (tail risk) |
| ^OVX | CBOE Crude Oil ETF VIX (2007 start) |
| ^GVZ | CBOE Gold ETF VIX (2008 start) |
| **^MOVE** | **ICE BofA MOVE Index — Treasury bond vol; bond-vol leads equity-vol per Cieslak & Pang 2021** |

### 2.6 Yields, credit, fixed income (10 tickers)

Citations: Estrella & Mishkin 1998 RES (yield curve predicts recessions); Welch & Goyal 2008 RFS (term + default spreads as equity premium predictors); Adrian, Crump & Moench 2013 (term-premium decomposition).

| Ticker | Description |
|--------|-------------|
| ^TNX | US 10Y Treasury Yield |
| ^FVX | US 5Y Treasury Yield |
| ^TYX | US 30Y Treasury Yield |
| ^IRX | US 13W Treasury Yield (3-month proxy) |
| TLT | iShares 20+Y Treasury Bond ETF |
| IEF | iShares 7-10Y Treasury Bond ETF |
| SHY | iShares 1-3Y Treasury Bond ETF |
| TIP | iShares TIPS Bond ETF (10y real-yield proxy) |
| HYG | iShares iBoxx HY Corporate (credit risk appetite) |
| LQD | iShares iBoxx IG Corporate |

### 2.7 Macro / commodities / FX (8 tickers)

Citations: Driesprong, Jacobsen & Maat 2008 (oil predicts equity); Akram 2009 (commodities and US monetary policy joint dynamics).

| Ticker | Description |
|--------|-------------|
| GC=F | Gold Futures |
| SI=F | Silver Futures |
| CL=F | WTI Crude Oil Futures |
| BZ=F | Brent Crude Futures |
| HG=F | Copper Futures (Dr. Copper) |
| DX-Y.NYB | US Dollar Index (DXY) |
| EURUSD=X | EUR/USD |
| USDJPY=X | USD/JPY |

### 2.8 International risk pulse (4 tickers)

| Ticker | Description |
|--------|-------------|
| ^N225 | Nikkei 225 (Tokyo) |
| ^FTSE | FTSE 100 (UK) |
| ^GDAXI | DAX (Germany) |
| ^HSI | Hang Seng (HK) |

### 2.9 Crypto / risk-on barometer (1 ticker)

Citation: Bouri et al. 2017 Finance Research Letters — Bitcoin correlation with risk-on flows post-2020.

| Ticker | Description |
|--------|-------------|
| BTC-USD | Bitcoin USD (starts 2014) |

### 2.10 Panel-learning targets (added 2026-04-29)

Per Gu, Kelly & Xiu 2020 RFS *"Empirical Asset Pricing via Machine Learning"* — multi-asset panel learning multiplies effective sample size and enables shared-trunk diversification at inference. **Used only by the panel-mode runner**, not the QQQ-only single-target setup.

#### 2.10.1 Top-30 NDX components by 2024 weight (Mag-7 + tier-1 large caps)

`AAPL, MSFT, NVDA, AMZN, META, GOOGL, GOOG, TSLA, AVGO, COST, NFLX, ADBE, AMD, PEP, CSCO, QCOM, INTC, TMUS, CMCSA, INTU, AMGN, AMAT, TXN, BKNG, ISRG, ADP, GILD, SBUX, MDLZ, MU`

#### 2.10.2 Adjacent indices for sub-market panel diversification (6 tickers)

`SPY, IWM, EEM, EFA, DIA, MDY` — US large/small/EM/intl-DM/Dow/midcap. Citation: Hou, Mo, Xue & Zhang 2014 — international predictability.

#### 2.10.3 Asia + Europe indices — leading-indicator edge (10 tickers)

These markets close BEFORE US market open on the same calendar day, giving leading-indicator information for QQQ day-T close. Citations: Lou, Polk, Skouras 2019 JFE *"A tug of war: Overnight versus intraday expected returns"*; Boudoukh, Richardson, Whitelaw 2007 RFS.

| Ticker | Description | ET close |
|--------|-------------|----------|
| ^N225 | Nikkei 225 (Tokyo) | 01:00 |
| ^HSI | Hang Seng (HK) | 04:00 |
| ^KS11 | KOSPI (Korea) | 02:30 |
| ^TWII | Taiwan Weighted | 02:30 |
| ^STI | Straits Times (Singapore) | 05:00 |
| ^AXJO | ASX 200 (Australia) | 02:00 |
| ^STOXX50E | Euro Stoxx 50 | 11:30 |
| ^FTSE | FTSE 100 | 11:30 |
| ^GDAXI | DAX (Germany) | 11:30 |
| ^FCHI | CAC 40 (France) | 11:30 |

All cleanly causal for predicting QQQ day-T close.

#### 2.10.4 Asia megacaps (ADRs preferred for clean US-aligned trading days) (8 tickers)

| Ticker | Description |
|--------|-------------|
| TSM | TSMC ADR |
| BABA | Alibaba ADR |
| JD | JD.com ADR |
| PDD | PDD Holdings ADR |
| SONY | Sony Group ADR |
| TM | Toyota Motor ADR |
| HMC | Honda Motor ADR |
| BIDU | Baidu ADR |

#### Panel total

`PRIMARY (1) + NDX-30 + ASIA_EUROPE_INDICES (10) + ASIA_MEGACAPS (8) + ADJACENT (6) = 55 parallel prediction targets` for the panel-mode runner. Source: `data/download.py:130-136` (`PANEL_TARGETS`).

### 2.11 Data integrity invariants

- **Hard cap 2026-01-01**: any row dated `>= 2026-01-01` is dropped post-fetch with a logged warning. Enforced in `_enforce_no_2026()`.
- **Cache is parquet**: every `(ticker, start, end)` triple persists to `.data_cache_qqq/`. Re-runs read from cache (verified via cache-hit logs).
- **Defensive against missing tickers**: `download_all` logs and skips any ticker that 404s, returns empty, or fails network. Feature builder is defensive — features computed from absent tickers are simply omitted, never NaN-padded.
- **`auto_adjust=True`**: yfinance returns split- and dividend-adjusted prices.

---

## 3. Engineered features — every block, every formula, every citation

Source: `data/features.py`. Final feature count after late-column drop: **205 features × 4859 rows** (2006-12-28 → 2026-04-29 when full window downloaded).

The pipeline produces ~120 raw features; the final 205 figure comes from the per-ticker expansion (sector ETFs each contribute a compact OHLCV block) + cross-ticker derived ratios + calendar dummies. The exact final count varies by which tickers fetched successfully on a given run.

### 3.1 QQQ primary OHLCV block — `_ohlcv_features(qqq, "qqq", full=True)`

**Returns at multiple horizons** — Lo & MacKinlay 1988 (variance ratio test of random walk):
- `qqq_logret_1d`, `qqq_logret_5d`, `qqq_logret_20d`
- `qqq_logret_60d`, `qqq_logret_120d`, `qqq_logret_252d`

**Moving-average crossings** — Brock, Lakonishok & LeBaron 1992 JF:
- `qqq_close_to_sma{5,10,20,50,100,200}` — `close / SMA_n - 1`

**Oscillators**:
- `qqq_rsi14`, `qqq_rsi28` — Wilder 1978 RSI
- `qqq_macd_norm`, `qqq_macd_hist_norm` — Appel 2005 MACD line and histogram, both divided by close
- `qqq_bb_z20` — Bollinger 1980s 20-day Bollinger Z-score
- `qqq_stoch_k14` — 14-day stochastic %K
- `qqq_williams_r14` — Williams %R
- `qqq_natr14` — normalised ATR (Wilder 1978 ATR ÷ close)

**Realised vol estimators**:
- `qqq_park_vol_5d`, `qqq_park_vol_20d` — Parkinson 1980 high-low estimator
- `qqq_gk_vol_20d` — Garman & Klass 1980 OHLC estimator
- `qqq_yz_vol_20d`, `qqq_yz_vol_60d` — Yang & Zhang 2000 drift-independent estimator (recommended)

**Donchian channel position**:
- `qqq_donchian_pos20`, `qqq_donchian_pos50`, `qqq_donchian_pos252` — `(close - rolling_min) / (rolling_max - rolling_min)`

**52-week high / low distance** — George & Hwang 2004 JF:
- `qqq_dist_52w_high`, `qqq_dist_52w_low`

**Momentum + reversal**:
- `qqq_mom_3m, _6m, _9m, _12m` — Jegadeesh & Titman 1993 JF
- `qqq_mom_12_1`, `qqq_mom_12_2` — skipped-month momentum (Asness, Moskowitz & Pedersen 2013)
- `qqq_rev_1w`, `qqq_rev_1m` — short-term reversal (Lehmann 1990)
- `qqq_max_1m_ret` — MAX feature (Bali, Cakici & Whitelaw 2011 JFE)

**Volume / microstructure** (when volume is available):
- `qqq_vol_z20`, `qqq_vol_z60` — log-volume Z-scores
- `qqq_amihud_20d` — Amihud 2002 illiquidity, log-transformed
- `qqq_obv_z60` — On-balance-volume Z-score

### 3.2 Sector ETF blocks — `_ohlcv_features(sector_df, "<sec>", full=False)`

For each of the 11 sectors that survives the late-column filter (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLU, XLB — XLRE / XLC drop because they start post-2007):

- `<sec>_logret_1d, _5d, _20d`
- `<sec>_ma20_z` — Z-score of close vs 20-day MA

= 4 features × 9 surviving sectors = 36 sector-block features.

### 3.3 Vol-regime block — `_vol_regime_features`

For each VIX-family signal that survives:

- `vix, vix_logret_1d, vix_logret_5d, vix_z60`
- `vxn, vxn_logret_1d, vxn_logret_5d, vxn_z60` — **QQQ-native fear gauge**
- `vxn_over_vix` — tech-vol vs broad-vol ratio
- `move, move_logret_5d, move_z60` — Treasury vol
- `move_over_vix` — bond-stress vs equity-stress
- `vix_over_vix3m`, `vix_over_vix6m` — term-structure ratios (contango / backwardation)
- `skew, skew_change_5d`

**Variance Risk Premium proxy** — Bollerslev, Tauchen & Zhou 2009 RFS:
- `vrp_proxy = (VIX/100)^2 - rolling_realised_var_20d_QQQ * 252`

`vix9d_over_vix`, `vvix_over_vix`, `ovx_over_vix`, `gvz_over_vix` are computed when both legs exist but several get dropped by the late-column filter (they start post-2007).

### 3.4 Yields & credit block — `_yields_credit_features`

**Yield levels and changes**:
- `yld_10y, yld_10y_5d, yld_10y_20d` (^TNX)
- `yld_5y` (^FVX), `yld_30y` (^TYX), `yld_3m` (^IRX)

**Term spreads** — Estrella & Mishkin 1998 RES:
- `term_10y_3m` — recession indicator
- `term_10y_5y`, `term_30y_10y`

**Bond-ETF returns** (smoothed yield-change proxies):
- `tlt_logret_5d, tlt_logret_20d` (TLT 20+Y)
- `ief_logret_5d, ief_logret_20d` (IEF 7-10Y)
- `shy_logret_5d, shy_logret_20d` (SHY 1-3Y)
- `tip_logret_5d, tip_logret_20d` (TIP)

**Credit risk appetite**:
- `hyg_over_lqd`, `hyg_over_lqd_5d` — HY/IG ratio

**Real-yield proxy**:
- `tip_logret_20d` — when TIP rises, real yields fall

### 3.5 Macro / FX / commodity block — `_macro_fx_features`

For each of `DX-Y.NYB, GC=F, SI=F, CL=F, BZ=F, HG=F, EURUSD=X, USDJPY=X`:
- `<name>_logret_1d, _5d, _20d` (e.g. `dxy_logret_5d`, `wti_logret_20d`)

**Cross ratios for risk-on / risk-off**:
- `gold_over_qqq` — flight-to-safety ratio
- `copper_gold_ratio` — Dr. Copper vs gold (cyclical vs defensive)

### 3.6 Cross-sectional / breadth block — `_cross_sectional_features`

**Benchmark relative strength** — Lo & MacKinlay 1990 RFS:
For each of `SPY, DIA, IWM, EFA, EEM`:
- `<bench>_logret_5d`
- `qqq_over_<bench>` — ratio
- `qqq_over_<bench>_5d` — log-ratio change

**Sector returns and dispersion** — Brown & Cliff 2004:
- `sec_<xl*>_logret_5d` for each sector (×9 surviving sectors)
- `sector_dispersion_5d`, `sector_dispersion_20d` — std across sector returns
- `sector_breadth_20d` — fraction of sectors with positive 20d return
- `xlk_minus_xlp_5d` — tech vs staples spread (cyclicals vs defensives)
- `xly_minus_xlp_5d` — discretionary vs staples

**International risk pulse** — for each of `^N225, ^FTSE, ^GDAXI, ^HSI`:
- `<index>_logret_5d`

**Nasdaq-Composite vs QQQ** (concentration vs breadth signal):
- `qqq_over_ixic` — ratio (Mag-7 concentration vs broad-tech breadth)
- `qqq_over_ixic_5d` — log-ratio change

**Semiconductor pulse**:
- `soxx_logret_5d, soxx_logret_20d, qqq_over_soxx_5d`
- `smh_logret_5d, smh_logret_20d, qqq_over_smh_5d`

**Biotech**:
- `ibb_logret_5d, ibb_logret_20d`

**Innovation / risk-on**:
- `arkk_logret_5d, arkk_logret_20d`
- `arkk_over_qqq` — speculative risk-on indicator
- `arkk_over_qqq_20d`

**Total bond market**:
- `agg_logret_5d, agg_logret_20d`

**Crypto risk-on** (when BTC available):
- `btc_logret_1d, _5d, _20d`

### 3.7 Calendar / seasonality block — `_calendar_features`

Citations: French 1980 + Cross 1973 (day-of-week effect); Ariel 1987 (turn-of-month); Rozeff & Kinney 1976 (January effect); Lucca & Moench 2015 JF (FOMC drift); Stivers & Sun 2002 (options expiration); Haug & Hirschey 2006 (Santa rally / December effect).

**Day-of-week dummies**: `dow_mon, dow_tue, dow_wed, dow_thu, dow_fri`

**Month**:
- `month` (1-12 integer)
- `month_sin, month_cos` — cyclical encoding
- `jan_effect`, `dec_effect`

**Turn-of-month** (Ariel 1987):
- `turn_of_month` — 1 if `day <= 3` or `day >= 28`
- `santa_rally` — 1 if December and `day >= 24`

**FOMC** (Lucca & Moench 2015 JF — equity returns concentrated around FOMC):
- `fomc_week` — week containing a FOMC meeting decision day
- `fomc_day` — actual meeting day
- Manually encoded FOMC date list 2003-2025 (one per ~6 weeks; ~180 dates).

**Options expiration** (Stivers & Sun 2002):
- `opex_friday` — third Friday of the month
- `opex_week` — 5-day window containing the third Friday

**Earnings season**:
- `earnings_season` — weeks 2-5 of Jan/Apr/Jul/Oct

### 3.8 Autoregressive block — `_autoregressive_features`

Citations: Lo & MacKinlay 1988 (variance ratio test); Conrad & Kaul 1988 (autoregressive predictability).

**Lagged QQQ returns**:
- `lag_ret_1d` — yesterday's 1-day return
- `lag_ret_2d` — 2 days ago
- `lag_ret_5d` — last week's 5-day return

**Variance ratios** — random-walk tests:
- `var_ratio_5_1 = Var(r_5) / (5 * Var(r_1))`
- `var_ratio_20_1 = Var(r_20) / (20 * Var(r_1))`

**Drawdown depth**:
- `dd_from_252max = close / rolling_252d_max - 1.0`

---

## 4. Cross-asset relationships (every ratio / spread used)

| Feature | Formula | Economic meaning |
|---------|---------|------------------|
| `qqq_over_spy`, `qqq_over_dia`, `qqq_over_iwm`, `qqq_over_efa`, `qqq_over_eem` | `QQQ / <bench>` | Tech-tilt vs broad market / small / international / EM |
| `qqq_over_ixic` | `QQQ / Nasdaq-Composite` | Mag-7 concentration vs broad tech breadth |
| `qqq_over_soxx_5d`, `qqq_over_smh_5d` | log-diff | QQQ vs pure-play semis |
| `arkk_over_qqq` | `ARKK / QQQ` | Speculative innovation vs broad QQQ |
| `vxn_over_vix` | `VXN / VIX` | Tech-vol vs broad-market-vol |
| `move_over_vix` | `MOVE / VIX` | Treasury-stress vs equity-stress |
| `vix_over_vix3m`, `vix_over_vix6m`, `vix9d_over_vix` | term-structure | Contango (calm) vs backwardation (stress) |
| `vvix_over_vix` | `VVIX / VIX` | Vol-of-vol regime |
| `ovx_over_vix`, `gvz_over_vix` | cross-asset vol | Oil-vol or gold-vol vs equity-vol |
| `term_10y_3m`, `term_10y_5y`, `term_30y_10y` | yield diff | Curve slope (Estrella-Mishkin recession indicator) |
| `hyg_over_lqd` | `HYG / LQD` | High-yield vs investment-grade — credit risk appetite |
| `gold_over_qqq` | `Gold / QQQ` | Flight to safety |
| `copper_gold_ratio` | `HG / GC` | Cyclical vs defensive commodity |
| `xlk_minus_xlp_5d` | sector spread | Tech vs staples (cyclicals vs defensives) |
| `xly_minus_xlp_5d` | sector spread | Discretionary vs staples |
| `vrp_proxy` | `(VIX/100)^2 - RV20 * 252` | Variance Risk Premium (Bollerslev-Tauchen-Zhou 2009) |

---

## 5. Walk-forward folds — 7 super-folds covering 2004 → 2025

Source: `data/splits.py`. Citations: Pagan & Sossounov 2003 (regime dating); Lunde & Timmermann 2004 (duration dependence); Hamilton 1989 (regime-switching); López de Prado 2018 *"Advances in Financial ML"* §7 (walk-forward + purge + embargo).

| Fold | Regime | Train end | Val | Test |
|------|--------|-----------|-----|------|
| 1 | **GFC peak crash** (Lehman + Mar-2009 bottom) | 2008-03 | 2008-04 → 2008-09 | 2008-10 → 2009-03 |
| 2 | **2011 US-downgrade + EU debt** | 2011-03 | 2011-04 → 2011-08 | 2011-09 → 2012-03 |
| 3 | **Taper tantrum + 2014 H1** | 2013-09 | 2013-10 → 2013-12 | 2014-01 → 2014-09 |
| 4 | **China devaluation + oil crash** | 2015-03 | 2015-04 → 2015-08 | 2015-09 → 2016-04 |
| 5 | **2018 Vol-mageddon + Q4 sell-off** | 2018-04 | 2018-05 → 2018-07 | 2018-08 → 2019-04 |
| 6 | **COVID crash + V-recovery** | 2019-09 | 2019-10 → 2020-01 | 2020-02 → 2020-12 |
| 7 | **Inflation bear, AI rally, 2025** | 2023-09 | 2023-10 → 2024-03 | 2024-04 → 2025-12 |

**Invariants (programmatically verified before every run):**
- Train start always 2004-01 (expanding window).
- Each fold's test window is in a NAMED regime — per-fold breakdowns reveal which market state the model wins or loses in.
- All val + test windows are pairwise disjoint across folds.
- **Purge = 90 days** (López de Prado purge horizon).
- **Embargo = 21 days** (post-test no-overlap buffer).
- **Label-horizon buffer = 10 days** (for the 5-day forward target + slack).
- Last fold's test ends 2025-12 — no 2026 leakage.

**Production split (added 2026-05 for OOS)**:
- Train: 2004-01-01 → 2025-06-30
- Val: 2025-07-01 → 2025-09-30 (90-day tail for early-stop)
- OOS: 2025-10-01 → 2026-04-30 (forward-only inference)

---

## 6. Late-column drop filter — why some features are excluded

After the outer-join + reindex to QQQ trading days, columns whose first valid index is **after 2007-01-01** are dropped — otherwise they would force the early rows out via `dropna()` and we'd lose the GFC-onset regime (fold 1).

**Casualties of this filter** (typical run, 26 columns dropped):
- XLRE block (XLRE inception Oct 2015) — 4 features dropped
- XLC block (XLC inception Jun 2018) — 4 features dropped
- ^VIX9D-derived ratios (~2011 start) — 1-2 features dropped
- ^VVIX-derived (~2007 borderline)
- ^OVX (2007), ^GVZ (2008)

We still keep: ^VIX, ^SKEW, sector ETFs (1998), bond ETFs (2002), all QQQ/SPY/IWM/etc.

After: warmup 252 rows dropped + remaining NaN rows dropped → final feature matrix.

---

## 7. Running the data pipeline

```python
from autoresearchindexstock.data.download import download_all
from autoresearchindexstock.data.features import compute_qqq_features, compute_qqq_targets

raw = download_all(start="2004-01-01", end="2025-12-31")  # uses .data_cache_qqq/
features = compute_qqq_features(raw)  # ~205 features × ~4859 rows
targets = compute_qqq_targets(raw)    # 4 target variants
```

Both are pure functions of the downloaded dict — no global state, no hidden caches beyond the parquet store, no internet access after the first call (cache is parquet on disk).

For panel-mode multi-asset training:
```python
from autoresearchindexstock.data.download import download_panel_targets
panel = download_panel_targets()  # tidy long-format (date, asset, OHLCV) — 55 assets
```

---

## 8. Citations — full list

| # | Citation |
|---|----------|
| 1 | Lo & MacKinlay 1988 *J. Financial Economics* "Stock market prices do not follow random walks" |
| 2 | Lo & MacKinlay 1990 *RFS* "When are contrarian profits due to stock market overreaction?" |
| 3 | Brock, Lakonishok & LeBaron 1992 *JF* "Simple technical trading rules and the stochastic properties of stock returns" |
| 4 | Wilder 1978 — *New Concepts in Technical Trading Systems* (RSI, ATR) |
| 5 | Appel 2005 — *Technical Analysis: Power Tools for Active Investors* (MACD) |
| 6 | Bollinger 1980s — Bollinger Bands |
| 7 | Parkinson 1980 *J. Business* "The extreme value method for estimating the variance of the rate of return" |
| 8 | Garman & Klass 1980 *J. Business* "On the estimation of security price volatilities from historical data" |
| 9 | Yang & Zhang 2000 *J. Business* "Drift-independent volatility estimation" |
| 10 | George & Hwang 2004 *JF* "The 52-week high and momentum investing" |
| 11 | Jegadeesh & Titman 1993 *JF* "Returns to buying winners and selling losers" |
| 12 | Asness, Moskowitz & Pedersen 2013 *JF* "Value and momentum everywhere" |
| 13 | Lehmann 1990 *QJE* "Fads, martingales, and market efficiency" |
| 14 | Bali, Cakici & Whitelaw 2011 *JFE* "Maxing out: Stocks as lotteries and the cross-section of expected returns" |
| 15 | Amihud 2002 *J. Financial Markets* "Illiquidity and stock returns" |
| 16 | Bollerslev, Tauchen & Zhou 2009 *RFS* "Expected stock returns and variance risk premia" |
| 17 | Pan & Poteshman 2006 *RFS* "The information in option volume for future stock prices" |
| 18 | Whaley 2009 *J. Portfolio Management* "Understanding the VIX" |
| 19 | Cieslak & Pang 2021 *RFS* "Common shocks in stocks and bonds" |
| 20 | Estrella & Mishkin 1998 *RES* "Predicting US recessions: Financial variables as leading indicators" |
| 21 | Welch & Goyal 2008 *RFS* "A comprehensive look at the empirical performance of equity premium prediction" |
| 22 | Adrian, Crump & Moench 2013 *J. Financial Economics* "Pricing the term structure with linear regressions" |
| 23 | Driesprong, Jacobsen & Maat 2008 *JFE* "Striking oil: Another puzzle?" |
| 24 | Akram 2009 *Energy Economics* "Commodity prices, interest rates and the dollar" |
| 25 | Brown & Cliff 2004 *JFE* "Investor sentiment and the near-term stock market" |
| 26 | Bouri et al. 2017 *Finance Research Letters* — Bitcoin / risk-on correlation |
| 27 | Yermack 2015 — Bitcoin macro-asset character |
| 28 | French 1980 *JFE* "Stock returns and the weekend effect" |
| 29 | Cross 1973 — Day-of-week effect |
| 30 | Ariel 1987 *JFE* "A monthly effect in stock returns" |
| 31 | Rozeff & Kinney 1976 *JFE* "Capital market seasonality: The case of stock returns" |
| 32 | Lucca & Moench 2015 *JF* "The pre-FOMC announcement drift" |
| 33 | Stivers & Sun 2002 — Options-expiration effects |
| 34 | Haug & Hirschey 2006 *FAJ* "The January effect" |
| 35 | Conrad & Kaul 1988 *RFS* "Time-variation in expected returns" |
| 36 | Pagan & Sossounov 2003 *J. Applied Econometrics* "A simple framework for analysing bull and bear markets" |
| 37 | Lunde & Timmermann 2004 *JBES* "Duration dependence in stock prices" |
| 38 | Hamilton 1989 *Econometrica* "A new approach to the economic analysis of nonstationary time series" |
| 39 | López de Prado 2018 *Advances in Financial Machine Learning* §7 — walk-forward + purge + embargo |
| 40 | Gu, Kelly & Xiu 2020 *RFS* "Empirical asset pricing via machine learning" — panel learning |
| 41 | Hou, Mo, Xue & Zhang 2014 — international predictability |
| 42 | Lou, Polk, Skouras 2019 *JFE* "A tug of war: Overnight versus intraday expected returns" |
| 43 | Boudoukh, Richardson, Whitelaw 2007 *RFS* "The myth of long-horizon predictability" |
| 44 | Kendall & Gal 2017 NeurIPS arXiv:1612.01474 — heteroscedastic loss for uncertainty |

---

## 9. File map

| File | Role |
|------|------|
| `autoresearchindexstock/data/download.py` | Ticker definitions, yfinance fetcher, parquet cache, panel target loader |
| `autoresearchindexstock/data/features.py` | Feature engineering — every block in §3 |
| `autoresearchindexstock/data/splits.py` | 7 walk-forward folds + purge/embargo/buffer logic |
| `autoresearchindexstock/.data_cache_qqq/` | Parquet cache (one file per ticker × window) |

Authoritative file checksums for "what features were used in this experiment" come from the model checkpoint's `feature_columns` field (saved at training time). See `winners/<exp>/model_checkpoint.pt` `["feature_columns"]`.
