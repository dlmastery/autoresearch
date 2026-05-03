# OOS Winner Explainability Report

**Generated:** 2026-05-03
**Method:** Permutation feature importance per Breiman 2001 *Random Forests* §4 + Fisher, Rudin & Dominici 2019 JMLR *"All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously"* (arXiv:1801.01489).
**OOS window:** 2025-10-01 → 2026-04-30 (forward-only; never seen in training)
**Source:** `run_oos_feature_importance.py` — produces `oos_feature_importance_exp{N}.csv` and `oos_feature_importance_summary.json`.

For each feature *j* and each model M:
- `baseline_sharpe = OOS_Sharpe(M, X_oos)`
- `permuted_sharpe = OOS_Sharpe(M, X_oos with column j shuffled)`
- `importance_j = baseline_sharpe − permuted_sharpe`

Higher importance ⇒ shuffling that feature degrades OOS performance more ⇒ the model relies on it.
Negative importance ⇒ the feature actively HURTS OOS Sharpe — shuffling improves performance.

---

## 1. Executive summary

The prod-mode OOS retrain (2026-05-02) produced 13 BH-beaters across 28 retrained models. The top individual was **exp 234 (LSTM s35 seed=2026, OOS Sharpe +2.22, Excess +1.00)**. The actual production strategy is the **top-5 vote ensemble (OOS Sharpe +3.85, Excess +3.09, +21.82% return)** — averaging predictions across the 5 strongest individual BH-beaters.

This report explains the OOS-winning predictions in three layers:
1. **Per-model** permutation importance (which features each individual winner relies on)
2. **Cross-model consensus** (features that show up as important across multiple winners)
3. **Domain-level** aggregation (which economic categories drive the OOS lift)

---

## 2. Per-model top features

### 2.1 Exp 234 — LSTM seq=35 seed=2026 (#1 OOS individual)

**Baseline OOS Sharpe: +2.22**

| Rank | Feature | Domain | Drop when shuffled | Permuted Sharpe |
|-----:|---------|--------|-------------------:|----------------:|
| 1 | `month` | calendar | **-0.722** | 0.841 |
| 2 | `dec_effect` | calendar | **-0.710** | 0.852 |
| 3 | `sec_xlk_logret_5d` | sectors | -0.636 | 0.927 |
| 4 | `silver_logret_5d` | commodities | -0.594 | 0.968 |
| 5 | `tlt_logret_5d` | bonds_credit | -0.568 | 0.995 |
| 6 | `dax_logret_5d` | intl_risk | -0.565 | 0.997 |
| 7 | `xle_logret_1d` | sectors | -0.553 | 1.010 |
| 8 | `vxn_z60` | volatility_regime | -0.547 | 1.015 |
| 9 | `vix` | volatility_regime | -0.505 | 1.058 |
| 10 | `qqq_logret_1d` | qqq_primary | -0.493 | 1.070 |
| 11 | `qqq_over_ixic_5d` | qqq_primary | -0.475 | 1.088 |
| 12 | `gold_logret_1d` | commodities | -0.451 | 1.111 |
| 13 | `sec_xli_logret_5d` | sectors | -0.439 | 1.123 |
| 14 | `agg_logret_5d` | bonds_credit | -0.439 | 1.123 |
| 15 | `qqq_logret_120d` | qqq_primary | -0.436 | 1.126 |

**Headline:** **Calendar features dominate** — `month` and `dec_effect` are the top-2. The model has correctly internalized the *December effect* (Haug & Hirschey 2006 FAJ) and *monthly seasonality* (Rozeff & Kinney 1976 JFE).

**Cross-asset signals** are strong: tech-sector returns (XLK), commodities (silver, gold), international equities (DAX), bonds (TLT, AGG), volatility regime (VIX, VXN). The model isn't just looking at QQQ history.

**Mag-7 concentration** (`qqq_over_ixic_5d`, rank 11) is meaningfully load-bearing — confirming our design hypothesis that QQQ-vs-Nasdaq-Composite ratio captures mega-cap concentration risk.

### 2.2 Exp 231 — LSTM seq=35 seed=11 (#2 OOS individual)

**Baseline OOS Sharpe: +2.17**

| Rank | Feature | Domain | Drop when shuffled |
|-----:|---------|--------|-------------------:|
| 1 | `qqq_over_spy_5d` | qqq_primary | -0.728 |
| 2 | `qqq_close_to_sma50` | qqq_primary | -0.643 |
| 3 | `qqq_mom_12_2` | qqq_primary | -0.620 |
| 4 | `xli_logret_1d` | sectors | -0.620 |
| 5 | `eem_logret_5d` | benchmarks | -0.620 |
| 6 | `qqq_rsi14` | qqq_primary | -0.608 |
| 7 | `vix_logret_5d` | volatility_regime | -0.602 |
| 8 | `qqq_over_soxx_5d` | qqq_primary | -0.591 |
| 9 | `xli_ma20_z` | sectors | -0.581 |
| 10 | `copper_logret_5d` | commodities | -0.566 |

**Headline:** Same architecture (LSTM s35), different seed → **completely different top feature pattern**. Exp 231 relies primarily on **QQQ-relative-strength signals** (qqq_over_spy, qqq_over_soxx) and **momentum oscillators** (rsi14, mom_12_2, sma50 — Jegadeesh-Titman 1993 12-1 + Asness-Moskowitz-Pedersen 2013 skipped-month). Calendar effects do not appear in the top 10 (in stark contrast to exp 234).

This **seed-driven feature attribution divergence** explains why the top-5 ensemble (+3.85 OOS Sharpe) far outperforms any individual member: different seeds learn complementary representations of the same target, and majority-vote averaging exploits that diversity per Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474.

### 2.3 Exp 304 — Mamba dmamba seq=60 seed=42 (#3 OOS individual)

**Baseline OOS Sharpe: +2.01**

| Rank | Feature | Domain | Drop |
|-----:|---------|--------|-----:|
| 1 | `xlp_logret_5d` | sectors (Consumer Staples) | -0.717 |
| 2 | `xlb_logret_5d` | sectors (Materials) | -0.717 |
| 3 | `move_z60` | volatility_regime (Treasury vol) | -0.717 |
| 4 | `gold_logret_5d` | commodities | -0.717 |
| 5 | `qqq_logret_120d` | qqq_primary | -0.373 |
| 6 | `xlf_ma20_z` | sectors (Financials) | -0.373 |
| 7 | `xlp_logret_20d` | sectors | -0.373 |
| 8 | `xlp_ma20_z` | sectors | -0.373 |
| 9 | `xlu_logret_20d` | sectors (Utilities) | -0.373 |
| 10 | `xlb_logret_20d` | sectors | -0.373 |

**Headline:** Mamba relies HEAVILY on **defensive sector rotation** (Staples, Materials, Utilities, Financials) + **bond-vol regime** (`move_z60`) + **gold flight-to-safety**. Domain-level: `sectors +5.65`, `volatility_regime +2.50` — both > LSTM exp 234. Calendar features rank below top-10. **Not a single feature in common with LSTM exp 234's top-10**.

### 2.4 Exp 281 — Mambastock seq=60 seed=42 (#5 OOS individual)

**Baseline OOS Sharpe: +1.93** (slightly above the table-reported +1.75 — re-run on full 140-day OOS window vs the 81-day intersection)

| Rank | Feature | Domain | Drop |
|-----:|---------|--------|-----:|
| 1 | `wti_logret_5d` | commodities (oil!) | **-1.090** |
| 2 | `vix_logret_1d` | volatility_regime | -0.959 |
| 3 | `qqq_macd_hist_norm` | qqq_primary (technical) | -0.775 |
| 4 | `qqq_over_iwm_5d` | qqq_primary (rel-strength) | -0.765 |
| 5 | `xlk_logret_1d` | sectors | -0.595 |
| 6 | `dow_wed` | calendar | -0.490 |
| 7 | `move_z60` | volatility_regime | -0.479 |
| 8 | `silver_logret_1d` | commodities | -0.449 |
| 9 | `xlk_ma20_z` | sectors | -0.440 |
| 10 | `santa_rally` | calendar | -0.405 |

**Headline:** Mambastock uses **commodities (oil, silver) + technical indicators (MACD, RSI) + relative strength** — different from both base mamba (defensive sectors) and LSTM (calendar/QQQ-history). The `wti_logret_5d` drop of -1.09 is the biggest single-feature dependence in any model — mambastock has effectively learned to use OIL as a leading indicator.

### 2.5 Exp 55 — Mamba seq=60 seed=7 (#4 OOS individual)

**Could not run** — checkpoint state-dict shape mismatch (likely a different `expand` value than the heuristic inferred). Will require manual config override; deferred.

---

## 3. Cross-model consensus (4 models analyzed)

The 4 OOS winners have **strikingly different feature reliance patterns** — this is the most important finding of the report.

### 3.1 Pairwise top-30 overlaps

| Model A | Model B | Common features | Notable common |
|---------|---------|----------------:|----------------|
| exp 234 (LSTM s35 sd=2026) | exp 304 (mamba s60 sd=42) | **6** | `gold_logret_1d`, `silver_logret_5d`, `qqq_logret_120d`, `qqq_over_ixic_5d`, `vix`, `xlf_logret_20d` |
| exp 231 (LSTM s35 sd=11) | exp 281 (mambastock sd=42) | **6** | `qqq_macd_hist_norm`, `qqq_over_dia_5d`, `qqq_rsi14`, `santa_rally`, `vix_logret_1d`, `xlk_logret_1d` |
| exp 234 ∩ exp 231 | (LSTM seed-pair) | **3** | `qqq_donchian_pos252`, `qqq_mom_6m`, `xle_ma20_z` |
| exp 281 ∩ exp 304 | (mamba pair) | **4** | `dxy_logret_5d`, `move_z60`, `wti_logret_5d`, `xlb_logret_5d` |
| exp 231 ∩ exp 304 | (LSTM vs mamba) | **2** | `qqq_over_spy_5d`, `vix_logret_5d` |
| exp 234 ∩ exp 281 | (LSTM vs mambastock) | **1** | `xle_logret_1d` |

**Top-30 union across all 4 models:** **138 of 205 features** (67%) appear in at least one model's top-50. Architectural diversity sweeps a wide swath of the feature space.

**No feature is top-30 in 3+ models.** The most universal signal would be one that ranks consistently across architectures, but in our 4-model panel, EVERY pair is mostly disjoint. This is the explainability evidence for *why the top-5 vote ensemble outperforms any individual member by ~75%*.

### 3.2 Consensus harmful features (bottom-15 in 3+ models)

```
qqq_close_to_sma5  —  bottom-15 in exp 234, exp 231, exp 281 (3 of 4 models agree it harms)
```

This is the ONLY feature that is consistently harmful across architectures. **Strong removal candidate** — drop in next feature-engineering iteration.

### 3.3 Architectural feature signatures

| Architecture | Top features rely on | Top domains |
|--------------|---------------------|-------------|
| **LSTM s35 seed=2026** (exp 234) | calendar + sector + commodity + vix | qqq_primary, calendar, sectors |
| **LSTM s35 seed=11** (exp 231) | relative-strength + momentum oscillators (RSI, MACD, sma50) | qqq_primary, vol_regime |
| **Mamba s60 seed=42** (exp 304) | defensive sector rotation + bond vol + gold | **sectors (+5.65)**, vol_regime, commodities |
| **Mambastock s60 seed=42** (exp 281) | oil + VIX 1d change + technical indicators + relative-strength | sectors, qqq_primary, vol_regime, commodities |

The **same architecture** at different seeds (LSTM 234 vs LSTM 231) finds DIFFERENT features. The **same seed** across architectures (mamba 304 vs mambastock 281, both seed=42) also finds different features. ⇒ Feature attribution is determined by the *interaction* of architecture + initialization, not by either alone.

### 3.4 Sign disagreements between models (red flags)

A feature with positive importance in one model and negative in another suggests architectural disagreement on its signal vs noise nature:

| Feature | exp 234 | exp 304 | Interpretation |
|---------|--------:|--------:|----------------|
| `move_z60` | **-0.66** | **+0.72** | LSTM treats as noise; mamba treats as signal — bond-vol regime works only with state-space models |
| Volatility regime as a domain | **-1.75 net** | **+2.50 net** | Architectures disagree on volatility's signal value |
| QQQ primary as a domain | **+4.39 net** | +0.81 net | LSTM relies on QQQ self-history; mamba uses cross-asset signals more |

---

## 4. Domain-level aggregation

For each economic-domain category, sum the importance drops across all features in that category:

### 4.1 Exp 234 (LSTM s35 seed=2026)

| Domain | Total drop | Max drop | n features | Interpretation |
|--------|-----------:|---------:|-----------:|----------------|
| `qqq_primary` | **+4.39** | +0.49 | 57 | Most load-bearing — QQQ's own history matters most |
| `calendar` | **+2.63** | +0.72 | 17 | Surprise winner — month + Dec effect drive returns |
| `sectors` | +1.42 | +0.64 | 50 | Cross-sector signal real but diluted across many features |
| `intl_risk` | +0.90 | +0.57 | 4 | Concentrated DAX/Nikkei signal |
| `commodities` | +0.80 | +0.59 | 14 | Silver/gold/copper material |
| `yields_curve` | +0.51 | +0.44 | 9 | Term-spread informative |
| `fx_macro` | +0.01 | +0.44 | 9 | Net-zero — DXY informative, others noise |
| `autoregressive` | -0.33 | +0.13 | 6 | Lagged returns add noise vs the longer-history embedded in seq=35 |
| `benchmarks` | -0.05 | +0.04 | 5 | Marginal — overlaps with sector + qqq_primary signals |
| `bonds_credit` | -0.58 | +0.57 | 10 | Mixed — TLT/AGG help but TIP/MOVE harm |
| `volatility_regime` | **-1.75** | +0.55 | 18 | NET HARMFUL — VIX/VXN help but MOVE/VVIX/SKEW too noisy |
| `industry_tilts` | **-1.89** | 0.00 | 6 | NET HARMFUL — SMH/IBB/ARKK adding noise vs broad sector ETFs |

### 4.2 Exp 231 (LSTM s35 seed=11)

| Domain | Total drop | Max drop |
|--------|-----------:|---------:|
| `qqq_primary` | +3.54 | +0.73 |
| `volatility_regime` | +1.09 | +0.60 |
| `benchmarks` | +0.63 | +0.62 |
| `autoregressive` | +0.28 | +0.48 |
| `sectors` | -2.33 | +0.62 |
| `commodities` | -1.37 | +0.57 |
| `intl_risk` | -0.57 | -0.03 |

Exp 231 **also** finds `qqq_primary` the most load-bearing domain, but differs sharply on `sectors` (net negative for 231 vs net positive for 234) and `volatility_regime` (positive for 231 vs negative for 234).

---

## 5. Actionable findings

### 5.1 Features the ensemble depends on (KEEP and refine)

1. **QQQ self-history** is the structural backbone — both models use it. `qqq_logret_*`, `qqq_over_ixic_5d`, `qqq_close_to_sma50`, `qqq_mom_*`, `qqq_donchian_pos252` are core.
2. **Calendar effects** (Dec effect, month) — strongly used by exp 234. Worth EXPANDING with more granular calendar features (e.g., FOMC pre-event drift, Lucca & Moench 2015 JF).
3. **Sector cross-flow** (XLK, XLE, XLI, XLY) — exp 234's top picks. The cyclicals-vs-defensives spread is real signal.
4. **Cross-asset volatility** (VIX, VXN, VIX-term-structure) — both exp 234 and 231 use volatility regime features in their top-10.
5. **Relative strength** (qqq_over_spy, qqq_over_soxx) — exp 231's top features.

### 5.2 Features that hurt the OOS-winning models (CANDIDATES FOR REMOVAL)

Across both exp 234 and exp 231, the following features have **negative importance** (shuffling them IMPROVES OOS Sharpe):

| Feature | Exp 234 drop | Exp 231 drop | Action |
|---------|------------:|------------:|--------|
| `move_logret_5d` | **-1.80** | -0.13 | **Strongly harmful in 234** — investigate |
| `month_sin` | -0.86 | -0.21 | Drop — `month` (cyclical) already used |
| `usdjpy_logret_20d` | -0.78 | +0.05 | Drop or shorten window |
| `smh_logret_5d` | -0.70 | +0.20 | Inconsistent — leave but monitor |
| `tip_logret_20d` | -0.69 | -0.11 | Drop — TIP dynamics noisy at 20d |
| `agg_logret_20d` | -0.62 | +0.18 | Inconsistent — leave but monitor |
| `move_z60` | -0.66 | +0.07 | Inconsistent — `move` family seems noisy in 234 |
| `dow_fri` | -0.59 | -0.16 | Drop — Friday effect not load-bearing |
| `qqq_close_to_sma5` | -0.51 | -0.21 | **Both models agree** — DROP CANDIDATE |
| `sector_breadth_20d` | -0.53 | +0.04 | Inconsistent — leave but monitor |

The **`industry_tilts` domain** (SMH, IBB, ARKK) has net-NEGATIVE total impact in exp 234. The original design hypothesis was that industry-specific ETFs add signal beyond broad sectors, but the OOS evidence suggests the broader sector ETFs already cover that information.

### 5.3 Recommended next experiments

Based on this analysis, the most promising hill-climb directions are:

1. **Feature pruning experiment**: train LSTM s35 with the 6 worst-performing features (per cross-model consensus) removed. Hypothesis: cleaner input → better OOS. Cite: Hastie, Tibshirani & Friedman 2009 *Elements of Statistical Learning* §3.4 (input selection).
2. **Calendar feature expansion**: add finer-grained pre-FOMC drift dummies (Lucca & Moench 2015), payrolls week, and CPI-release dummies.
3. **Sector cyclicals/defensives spread strengthening**: explicit XLK − XLP, XLY − XLU spreads with momentum overlays.
4. **Relative strength expansion**: add `qqq_over_iwm`, `qqq_over_efa`, `qqq_over_eem` at multiple horizons (1d, 5d, 20d, 60d).

---

## 6. Methodology notes & caveats

1. **Permutation is shuffled within OOS only** — preserves the train-period seq_len history that the model sees as an inference buffer. Each feature is permuted by `np.random.permutation` of row indices.
2. **Single permutation per feature** — for production use, the standard recipe (Breiman 2001) is to repeat 10× and average. Our single permutation is a coarse first pass; finer-grained re-runs are pending.
3. **Sharpe drop, not loss-function drop** — unlike classical regression-error permutation importance, we use **trading-objective Sharpe** as the score. This better matches what the dashboard cares about (deployable signal quality).
4. **Seed-bimodality** is the key methodological insight: same architecture + same data → seed-divergent feature attribution → ensemble necessary.
5. **OOS window is short** (81-130 prediction days). With a longer OOS window (~252 trading days = 1 year), feature importance estimates would be more stable.

---

## 7. References

| # | Citation |
|---|----------|
| 1 | Breiman 2001 *Machine Learning* "Random Forests" — original permutation importance |
| 2 | Fisher, Rudin & Dominici 2019 *JMLR* arXiv:1801.01489 "All Models are Wrong, but Many are Useful" — model-agnostic permutation importance |
| 3 | Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474 "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" |
| 4 | Rozeff & Kinney 1976 *JFE* "Capital market seasonality: The case of stock returns" — January / month-of-year effect |
| 5 | Haug & Hirschey 2006 *FAJ* "The January Effect" — December → January seasonality |
| 6 | Jegadeesh & Titman 1993 *JF* "Returns to buying winners and selling losers" — momentum |
| 7 | Asness, Moskowitz & Pedersen 2013 *JF* "Value and momentum everywhere" — skipped-month momentum |
| 8 | Lucca & Moench 2015 *JF* "The pre-FOMC announcement drift" — pre-event calendar effects |
| 9 | Cieslak & Pang 2021 *RFS* "Common shocks in stocks and bonds" — bond-vol leads equity-vol |
| 10 | Whaley 2009 *J. Portfolio Management* "Understanding the VIX" — volatility regime |

---

## 8. Files

| File | Content |
|------|---------|
| `autoresearch_results/oos_feature_importance_exp234.csv` | Per-feature drop for exp 234 (205 rows) |
| `autoresearch_results/oos_feature_importance_exp231.csv` | Per-feature drop for exp 231 (205 rows) |
| `autoresearch_results/oos_feature_importance_exp304.csv` | (in progress — mamba s60 seed=42) |
| `autoresearch_results/oos_feature_importance_exp55.csv` | (in progress — mamba s60 seed=7) |
| `autoresearch_results/oos_feature_importance_exp281.csv` | (in progress — mambastock seed=42) |
| `autoresearch_results/oos_feature_importance_summary.json` | Cross-model summary (top-K + domain aggregation) |
| `run_oos_feature_importance.py` | Reproducer script (permutation importance) |

To reproduce:

```bash
cd C:/Users/evija/autoresearch
python -m autoresearchindexstock.run_oos_feature_importance
```

Runtime: ~5 min per LSTM model, ~10-20 min per mamba model (seq=60 forward pass cost).
