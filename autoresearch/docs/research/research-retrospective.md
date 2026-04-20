# Research Retrospective: Currency Prediction Design

**Date:** 2026-04-04
**Session type:** Brainstorming + research + design

This document traces how the design evolved through collaborative discussion. Each section records what was proposed, what feedback was given, and how it improved the outcome.

---

## 1. Pivot from Colab to Local CLI

**Context:** The existing autoresearch project was a Colab notebook (GPT training optimizer). Modifying cells triggered confirmation dialogs, making the agent loop impractical.

**Decision:** Pivot to a locally runnable CLI tool with a new ML problem — currency trading prediction.

**Impact:** Eliminated the Colab friction entirely. CLI gives the optimizer full control over code modification + execution without interactive confirmations.

---

## 2. Currency Pair Selection: "Major + Cross"

**Initial proposal:** 4 options ranging from single pair to user-defined.

**My recommendation:** Single pair (EUR/USD) for a cleaner optimization target.

**User feedback:** "Start with B" — major + cross pairs.

**How this improved the design:** Multi-pair prediction forces the model to learn cross-currency relationships (e.g., EUR/USD and EUR/GBP share EUR dynamics). This is more realistic than single-pair and gives the model more signal. It also makes the problem harder, which gives the optimizer more room to explore (cross-pair features, attention across pairs, etc.).

---

## 3. Prediction Target: Multi-Horizon Returns

**Options presented:** Next-day direction, next-day magnitude, multi-horizon returns, volatility.

**My recommendation:** Next-day return magnitude (simpler, continuous metric).

**User feedback:** "C" — multi-horizon returns with Sharpe metric.

**How this improved the design:** Multi-horizon is significantly better because:
- Single-horizon models can overfit to one time scale
- Multi-horizon forces the model to learn structure at multiple frequencies
- Sharpe ratio as metric is harder to game than MSE (risk-adjusted, penalizes volatile predictions)
- More realistic for any actual trading use case

---

## 4. Overfitting Warning

**User feedback:** "Note this is a specific ML problem — make sure you don't overfit. This is just a setup of some ML problem."

**How this shaped the design:** This comment reframed the entire approach. Instead of building the most powerful model possible, the goal became building a **clean, leakage-free evaluation framework** that the optimizer can improve against. Led directly to:
- Multiple disjoint test sets (not just one train/test split)
- Purge gaps between splits
- Average Sharpe across all test sets as the optimization target
- The model is intentionally a baseline — the optimizer does the improving

---

## 5. Architecture: From "Simple LSTM" to LFM2.5

**Initial options presented:** TFT, PatchTST, TimesNet, or simple LSTM baseline.

**My recommendation:** Simple LSTM baseline (give optimizer room to improve).

**User feedback:** "Check the latest 2025/2026 arxiv research first."

**Research conducted:** Three parallel web searches covering:
- State-of-the-art financial time series models (found ms-Mamba, SST, xLSTM-Mixer, AdaMamba)
- Foundation models for time series (found Chronos-2, TimesFM-2.5, FinCast)
- Multi-horizon transformer/mamba hybrids

**How this improved the design:** Shifted from a generic baseline to a research-grounded starting point. The arxiv survey revealed that:
- **ms-Mamba** (March 2026) was the freshest SOTA for multi-scale time series
- **xLSTM** had the best risk-adjusted financial performance (Sharpe 1.79-1.99)
- Foundation models (Chronos-2) could serve as feature extractors
- The field had moved decisively toward hybrid Mamba-Transformer architectures

---

## 6. Liquid Foundation Model (LFM) Instead of LTC

**After the arxiv survey, I proposed:** LTC (Liquid Time-Constant) network as baseline, from the original 2020 paper.

**User feedback:** "Can you use LFM not LTC" — use the actual Liquid Foundation Model, not the old research network.

**User then added:** "Also look for ChromaDB foundation model based stuff, AutoGluon, Chronos."

**Additional research conducted:**
- LFM2.5 release details (March 31, 2026 — literally the previous week)
- AutoGluon v1.5 capabilities (wraps Chronos-2 + statistical models)
- ChromaDB + retrieval-augmented time series forecasting papers

**How this improved the design:** Massive upgrade. LFM2.5 vs LTC:
- LFM2.5 is a 354.5M param pretrained foundation model; LTC is a ~100-line research architecture
- LFM2.5 has already learned rich sequential representations from 28T tokens of training
- LFM2.5 has `inputs_embeds` pathway so we can feed numeric data directly
- LFM2.5 is available as a standard HuggingFace model (`pip install transformers`)
- LTC would need to be trained from scratch on our small dataset

---

## 7. "Fix 350M Model First"

**User feedback:** After the architecture was proposed, user said "Fix 350M model first" before building anything on top.

**What we did:** Systematic verification:
1. Checked Python/torch/transformers versions
2. Installed all dependencies
3. Downloaded `LiquidAI/LFM2.5-350M-Base` from HuggingFace
4. Verified `Lfm2Model` loads (354.5M params, hidden_size=1024, 16 layers)
5. Verified `inputs_embeds` pathway works (can bypass text tokenizer)
6. Confirmed CPU inference works (no GPU on this machine)

**How this improved the design:** Caught potential blockers before building:
- Confirmed no GPU = must use float32 (bf16 not well supported on CPU)
- Confirmed hidden_size=1024 (not 2560 as the default config suggested)
- Confirmed model downloads without authentication issues
- Proved the `inputs_embeds` bypass works — critical for feeding numeric financial data

---

## 8. Perfect Backtesting: No Leakage

**User feedback:** "Have multiple disjoint test data sets and multiple disjoint train data sets. Absolutely no leakage. Enough gap (few months) between test data sets and adjacent train data sets. Absolute perfect backtesting with no leakage. Have train test hold out validation data sets perfect."

**How this shaped the design:** This became a hard constraint, not a nice-to-have. Led to:
- 3-month purge gaps between every train/val and val/test boundary
- 7 disjoint test sets (not the usual single 80/20 split)
- Automated leakage detection tests planned (`test_leakage.py`)
- Scaler fit on train only
- All features backward-looking only
- The optimization metric averages across ALL test sets

---

## 9. Regime-Aware Splits: "More Splits, Analyze Downturns/Upturns/Plateaus"

**Initial proposal:** 3 walk-forward splits with purge gaps.

**User feedback:** "More splits. Analyze and see cases where there is downturn, upturn, and plateau and all."

**What we did:** Downloaded 20+ years of EUR/USD data and ran empirical regime analysis:
- Computed 3-month rolling annualized returns and volatility per quarter
- Classified each quarter as STRONG_UP, STRONG_DOWN, PLATEAU, HIGH_VOL
- Identified 14 strong downturn quarters, 8 strong upturn quarters, 17 plateau quarters
- Found extreme events: 2008Q4 (55% annualized vol!), 2015Q1 (-10.5%), 2010Q3 (+9.0%)

**How this improved the design:** Went from 3 generic splits to 7 regime-targeted splits where each test set deliberately covers a different market condition:
- Crisis onset, post-crash recovery, debt plateau, strong trend, low-vol calm, downturn, recent mixed
- The model must perform across ALL regimes to score well
- This is much harder to overfit than testing on a single period that might be "easy"

---

## Key Takeaways

1. **"Check the research first"** was the highest-impact comment — shifted from a generic baseline to a cutting-edge foundation model released days earlier
2. **"Use LFM not LTC"** caught a gap between academic research and production tooling — always use the real product, not the paper's toy implementation
3. **"Fix the model first"** is sound engineering — verify the core dependency works before building the system around it
4. **"No leakage" as a hard constraint** forced rigorous split design that became the strongest part of the system
5. **"More splits, analyze regimes"** turned a routine train/test split into a regime-aware evaluation framework that's genuinely resistant to overfitting

---

## What's Next

1. Write implementation plan (project structure, module-by-module build order)
2. Build data pipeline with leakage tests
3. Build model with LFM2.5 backbone
4. Train + evaluate baseline across 7 folds
5. Port autoresearch optimizer loop from Colab to CLI
6. Run optimizer
