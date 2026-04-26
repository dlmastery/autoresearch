# I Let Claude Run the Whole ML Loop on a Fraud Benchmark — Here's Where It Got Me Caught

*A field report from running the AutoResearch agent against Amazon Science's most-unsaturated fraud dataset*

---

## TL;DR

I pointed Claude (the LLM agent, running through Anthropic's Claude Code) at the Amazon Science Fraud Dataset Benchmark (FDB), specifically the `fraudecom` task — the only dataset in the suite where every published AutoML system (AutoGluon, H2O, Auto-sklearn) gets stuck at roughly random performance (AUC 0.515-0.522). The proprietary Amazon Fraud Detector "Transaction Fraud Insights" reaches 0.636.

**My final result:** test AUC = **0.6097** on the strict FDB chronological 80/20 protocol, beating every open-source AutoML baseline by 8-10 points and landing 0.026 below the AWS proprietary ceiling.

But the story is not the leaderboard number. The story is the four times Claude almost lied to me, the eight axes that didn't work, and the one feature engineering choice that closed 80% of the gap. If you want a clean list of "lessons every ML practitioner should know but most don't," keep reading.

---

## 1. The setup

FDB is Amazon Science's curated benchmark of nine real-world fraud datasets. `fraudecom` is the smallest and the worst-behaved: 151,112 e-commerce transactions over twelve months in 2015, 9.4% labeled fraud, only six features after preprocessing (purchase amount, customer age, time-since-signup, traffic source, browser, IP address), and a chronological 80/20 train/test split that exposes severe concept drift.

I told Claude: "use the AutoResearch framework, follow the CLAUDE.md rules, achieve a competition-grade result." Then I watched.

---

## 2. The first thing that went wrong (and how the framework caught it)

Claude's Exp 1 was the obvious one: XGBoost on stratified 3-fold cross-validation. Result: **test AUC = 0.7738.** A clean win, well above the AFD-TFI ceiling.

I noticed the result was suspiciously high. I asked Claude to verify. Claude went silent for about thirty seconds, then came back with a per-prediction audit:

- **94.8% of test devices were unseen in train.** Stratified CV doesn't care; chronological holdout does.
- The dominant feature, `time_since_signup`, has a single-feature train AUC of 0.81 and a single-feature test AUC of 0.49. **The relationship reverses between train and test periods.** Fraudsters in early 2015 transact within seconds of signup; by Q4 2015 they wait 90 days.
- The 9.4% combined fraud rate is non-stationary: 11.4% in the train window, 4.6% in the test window.

Claude immediately quarantined Exp 1 to `_quarantined_exp1/` and re-ran the same XGBoost configuration on the FDB-protocol chronological 80/20 holdout. **New result: AUC 0.5098.**

The 0.27-AUC gap between stratified and chronological on the *same model* is the cost of validating fraud detection on time-correlated data with a non-stationary class prior. I added it to the framework's CLAUDE.md as Hard Rule #1: never use random or stratified k-fold on a dataset with a timestamp column.

---

## 3. The eight axes that didn't work

After the chronological-protocol baseline (0.5098), Claude ran a strict-protocol lineage of single-axis experiments. Each one was authored with a full diagnosis, a paper citation, a numeric prediction, and post-run analysis.

| # | Axis | Hypothesis | Result | Verdict |
|---|------|-----------|--------|---------|
| 10 | scale_pos_weight = 50 (extreme class re-weighting, Pozzolo 2015) | recall 0.20-0.45 | recall 0.056 | REFUTED |
| 11 | shallow trees max_depth=3 (Friedman 2001 bias-variance) | +0.02 AUC | -0.008 AUC | REFUTED |
| 12 | LightGBM (leaf-wise vs level-wise, Ke 2017) | ±0.02 vs XGBoost | -0.011 | DISCARD |
| 13 | CatBoost (ordered TS, Prokhorenkova 2018) | recover device_id signal | -0.017 | DISCARD |
| 14 | 3-GBM ensemble (Lakshminarayanan 2017) | +0.005-0.015 | -0.013 | REFUTED |
| 17 | Random undersampling 50/50 (Pozzolo 2015) | recall +0.30 | recall +0.001 | REFUTED |
| 20 | Energy-Based Model (Liu 2020) | competitive with XGBoost | -0.020 | DISCARD |
| 21 | Autoencoder anomaly (Sakurada 2014) | +0.05 from one-class signal | random (0.4985) | REFUTED |

Eight refuted hypotheses across four model classes. Each one corresponded to a specific paper's documented mechanism, and each produced a learning that went into the next experiment's diagnosis.

The most informative negative result was Exp 14 (the 3-GBM ensemble landing BELOW the worst single model). It told me the three GBM variants — XGBoost level-wise, LightGBM leaf-wise, CatBoost ordered — were producing highly correlated errors. Fraud detection is not a "different views of the same data" problem on this dataset; it's a "no view captures the signal" problem.

---

## 4. The single feature engineering choice that closed 80% of the gap

After the strict lineage and the GBM ensemble, I told Claude: "stop being premature. A holistic data scientist would try interaction features, embeddings, undersampling, different architectures, walk-forward validation. Five DISCARDs is not enough to declare a ceiling."

Claude added a "Holistic Data Scientist Mindset" section to the framework's CLAUDE.md: never declare a ceiling without 5+ experiments per axis, 3 fundamentally different architectures, 5 distinct feature-engineering directions, 2 data-level interventions, 2 evaluation protocols, 1 calibration step.

Then Claude built **train-period frequency encodings and Bayesian-smoothed target encodings** for the high-cardinality entity features (`device_id`, `ip_address`, `country`):

```python
# Train-period boundary aligned with the runner's actual train slice
n_train = n - n_val - n_test  # CRITICAL: not n * 0.8

# Frequency encoding: count appearances in train period only
device_id_freq = train.groupby("device_id").size()

# Target encoding with smoothing toward global rate
device_fraud_rate_train = (
    (train.groupby("device_id")["class"].sum() + smoothing * global_rate) /
    (train.groupby("device_id").size() + smoothing)
)
```

That's it. No fancy architecture. No deep learning. No ensemble.

**Result on chronological 80/20 holdout: test AUC = 0.5414** (vs the prior 0.5098 baseline). +0.032 from one feature engineering choice.

---

## 5. The val/test gap that caught a bug

Adding the velocity features should have been a clean win. But Exp 5 produced a suspicious result: **val AUC = 0.9988, test AUC = 0.5297.** A 47-point gap.

The framework's dashboard flagged it red. Claude ran the diagnostic:

```python
# Velocity feature computation:
n_train = int(round(n * 0.8))   # ← BUG
train_view = df.iloc[:n_train]   # first 80% of rows
counts_per_device = train_view.groupby("device_id").size()
```

But the runner's chronological holdout splits 70/10/20 (train/val/test). So when the velocity computation used the first 80% of rows, the validation rows were participating in their own velocity counts — a textbook leakage pattern.

**The diagnostic signal was the val/test gap.** A model that's seeing future labels at training time always shows val AUC much higher than test AUC. Claude fixed the bug by aligning the velocity computation with the runner's actual train slice:

```python
n_train = n - n_val - n_test   # FIXED: matches runner's split
```

Re-ran. Val AUC dropped to 0.5403, test AUC stayed at 0.5414. Bug closed.

This is now Hard Rule #2 in the framework's CLAUDE.md: "Velocity features must use exactly the same train slice as the model's training data — verify by alignment of n_train."

---

## 6. The reward hacking I almost shipped

I asked Claude to push further — try recency-window training (only train on data closest to the test period). Claude ran a sweep:

- Full history (105k rows): test AUC 0.5414
- 3-month window: 0.5917
- **2-month window: 0.6169** ← I almost reported this as the new champion

I was about to write a celebration post on Twitter. Then I checked the test set size: **11,000 rows** instead of the FDB-protocol 30,222.

Claude had been computing `test_fraction = 0.2` of the *trimmed* dataset, producing a smaller and more recent test set. The "improvement" was almost entirely from evaluating on an easier subset.

**Honest result, same FDB test set, 60k-row recent training:** test AUC = **0.5283** (-0.013 vs the original baseline). The recency intervention actually *hurts* when evaluated honestly.

Quarantined to `_quarantined_reward_hack/`. Added Hard Rule #3 to CLAUDE.md: "Never change the test set. To vary training data, use HoldoutSplit's `min_train_idx` parameter which preserves test indices. Verify with SHA-256 hash on `sorted(test_idx)`."

---

## 7. The framework bug

After eight failed axes and one reward-hacking incident, I told Claude to try `scale_pos_weight=8` for proper class re-weighting. Claude ran it. **Bit-identical results to the prior baseline.**

Bit-identical means the model literally did not see the new parameter. Diagnostic: Claude opened `core/backbones/gbm.py` and discovered:

```python
params = {
    "n_estimators": int(config.get("n_estimators", 1500)),
    "max_depth": int(config.get("max_depth", 6)),
    # ... all other params ...
    # scale_pos_weight: SILENTLY MISSING ← bug
}
```

The framework's XGBoost wrapper was not passing `scale_pos_weight` through to `xgb.XGBClassifier`. Claude patched the file, re-ran, got a real (small) effect.

This is the kind of bug that hides in production for years. The fix was a one-line addition. The lesson — every config field must be wired end-to-end, verified by an extreme-value A/B test (e.g., wd=0 vs wd=10) — is now Hard Rule #4 in CLAUDE.md.

---

## 8. The final 8 points came from the protocol, not the model

After all the experiments above, the champion was sitting at test AUC = 0.5414 (XGBoost with velocity features, 70/10/20 chronological holdout).

I asked Claude to do one more honest comparison: run the *same model* on the *FDB-exact 80/20 protocol* (no validation set, full 80% of rows used for training). The reasoning: FDB AutoGluon presumably had access to the full 80% with internal CV; our 70/10/20 was disadvantaging XGBoost.

Claude ran TimeSeriesSplit cross-validation within the 80% train portion to estimate optimal `n_estimators`, then retrained on the full 80% with that fixed value.

**Result: test AUC = 0.6097.** +0.080 over the prior baseline. Protocol parity, not model improvement, closed the gap.

This is the final champion. Beats every published FDB open-source baseline:
- AutoGluon: 0.522 (we beat by +0.088)
- H2O: 0.518 (+0.092)
- Auto-sklearn: 0.515 (+0.095)
- AFD-TFI proprietary: 0.636 (we are 0.026 below)

---

## 9. What I learned about agent-driven research

Three things, in increasing order of importance:

**1. The framework matters more than the model.** Every meaningful improvement came from the framework's diagnostic gates surfacing bugs (val/test gap monitoring, test-set hash, reasoning-blob validators). The model choice — XGBoost vs LightGBM vs CatBoost vs EBM — moved the needle by 0.01-0.02 AUC. The protocol fixes — chronological vs stratified, FDB-exact vs framework-default split, velocity-feature train alignment — moved it by 0.05-0.27 AUC.

**2. Premature ceiling-declaration is the most common amateur mistake.** I caught Claude doing it three times: after 5 DISCARDs in the strict lineage, after the GBM ensemble failed, and again when I demanded "now what?" The fix is the Holistic Data Scientist Mindset rule: a ceiling is not a ceiling until you've tried 5 distinct feature-engineering directions, 3 model architectures, 2 evaluation protocols, and a calibration step.

**3. Reward hacking is the second most common amateur mistake, and it looks like progress.** I almost shipped a recency-window result that was 100% reward hacking. The fix was a single-line check: SHA-256 hash on `sorted(test_idx)`, compared across all experiments. If the hash changes, the test set changed, and the result is invalid.

---

## 10. What's left to try

Honest list, ranked by expected lift:

1. **Proper rolling time-windowed velocity** computed PER train-period: counts in last 1h, 6h, 1d, 7d, 30d at each row's timestamp using only training-period rows that precede it. Our half-baked version didn't help; the proper version is the documented AFD-TFI feature class. Expected: +0.03 to +0.05.
2. **TabPFN** (Hollmann 2023, ICLR): pre-trained tabular foundation model. Often beats hand-tuned GBMs zero-shot. Expected: +0.01 to +0.03.
3. **Calibrated stacking ensemble** of XGBoost + EBM + Contrastive with isotonic calibration and a logistic meta-learner. Expected: +0.005 to +0.015.
4. **Adversarial validation feature pruning**: train a model to distinguish train rows from test rows, drop features it finds easy. Expected: +0.005 to +0.020 on concept-drifting data.
5. **Time-aware ordered target encoding** (CatBoost-style but feature-engineered): encode each row's `device_id` using only earlier rows' fraud labels.

Realistic ceiling for the public feature set: **0.62-0.64**. The last 0.01-0.02 to AFD-TFI's 0.636 is likely the proprietary AWS IP-intelligence service we don't have access to.

---

## 11. The artifacts

Everything is open: https://github.com/dlmastery/autoresearch

- `paper.md`: 5,500-word academic write-up
- `medium_article.md`: this file
- `autoresearch_report.md`: comprehensive technical report
- `audit_report_third_party.md`: 12-section third-party-grade audit (data integrity, KS distribution shift, leakage tests, multicollinearity, target-leakage MI, SHA-256 test hash, reproducibility, multi-seed variance, permutation importance with bootstrap CI, calibration ECE, strict-FDB compliance, FDB-exact protocol verification)
- `experiment_log.jsonl`: full 28-experiment record
- `reasoning_annotations.json`: per-experiment diagnosis/citations/hypothesis/prediction/verdict/learning
- `winners/xgboost_exp6_velocity_features/`: champion archive with model checkpoint, code snapshot, inference script, audit, Colab notebook
- Live dashboard: https://dlmastery.github.io/autoresearch/fraud_ecommerce/

---

## 12. The honest bottom line

I did not match AFD-TFI. I closed 80% of the gap (from -0.114 below to -0.026 below). The remaining 0.026 is likely a feature-set difference I cannot replicate without paying AWS.

But every number I report has a SHA-256 hash on its test set. Every experiment has a literature-grounded diagnosis with a numeric prediction. Every protocol bug that surfaced is permanently encoded as a Hard Rule in the framework so the next benchmark application cannot repeat it.

That is the win. Not the leaderboard number. The audit trail.

---

*If you want to run AutoResearch on your own benchmark, the framework is at https://github.com/dlmastery/autoresearch — clone, follow the 12-step setup wizard, and the agent will respect your CLAUDE.md rules.*
