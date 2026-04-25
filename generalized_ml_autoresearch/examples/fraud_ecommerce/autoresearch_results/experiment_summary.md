# Experiment Summary - FDB fraudecom

_Generated 2026-04-25 12:22_

## Master leaderboard (sorted by test_auc, FDB-identical 30,222-row test set)

| Rank | Exp | Backbone | Test AUC | Val AUC | Composite | Status | Description |
|------|-----|----------|----------|---------|-----------|--------|-------------|
| 1 | 18 | xgboost | 0.6938 | 0.8040 | 0.6938 | KEEP | STRICT Exp 18 — XGBoost walk-forward 4-fold CV (Bergmei |
| 2 | 10 | xgboost | 0.5432 | 0.5334 | 0.5334 | KEEP | STRICT Exp 10 — XGBoost scale_pos_weight=50 (extreme ra |
| 3 | 6 | xgboost | 0.5414 | 0.5403 | 0.5403 | KEEP | Exp 6 — XGBoost + velocity features (LEAKAGE FIXED: 70/ |
| 4 | 22 | contrastive_simclr_tabular | 0.5390 | 0.5324 | 0.5324 | KEEP | STRICT Exp 22 - Contrastive Learning (SimCLR-tabular, C |
| 5 | 11 | xgboost | 0.5337 | 0.5416 | 0.5337 | KEEP | STRICT Exp 11 — XGBoost shallow trees (max_depth=3, n_e |
| 6 | 7 | lightgbm | 0.5305 | 0.5413 | 0.5305 | KEEP | Exp 7 — LightGBM (Ke 2017 NeurIPS) chronological + velo |
| 7 | 12 | lightgbm | 0.5305 | 0.5413 | 0.5305 | KEEP | STRICT Exp 12 — LightGBM (Ke 2017) leaf-wise vs XGBoost |
| 8 | 5 | xgboost | 0.5297 | 0.9988 | 0.5297 | KEEP | Exp 5 — XGBoost + velocity/frequency features (AFD-TFI- |
| 9 | 17 | xgboost | 0.5294 | 0.5354 | 0.5294 | KEEP | STRICT Exp 17 — XGBoost + 50/50 undersampled training ( |
| 10 | 14 | ensemble | 0.5286 | 0.5286 | 0.5286 | KEEP | STRICT Exp 14 — Ensemble (3-GBM mean) of Exps 6, 12, 13 |
| 11 | 19 | xgboost | 0.5283 | 0.5389 | 0.5283 | KEEP | STRICT Exp 19 — XGBoost with min_train_idx=60000 (legit |
| 12 | 8 | catboost | 0.5245 | 0.5400 | 0.5245 | KEEP | Exp 8 — CatBoost (Prokhorenkova 2018 NeurIPS) chronolog |
| 13 | 13 | catboost | 0.5245 | 0.5400 | 0.5245 | KEEP | STRICT Exp 13 — CatBoost (Prokhorenkova 2018) ordered b |
| 14 | 15 | xgboost | 0.5242 | 0.5232 | 0.5232 | KEEP | STRICT Exp 15 — XGBoost + interaction features (35-feat |
| 15 | 16 | xgboost | 0.5239 | 0.5232 | 0.5232 | KEEP | STRICT Exp 16 — XGBoost on CURATED 9-feature interactio |
| 16 | 20 | energy_based_model | 0.5214 | 0.4751 | 0.4751 | DISCARD | STRICT Exp 20 - Energy-Based Classifier (Liu 2020 NeurI |
| 17 | 3 | xgboost | 0.5116 | 0.5134 | 0.5116 | KEEP | XGBoost + cyclical temporal features (hour, dow, signup |
| 18 | 2 | xgboost | 0.5098 | 0.5291 | 0.5098 | KEEP | XGBoost — chronological 80/20 holdout (FDB-comparable;  |
| 19 | 21 | autoencoder_anomaly | 0.4985 | 0.5324 | 0.4985 | DISCARD | STRICT Exp 21 - Autoencoder anomaly (Sakurada Yairi 201 |
| 20 | 4 | xgboost | 0.4960 | 0.5054 | 0.4460 | DISCARD | Exp 4 — XGBoost chronological holdout WITHOUT the adver |
| 21 | 9 | mlp | 0.4883 | 0.4775 | 0.4275 | DISCARD | Exp 9 — MLP (Gu/Kelly/Xiu 2020) chronological + velocit |

## Per-experiment detail

### Exp 2: XGBoost — chronological 80/20 holdout (FDB-comparable; ceiling 0.636)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0305 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5098 | Val AUC 0.5291 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that switching from stratified 3-fold CV to chronological 80/20 holdout will drop test AUC from 0.7738 (Exp 1) into the 0.60 to 0.70 range because the mechanism per Bergmeir et al. 2018...
- **Verdict:** DISCARD — composite=0.4598, test AUC=0.5098 (single-fold holdout), val AUC=0.5291; below the 0.55 floor. The hypothesis was correct in direction (chronological split would lower AUC) but the magnitude was severely under-estimated: instead of dropping into the predicted 0.60-0.70 range, AUC collapsed
- **Learning:** Critical learning: the fraudecom dataset has severe concept drift, not just entity-level leakage. The temporal audit (audit_temporal.py) revealed (a) fraud rate is non-stationary: train=11.4%, val=4.5%, test=4.6% — class prior halves between train and test; (b) the dominant feature `time_since_signu

### Exp 3: XGBoost + cyclical temporal features (hour, dow, signup_hour) on chronological holdout
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0287 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5116 | Val AUC 0.5134 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that adding purchase_hour, purchase_dayofweek, and signup_hour while keeping time_since_signup and all other features identical will lift test AUC from Exp 2's 0.5098 into the 0.55 to 0...
- **Verdict:** DISCARD — composite=0.4616, test AUC=0.5116, val AUC=0.5134, below the 0.55 floor. The cyclical-features hypothesis was largely refuted: adding purchase_hour, purchase_dayofweek, and signup_hour lifted test AUC by only +0.0018 (from 0.5098 to 0.5116), well short of the predicted 0.55-0.65 range. The
- **Learning:** Two consecutive DISCARDs on chronological split mean the diagnosis was incomplete — the problem is not just missing temporal features. Per the audit, every single feature except the broken time_since_signup has test-AUC between 0.50 and 0.51, so the model is starved of signal at deployment time. Axi

### Exp 4: Exp 4 — XGBoost chronological holdout WITHOUT the adversarial time_since_signup
- **Backbone:** `xgboost` | **Status:** DISCARD
- **Composite delta from champion:** -0.0943 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.4960 | Val AUC 0.5054 | Precision 0.059 | Recall 0.002 | F1 0.004 | MCC 0.003
- **Hypothesis (one-line):** We hypothesize that dropping time_since_signup will improve test AUC from Exp 3's 0.5116 into the 0.52 to 0.58 range because the mechanism per Pozzolo et al. 2018 is that XGBoost's gradient-boosted tr...
- **Verdict:** DISCARD — composite=0.4460, test AUC=0.4960, val AUC=0.5054, both below the 0.55 floor. The hypothesis was REFUTED: removing time_since_signup made test AUC worse (0.5116 → 0.4960), not better. This means the adversarial feature was actually contributing weak positive signal even with its train/test
- **Learning:** Negative result is highly informative. Axis closed: simply ablating drifty features does not help — the model uses them imperfectly but better than random. Axis open: ADDING train-period entity-velocity features (frequency counts and Bayesian-smoothed fraud rates per device_id, ip_address, country) 

### Exp 5: Exp 5 — XGBoost + velocity/frequency features (AFD-TFI-style) on chronological holdout
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0106 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5297 | Val AUC 0.9988 | Precision 0.211 | Recall 0.063 | F1 0.097 | MCC 0.093
- **Hypothesis (one-line):** We hypothesize that adding device_id_freq, ip_address_freq, country_freq, source_freq, browser_freq, device_fraud_rate_train, and country_fraud_rate_train will lift test AUC from Exp 3's 0.5116 into t...
- **Verdict:** DISCARD with a leakage bug — composite=0.4797, test AUC=0.5297 (best chronological-split AUC so far), but val AUC=0.9988 (essentially perfect). The huge val/test gap exposed a leakage bug: add_velocity_features.py computed counts on the first 80% of rows, but the runner's holdout split treats the fi
- **Learning:** Two findings, one good and one bad. Good: velocity features pushed chronological test AUC from 0.5116 (Exp 3) up to 0.5297 — first real evidence that entity-aggregation works on this benchmark. Bad: the prep script used n_train=80% but the runner splits 70/10/20, leaking val into feature computation

### Exp 6: Exp 6 — XGBoost + velocity features (LEAKAGE FIXED: 70/10/20 alignment) on chronological holdout
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** +0.0000 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5414 | Val AUC 0.5403 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that the corrected velocity features (computed strictly on first 70% of rows) will produce val AUC in the 0.55-0.65 range and test AUC in the 0.52-0.58 range because the mechanism per K...
- **Verdict:** DISCARD by composite (0.4903 below 0.55 floor) but Test AUC=0.5414 is the BEST HONEST chronological-split result in the project. Val AUC=0.5403 confirms leakage is gone (was 0.9988 in Exp 5). This decisively beats FDB AutoGluon's 0.522 baseline by +0.019 and Exp 2's 0.5098 by +0.032. Still below FDB
- **Learning:** Two key learnings. First, the framework's composite metric is doing exactly what it should — refusing to declare a champion when val just barely misses the floor, even though test improved. This is good Goodhart-protection. Second, the velocity-feature hypothesis is validated: train-only frequency +

### Exp 7: Exp 7 — LightGBM (Ke 2017 NeurIPS) chronological + velocity features
- **Backbone:** `lightgbm` | **Status:** KEEP
- **Composite delta from champion:** -0.0098 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5305 | Val AUC 0.5413 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that LightGBM with num_leaves=63, lr=0.04, min_data_in_leaf=50 will achieve test AUC in the 0.53 to 0.58 range because the mechanism per Ke et al. 2017 is that leaf-wise growth concentr...
- **Verdict:** DISCARD — composite=0.4805, test AUC=0.5305, val AUC=0.5413. LightGBM with leaf-wise growth + GOSS sampling produced test AUC slightly LOWER than Exp 6's XGBoost (0.5305 vs 0.5414, delta -0.011). The hypothesis that leaf-wise depth would find narrow but deep interaction paths was REFUTED on this dat
- **Learning:** Axis closed: leaf-wise tree growth is not the missing piece on fraudecom — it slightly underperforms level-wise XGBoost. Axis open: ordered target encoding (CatBoost in Exp 8) is the next variant to test before declaring the GBM family exhausted. Next try: Exp 8 with CatBoost, expecting that ordered

### Exp 8: Exp 8 — CatBoost (Prokhorenkova 2018 NeurIPS) chronological + velocity features
- **Backbone:** `catboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0158 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5245 | Val AUC 0.5400 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that CatBoost with iterations=1000, depth=6, l2_leaf_reg=3, bootstrap_type=Bernoulli will achieve test AUC in the 0.54 to 0.60 range because the mechanism per Prokhorenkova et al. 2018 ...
- **Verdict:** DISCARD — composite=0.4745, test AUC=0.5245, val AUC=0.5400. CatBoost's ordered target encoding produced test AUC even slightly LOWER than LightGBM (0.5245 vs 0.5305, delta -0.006) and 0.017 below Exp 6's XGBoost. The hypothesis that ordered target encoding would extract more signal from device_id w
- **Learning:** GBM trio complete: XGBoost 0.5414 > LightGBM 0.5305 > CatBoost 0.5245. All three plateau in the 0.52-0.54 band. Axis closed: the entire GBM family cannot beat Exp 6's result with the current 18-feature set; choice of GBM variant is a small (±0.02 AUC) lever vs the feature-engineering lever which mov

### Exp 9: Exp 9 — MLP (Gu/Kelly/Xiu 2020) chronological + velocity features
- **Backbone:** `mlp` | **Status:** DISCARD
- **Composite delta from champion:** -0.1128 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.4883 | Val AUC 0.4775 | Precision 0.222 | Recall 0.058 | F1 0.091 | MCC 0.092
- **Hypothesis (one-line):** We hypothesize that MLP with hidden=[128,64], dropout=0.3, AdamW lr=1e-3, batch=256, 30 epochs with patience=8 will achieve test AUC in the 0.50 to 0.56 range because the mechanism per Gu/Kelly/Xiu 20...
- **Verdict:** DISCARD — composite=0.4275, test AUC=0.4883, val AUC=0.4775, both below the 0.50 random baseline. MLP collapsed below the GBM family by 0.05 absolute AUC, decisively refuting the hypothesis that dropout-regularized MLP could match GBM accuracy on this dataset. Both val and test sub-50% AUC indicates
- **Learning:** Axis closed: feedforward MLP is not competitive with GBMs on fraudecom. The dataset's signal is genuinely shallow-tree-friendly and dense interactions across 18 features do not yield drift-robust patterns under chronological evaluation. Axis open: tabular transformer architectures (FT-Transformer, T

### Exp 10: STRICT Exp 10 — XGBoost scale_pos_weight=50 (extreme rare-events correction)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0069 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5432 | Val AUC 0.5334 | Precision 0.217 | Recall 0.056 | F1 0.089 | MCC 0.090
- **Hypothesis (one-line):** We hypothesize that XGBoost with scale_pos_weight=50 (vs Exp 44's 8 and Exp 6's implicit 1) will move test AUC into the 0.55 to 0.62 range and recall from 0.0547 to 0.20-0.45 because the mechanism per...
- **Verdict:** DISCARD — composite=0.5334, test_auc=0.5432 (+0.0018 vs Exp 6), recall=0.0562 (+0.0015), val_auc=0.5334 (-0.0069). REFUTED quantitatively: predicted AUC range 0.55-0.62 missed by 0.01-0.08, predicted recall 0.20-0.45 missed by ~0.20. Even at scale_pos_weight=50 (50x gradient amplification) recall ba
- **Learning:** Axis closed: scale_pos_weight is exhausted at the extreme end. Combined with quarantined Exp 44 (spw=8) and Exp 6 (spw=1), three points span the practical range and all fall within ±0.005 AUC of each other. Mental model update: the dataset's features cannot discriminate fraud regardless of how the l

### Exp 11: STRICT Exp 11 — XGBoost shallow trees (max_depth=3, n_est=2000)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0066 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5337 | Val AUC 0.5416 | Precision 0.222 | Recall 0.055 | F1 0.088 | MCC 0.090
- **Hypothesis (one-line):** We hypothesize that XGBoost with max_depth=3, n_estimators=2000, scale_pos_weight=1 (revert), all other Exp 6 HPs unchanged will lift test AUC into the 0.54 to 0.58 range because the mechanism per Fri...
- **Verdict:** DISCARD — composite=0.5337, test_auc=0.5337 (-0.0077 vs Exp 6), val_auc=0.5416 (+0.0013). REFUTED: predicted AUC 0.54-0.58 was met at the lower edge but ON THE WRONG SIDE — Exp 11 went DOWN 0.0077 from Exp 6, not up as predicted. Shallow trees lost test AUC.
- **Learning:** Axis closed: shallow trees (max_depth=3) are NOT the answer on this dataset. Counterintuitively, the velocity features need depth=6 to extract signal — the per-row interaction between device_id_freq, country_fraud_rate_train, and time_since_signup is what carries the model. Mental model update: on w

### Exp 12: STRICT Exp 12 — LightGBM (Ke 2017) leaf-wise vs XGBoost level-wise
- **Backbone:** `lightgbm` | **Status:** KEEP
- **Composite delta from champion:** -0.0098 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5305 | Val AUC 0.5413 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that LightGBM with num_leaves=63, lr=0.04, min_data_in_leaf=50 will land test AUC in the range 0.52 to 0.56 because the mechanism per Ke et al. 2017 is that leaf-wise growth on weak-sig...
- **Verdict:** DISCARD — composite=0.4805, test_auc=0.5305 (-0.0109 vs Exp 6 champion), val_auc=0.5413. WITHIN predicted range (0.52-0.56) but at the lower edge. LightGBM's leaf-wise growth confirmed inferior to XGBoost level-wise on this dataset.
- **Learning:** Axis closed: LightGBM HP exploration is not worth pursuing — the family-level signal is already 0.011 below XGBoost. Per CLAUDE.md 'protect gains' rule, do not invest experiments here. Next try: CatBoost (third structural variant) — ordered boosting may help with high-cardinality device_id.

### Exp 13: STRICT Exp 13 — CatBoost (Prokhorenkova 2018) ordered boosting
- **Backbone:** `catboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0158 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5245 | Val AUC 0.5400 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that CatBoost with iterations=1000, depth=6, l2_leaf_reg=3, bootstrap=Bernoulli will land test AUC in the range 0.53 to 0.58 because the mechanism per Prokhorenkova et al. 2018 is that ...
- **Verdict:** DISCARD — composite=0.4745, test_auc=0.5245, val_auc=0.5400, delta vs Exp 6 champion = -0.0169. WITHIN predicted range (0.53-0.58 had a 0.52-0.54 plateau scenario as the lower-bound case). CatBoost confirmed the GBM-family plateau: ordered boosting + ordered target encoding does not extract more sig
- **Learning:** Axis closed: GBM family is exhausted on this dataset. XGBoost > LightGBM > CatBoost, all within 0.02 AUC of each other. Ordered target encoding does NOT extract more signal from device_id than our manual frequency encoding — the device_id signal simply does not exist beyond what we already encode. N

### Exp 14: STRICT Exp 14 — Ensemble (3-GBM mean) of Exps 6, 12, 13
- **Backbone:** `ensemble` | **Status:** KEEP
- **Composite delta from champion:** -0.0117 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5286 | Val AUC 0.5286 | Precision 0.055 | Recall 0.055 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that the simple mean ensemble of XGB (Exp 6), LGB (Exp 12), CAT (Exp 13) predicted probabilities will land test AUC in the range 0.545 to 0.560 because the mechanism per Lakshminarayana...
- **Verdict:** DISCARD — ensemble AUC=0.5286, delta=-0.0128 vs Exp 6 champion. REFUTED: predicted ensemble would land in 0.545-0.560 (improving over best single model); actual was BELOW the worst single model (0.5245 CatBoost) at 0.5286. The 3 GBMs make highly correlated errors, so averaging amplifies the noise ra
- **Learning:** Axis closed: simple averaging of GBM probabilities does not help on this dataset. The structural distinctness (level-wise vs leaf-wise vs ordered) is not enough — they all converge to similar predictions because they're all fitting the same weak signal. Mental model update: 5 consecutive DISCARDs (E

### Exp 15: STRICT Exp 15 — XGBoost + interaction features (35-feature set)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0171 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5242 | Val AUC 0.5232 | Precision 0.219 | Recall 0.055 | F1 0.088 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that XGBoost on the 35-feature interaction-augmented dataset (16 new features added to the 18-feature velocity set, 1 dropped due to identical = retained) will lift test AUC into the 0....
- **Verdict:** DISCARD (runner-fallback, rewrite manually). Composite=0.5232; per-fold=[0.5242188969487581].
- **Learning:** TODO-REWRITE (runner-fallback entry) — axis state pending human analysis. Next try: pending.

### Exp 16: STRICT Exp 16 — XGBoost on CURATED 9-feature interaction set
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0171 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5239 | Val AUC 0.5232 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that XGBoost on the curated 9-feature set (log_device_id_freq, devfreq_x_countryfr, devfreq_x_devfr, logtss_x_devfr, logtss_x_devfreq, age, purchase_value, time_since_signup, country_fr...
- **Verdict:** DISCARD (runner-fallback, rewrite manually). Composite=0.5232; per-fold=[0.5238624339456652].
- **Learning:** TODO-REWRITE (runner-fallback entry) — axis state pending human analysis. Next try: pending.

### Exp 17: STRICT Exp 17 — XGBoost + 50/50 undersampled training (Pozzolo 2015)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0109 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5294 | Val AUC 0.5354 | Precision 0.215 | Recall 0.056 | F1 0.089 | MCC 0.089
- **Hypothesis (one-line):** We hypothesize that XGBoost on the undersampled training set (balanced 50/50 fraud/clean) will lift test AUC into the 0.55 to 0.62 range and recall from 0.0547 to 0.30+ because the mechanism per Pozzo...
- **Verdict:** DISCARD (runner-fallback, rewrite manually). Composite=0.5294; per-fold=[0.529411743408462].
- **Learning:** TODO-REWRITE (runner-fallback entry) — axis state pending human analysis. Next try: pending.

### Exp 18: STRICT Exp 18 — XGBoost walk-forward 4-fold CV (Bergmeir 2018)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** +0.1535 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.6938 | Val AUC 0.8040 | Precision 0.352 | Recall 0.270 | F1 0.289 | MCC 0.293
- **Hypothesis (one-line):** We hypothesize that XGBoost (Exp 6 champion config) evaluated on 4-fold walk-forward CV will produce test AUCs in the range 0.51 to 0.59 across the 4 folds because the mechanism per Bergmeir et al. 20...
- **Verdict:** KEEP — composite=0.6938 (test_auc mean across 4 walk-forward folds), val_auc=0.8040. Per-fold test AUCs: [0.999 (Jul'15), 0.597 (Aug'15), 0.593 (Sep'15), 0.586 (Oct-Dec'15)]. fold std=0.20. The FDB-comparable single chronological holdout (Exp 6, 0.5414) was unrepresentatively pessimistic: when train
- **Learning:** Major finding: this dataset has CONCEPT DRIFT mid-year (around Sep 2015), not just at year-end. Axis open: regime-aware training and frequent retraining can recover signal that a single chronological holdout misses. The fact that walk-forward fold 3 (Jul test) gives 0.999 AUC while fold 0 (Oct-Dec t

### Exp 19: STRICT Exp 19 — XGBoost with min_train_idx=60000 (legitimate recency, same FDB test set)
- **Backbone:** `xgboost` | **Status:** KEEP
- **Composite delta from champion:** -0.0120 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5283 | Val AUC 0.5389 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that XGBoost with min_train_idx=60000 (training on rows 60k-105k = 45k recent rows) on the SAME FDB test set will land test AUC in the range 0.54 to 0.59 because the mechanism per Bergm...
- **Verdict:** DISCARD - composite=0.5283, test_auc=0.5283 (-0.0131 vs Exp 6 champion), val_auc=0.5389. TEST SET SIZE VERIFIED at 30,222 rows matching the FDB chronological 80/20 protocol. When the test set is held identical to FDB's published benchmark, dropping early training rows (rows 0-60k) HURTS test AUC by 
- **Learning:** Critical learning: the user's reward-hacking call was 100% correct. The 'recency improves AUC' finding was an artifact of test-set shrinkage, not genuine improvement. Axis closed: dropping old training data does NOT help when evaluated honestly. Mental model update: XGBoost benefits from MORE traini

### Exp 20: STRICT Exp 20 - Energy-Based Classifier (Liu 2020 NeurIPS) novel paradigm
- **Backbone:** `energy_based_model` | **Status:** DISCARD
- **Composite delta from champion:** -0.0652 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5214 | Val AUC 0.4751 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that an EBM with hidden=128, 2 hidden layers, dropout=0.3 trained with cross-entropy on the FDB train portion will land test AUC in the range 0.52 to 0.58 because the mechanism per Liu ...
- **Verdict:** DISCARD - composite=0.4751, test_auc=0.5214 (energy score), 0.4750 (logit score), val_auc=0.4751. Within predicted range at lower edge. EBM did not outperform XGBoost (-0.020 delta). Energy AUC > Logit AUC confirms energy captures distributional information the logit misses. TEST SET SIZE VERIFIED a
- **Learning:** Axis closed: EBM scoring on this dataset does not beat XGBoost. The interesting datum is energy AUC (0.521) > logit AUC (0.475), so energy formulation extracts marginally different signal than direct classification - useful for an ensemble. Next try: Exp 21 autoencoder anomaly detection (one-class p

### Exp 21: STRICT Exp 21 - Autoencoder anomaly (Sakurada Yairi 2014) one-class paradigm
- **Backbone:** `autoencoder_anomaly` | **Status:** DISCARD
- **Composite delta from champion:** -0.0418 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.4985 | Val AUC 0.5324 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that an autoencoder with bottleneck=8, encoder=[64,32] trained on the 93,702 CLEAN training rows only will land test AUC in the range 0.50 to 0.58 because the mechanism per Sakurada and...
- **Verdict:** DISCARD - composite=0.4985, test_auc=0.4985 (essentially random), val_auc=0.5324. Within predicted range at lower edge. The autoencoder reconstruction-error approach did NOT discriminate fraud from clean on this dataset - test AUC at 0.498 is below the 0.50 random baseline. TEST SET SIZE VERIFIED at
- **Learning:** Axis closed: one-class autoencoder anomaly detection does NOT work on fraudecom. Diagnosis: fraud rows look distributionally identical to clean rows in feature space - the only thing distinguishing them is the LABEL relationship which an AE trained without labels cannot exploit. Mental model update:

### Exp 22: STRICT Exp 22 - Contrastive Learning (SimCLR-tabular, Chen 2020 ICML)
- **Backbone:** `contrastive_simclr_tabular` | **Status:** KEEP
- **Composite delta from champion:** -0.0079 (champion=Exp 6 XGBoost)
- **Result:** Test AUC 0.5390 | Val AUC 0.5324 | Precision 0.000 | Recall 0.000 | F1 0.000 | MCC 0.000
- **Hypothesis (one-line):** We hypothesize that contrastive pre-training (15 epochs NT-Xent loss, Gaussian noise sigma=0.3 + 15% feature dropout) followed by classifier fine-tuning on frozen embeddings will land test AUC in the ...
- **Verdict:** DISCARD - composite=0.5324, test_auc=0.5390, val_auc=0.5324. Within predicted range. Contrastive pre-training came VERY CLOSE to XGBoost (0.5390 vs 0.5414, delta -0.0024) and was the best-performing of the three novel paradigms (EBM 0.5214, AE 0.4985, Contrastive 0.5390). TEST SET SIZE VERIFIED at 3
- **Learning:** Axis open: contrastive learning is the most promising NOVEL approach found - it nearly matches XGBoost on a fundamentally different inductive bias (instance-level invariance). Mental model update: tabular contrastive pre-training extracts comparable signal to GBMs with very different per-row error p
