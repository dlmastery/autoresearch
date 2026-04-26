# Forensic Checkpoint — FDB fraudecom

_Audit-grade snapshot of the project state. Updated on every major commit. Acts as the single source of truth for the project's current claim, the test-set integrity attestation, and the reproducibility instructions._

---

## Current claim (as of last update)

- **Champion:** Exp 25 — XGBoost FDB-exact 80/20 with engineered velocity and target-encoded entity features.
- **Test AUC-ROC:** 0.6097 on the FDB-protocol 30,222-row test set.
- **vs FDB published baselines:**
  - AFD-TFI (proprietary): -0.026
  - AutoGluon: +0.088
  - H2O: +0.092
  - Auto-sklearn: +0.095

---

## Test-set integrity attestation

- **Test set rows:** indices 120,890 through 151,111 of the dataset sorted ascending by `purchase_time`.
- **Test set size:** 30,222 rows. ✅ matches FDB documented exactly.
- **Test fraud rate:** 4.60% (1,389 of 30,222 positive).
- **SHA-256 hash on sorted test indices (first 16 hex):** `cba9f0e8d8b7a4c2`. ✅ locked.
- **Verification:** every experiment in the live log has `per_fold_test_reports[0].n == 30222` (except the quarantined stratified-CV and the walk-forward CV, both quarantined or differently labeled).

---

## Quarantined experiments

| Folder | Count | Reason |
|--------|-------|--------|
| `_quarantined_exp1/` | 1 | Stratified CV → AUC inflated by 0.27 vs honest chronological |
| `_quarantined_blind_sweep/` | 35 | Blind hyperparameter grid sweep (no per-experiment diagnosis) |
| `_quarantined_reward_hack/` | 5 | Test set size changed (11k vs FDB-protocol 30,222) |

Each quarantine has a `WHY_QUARANTINED.md` documenting the violation.

---

## Reproducibility ritual

```bash
git clone https://github.com/dlmastery/autoresearch.git
cd autoresearch
git checkout master
pip install -e .
pip install xgboost==3.2.0 lightgbm==4.6.0 catboost==1.2.10 interpret==0.7.8

python generalized_ml_autoresearch/examples/fraud_ecommerce/prepare_data.py
python generalized_ml_autoresearch/examples/fraud_ecommerce/add_velocity_features.py
python generalized_ml_autoresearch/examples/fraud_ecommerce/run_exp25_fdb_exact.py
```

Expected output: `composite=0.6097` ± 0.005 (5-seed std=0.006).

---

## Framework changes encoded as Hard Rules

(Cross-references to `templates/CLAUDE_template.md`)

1. Reward Hacking Prohibition — never change the test set; verify SHA-256 hash on sorted test indices.
2. Velocity-feature train-period alignment — use `n_train = n - n_val - n_test`.
3. Holistic Data Scientist Mindset — no ceiling declaration without 5+ experiments per axis, 3 architectures, 5 feature-engineering directions, 2 protocols, 1 calibration.
4. Wiring verification — extreme-value A/B test for every new config field.
5. Stratified-CV ban for time-ordered data.
6. Composite-floor rationale — set to "must beat random" not "wishful target."

---

## Audit signature

| Item | Status |
|------|--------|
| Data integrity (no duplicates, no missing) | ✅ PASS |
| Test set hash locked | ✅ PASS |
| Class balance per split + chi-square | ✅ DOCUMENTED (drift expected) |
| KS distribution shift per feature | ✅ DOCUMENTED |
| Multicollinearity audit | ✅ PASS (no |r| > 0.85) |
| Target leakage detection (MI shuffled = 0) | ✅ PASS |
| Reproducibility (same seed → byte-identical) | ✅ PASS |
| Multi-seed variance (std < 0.02) | ✅ PASS (0.006) |
| Permutation feature importance with bootstrap CI | ✅ DONE |
| Calibration (ECE, Brier) | ✅ MEASURED (ECE=0.029) |
| Strict-FDB feature compliance ablation | ✅ DONE (Exp 23) |
| FDB-exact 80/20 protocol verification | ✅ DONE (Exp 25) |

---

_Last updated: 2026-04-25. See `autoresearch_results/forensic_report.md` for the full forensic write-up._
