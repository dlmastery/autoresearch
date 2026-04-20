---
name: Autoresearch Checkpoint
description: 174 exps. GLOBAL CHAMP LSTM Exp35 +6.4242. PATCHTST PHASE STARTED (1/50). Mamba 22/50 DONE, champion dmamba e=4 +5.5996.
type: project
---

## PHASE ROLLING SUMMARY

| Backbone | Phase | Exps | Status | Family Champion |
|----------|-------|------|--------|-----------------|
| mlp | DONE | 54 | ✓ | residual MLP Exp32 +5.499 |
| lfm2-350m | SKIPPED | 43 | ✗ (user 2026-04-19) | Exp20 +1.77 |
| lstm | DONE | 46/50 | ✓ (user halt) | Exp35 wd=7e-4 bs=16 **+6.4242 GLOBAL CHAMP** |
| mamba | DONE | 22/50 | ✓ (user halt 2026-04-20) | dmamba e=4 +5.5996 |
| **patchtst** | **IN PROGRESS** | **1/50** | **running** | TBD |
| patchtsmixer | queued | 0/50 | pending | — |
| xgboost | queued | 0/50 | pending | — |
| lightgbm | queued | 0/50 | pending | — |
| catboost | queued | 0/50 | pending | — |
| ensemble (phase b) | queued | — | AFTER all above | seed-ensemble code + cross-backbone |

## 🏆 GLOBAL CHAMPION (unchanged)
**LSTM Exp35 (wd=7e-4 bs=16 seed=42)** composite **+6.4242** | test Sharpe +6.5242 | val Sharpe +7.1539 | 7/7 positive test folds | +1122% return
Archived: `winners/lstm_exp35_wd7e4_bs16_seed42/`

## Current Experiment: PatchTST Exp1/50 (JSONL 174)
Running in background (ID bjtiunacg) with SOTA recipe per Nie 2023 ICLR (arXiv:2211.14730):
- seq_len=60 (fixes the earlier Exp117 violation at seq=10 → −1.72)
- ep=100, pat=20, lr=1e-4, bs=32, wd=1e-4, warmup=10, hd=0.15, huber=1.0, seed=42

Pre-authored reasoning annotation passes all Citation Rigor + Completeness gates.

## User Plan (2026-04-20)
1. Finish (c): PatchTST 50 → PatchTSMixer 50 → XGBoost 50 → LightGBM 50 → CatBoost 50
2. Then (b): build seed-ensemble code, re-run all champion configs under ensemble
3. User confirmed this order: "c seems best ... then after all the things are over then we move to b"

## Checkpoint Discipline
- Every experiment: update JSONL (auto), trade logs (auto), reasoning_annotations (auto + Claude post-analysis)
- Every experiment: pre-author reasoning entry BEFORE launch (blocks on missing entry)
- Every 10 min: commit + push
- No orphan TODO-REWRITE entries

## Mamba Phase Summary (archived, 22/50)
Architectural axes ALL closed at Mamba paper defaults (Gu & Dao 2024):
variant=dmamba (Liu 2025 arXiv:2602.09081), d_state=16, expand=4, nl=2,
bs=32, lr=5e-4, wd=0.1, hd=0.1, warmup=10. Best fold-2 across all
backbones: Mamba Exp7 test fold 2 **+3.76** (vs LSTM +0.40 — 9x lift on
hardest regime — strong ensemble candidate per (phase b)).

7-seed variance on champion: mean +4.45, std +0.89, range 2.16,
median +4.39. Champion seed=42 is +1.4σ above mean (lucky high).
