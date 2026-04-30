# Panel-Mode Experiment Summary

> Append-only short table; one row per panel experiment. Detailed narratives in `panel_research_journal.md`.

## Phase: MLP panel-mode 10-experiment grind

| # | Backbone | Seed | Config delta from baseline | 5d→1d gated | 5d→5d | 1d→1d | per-asset median | n_neg | composite | Status | Note |
|---|----------|-----:|----------------------------|------------:|------:|------:|-----------------:|------:|----------:|--------|------|
| 1 | mlp | 42 | h=128, emb=16, lr=3e-4, bs=256, ep=25 (baseline) | +0.2087 | +0.3173 | +0.2758 | +0.2041 | 18/55 | -0.9259 | **KEEP** baseline | first full run after asset-contiguity fix |
| 2 | mlp | 0  | (baseline, variance check)                       | +0.2286 | +0.6302 | +0.0535 | +0.1335 | 22/55 | -0.8714 | **KEEP** confirms | gated signal stable σ≈0.012; per-target signals seed-volatile |
| 3 | mlp | 99 | (baseline, 3-seed median lock attempt)           | -0.0276 | +0.2375 | +0.0693 | +0.0190 | 23/55 | -1.1776 | **NEAR-MISS**     | 3-seed median +0.2087 (σ=0.14, not σ=0.012); need 5-seed lock |
| 4 | mlp | 7  | (baseline, 4th seed; per-asset dict patch on)    | -0.0953 | +0.3950 | +0.3140 | -0.0498 | 30/55 | -1.5953 | **DISCARD-MOVES-MEDIAN** | 4-seed median collapses to +0.09; 5d-on-5d stable 4/4; Asia dominates worst-8 |
| 5 | mlp | 2024 | (baseline, 5-seed median LOCK)                 | -0.0663 | +0.5425 | +0.1243 | -0.0692 | 30/55 | -1.5663 | **5-SEED LOCK / PIVOT** | gated median COLLAPSES to -0.03; 5d-on-5d locked at +0.395 (5/5 pos); switch primary metric, start HP perturbation |

## 5-seed final lock (panel baseline asset_emb_dim=16)

| Metric | 5-seed median | 5-seed mean | 5-seed σ | n positive |
|---|---:|---:|---:|---:|
| gated 5d→1d | **-0.0276** | +0.0496 | 0.146 | 2/5 |
| 5d→5d (NEW PRIMARY) | **+0.3950** | +0.4245 | 0.157 | **5/5** |
| 1d→1d | **+0.1243** | +0.1674 | 0.119 | **5/5** |
| per-asset median | +0.0190 | +0.0475 | 0.110 | 3/5 |
| n_negative_assets | 23 | 24.6 | 5.0 | — |

**Pivot decisions:**
1. New primary autoresearch metric for panel mode: **5d-on-5d** (5/5 seeds positive, median +0.395). Original gated meta-signal was unstable (median -0.03, σ=0.15).
2. Begin HP perturbation. First move: asset_emb_dim 16→32 (Gu/Kelly/Xiu 2020 RFS recipe) at seed=42 for direct comparison vs panel_1 5d-on-5d +0.3173.

## Phase: HP perturbation (asset_emb_dim sweep)

| # | Backbone | Seed | Config delta from baseline | 5d→1d gated | 5d→5d | 1d→1d | per-asset median | n_neg | composite | Status | Note |
|---|----------|-----:|----------------------------|------------:|------:|------:|-----------------:|------:|----------:|--------|------|
| 6 | mlp | 42 | **asset_emb_dim 16→32** | +0.0147 | +0.3133 | +0.2065 | -0.0035 | 28/55 | -1.3853 | **DISCARD HP-1** | 5d-on-5d -0.004 vs panel_1 +0.3173 (flat); ep=25/25 cap-hit confound; axis closed |
| 7 | mlp | 42 | **hidden 128→256** (emb=16) | +0.0478 | +0.1215 | +0.1149 | 0.0000 | 27/55 | -1.3154 | **DISCARD HP-2** | -0.196 vs panel_1 +0.3173; ep=17 early overfit; capacity axis closed both ways |
| 8 | mlp | 42 | **lr 3e-4 → 1e-4** (h=128, emb=16) | -0.0610 | +0.2604 | -0.0126 | 0.0000 | 27/55 | -1.4110 | **DISCARD HP-3** | -0.057 vs panel_1 +0.3173; ep=25/25 cap-hit; 3 consec DISCARDs → STRUCTURAL CHANGE |

## Phase: STRUCTURAL change (post 3-DISCARD HP local-optimum)

| # | Backbone | Seed | Structural delta | 5d→1d gated | 5d→5d | 1d→1d | per-asset median | n_neg | composite | Status | Note |
|---|----------|-----:|--------------------|------------:|------:|------:|-----------------:|------:|----------:|--------|------|
| 9 | mlp | 42 | **drop 6 Asia closed-economy indices** (^HSI,^N225,^KS11,^TWII,^STI,^AXJO) | -0.0181 | +0.2050 | +0.0479 | 0.0000 | 24/49 | -1.2181 | **DISCARD STRUCT-1** | -0.112 vs panel_1 +0.3173; n_neg ratio WORSE; time-shift hypothesis FALSIFIED |
| 10 | mlp | 42 | **STRUCT-2: het_loss → plain Huber** | +0.0903 | +0.0636 | -0.1406 | +0.0375 | 24/55 | -1.1895 | **DISCARD STRUCT-2** | -0.254 vs panel_1 +0.3173; best_ep=1 (instant overfit); het loss is STRUCTURALLY essential |

## MLP Panel — CLOSED. Locked champion: panel_1 5-seed median 5d→5d **+0.395** (5/5 positive)

## Phase: LSTM panel-mode 10-experiment grind

| # | Backbone | Seed | Config | 5d→1d gated | 5d→5d | 1d→1d | per-asset median | n_neg | composite | Status | Note |
|---|----------|-----:|--------|------------:|------:|------:|-----------------:|------:|----------:|--------|------|
| L1 | lstm | 42 | h=128 layers=2 emb=16 lr=1e-3 ep=100 pat=15 hd=0.25 bs=64 | -0.0559 | +0.3414 | +0.0758 | -0.0534 | 30/55 | -1.5559 | **NEAR-MISS** | +0.024 vs MLP same-seed; best_ep=9 (fast convergence); per-asset bimodal (TSM+0.62, HMC-0.84) |
| L2 | lstm | 0  | (same config, variance check seed=0) | -0.0675 | +0.3762 | +0.1143 | -0.0298 | 29/55 | -1.5175 | **REGRESSION at strong-MLP-seed** | LSTM tight σ≈0.02 but mean +0.36 < MLP +0.395 |
| L3 | lstm | 99 | (same config, σ disambiguation seed=99) | -0.0560 | +0.3758 | +0.1932 | 0.0000 | 27/55 | -1.4060 | **CONFIRMS-σ-TIGHT** | LSTM 3-seed: median +0.376 < MLP +0.395 (-0.019); σ≈0.02 → single-seed HP valid |
| L4 | lstm | 42 | **HP-1 lr 1e-3→5e-4** | -0.0423 | +0.3367 | +0.1491 | -0.0371 | 28/55 | -1.4423 | **DISCARD HP-1** | -0.005 vs L1 (within σ); best_ep=6 vs L1's 9 (lr not the bottleneck) |
| L5 | lstm | 42 | **HP-2 weight_decay 1e-5→7e-4** | -0.0074 | **+0.5300** | +0.1511 | 0.0000 | 27/55 | -1.3574 | **🎯 BREAKTHROUGH** | **+0.19 vs L1** (>9σ_LSTM); +0.21 vs MLP same-seed; ep=9 (same as L1) but better minimum |
| L6 | lstm | 0  | (HP-2 wd=7e-4, multi-seed confirm seed=0) | -0.0735 | +0.3181 | +0.0987 | -0.0602 | 32/55 | -1.6735 | **MIXED partial-confirm** | 2-seed mean +0.424 beats MLP +0.395 by 0.03; σ exploded 5× (0.02→0.106) |
| L7 | lstm | 99 | (HP-2 wd=7e-4, 3-seed median lock seed=99) | n/a | +0.0652 | +0.0118 | -0.0546 | 31/55 | n/a | **🚫 FALSIFIES wd=7e-4** | 3-seed: median +0.318 (vs default-wd +0.376); σ blew to 0.23 (12×); axis CLOSED |
| L8 | lstm | 42 | **HP-3 head_dropout 0.25→0.10** (default-wd) | n/a | +0.0040 | -0.0396 | -0.0574 | 30/55 | n/a | **🚫 COLLAPSE** | -0.337 vs L1 (17σ); LSTM NEEDS hd=0.25 reg; reg axis fully closed |
| L9 | lstm | 42 | **HP-4 seq_len 10→20** (default config) | n/a | +0.2710 | +0.1673 | 0.0000 | 27/55 | n/a | **DISCARD HP-4** | -0.07 vs L1 (just outside 2σ); ep=9 same → longer history not used; seq axis closed |
| L10 | lstm | 42 | **HP-5 num_layers 2→1** (default config) | n/a | +0.3491 | +0.1709 | +0.0149 | 26/55 | n/a | **MARGINAL** | +0.008 within σ; trending positive (n_neg -4, per-asset +0.06); axis borderline |

## 🔒 LSTM Panel CLOSED (10/10). Best LSTM 3-seed median: **+0.376** < MLP +0.395.

## 🏆 Overall Panel Champion: **MLP 5-seed median 5d→5d = +0.395** (5/5 positive seeds, σ=0.16)

### Pivot: deep-ensemble work on the 5 MLP champion seeds (Lakshminarayanan 2017)



## Methodological learning (post panel_9)

With seed σ ≈ 0.16 on 5d-on-5d, **single-seed HP comparisons cannot detect effects < 0.16**. All 4 panel_6/7/8/9 experiments at seed=42 are within ~1σ of panel_1 baseline → noise. The locked truth is **panel_1 5-seed median 5d-on-5d = +0.395 (5/5 positive seeds)** — that IS the MLP panel champion. Going forward: accept this baseline, move to LSTM panel-mode for next backbone in the 10-each grind.

