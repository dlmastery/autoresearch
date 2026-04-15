# LFM2.5-350M Hyperparameter Research

## Sources

- [LFM2 Technical Report (arXiv:2511.23404)](https://arxiv.org/pdf/2511.23404)
- [Unsloth LFM2.5 Fine-tuning Guide](https://unsloth.ai/docs/models/tutorials/lfm2.5)
- [Houlsby et al., 2019 — Parameter-Efficient Transfer Learning](https://arxiv.org/abs/1902.00751)
- [How to jointly tune lr and weight decay for AdamW (Schaipp, 2024)](https://fabian-sp.github.io/posts/2024/02/decoupling/)
- [PatchTST Official Repo](https://github.com/yuqinie98/PatchTST)
- [Linear Decay-to-Zero Schedule (2025)](https://arxiv.org/html/2510.19093v1)
- [Fine-Tuning Pre-Trained Models the Right Way (Swain, 2026)](https://lalatenduswain.medium.com/fine-tuning-pre-trained-models-the-right-way-a-step-by-step-guide-to-learning-rate-strategy-b3d9c0307222)

## Frozen Backbone + Learned Adapter (Our Setup)

Our LFM2.5-350M setup: 350M frozen params, ~636K trainable (projection linear + prediction heads).
This is a **linear probe / adapter** setup, NOT full fine-tuning.

### Learning Rate

| Source | Recommendation | Context |
|--------|---------------|---------|
| Adapter literature (Houlsby 2019) | 1e-4 to 3e-4 | Adapters with ~2-5% trainable params |
| Frozen backbone best practice | 2e-5 to 1e-4 | Linear probes on frozen features |
| LLM fine-tuning consensus (2025) | 2e-5 for full, 1e-4 for LoRA/adapter | Larger models need lower lr |
| Our empirical (pre-fix, contaminated) | 1e-4 worked, gave Sharpe=1.19 | 5 epochs, batch=64 |
| Our empirical (clean data, exp 1) | 1e-4 gave Sharpe=-0.31 | 20 epochs, batch=32, super-fold eval |

**Key insight:** Our adapter is VERY thin (636K params / 350M total = 0.18%). This is even smaller than typical adapters (2-5%). Literature suggests LOWER lr for thinner adapters because each parameter update has outsized impact on the output.

**Recommended range:** 1e-5 to 1e-4 (not higher)

### Batch Size

| Source | Recommendation | Context |
|--------|---------------|---------|
| General consensus | 32-128 for fine-tuning | Depends on dataset size |
| Small dataset (<5000 samples) | 16-32 | More gradient noise = implicit regularization |
| lr-batch coupling rule | lr scales by sqrt(batch_ratio) | If doubling batch, multiply lr by 1.41x |
| Our training set | 2340 samples (contiguous) | Very small — favors smaller batch |

### Weight Decay

| Source | Recommendation | Context |
|--------|---------------|---------|
| AdamW paper (Loshchilov 2019) | 0.01 typical | For full training |
| Adapter fine-tuning | 1e-5 to 1e-3 | Less weight decay for fewer params |
| Diagnostic rule | Increase if val > train (overfitting) | Decrease if both plateau early |

### Epochs / Early Stopping

| Source | Recommendation | Context |
|--------|---------------|---------|
| Original run (contaminated) | 5 epochs was sufficient | Converged quickly with 1e-4 lr |
| Clean data run | 20 epochs, patience=5 | Val loss plateaus around epoch 16 |
| Adapter literature | 10-30 epochs typical | With early stopping |

**20 epochs with patience 5 is sufficient.** Val loss plateau observed empirically at epoch 16.

### Sequence Length (LFM2-specific)

| Source | Recommendation | Context |
|--------|---------------|---------|
| LFM2 architecture | SSM-based, handles long sequences | Pre-trained on long context |
| FX microstructure | Autocorrelation decays in 5-10 days | Short-term signal |
| Our default | 60 days (~3 months) | May be too long for 1d prediction |
| Trade-off | Longer = fewer training samples | 2340 samples at seq=60 vs ~3000 at seq=20 |

**Hypothesis:** seq=60 may dilute recent signal with stale noise. Worth testing seq=20-40 with the research-backed lr.
