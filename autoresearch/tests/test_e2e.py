"""
End-to-end smoke test: downloads data, computes features, splits,
creates model, does a forward pass, evaluates Sharpe.
Does NOT call Claude API or run full training.
"""
import pytest
import torch
import numpy as np
from autoresearch.data.download import download_pair
from autoresearch.data.features import compute_single_pair_features, compute_targets
from autoresearch.data.splits import FOLDS, split_data
from autoresearch.model.backbone import CurrencyLFM
from autoresearch.model.train import create_dataset, SEQ_LEN
from autoresearch.evaluation.metrics import sharpe_ratio


@pytest.mark.slow
def test_e2e_single_fold():
    # Download EUR/USD (recent data only for speed)
    df = download_pair("EURUSD=X", start="2020-01-01", end="2024-12-31")
    features = compute_single_pair_features(df, prefix="EURUSD")

    # Drop warmup NaNs
    features = features.iloc[63:].dropna()
    targets = compute_targets(df)

    # Align
    common = features.index.intersection(targets.index)
    features = features.loc[common]
    targets = targets.loc[common]

    assert len(features) > SEQ_LEN * 2, f"Not enough data: {len(features)} rows"
    assert features.isna().sum().sum() == 0, "Features contain NaN"

    # Use fold 7 (most recent data matches our range)
    fold = FOLDS[6]
    train_feat, val_feat, test_feat = split_data(features, fold)
    train_tgt, val_tgt, test_tgt = split_data(targets, fold)

    assert len(train_feat) > SEQ_LEN, f"Train too small: {len(train_feat)}"

    # Create model
    model = CurrencyLFM(n_input_features=features.shape[1], freeze_backbone=True)

    # Create dataset
    train_ds = create_dataset(train_feat, train_tgt, seq_len=SEQ_LEN)
    assert len(train_ds) > 0, "Empty training dataset"

    # Forward pass
    x, y = train_ds[0]
    model.eval()
    with torch.no_grad():
        preds = model(x.unsqueeze(0))

    assert "ret_1d" in preds
    assert "ret_5d" in preds
    assert preds["ret_1d"].shape[1] == 6  # 6 pairs
    assert "ret_1d_log_var" in preds, "Missing aleatoric uncertainty output"
    assert preds["ret_1d_log_var"].shape[1] == 6

    # Sharpe on random returns (verify metric works end-to-end)
    fake_returns = np.random.randn(100) * 0.01
    s = sharpe_ratio(fake_returns)
    assert isinstance(s, float)
    print(f"E2E smoke test passed: {features.shape[1]} features, "
          f"train={len(train_feat)}, model output shape={preds['ret_1d'].shape}")
