"""Tests for model.train – FXDataset and training loop helpers."""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from autoresearch.model.train import (
    BATCH_SIZE,
    EPOCHS,
    FXDataset,
    LEARNING_RATE,
    SEQ_LEN,
    create_dataset,
)


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class _MockModel(nn.Module):
    """Minimal model that satisfies the forward(x)->dict interface."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 2)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (batch, seq_len, n_features) -> take last timestep
        out = self.linear(x[:, -1, :])  # (batch, 2)
        return {"ret_1d": out[:, 0:1], "ret_5d": out[:, 1:2]}


# ---------------------------------------------------------------------------
# Tests – Dataset creation
# ---------------------------------------------------------------------------


class TestDatasetCreation:
    """test_dataset_creation: basic shape and indexing checks."""

    def test_length(self):
        n_rows, n_feat, seq_len = 100, 10, 30
        features = np.random.randn(n_rows, n_feat)
        targets = np.random.randn(n_rows, 2)
        ds = FXDataset(features, targets, seq_len)
        assert len(ds) == n_rows - seq_len  # 100 - 30 = 70

    def test_shapes(self):
        n_rows, n_feat, seq_len = 100, 10, 30
        features = np.random.randn(n_rows, n_feat)
        targets = np.random.randn(n_rows, 2)
        ds = FXDataset(features, targets, seq_len)
        x, y = ds[0]
        assert x.shape == (seq_len, n_feat)  # (30, 10)
        assert y.shape == (2,)

    def test_tensor_dtypes(self):
        features = np.random.randn(50, 5)
        targets = np.random.randn(50, 2)
        ds = FXDataset(features, targets, seq_len=10)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests – No future leak
# ---------------------------------------------------------------------------


class TestDatasetNoFutureLeak:
    """test_dataset_no_future_leak: verify window alignment."""

    def test_feature_window_alignment(self):
        """ds[5] must contain feature rows [5:35], with last row == row 34."""
        n_rows, n_feat, seq_len = 100, 10, 30
        # Sequential integer features so each row is identifiable
        features = np.arange(n_rows * n_feat).reshape(n_rows, n_feat).astype(float)
        targets = np.random.randn(n_rows, 2)
        ds = FXDataset(features, targets, seq_len)

        x, y = ds[5]

        # x should be features[5:35]
        expected_x = features[5:35]
        np.testing.assert_array_equal(x.numpy(), expected_x)

        # Last row of window should be row 34's values
        np.testing.assert_array_equal(x[-1].numpy(), features[34])

    def test_target_alignment(self):
        """Target for ds[idx] should be targets[idx + seq_len - 1]."""
        n_rows, n_feat, seq_len = 100, 10, 30
        features = np.random.randn(n_rows, n_feat)
        targets = np.arange(n_rows * 2).reshape(n_rows, 2).astype(float)
        ds = FXDataset(features, targets, seq_len)

        _, y = ds[5]
        expected_y = targets[5 + seq_len - 1]  # targets[34]
        np.testing.assert_array_equal(y.numpy(), expected_y)


# ---------------------------------------------------------------------------
# Tests – create_dataset helper
# ---------------------------------------------------------------------------


class TestCreateDataset:
    """Verify create_dataset aligns on common index."""

    def test_common_index_alignment(self):
        idx_a = pd.date_range("2020-01-01", periods=50, freq="B")
        idx_b = pd.date_range("2020-01-15", periods=50, freq="B")

        features = pd.DataFrame(np.random.randn(50, 5), index=idx_a)
        targets = pd.DataFrame(np.random.randn(50, 2), index=idx_b)

        ds = create_dataset(features, targets, seq_len=10)
        common = idx_a.intersection(idx_b)
        assert len(ds) == len(common) - 10


# ---------------------------------------------------------------------------
# Tests – Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_seq_len(self):
        assert SEQ_LEN == 60  # global default (LFM2.5); non-LFM backbones use 10 via get_seq_len()

    def test_batch_size(self):
        assert BATCH_SIZE == 8

    def test_learning_rate(self):
        assert LEARNING_RATE == pytest.approx(1e-4)

    def test_epochs(self):
        assert EPOCHS == 5
