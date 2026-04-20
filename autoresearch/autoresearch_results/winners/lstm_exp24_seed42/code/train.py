"""Training loop for FX prediction models.

Provides FXDataset, dataset creation helpers, and a single-fold training
routine with production-grade best practices:
  - Huber loss (robust to fat-tailed return distributions)
  - Gradient clipping (prevents exploding gradients)
  - Cosine annealing LR schedule
  - Early stopping with patience
  - Best-model checkpointing
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN: int = 60       # global fallback; per-backbone defaults in backbone.get_seq_len()
BATCH_SIZE: int = 32    # smaller batches for noisy FX data (better gradient estimates)
LEARNING_RATE: float = 3e-4  # AdamW sweet spot for fine-tuning (Loshchilov & Hutter, 2019)
EPOCHS: int = 20        # more epochs with early stopping -- let patience decide
GRAD_CLIP: float = 1.0
PATIENCE: int = 5       # 5-epoch patience avoids premature stopping on noisy val loss

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FXDataset(Dataset):
    """Sliding-window dataset over pre-computed feature / target arrays."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_dataset(
    features: pd.DataFrame, targets: pd.DataFrame, seq_len: int
) -> FXDataset:
    """Align *features* and *targets* on their common index and return an
    :class:`FXDataset` backed by numpy arrays."""
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]
    return FXDataset(features.values, targets.values, seq_len)


def find_contiguous_segments(
    index: pd.DatetimeIndex,
    max_gap_days: int = 5,
) -> list[tuple[int, int]]:
    """Return (start, end) integer slices for contiguous date blocks.

    A new segment starts whenever two consecutive index entries are more
    than *max_gap_days* calendar days apart (normal weekends are 2--3 days).
    """
    if len(index) == 0:
        return []
    diffs = pd.Series(index).diff()
    gap_mask = diffs > pd.Timedelta(days=max_gap_days)
    split_positions = gap_mask[gap_mask].index.tolist()

    segments = []
    prev = 0
    for pos in split_positions:
        segments.append((prev, pos))
        prev = pos
    segments.append((prev, len(index)))
    return segments


def create_contiguous_datasets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    seq_len: int,
    max_gap_days: int = 5,
) -> torch.utils.data.ConcatDataset:
    """Create a ConcatDataset from non-contiguous DatetimeIndex data.

    Splits the data at gaps larger than *max_gap_days* calendar days,
    creates a separate FXDataset per contiguous segment (skipping any
    segment shorter than seq_len), and returns the concatenation.

    This avoids sliding windows that span time gaps -- each window
    stays within a single contiguous block of trading days.
    """
    common_idx = features.index.intersection(targets.index)
    features = features.loc[common_idx]
    targets = targets.loc[common_idx]

    if len(features) == 0:
        return torch.utils.data.ConcatDataset([])

    segments = find_contiguous_segments(features.index, max_gap_days)

    datasets = []
    for start, end in segments:
        seg_feat = features.iloc[start:end]
        seg_tgt = targets.iloc[start:end]
        if len(seg_feat) >= seq_len + 1:
            datasets.append(FXDataset(seg_feat.values, seg_tgt.values, seq_len))

    return torch.utils.data.ConcatDataset(datasets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


WEIGHT_DECAY: float = 1e-5

def train_one_fold(
    model,
    train_features: pd.DataFrame,
    train_targets: pd.DataFrame,
    val_features: pd.DataFrame,
    val_targets: pd.DataFrame,
    scaler: StandardScaler | None = None,
    seq_len: int = SEQ_LEN,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_epochs: int = 0,
    huber_delta: float = 1.0,
    patience: int = PATIENCE,
    grad_clip: float = GRAD_CLIP,
) -> dict:
    """Train *model* for one cross-validation fold and return results.

    Uses Huber loss, gradient clipping, cosine LR schedule, and early
    stopping with best-model restoration.
    """
    # 1. Scaler ----------------------------------------------------------
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_features.values)

    train_feat_scaled = pd.DataFrame(
        scaler.transform(train_features.values),
        index=train_features.index,
        columns=train_features.columns,
    )
    val_feat_scaled = pd.DataFrame(
        scaler.transform(val_features.values),
        index=val_features.index,
        columns=val_features.columns,
    )

    # 2. Datasets & loaders ----------------------------------------------
    # Training data may have holes (punched-out val/test windows).
    # Use contiguous segments to avoid sliding windows across gaps.
    train_ds = create_contiguous_datasets(train_feat_scaled, train_targets, seq_len)
    val_ds = create_contiguous_datasets(val_feat_scaled, val_targets, seq_len)

    use_cuda = next(model.parameters()).is_cuda
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=use_cuda, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=0,
    )

    # 3. Optimiser, loss, scheduler --------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    huber_fn = torch.nn.HuberLoss(delta=huber_delta, reduction="none")

    # LR schedule: optional linear warmup then cosine decay.
    # Warmup stabilises randomly-initialised projection/head layers before
    # full-strength gradients hit (Devlin et al. 2019; Hu et al. 2022 LoRA).
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Device ----------------------------------------------------------
    device = next(model.parameters()).device

    # 5. Training loop with early stopping -------------------------------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- train ------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        het_mode = hasattr(model, 'het_loss') and model.het_loss
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            mu_stack = torch.stack(
                [preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1
            )
            if het_mode:
                logvar_stack = torch.stack(
                    [preds["ret_1d_log_var"][:, 0], preds["ret_5d_log_var"][:, 0]], dim=1
                )
                huber_elem = huber_fn(mu_stack, y)
                loss = (torch.exp(-logvar_stack) * huber_elem + 0.5 * logvar_stack).mean()
            else:
                loss = huber_fn(mu_stack, y).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_n, 1)

        # --- val --------------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                mu_stack = torch.stack(
                    [preds["ret_1d"][:, 0], preds["ret_5d"][:, 0]], dim=1
                )
                if het_mode:
                    logvar_stack = torch.stack(
                        [preds["ret_1d_log_var"][:, 0], preds["ret_5d_log_var"][:, 0]], dim=1
                    )
                    huber_elem = huber_fn(mu_stack, y)
                    loss = (torch.exp(-logvar_stack) * huber_elem + 0.5 * logvar_stack).mean()
                else:
                    loss = huber_fn(mu_stack, y).mean()
                val_loss_sum += loss.item() * x.size(0)
                val_n += x.size(0)

        avg_val_loss = val_loss_sum / max(val_n, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train={avg_train_loss:.6f}  "
            f"val={avg_val_loss:.6f}  "
            f"best={best_val_loss:.6f}  "
            f"lr={current_lr:.2e}  "
            f"patience={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # 6. Restore best model ----------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss, "scaler": scaler}
