"""
train_model.py
==============
Behavioral Cloning – NVIDIA PilotNet Training Script
Hardware target : Pop!_OS · NVIDIA RTX 4050 (CUDA)
Output          : robogo_pilotnet.pth  (best validation loss checkpoint)

Required packages
-----------------
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install pandas pillow tqdm

Run
---
    python3 train_model.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import time

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR  = "my_dataset"
IMG_DIR      = os.path.join(DATASET_DIR, "images")
LOG_PATH     = os.path.join(DATASET_DIR, "driving_log.csv")
CHECKPOINT   = "robogo_pilotnet.pth"

IMG_W, IMG_H = 200, 66          # PilotNet canonical input size
BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3
TRAIN_RATIO  = 0.80
NUM_WORKERS  = 4                 # set 0 if you hit DataLoader issues on Windows
PIN_MEMORY   = True

# ImageNet statistics – safe default for transfer; re-calculate if needed
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
if device.type == 'cuda':
    print(f"       GPU : {torch.cuda.get_device_name(0)}")
    print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DrivingDataset(Dataset):
    """
    Loads (image, steering_angle) pairs from driving_log.csv.

    Augmentation (training only):
      • Random horizontal flip  → invert steering angle sign
      • ColorJitter             → brightness / contrast robustness
    """

    # Shared resize + tensor + normalise (no randomness – safe for val too)
    _base_tf = T.Compose([
        T.Resize((IMG_H, IMG_W)),           # H×W order for transforms
        T.ToTensor(),                        # [0,255] uint8  →  [0,1] float32
        T.Normalize(mean=MEAN, std=STD),
    ])

    # Extra augmentations applied only during training
    _color_jitter = T.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.05,
    )

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        img_path = os.path.join(IMG_DIR, row["image_path"])
        angle    = float(row["steering_angle"])

        img = Image.open(img_path).convert("RGB")

        if self.augment:
            # ── Random horizontal flip ─────────────────────────────────────
            if random.random() > 0.5:
                img   = TF.hflip(img)
                angle = -angle

            # ── Colour jitter (brightness / contrast) ─────────────────────
            img = self._color_jitter(img)

        img_tensor    = self._base_tf(img)
        angle_tensor  = torch.tensor([angle], dtype=torch.float32)

        return img_tensor, angle_tensor


def build_loaders(log_path: str):
    """Read CSV, split 80/20, return train & val DataLoaders."""
    df = pd.read_csv(log_path, names=["image_path", "steering_angle", "speed"],
                     header=0)

    # Drop rows whose image file is missing (safety guard)
    df = df[df["image_path"].apply(
        lambda f: os.path.isfile(os.path.join(IMG_DIR, f))
    )].reset_index(drop=True)

    print(f"[INFO] Total valid samples: {len(df)}")

    n_train = int(len(df) * TRAIN_RATIO)
    n_val   = len(df) - n_train

    # Deterministic split via indices so images & labels stay aligned
    indices = torch.randperm(len(df), generator=torch.Generator().manual_seed(42)).tolist()
    train_df = df.iloc[indices[:n_train]]
    val_df   = df.iloc[indices[n_train:]]

    print(f"[INFO] Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_ds = DrivingDataset(train_df, augment=True)
    val_ds   = DrivingDataset(val_df,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE * 2,       # no gradients → fits 2× batch
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Model – NVIDIA PilotNet
# ─────────────────────────────────────────────────────────────────────────────
class PilotNet(nn.Module):
    """
    Exact NVIDIA PilotNet architecture
    Paper: https://arxiv.org/abs/1604.07316

    Input  : (B, 3, 66, 200)   – normalised RGB
    Output : (B, 1)            – predicted steering angle in [-1, 1]
    """

    def __init__(self, dropout_p: float = 0.1):
        super().__init__()

        # ── Convolutional backbone ─────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1 – 24 × 5×5, stride 2  →  (B, 24, 31, 98)
            nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),

            # Block 2 – 36 × 5×5, stride 2  →  (B, 36, 14, 47)
            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(36),
            nn.ELU(inplace=True),

            # Block 3 – 48 × 5×5, stride 2  →  (B, 48, 5, 22)
            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),

            # Block 4 – 64 × 3×3, stride 1  →  (B, 64, 3, 20)
            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),

            # Block 5 – 64 × 3×3, stride 1  →  (B, 64, 1, 18)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )

        # ── Fully-connected head ───────────────────────────────────────────
        # Flat size after conv stack: 64 × 1 × 18 = 1152
        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1152, 100),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(100, 50),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_p),

            nn.Linear(50, 10),
            nn.ELU(inplace=True),

            nn.Linear(10, 1),   # raw steering angle – no activation
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.regressor(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"  Train E{epoch:02d}", leave=False,
                unit="batch", dynamic_ncols=True)

    for imgs, angles in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass (AMP)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            preds = model(imgs)
            loss  = criterion(preds, angles)

        scaler.scale(loss).backward()
        # Gradient clipping – avoids exploding gradients on small datasets
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.5f}"})

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    for imgs, angles in loader:
        imgs   = imgs.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            preds = model(imgs)
            loss  = criterion(preds, angles)

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RoboGo PilotNet – Behavioral Cloning Training")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loaders(LOG_PATH)

    # ── Model ─────────────────────────────────────────────────────────────
    model = PilotNet(dropout_p=0.1).to(device)
    print(f"[INFO] PilotNet parameters: {count_parameters(model):,}")

    # ── Loss / Optimiser / Scheduler ──────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing: smoothly decays LR to near-zero over EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # AMP GradScaler (no-op on CPU)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    print(f"\n{'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>10} | "
          f"{'LR':>8} | {'Time':>6} | {'Best':>5}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler, epoch)
        val_loss   = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed  = time.time() - t0
        cur_lr   = optimizer.param_groups[0]["lr"]
        is_best  = val_loss < best_val_loss

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        flag = " ✓" if is_best else ""
        print(f"{epoch:>5} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
              f"{cur_lr:>8.2e} | {elapsed:>5.1f}s |{flag}")

        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch"          : epoch,
                "model_state"    : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"       : best_val_loss,
                "img_w"          : IMG_W,
                "img_h"          : IMG_H,
                "mean"           : MEAN,
                "std"            : STD,
            }, CHECKPOINT)

    # ── Summary ───────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"[INFO] Training complete.")
    print(f"[INFO] Best validation MSE : {best_val_loss:.6f}")
    print(f"[INFO] Best RMSE           : {best_val_loss ** 0.5:.6f}  "
          f"(steering units)")
    print(f"[INFO] Checkpoint saved    : {os.path.abspath(CHECKPOINT)}")


if __name__ == "__main__":
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # Deterministic ops where possible
        torch.backends.cudnn.benchmark   = True   # fastest conv for fixed input size
        torch.backends.cudnn.deterministic = False # keep benchmark=True effective

    main()
