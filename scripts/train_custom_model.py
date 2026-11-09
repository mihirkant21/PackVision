"""
PackVision AI - Custom Model Training Pipeline
==============================================
Trains PackVisionNet on the custom paired packaging dataset:
- PyTorch DataLoader with geometric & photometric augmentations.
- FiLM-conditioned Multi-Scale U-Net optimization via AdamW.
- Multi-term objective: L1 reconstruction + Sobel edge fidelity + SSIM.
- Validation tracking and automatic best checkpoint saving (.pt).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.packvision_net import PackVisionNet, PackagingLoss


# Category & Attribute Label Mappings
CATEGORY_MAP = {
    "carton": 0, "box": 1, "pouch": 2, "bottle": 3,
    "jar": 4, "can": 5, "tube": 6, "sleeve": 7,
}
MATERIAL_MAP = {
    "kraft_cardboard": 0, "corrugated_box": 1, "matte_carton": 2,
    "coated_sbs": 3, "flexible_pouch": 4, "frosted_glass": 5,
    "brushed_aluminum": 6, "molded_pulp": 7,
}
FINISH_MAP = {
    "gold_foil": 0, "silver_foil": 1, "spot_uv": 2,
    "embossed": 3, "debossed": 4, "holographic": 5,
    "matte": 6, "uncoated": 7,
}


class PackagingDataset(Dataset):
    """PyTorch Dataset loading paired (sketch, render, attributes) samples."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_root: Union[str, Path] = "data/dataset",
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.is_train = is_train
        self.records: List[Dict] = []

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        print(f"[PackagingDataset] Loaded {len(self.records)} samples from {manifest_path.name}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # 1. Load Sketch Image (1 channel)
        sketch_path = self.data_root / rec["sketch_file"]
        if not sketch_path.exists():
            # Try direct relative path
            sketch_path = PROJECT_ROOT / rec["sketch_file"]

        sketch_img = Image.open(sketch_path).convert("L")
        sketch_img = sketch_img.resize(self.image_size, Image.Resampling.LANCZOS)
        sketch_np = np.array(sketch_img, dtype=np.float32) / 255.0

        # Invert so lines are white on black for neural edge conditioning
        sketch_np = 1.0 - sketch_np

        # 2. Load Target Render Image (3 channels RGB)
        render_path = self.data_root / rec["render_file"]
        if not render_path.exists():
            render_path = PROJECT_ROOT / rec["render_file"]

        render_img = Image.open(render_path).convert("RGB")
        render_img = render_img.resize(self.image_size, Image.Resampling.LANCZOS)
        render_np = np.array(render_img, dtype=np.float32) / 255.0

        # Data augmentation (training only)
        if self.is_train and random.random() > 0.5:
            # Horizontal flip
            sketch_np = np.fliplr(sketch_np).copy()
            render_np = np.fliplr(render_np).copy()

        # Convert to PyTorch tensors (C, H, W)
        sketch_tensor = torch.from_numpy(sketch_np).unsqueeze(0).float()
        render_tensor = torch.from_numpy(render_np).permute(2, 0, 1).float()

        # Attribute index tensors
        cat_idx = CATEGORY_MAP.get(rec.get("category", "carton"), 0)
        mat_idx = MATERIAL_MAP.get(rec.get("material", "kraft_cardboard"), 0)
        fin_idx = FINISH_MAP.get(rec.get("finish", "matte"), 0)

        return {
            "sketch": sketch_tensor,
            "render": render_tensor,
            "category_idx": torch.tensor(cat_idx, dtype=torch.long),
            "material_idx": torch.tensor(mat_idx, dtype=torch.long),
            "finish_idx": torch.tensor(fin_idx, dtype=torch.long),
            "sample_id": rec.get("sample_id", f"sample_{idx}"),
        }


def train_model(
    data_root: str = "data/dataset",
    output_dir: str = "models/checkpoints",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    image_size: int = 256,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the full PackVisionNet training and validation cycle."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting PackVision AI Model Training on {device.upper()} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. DataLoaders
    train_manifest = Path(data_root) / "train_manifest.jsonl"
    val_manifest = Path(data_root) / "val_manifest.jsonl"

    if not train_manifest.exists() or not val_manifest.exists():
        # Fallback to general metadata.jsonl
        meta_manifest = Path(data_root) / "metadata.jsonl"
        train_manifest = meta_manifest
        val_manifest = meta_manifest

    train_dataset = PackagingDataset(
        manifest_path=train_manifest,
        data_root=data_root,
        image_size=(image_size, image_size),
        is_train=True,
    )
    val_dataset = PackagingDataset(
        manifest_path=val_manifest,
        data_root=data_root,
        image_size=(image_size, image_size),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model & Loss
    model = PackVisionNet(
        in_channels=1,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2, 4, 8),
        embedding_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture] PackVisionNet initialized ({total_params:,} trainable parameters)")

    criterion = PackagingLoss(lambda_edge=0.4, lambda_ssim=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_checkpoint_path = output_path / "packvision_custom_v1.pt"

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss_accum = 0.0
        train_l1_accum = 0.0
        train_edge_accum = 0.0
        train_ssim_accum = 0.0
        num_train_batches = 0

        for batch in train_loader:
            sketches = batch["sketch"].to(device)
            renders = batch["render"].to(device)
            cat_idx = batch["category_idx"].to(device)
            mat_idx = batch["material_idx"].to(device)
            fin_idx = batch["finish_idx"].to(device)

            optimizer.zero_grad()
            predictions = model(sketches, cat_idx, mat_idx, fin_idx)
            losses = criterion(predictions, renders)

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            train_l1_accum += losses["l1"].item()
            train_edge_accum += losses["edge"].item()
            train_ssim_accum += losses["ssim"].item()
            num_train_batches += 1

        scheduler.step()

        # Epoch Train Averages
        avg_train_loss = train_loss_accum / max(1, num_train_batches)
        avg_train_l1 = train_l1_accum / max(1, num_train_batches)
        avg_train_ssim = train_ssim_accum / max(1, num_train_batches)

        # Validation Pass
        model.eval()
        val_loss_accum = 0.0
        val_l1_accum = 0.0
        val_edge_accum = 0.0
        val_ssim_accum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                sketches = batch["sketch"].to(device)
                renders = batch["render"].to(device)
                cat_idx = batch["category_idx"].to(device)
                mat_idx = batch["material_idx"].to(device)
                fin_idx = batch["finish_idx"].to(device)

                predictions = model(sketches, cat_idx, mat_idx, fin_idx)
                losses = criterion(predictions, renders)

                val_loss_accum += losses["loss"].item()
                val_l1_accum += losses["l1"].item()
                val_edge_accum += losses["edge"].item()
                val_ssim_accum += losses["ssim"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_accum / max(1, num_val_batches)
        avg_val_l1 = val_l1_accum / max(1, num_val_batches)
        avg_val_ssim = val_ssim_accum / max(1, num_val_batches)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, SSIM: {avg_train_ssim:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}, SSIM: {avg_val_ssim:.4f}) | "
            f"LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_ssim": round(avg_train_ssim, 4),
            "val_loss": round(avg_val_loss, 5),
            "val_l1": round(avg_val_l1, 5),
            "val_ssim": round(avg_val_ssim, 4),
            "lr": current_lr,
            "epoch_time_s": round(epoch_duration, 2),
        })

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_ssim": avg_val_ssim,
                "model_config": {
                    "in_channels": 1,
                    "out_channels": 3,
                    "base_channels": 32,
                    "channel_multipliers": (1, 2, 4, 8),
                    "embedding_dim": 64,
                },
                "category_map": CATEGORY_MAP,
                "material_map": MATERIAL_MAP,
                "finish_map": FINISH_MAP,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    total_training_time = time.time() - start_training_time

    # Save training trajectory report
    history_file = output_path / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_epochs": epochs,
            "total_time_seconds": round(total_training_time, 2),
            "best_val_loss": round(best_val_loss, 5),
            "final_val_ssim": round(history[-1]["val_ssim"], 4),
            "parameters": total_params,
            "history": history,
        }, f, indent=2)

    print(f"\n=== Training Complete in {total_training_time:.1f}s ===")
    print(f"Best Checkpoint: {best_checkpoint_path}")
    print(f"Training History: {history_file}")

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_loss": best_val_loss,
        "final_ssim": history[-1]["val_ssim"],
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PackVisionNet on custom packaging dataset.")
    parser.add_argument("--data_root", type=str, default="data/dataset")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
    )
