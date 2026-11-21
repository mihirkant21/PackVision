"""
PackVision AI - Custom Model Training Pipeline (v2.0)
=====================================================
Trains PackVisionNet-v2 on custom paired packaging datasets:
- PyTorch DataLoader with geometric, line stroke & photometric augmentations.
- FiLM + Structural Edge Attention (SEA) conditioned Multi-Scale U-Net optimization.
- Composite objective:
    L_total = lambda_l1 * L1 + lambda_ssim * (1 - SSIM) + lambda_edge * SobelEdge + lambda_color * ColorConsistency
- Cosine Annealing learning rate scheduler with linear warmup.
- Mixed precision (BFloat16 on CPU / Float16 on CUDA) with gradient accumulation.
- Comprehensive validation tracking: SSIM, PSNR, Edge IoU, L1, Color Loss.
- Automatic best checkpoint saving with complete model metadata (.pt).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.packvision_net import (
    PackVisionNet,
    PackagingLoss,
    MultiScaleSobelLoss,
    ColorConsistencyLoss,
)


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
        self.records: List[Dict[str, Any]] = []

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        print(f"[PackagingDataset] Loaded {len(self.records)} samples from {manifest_path.name}", flush=True)

        # Preload and cache all samples in RAM for instant batch retrieval
        self.cached_samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for idx, rec in enumerate(self.records):
            sketch_path = self._resolve_file(rec["sketch_file"])
            sketch_img = Image.open(sketch_path).convert("L").resize(self.image_size, Image.Resampling.LANCZOS)
            sketch_np = 1.0 - (np.array(sketch_img, dtype=np.float32) / 255.0)

            render_path = self._resolve_file(rec["render_file"])
            render_img = Image.open(render_path).convert("RGB").resize(self.image_size, Image.Resampling.LANCZOS)
            render_np = np.array(render_img, dtype=np.float32) / 255.0

            self.cached_samples[idx] = (sketch_np, render_np)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_file(self, rel_path: str) -> Path:
        """Resolves file path against data_root and project root."""
        p1 = self.data_root / rel_path
        if p1.exists():
            return p1
        p2 = PROJECT_ROOT / rel_path
        if p2.exists():
            return p2
        p3 = self.data_root / Path(rel_path).name
        if p3.exists():
            return p3
        return p1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        sketch_np, render_np = self.cached_samples[idx]
        sketch_np = sketch_np.copy()
        render_np = render_np.copy()

        # Data Augmentations (training split only)
        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                sketch_np = np.fliplr(sketch_np).copy()
                render_np = np.fliplr(render_np).copy()

            # Random line stroke thickness jitter (dilation/erosion)
            if random.random() > 0.6:
                kernel = np.ones((2, 2), np.uint8)
                if random.random() > 0.5:
                    sketch_np = cv2.dilate(sketch_np, kernel, iterations=1)
                else:
                    sketch_np = cv2.erode(sketch_np, kernel, iterations=1)

            # Random subtle photometric brightness jitter
            if random.random() > 0.6:
                factor = random.uniform(0.92, 1.08)
                render_np = np.clip(render_np * factor, 0.0, 1.0)

        # Convert to PyTorch tensors (C, H, W)
        sketch_tensor = torch.from_numpy(sketch_np).unsqueeze(0).float()
        render_tensor = torch.from_numpy(render_np).permute(2, 0, 1).float()

        # Attribute category indices
        cat_idx = CATEGORY_MAP.get(rec.get("category", "carton").lower(), 0)
        mat_idx = MATERIAL_MAP.get(rec.get("material", "kraft_cardboard").lower(), 0)
        fin_idx = FINISH_MAP.get(rec.get("finish", "matte").lower(), 0)

        return {
            "sketch": sketch_tensor,
            "render": render_tensor,
            "category_idx": torch.tensor(cat_idx, dtype=torch.long),
            "material_idx": torch.tensor(mat_idx, dtype=torch.long),
            "finish_idx": torch.tensor(fin_idx, dtype=torch.long),
            "sample_id": rec.get("sample_id", f"sample_{idx}"),
        }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sobel_filter: MultiScaleSobelLoss,
    edge_threshold: float = 0.15,
) -> Dict[str, float]:
    """Computes SSIM, PSNR, Edge IoU, and L1 metrics between pred and target."""
    with torch.no_grad():
        # L1
        l1 = F.l1_loss(pred, target).item()

        # PSNR
        mse = F.mse_loss(pred, target).item()
        psnr = 100.0 if mse < 1e-10 else 10.0 * math.log10(1.0 / mse)

        # SSIM
        mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
        mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
        sigma_x = F.avg_pool2d(pred * pred, 11, stride=1, padding=5) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + 1e-6
        )
        ssim = ssim_map.mean().item()

        # Edge IoU via Sobel filter
        gray_p = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
        gray_t = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]
        ep = sobel_filter._gradient_magnitude(gray_p)
        et = sobel_filter._gradient_magnitude(gray_t)
        mask_p = ep > edge_threshold
        mask_t = et > edge_threshold
        intersection = (mask_p & mask_t).float().sum()
        union = (mask_p | mask_t).float().sum()
        edge_iou = (intersection / (union + 1e-6)).item()

    return {
        "l1": l1,
        "psnr": psnr,
        "ssim": ssim,
        "edge_iou": edge_iou,
    }


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine Annealing learning rate schedule with linear warmup."""
    base_lr = optimizer.param_groups[0]["lr"]
    min_ratio = min_lr / base_lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear ramp-up
            return 0.1 + 0.9 * ((epoch + 1) / max(1, warmup_epochs))
        # Cosine decay down to min_ratio
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    data_root: str = "data/dataset",
    output_dir: str = "models/checkpoints",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2e-4,
    image_size: int = 256,
    accum_steps: int = 2,
    warmup_epochs: int = 2,
    lambda_l1: float = 1.0,
    lambda_ssim: float = 0.5,
    lambda_edge: float = 0.4,
    lambda_color: float = 0.3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the full PackVisionNet-v2 training and validation cycle."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        # Maximize utilization of available CPU cores
        threads = min(12, os.cpu_count() or 4)
        torch.set_num_threads(threads)
        print(f"[*] Configured PyTorch CPU threads: {torch.get_num_threads()}")

    print(f"\n=======================================================================")
    print(f"    PackVision AI - PackVisionNet-v2 Training Suite on {device.upper()}")
    print(f"=======================================================================")
    print(f"Resolution: {image_size}x{image_size} | Batch Size: {batch_size} (Accum: {accum_steps})")
    print(f"Loss weights: L1={lambda_l1}, SSIM={lambda_ssim}, Edge={lambda_edge}, Color={lambda_color}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & DataLoaders
    data_path = Path(data_root)
    train_manifest = data_path / "train_manifest.jsonl"
    val_manifest = data_path / "val_manifest.jsonl"

    if not train_manifest.exists() or not val_manifest.exists():
        # Previously this silently fell back to metadata.jsonl for BOTH splits,
        # which trains and validates on the same samples (data leakage).
        # Require explicit split manifests instead.
        missing = [p.name for p in (train_manifest, val_manifest) if not p.exists()]
        raise FileNotFoundError(
            f"Missing split manifest(s) in {data_path}: {', '.join(missing)}. "
            f"Refusing to train with val == train (data leakage). "
            f"Generate them with the dataset generator "
            f"(python data/dataset_generator.py) or repair them from disk with "
            f"python data/repair_manifests.py --write."
        )

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

    # 2. Instantiate Model, Loss & Optimizer
    model_config = {
        "in_channels": 1,
        "out_channels": 3,
        "base_channels": 32,
        "channel_multipliers": (1, 2, 4, 8),
        "num_categories": 16,
        "num_materials": 16,
        "num_finishes": 16,
        "embedding_dim": 64,
        "edge_attention": True,
        "use_attention_gate": True,
        "progressive_refinement": True,
        "image_size": image_size,
        "version": "2.0",
    }

    model = PackVisionNet(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        base_channels=model_config["base_channels"],
        channel_multipliers=model_config["channel_multipliers"],
        num_categories=model_config["num_categories"],
        num_materials=model_config["num_materials"],
        num_finishes=model_config["num_finishes"],
        embedding_dim=model_config["embedding_dim"],
        edge_attention=model_config["edge_attention"],
        use_attention_gate=model_config["use_attention_gate"],
        progressive_refinement=model_config["progressive_refinement"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture] PackVisionNet-v2 initialized ({total_params:,} trainable parameters)")

    criterion = PackagingLoss(
        lambda_l1=lambda_l1,
        lambda_ssim=lambda_ssim,
        lambda_edge=lambda_edge,
        lambda_color=lambda_color,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr=1e-5,
    )

    # Precision setup: Autocast on CUDA GPU, native float32 vectorization on CPU
    use_autocast = (device == "cuda")
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32

    # 3. Training Loop
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_ssim = 0.0
    best_val_psnr = 0.0
    best_val_edge_iou = 0.0
    best_epoch = 1
    best_checkpoint_path = output_path / "packvision_custom_v1.pt"

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss_accum = 0.0
        train_l1_accum = 0.0
        train_ssim_accum = 0.0
        train_edge_accum = 0.0
        train_color_accum = 0.0
        num_train_batches = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            step_start = time.time()
            sketches = batch["sketch"].to(device)
            renders = batch["render"].to(device)
            cat_idx = batch["category_idx"].to(device)
            mat_idx = batch["material_idx"].to(device)
            fin_idx = batch["finish_idx"].to(device)

            if use_autocast:
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    predictions = model(sketches, cat_idx, mat_idx, fin_idx)
                    losses = criterion(predictions.float(), renders.float())
                    loss = losses["loss"] / accum_steps
            else:
                predictions = model(sketches, cat_idx, mat_idx, fin_idx)
                losses = criterion(predictions, renders)
                loss = losses["loss"] / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss_accum += losses["loss"].item()
            train_l1_accum += losses["l1"].item()
            train_ssim_accum += losses["ssim"].item()
            train_edge_accum += losses["edge"].item()
            train_color_accum += losses["color"].item()
            num_train_batches += 1

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                cur_l = losses["loss"].item()
                cur_ssim = losses["ssim"].item()
                step_dt = time.time() - step_start
                print(
                    f"  [Epoch {epoch:02d} | Step {step+1:02d}/{len(train_loader):02d}] "
                    f"Batch Loss: {cur_l:.4f} (SSIM: {cur_ssim:.4f}) | {step_dt:.2f}s/step",
                    flush=True,
                )

        scheduler.step()

        # Epoch Train Averages
        avg_train_loss = train_loss_accum / max(1, num_train_batches)
        avg_train_l1 = train_l1_accum / max(1, num_train_batches)
        avg_train_ssim = train_ssim_accum / max(1, num_train_batches)
        avg_train_edge = train_edge_accum / max(1, num_train_batches)
        avg_train_color = train_color_accum / max(1, num_train_batches)

        # Validation Pass
        model.eval()
        val_loss_accum = 0.0
        val_l1_accum = 0.0
        val_ssim_accum = 0.0
        val_psnr_accum = 0.0
        val_edge_iou_accum = 0.0
        val_color_accum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                sketches = batch["sketch"].to(device)
                renders = batch["render"].to(device)
                cat_idx = batch["category_idx"].to(device)
                mat_idx = batch["material_idx"].to(device)
                fin_idx = batch["finish_idx"].to(device)

                with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=use_autocast):
                    predictions = model(sketches, cat_idx, mat_idx, fin_idx)
                    losses = criterion(predictions.float(), renders.float())

                # Detailed metrics
                metrics = compute_metrics(
                    predictions.float(),
                    renders.float(),
                    sobel_filter=criterion.sobel_loss,
                )

                val_loss_accum += losses["loss"].item()
                val_l1_accum += metrics["l1"]
                val_ssim_accum += metrics["ssim"]
                val_psnr_accum += metrics["psnr"]
                val_edge_iou_accum += metrics["edge_iou"]
                val_color_accum += losses["color"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_accum / max(1, num_val_batches)
        avg_val_l1 = val_l1_accum / max(1, num_val_batches)
        avg_val_ssim = val_ssim_accum / max(1, num_val_batches)
        avg_val_psnr = val_psnr_accum / max(1, num_val_batches)
        avg_val_edge_iou = val_edge_iou_accum / max(1, num_val_batches)
        avg_val_color = val_color_accum / max(1, num_val_batches)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, SSIM: {avg_train_ssim:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"SSIM: {avg_val_ssim:.4f} | PSNR: {avg_val_psnr:.2f}dB | EdgeIoU: {avg_val_edge_iou:.4f} | "
            f"LR: {current_lr:.2e}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_ssim": round(avg_train_ssim, 4),
            "train_edge": round(avg_train_edge, 5),
            "train_color": round(avg_train_color, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_l1": round(avg_val_l1, 5),
            "val_ssim": round(avg_val_ssim, 4),
            "val_psnr": round(avg_val_psnr, 2),
            "val_edge_iou": round(avg_val_edge_iou, 4),
            "val_color": round(avg_val_color, 5),
            "lr": current_lr,
            "epoch_time_s": round(epoch_duration, 2),
        })

        # Save Best Checkpoint based on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_ssim = avg_val_ssim
            best_val_psnr = avg_val_psnr
            best_val_edge_iou = avg_val_edge_iou
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_ssim": best_val_ssim,
                "val_psnr": best_val_psnr,
                "val_edge_iou": best_val_edge_iou,
                "model_config": model_config,
                "category_map": CATEGORY_MAP,
                "material_map": MATERIAL_MAP,
                "finish_map": FINISH_MAP,
                "training_history": history,
                "loss_weights": {
                    "lambda_l1": lambda_l1,
                    "lambda_ssim": lambda_ssim,
                    "lambda_edge": lambda_edge,
                    "lambda_color": lambda_color,
                },
                "version": "2.0",
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f}, SSIM: {best_val_ssim:.4f})")

    total_training_time = time.time() - start_training_time

    # Save comprehensive training trajectory report
    history_file = output_path / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "PackVisionNet-v2",
            "version": "2.0",
            "total_epochs": epochs,
            "total_time_seconds": round(total_training_time, 2),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 5),
            "best_val_ssim": round(best_val_ssim, 4),
            "best_val_psnr": round(best_val_psnr, 2),
            "best_val_edge_iou": round(best_val_edge_iou, 4),
            "parameters": total_params,
            "model_config": model_config,
            "history": history,
        }, f, indent=2)

    print(f"\n=== Training Complete in {total_training_time:.1f}s ===")
    print(f"Best Checkpoint: {best_checkpoint_path} (Epoch {best_epoch})")
    print(f"Validation SSIM: {best_val_ssim:.4f} | PSNR: {best_val_psnr:.2f}dB | Edge IoU: {best_val_edge_iou:.4f}")
    print(f"Training History: {history_file}")

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_loss": best_val_loss,
        "best_val_ssim": best_val_ssim,
        "best_val_psnr": best_val_psnr,
        "best_val_edge_iou": best_val_edge_iou,
        "best_epoch": best_epoch,
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PackVisionNet-v2 on custom packaging dataset.")
    parser.add_argument("--data_root", "--dataset-dir", "--dataset_dir", dest="data_root", type=str, default="data/dataset")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=2e-4)
    parser.add_argument("--image_size", "--image-size", dest="image_size", type=int, default=256)
    parser.add_argument("--accum_steps", "--gradient-accumulation-steps", dest="accum_steps", type=int, default=2)
    parser.add_argument("--warmup_epochs", "--warmup-epochs", dest="warmup_epochs", type=int, default=2)
    parser.add_argument("--lambda_l1", "--lambda-l1", dest="lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_ssim", "--lambda-ssim", dest="lambda_ssim", type=float, default=0.5)
    parser.add_argument("--lambda_edge", "--lambda-edge", dest="lambda_edge", type=float, default=0.4)
    parser.add_argument("--lambda_color", "--lambda-color", dest="lambda_color", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge,
        lambda_color=args.lambda_color,
        device=args.device,
    )
