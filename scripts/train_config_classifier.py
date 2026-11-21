"""
PackVision AI - Train the Local Packaging Configuration Classifier
==================================================================
Trains ``src.config_classifier.ConfigClassifierCNN`` on CPU from the synthetic
dataset triplets (sketch + render as independent samples, same labels):

  * Input space: both designer sketches AND photorealistic renders, so one
    model handles whatever the user uploads.
  * Augmentation (train only): random resized crop, +-8 deg rotation,
    horizontal flip, brightness/contrast jitter, gaussian noise, mild blur --
    K augmented copies per image plus the clean original.
  * Validation: CLEAN originals only, from the canonical val split of the
    dataset manifests (stale-duplicate records excluded, plus an id-leakage
    guard), reported per head and per input domain.
  * Labels: dataset legacy keys -> canonical taxonomy keys via
    ``src.taxonomy.normalize``; heads span the full canonical vocabulary.

Checkpoint (best val mean accuracy) is saved to
``models/checkpoints/config_classifier.pt`` as a torch.save dict holding the
state_dict, label maps and val metrics (same convention as the existing
packvision checkpoint).

Typical run: ``python scripts/train_config_classifier.py`` (a few minutes on CPU).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_classifier import (  # noqa: E402
    CATEGORY_KEYS,
    CHECKPOINT_PATH,
    ConfigClassifierCNN,
    FINISH_KEYS,
    INPUT_SIZE,
    MATERIAL_KEYS,
)
from src.taxonomy import normalize as taxonomy_normalize  # noqa: E402

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
DOMAINS = ("sketch", "render")
AUG_SOURCE_SIZE = 160  # augments run at this size before final downsample


# =============================================================================
# DATASET ASSEMBLY
# =============================================================================

def load_records(data_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read metadata.jsonl -> (train_records, clean_val_records).

    Val keeps only canonical records (no ``stale_duplicate_of`` provenance) and
    drops any id that also exists in the train split (leakage guard).
    """
    meta_path = data_root / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    records: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    train = [r for r in records if r.get("split") == "train"]
    train_ids = {r["sample_id"] for r in train}
    val = [
        r for r in records
        if r.get("split") == "val"
        and "stale_duplicate_of" not in r
        and r["sample_id"] not in train_ids
    ]
    return train, val


def build_label_maps() -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Canonical key -> index maps for the three heads."""
    return (
        {k: i for i, k in enumerate(CATEGORY_KEYS)},
        {k: i for i, k in enumerate(MATERIAL_KEYS)},
        {k: i for i, k in enumerate(FINISH_KEYS)},
    )


def record_to_indices(rec: Dict[str, Any], maps) -> Tuple[int, int, int] | None:
    """Map a record's legacy labels to canonical head indices (None if unmappable)."""
    cat_map, mat_map, fin_map = maps
    cat = taxonomy_normalize(str(rec.get("category", "")), "package_types")
    mat = taxonomy_normalize(str(rec.get("material", "")), "materials")
    fin = taxonomy_normalize(str(rec.get("finish", "")), "finishes")
    if cat is None or mat is None or fin is None:
        return None
    return cat_map[cat], mat_map[mat], fin_map[fin]


def collect_image_items(
    records: List[Dict[str, Any]],
    data_root: Path,
    maps,
) -> List[Dict[str, Any]]:
    """Expand each triplet record into per-domain items with canonical indices."""
    items: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for rec in records:
        indices = record_to_indices(rec, maps)
        if indices is None:
            skipped.append(rec["sample_id"])
            continue
        for domain in DOMAINS:
            rel = rec.get(f"{domain}_file")
            if not rel:
                skipped.append(rec["sample_id"])
                continue
            path = data_root / rel
            if not path.exists():
                skipped.append(rec["sample_id"])
                continue
            items.append({
                "path": path,
                "domain": domain,
                "sample_id": rec["sample_id"],
                "category": indices[0],
                "material": indices[1],
                "finish": indices[2],
            })
    if skipped:
        print(f"[warn] skipped {len(skipped)} unusable records/paths (unknown labels or missing files)")
    return items


# =============================================================================
# AUGMENTATION
# =============================================================================

def _border_fill(img: Image.Image) -> Tuple[int, int, int]:
    """Mean border colour -- sensible rotation fill for sketches and renders."""
    arr = np.asarray(img.convert("RGB"))
    border = np.concatenate([
        arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]
    ]).mean(axis=0)
    return tuple(int(round(v)) for v in border)


def augment_image(base: Image.Image, rng: random.Random) -> Image.Image:
    """Stochastic augmentation chain: crop -> rotate -> flip -> tone -> noise."""
    img = base.convert("RGB")

    # 1. Random square crop (scale 0.80-1.00 of the aug canvas), back to source size.
    scale = rng.uniform(0.80, 1.00)
    w, h = img.size
    cw = max(16, int(w * scale))
    ch = max(16, int(h * scale))
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BILINEAR)

    # 2. Small rotation.
    angle = rng.uniform(-8.0, 8.0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=_border_fill(img))

    # 3. Horizontal flip.
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 4. Photometric jitter.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))

    # 5. Mild blur.
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))

    return img


def add_gaussian_noise(arr: np.ndarray, noise_rng: np.random.Generator, sigma_max: float = 6.0) -> np.ndarray:
    sigma = float(noise_rng.uniform(0.0, sigma_max))
    if sigma <= 0:
        return arr
    noisy = arr.astype(np.float32) + noise_rng.normal(0.0, sigma, arr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def load_base_image(path: Path, rng: random.Random, augment: bool) -> Image.Image:
    """Load an image at AUG_SOURCE_SIZE; optionally apply the augmentation chain."""
    img = Image.open(path).convert("RGB")
    if img.size != (AUG_SOURCE_SIZE, AUG_SOURCE_SIZE):
        img = img.resize((AUG_SOURCE_SIZE, AUG_SOURCE_SIZE), Image.Resampling.LANCZOS)
    if augment:
        img = augment_image(img, rng)
    return img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)


def build_split_arrays(
    items: List[Dict[str, Any]],
    aug_copies: int,
    seed: int,
    augment: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Tuple[str, int]]]:
    """Materialize images into in-memory uint8 arrays.

    Returns (images uint8 [N,128,128,3], labels {head: int64 [N]}, meta list).
    Train mode yields the clean original + ``aug_copies`` augmented copies per
    item; val mode yields only the clean original.
    """
    images: List[np.ndarray] = []
    labels: Dict[str, List[int]] = {"category": [], "material": [], "finish": []}
    meta: List[Tuple[str, int]] = []
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    t0 = time.time()

    repeats = (1 + aug_copies) if augment else 1
    for item in items:
        base = load_base_image(item["path"], rng, augment=False)
        for copy_idx in range(repeats):
            if copy_idx == 0:
                img = base
            else:
                # Fresh augmented copy (augmentation chain + sensor noise).
                img = load_base_image(item["path"], rng, augment=True)
            arr = np.asarray(img, dtype=np.uint8)
            if augment and copy_idx > 0:
                arr = add_gaussian_noise(arr, noise_rng)
            images.append(arr)
            labels["category"].append(item["category"])
            labels["material"].append(item["material"])
            labels["finish"].append(item["finish"])
            meta.append((item["domain"], copy_idx))

    img_arr = np.stack(images)
    label_arr = {k: np.asarray(v, dtype=np.int64) for k, v in labels.items()}
    print(f"  materialized {len(img_arr)} images in {time.time() - t0:.1f}s")
    return img_arr, label_arr, meta


class ConfigDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: Dict[str, np.ndarray],
        meta: List[Tuple[str, int]] | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.meta = meta if meta is not None else [("?", 0)] * len(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        arr = self.images[idx].astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        target = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.labels.items()}
        domain, copy_idx = self.meta[idx]
        return tensor, target, domain, copy_idx


# =============================================================================
# TRAINING
# =============================================================================

@torch.no_grad()
def evaluate(
    model: ConfigClassifierCNN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Per-head top-1 accuracy overall and per input domain (sketch / render)."""
    model.eval()
    correct = Counter()
    total = 0
    domain_correct = Counter()  # (domain, head) -> correct count
    domain_total = Counter()    # (domain, head) -> sample count
    for tensors, targets, domains, _copy_idx in loader:
        tensors = tensors.to(device)
        outputs = model(tensors)
        for head in ("category", "material", "finish"):
            pred = outputs[head].argmax(dim=1).cpu()
            matches = pred == targets[head]
            correct[head] += int(matches.sum())
            for d, m in zip(domains, matches.tolist()):
                domain_total[(d, head)] += 1
                domain_correct[(d, head)] += int(m)
        total += len(tensors)
    heads = ("category", "material", "finish")
    metrics = {f"{h}_acc": correct[h] / max(1, total) for h in heads}
    metrics["mean_acc"] = sum(metrics[f"{h}_acc"] for h in heads) / 3.0
    for d in sorted({d for d, _ in domain_total}):
        for h in heads:
            metrics[f"{d}_{h}_acc"] = domain_correct[(d, h)] / max(1, domain_total[(d, h)])
    metrics["val_n"] = float(total)
    return metrics


def train(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if args.device else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Device: {device} | torch {torch.__version__} | threads {torch.get_num_threads()}")
    maps = build_label_maps()

    data_root = Path(args.data_root)
    print(f"Loading dataset from {data_root}")
    train_records, val_records = load_records(data_root)
    print(f"  triplets: train={len(train_records)} val={len(val_records)}")
    train_items = collect_image_items(train_records, data_root, maps)
    val_items = collect_image_items(val_records, data_root, maps)
    per_domain = Counter(i["domain"] for i in train_items)
    print(f"  base images: train={len(train_items)} ({dict(per_domain)}) val={len(val_items)}")
    if not train_items or not val_items:
        raise SystemExit("No usable training or validation images found.")

    print(f"Materializing train images (+{args.aug_copies} augmented copies each)...")
    train_x, train_y, train_meta = build_split_arrays(train_items, args.aug_copies, args.seed, augment=True)
    print(f"Materializing clean val images...")
    val_x, val_y, val_meta = build_split_arrays(val_items, 0, args.seed, augment=False)

    train_ds = ConfigDataset(train_x, train_y, train_meta)
    val_ds = ConfigDataset(val_x, val_y, val_meta)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=gen, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = ConfigClassifierCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best = {"mean_acc": -1.0, "epoch": 0, "metrics": None, "state": None}
    wall_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        t_ep = time.time()
        for tensors, targets, _domains, _copy_idx in train_loader:
            tensors = tensors.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(tensors)
            loss = sum(criterion(outputs[h], targets[h]) for h in outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        scheduler.step()
        train_time = time.time() - t_ep

        metrics = evaluate(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(1, batches), 4),
            "lr": scheduler.get_last_lr()[0],
            "train_epoch_seconds": round(train_time, 1),
            **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
        }
        history.append(entry)
        marker = ""
        if metrics["mean_acc"] > best["mean_acc"]:
            best = {"mean_acc": metrics["mean_acc"], "epoch": epoch,
                    "metrics": metrics, "state": {k: v.detach().cpu().clone()
                                                  for k, v in model.state_dict().items()}}
            marker = "  <-- best"
        print(f"epoch {epoch:02d}/{args.epochs} | loss {entry['train_loss']:.4f} | "
              f"lr {entry['lr']:.5f} | {train_time:.1f}s | "
              f"cat {metrics['category_acc']:.3f} mat {metrics['material_acc']:.3f} "
              f"fin {metrics['finish_acc']:.3f} | mean {metrics['mean_acc']:.3f} | "
              f"(sketch-cat {metrics.get('sketch_category_acc', 0):.3f} "
              f"render-cat {metrics.get('render_category_acc', 0):.3f}){marker}")

    wall = time.time() - wall_start
    if best["state"] is None:
        raise SystemExit("Training produced no usable checkpoint.")

    # Reload best weights for the final metrics snapshot.
    model.load_state_dict(best["state"])
    final_metrics = evaluate(model, val_loader, device)

    checkpoint = {
        "version": 1,
        "kind": "config_classifier",
        "epoch": best["epoch"],
        "model_state_dict": best["state"],
        "model_config": {
            "input_size": INPUT_SIZE,
            "num_categories": len(CATEGORY_KEYS),
            "num_materials": len(MATERIAL_KEYS),
            "num_finishes": len(FINISH_KEYS),
            "base_channels": 32,
            "dropout": 0.15,
        },
        "category_map": maps[0],
        "material_map": maps[1],
        "finish_map": maps[2],
        "val_metrics": {
            "category_acc": round(final_metrics["category_acc"], 4),
            "material_acc": round(final_metrics["material_acc"], 4),
            "finish_acc": round(final_metrics["finish_acc"], 4),
            "mean_acc": round(final_metrics["mean_acc"], 4),
            "sketch_category_acc": round(final_metrics.get("sketch_category_acc", 0.0), 4),
            "render_category_acc": round(final_metrics.get("render_category_acc", 0.0), 4),
            "sketch_material_acc": round(final_metrics.get("sketch_material_acc", 0.0), 4),
            "render_material_acc": round(final_metrics.get("render_material_acc", 0.0), 4),
            "sketch_finish_acc": round(final_metrics.get("sketch_finish_acc", 0.0), 4),
            "render_finish_acc": round(final_metrics.get("render_finish_acc", 0.0), 4),
            "epoch": best["epoch"],
            "val_n": int(final_metrics["val_n"]),
        },
        "training_meta": {
            "train_images": int(len(train_x)),
            "val_images": int(len(val_x)),
            "train_base_images": len(train_items),
            "val_base_images": len(val_items),
            "train_triplets": len(train_records),
            "val_triplets": len(val_records),
            "aug_copies_per_image": args.aug_copies,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "label_smoothing": args.label_smoothing,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "parameter_count": n_params,
            "wall_seconds": round(wall, 1),
        },
        "val_history": history,
    }
    torch.save(checkpoint, output_path)
    print(f"\nSaved best checkpoint (epoch {best['epoch']}) -> {output_path}")
    print(f"Final val metrics: cat {final_metrics['category_acc']:.4f} | "
          f"mat {final_metrics['material_acc']:.4f} | fin {final_metrics['finish_acc']:.4f} | "
          f"mean {final_metrics['mean_acc']:.4f}")
    print(f"Training wall time: {wall:.1f}s")
    return checkpoint


# =============================================================================
# EVAL-ONLY MODE
# =============================================================================

def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, float]:
    """Re-evaluate a saved checkpoint on the clean val split and refresh its val_metrics."""
    device = torch.device(args.device if args.device else "cpu")
    maps = build_label_maps()
    data_root = Path(args.data_root)
    _, val_records = load_records(data_root)
    val_items = collect_image_items(val_records, data_root, maps)
    val_x, val_y, val_meta = build_split_arrays(val_items, 0, args.seed, augment=False)
    val_ds = ConfigDataset(val_x, val_y, val_meta)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    ck = torch.load(args.output, map_location="cpu", weights_only=False)
    model = ConfigClassifierCNN(
        num_categories=ck["model_config"].get("num_categories", len(CATEGORY_KEYS)),
        num_materials=ck["model_config"].get("num_materials", len(MATERIAL_KEYS)),
        num_finishes=ck["model_config"].get("num_finishes", len(FINISH_KEYS)),
        base_channels=ck["model_config"].get("base_channels", 32),
        dropout=ck["model_config"].get("dropout", 0.15),
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    metrics = evaluate(model, val_loader, device)

    ck.setdefault("val_metrics", {})
    ck["val_metrics"].update({
        "category_acc": round(metrics["category_acc"], 4),
        "material_acc": round(metrics["material_acc"], 4),
        "finish_acc": round(metrics["finish_acc"], 4),
        "mean_acc": round(metrics["mean_acc"], 4),
        "sketch_category_acc": round(metrics.get("sketch_category_acc", 0.0), 4),
        "render_category_acc": round(metrics.get("render_category_acc", 0.0), 4),
        "sketch_material_acc": round(metrics.get("sketch_material_acc", 0.0), 4),
        "render_material_acc": round(metrics.get("render_material_acc", 0.0), 4),
        "sketch_finish_acc": round(metrics.get("sketch_finish_acc", 0.0), 4),
        "render_finish_acc": round(metrics.get("render_finish_acc", 0.0), 4),
        "val_n": int(metrics["val_n"]),
    })
    torch.save(ck, args.output)
    print(f"Re-evaluated {args.output} on {metrics['val_n']} clean val images:")
    for h in ("category", "material", "finish"):
        print(f"  {h:9s} overall {metrics[f'{h}_acc']:.4f} | "
              f"sketch {metrics.get(f'sketch_{h}_acc', 0):.4f} | "
              f"render {metrics.get(f'render_{h}_acc', 0):.4f}")
    print(f"  mean {metrics['mean_acc']:.4f} (val_metrics refreshed in checkpoint)")
    return metrics


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output", default=str(CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--aug-copies", type=int, default=5,
                        help="Augmented copies per train image (0 disables augmentation).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; re-evaluate the existing checkpoint and "
                             "refresh its val_metrics.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli = parse_args()
    if cli.eval_only:
        evaluate_checkpoint(cli)
    else:
        train(cli)
