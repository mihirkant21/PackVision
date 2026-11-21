"""
PackVision AI - Local Packaging Configuration Classifier
========================================================
Small multi-task CNN (~1.8M params) that predicts packaging configuration
(category / material / finish) directly from an image, entirely offline on
CPU. It replaces the flaky free vision API for auto-detecting the packaging
config when a user uploads a picture.

Design notes
------------
* Trained on BOTH domains the pipeline sees: designer sketches (grayscale
  lineart) and photorealistic renders, each treated as an independent training
  sample carrying the same labels. Category (silhouette geometry) is learnable
  from both domains; material / finish are only visible in renders (sketches
  contain no texture signal), so on pure sketch inputs the material / finish
  heads fall back to the dataset priors -- expected and honest behaviour.
* Heads are sized over the FULL canonical taxonomy (10 package types, 12
  materials, 11 finishes) so the label space matches ``src.taxonomy`` exactly,
  including classes absent from the synthetic dataset (sleeve, mailer_box,
  flat_pouch, molded_pulp, ...). Dataset legacy keys (carton, corrugated_box,
  holographic, ...) are mapped to canonical keys via ``taxonomy.normalize``
  at training time, so predictions are canonical by construction.
* ``predict_config(image)`` returns the canonical key + softmax confidence for
  each attribute plus the detected input mode (reuses
  ``src.photo_mode.detect_input_mode``).

Checkpoint format (``models/checkpoints/config_classifier.pt``) is a plain
torch.save dict, mirroring the existing ``packvision_custom_v1.pt`` style::

    {
        "version": 1,
        "kind": "config_classifier",
        "epoch": <best epoch int>,
        "model_state_dict": {...},
        "model_config": {"input_size", "num_categories", "num_materials",
                         "num_finishes", "base_channels", "dropout"},
        "category_map":   {canonical_key: index, ...},
        "material_map":   {canonical_key: index, ...},
        "finish_map":     {canonical_key: index, ...},
        "val_metrics":    {"category_acc", "material_acc", "finish_acc",
                           "mean_acc", "epoch"},
        "training_meta":  {...dataset sizes / wall time...},
    }
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.photo_mode import detect_input_mode
from src.taxonomy import all_keys as taxonomy_all_keys
from src.taxonomy import normalize as taxonomy_normalize

__all__ = [
    "ConfigClassifierCNN",
    "build_model",
    "load_checkpoint",
    "preprocess_image",
    "predict_config",
    "CHECKPOINT_PATH",
    "INPUT_SIZE",
]

#: default checkpoint location
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "config_classifier.pt"

#: square input side the network was trained on
INPUT_SIZE = 128

#: canonical label spaces (single source of truth: src.taxonomy)
CATEGORY_KEYS = taxonomy_all_keys("package_types")
MATERIAL_KEYS = taxonomy_all_keys("materials")
FINISH_KEYS = taxonomy_all_keys("finishes")


# =============================================================================
# MODEL
# =============================================================================

def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Conv3x3 -> BatchNorm -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConfigClassifierCNN(nn.Module):
    """Small multi-task CNN backbone with three linear classification heads.

    Input: (B, 3, 128, 128) float tensor, normalized to [-1, 1].
    Output: dict of logits {"category", "material", "finish"}.
    """

    def __init__(
        self,
        num_categories: int = len(CATEGORY_KEYS),
        num_materials: int = len(MATERIAL_KEYS),
        num_finishes: int = len(FINISH_KEYS),
        base_channels: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        c = base_channels  # 32
        self.features = nn.Sequential(
            _conv_block(3, c),                # 128 x 128
            _conv_block(c, c * 2, stride=2),  # 64 x 64
            _conv_block(c * 2, c * 2),        # 64 x 64
            _conv_block(c * 2, c * 4, stride=2),  # 32 x 32
            _conv_block(c * 4, c * 4),        # 32 x 32
            _conv_block(c * 4, c * 8, stride=2),  # 16 x 16
            _conv_block(c * 8, c * 8),        # 16 x 16
            _conv_block(c * 8, c * 8, stride=2),  # 8 x 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = c * 8
        self.drop = nn.Dropout(dropout)
        self.category_head = nn.Linear(feat_dim, num_categories)
        self.material_head = nn.Linear(feat_dim, num_materials)
        self.finish_head = nn.Linear(feat_dim, num_finishes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.features(x)
        f = self.pool(f).flatten(1)
        f = self.drop(f)
        return {
            "category": self.category_head(f),
            "material": self.material_head(f),
            "finish": self.finish_head(f),
        }


def build_model(model_config: Optional[Dict[str, Any]] = None) -> ConfigClassifierCNN:
    """Build a model, optionally sized from a stored ``model_config`` dict."""
    cfg = model_config or {}
    return ConfigClassifierCNN(
        num_categories=cfg.get("num_categories", len(CATEGORY_KEYS)),
        num_materials=cfg.get("num_materials", len(MATERIAL_KEYS)),
        num_finishes=cfg.get("num_finishes", len(FINISH_KEYS)),
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.15),
    )


# =============================================================================
# CHECKPOINT LOADING (lazy, cached, thread-safe)
# =============================================================================

_BUNDLE: Optional[Dict[str, Any]] = None
_BUNDLE_LOCK = threading.Lock()


def load_checkpoint(path: Union[str, Path] = CHECKPOINT_PATH, force_reload: bool = False) -> Dict[str, Any]:
    """Load (and cache) the classifier checkpoint.

    Returns a bundle dict: {"model": eval-mode nn.Module, "checkpoint": dict,
    "index_to_category"/"index_to_material"/"index_to_finish": label lists}.
    """
    global _BUNDLE
    path = Path(path)
    if _BUNDLE is None or force_reload or _BUNDLE.get("path") != path:
        with _BUNDLE_LOCK:
            if _BUNDLE is None or force_reload or _BUNDLE.get("path") != path:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Config classifier checkpoint not found at {path}. "
                        f"Train it first: python scripts/train_config_classifier.py"
                    )
                ck = torch.load(path, map_location="cpu", weights_only=False)
                model = build_model(ck.get("model_config"))
                model.load_state_dict(ck["model_state_dict"], strict=True)
                model.eval()
                cat_map = ck.get("category_map", {k: i for i, k in enumerate(CATEGORY_KEYS)})
                mat_map = ck.get("material_map", {k: i for i, k in enumerate(MATERIAL_KEYS)})
                fin_map = ck.get("finish_map", {k: i for i, k in enumerate(FINISH_KEYS)})
                _BUNDLE = {
                    "path": path,
                    "model": model,
                    "checkpoint": ck,
                    "index_to_category": [None] * len(cat_map),
                    "index_to_material": [None] * len(mat_map),
                    "index_to_finish": [None] * len(fin_map),
                }
                for name, idx in cat_map.items():
                    _BUNDLE["index_to_category"][int(idx)] = name
                for name, idx in mat_map.items():
                    _BUNDLE["index_to_material"][int(idx)] = name
                for name, idx in fin_map.items():
                    _BUNDLE["index_to_finish"][int(idx)] = name
    return _BUNDLE


# =============================================================================
# INFERENCE
# =============================================================================

def preprocess_image(image: Image.Image, input_size: int = INPUT_SIZE) -> torch.Tensor:
    """PIL Image -> (1, 3, H, W) float tensor normalized to [-1, 1].

    Center-crops to square (ImageOps.fit) then resizes -- identical treatment
    for square inputs and sensible behaviour for arbitrary uploads.
    """
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    img = img.convert("RGB")
    if img.size != (input_size, input_size):
        img = ImageOps.fit(img, (input_size, input_size), method=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # same normalization used at training time
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _canonical(key: Optional[str], category: str) -> Optional[str]:
    """Defensive canonicalization: taxonomy.normalize on a canonical key is identity."""
    if key is None:
        return None
    return taxonomy_normalize(key, category) or key


def predict_config(
    image: Union[Image.Image, np.ndarray],
    checkpoint_path: Union[str, Path] = CHECKPOINT_PATH,
) -> Dict[str, Any]:
    """Predict packaging configuration from an image. Instant, offline, CPU.

    Args:
        image: PIL Image (or numpy RGB array) of a packaging sketch or photo.
        checkpoint_path: override checkpoint location (defaults to the
            trained ``models/checkpoints/config_classifier.pt``).

    Returns:
        dict with canonical taxonomy keys (``src.taxonomy``) and confidences::

            {
              "category": "folding_carton",       "category_confidence": 0.94,
              "material": "kraft_cardboard",      "material_confidence": 0.31,
              "finish":   "gold_foil",            "finish_confidence": 0.27,
              "input_mode": "sketch",             # via src.photo_mode
              "probabilities": {
                  "category": {key: prob, ...},
                  "material": {key: prob, ...},
                  "finish":   {key: prob, ...},
              },
            }
    """
    bundle = load_checkpoint(checkpoint_path)
    model = bundle["model"]

    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    input_mode = detect_input_mode(pil_image)

    x = preprocess_image(pil_image)
    with torch.no_grad():
        logits = model(x)
        probs = {head: F.softmax(logits[head], dim=1)[0] for head in logits}

    def top(head: str, labels):
        p = probs[head]
        idx = int(torch.argmax(p).item())
        conf = float(p[idx].item())
        return _canonical(labels[idx], head), conf

    category, cat_conf = top("category", bundle["index_to_category"])
    material, mat_conf = top("material", bundle["index_to_material"])
    finish, fin_conf = top("finish", bundle["index_to_finish"])

    def dist(head: str, labels):
        p = probs[head]
        return {
            _canonical(labels[i], head): float(p[i].item())
            for i in range(len(labels))
        }

    return {
        "category": category,
        "category_confidence": round(cat_conf, 4),
        "material": material,
        "material_confidence": round(mat_conf, 4),
        "finish": finish,
        "finish_confidence": round(fin_conf, 4),
        "input_mode": input_mode,
        "probabilities": {
            "category": dist("category", bundle["index_to_category"]),
            "material": dist("material", bundle["index_to_material"]),
            "finish": dist("finish", bundle["index_to_finish"]),
        },
    }
