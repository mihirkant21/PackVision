"""Input-mode detection: distinguishes hand-drawn sketches from real photos.

Used by the pipeline when ``input_mode="auto"`` so photos uploaded by the user
get a photo-appropriate cloud instruction (and skip ControlNet-style edge
preprocessing) while sketches keep the existing treatment.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

ImageLike = Union[Image.Image, np.ndarray]


def detect_input_mode(image: ImageLike) -> str:
    """Return "sketch" or "photo" for the given image.

    Heuristic (on a 256x256 thumbnail):
    - sketches: mostly near-white background, sparse dark strokes, low colour
    - photos: dense non-white coverage and/or meaningful colour saturation
    """
    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    thumb = img.convert("RGB").resize((256, 256))
    arr = np.asarray(thumb).astype(np.float32)

    gray = arr.mean(axis=2)
    near_white = float((gray > 235).mean())
    dark = float((gray < 100).mean())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())

    # Sketchy: lots of untouched white paper, few dark strokes, little colour.
    if near_white >= 0.55 and saturation < 45:
        return "sketch"
    # Heavily lined but still mostly paper (lineart with shading).
    if near_white >= 0.40 and dark < 0.25 and saturation < 60:
        return "sketch"
    return "photo"
