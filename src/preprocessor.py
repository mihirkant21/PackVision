"""
PackVision AI - Sketch Preprocessing Engine
============================================
Handles sketch normalization, adaptive Canny edge extraction, line dilation,
contrast enhancement for faint hand sketches, and pseudo-depth map estimation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class PreprocessResult:
    """Container for preprocessed sketch outputs and transformation metadata."""
    image: Image.Image
    canny: Image.Image
    depth: Image.Image
    mask: Image.Image
    original_size: Tuple[int, int]
    target_size: Tuple[int, int]
    scale: float
    padding: Tuple[int, int, int, int]  # top, bottom, left, right
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "original_size": self.original_size,
            "target_size": self.target_size,
            "scale": self.scale,
            "padding": self.padding,
            **self.metadata,
        }


class SketchPreprocessor:
    """
    Production-grade preprocessing engine for packaging sketches.
    
    Transforms raw user sketches (hand-drawn scans, vector line art, raster sketches)
    into high-fidelity condition inputs required by ControlNet and SDXL pipelines.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        sigma: float = 0.33,
        dilation_kernel_size: int = 2,
        auto_invert: bool = True,
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target (width, height) resolution, defaults to 1024x1024.
            low_threshold: Lower hysteresis threshold for Canny. If None, computed adaptively.
            high_threshold: Upper hysteresis threshold for Canny. If None, computed adaptively.
            sigma: Standard deviation factor for adaptive median-based Canny thresholds.
            dilation_kernel_size: Dilation kernel size to bridge broken pencil strokes (0 to disable).
            auto_invert: If True, automatically detects dark-on-light sketches and inverts them.
        """
        self.target_size = target_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sigma = sigma
        self.dilation_kernel_size = dilation_kernel_size
        self.auto_invert = auto_invert

    # -------------------------------------------------------------------------
    # Format and Color Space Converters
    # -------------------------------------------------------------------------

    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to OpenCV BGR or Grayscale numpy array.
        
        Handles RGBA, RGB, L, and 1-bit modes cleanly.
        """
        if not HAS_CV2:
            if image.mode == "RGBA":
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                canvas.paste(image, mask=image.split()[3])
                return np.array(canvas)
            elif image.mode == "RGB":
                return np.array(image)
            elif image.mode in ("L", "1"):
                return np.array(image.convert("L"))
            else:
                return np.array(image.convert("RGB"))

        if image.mode == "RGBA":
            # Composite onto white background before converting
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.split()[3])
            arr = np.array(canvas)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif image.mode == "RGB":
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif image.mode in ("L", "1"):
            return np.array(image.convert("L"))
        else:
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(image: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR or Grayscale numpy array to PIL Image."""
        if not HAS_CV2:
            if image.ndim == 2:
                return Image.fromarray(image.astype(np.uint8), mode="L")
            elif image.ndim == 3:
                return Image.fromarray(image.astype(np.uint8), mode="RGB")
            raise ValueError(f"Unsupported image array shape: {image.shape}")

        if image.ndim == 2:
            return Image.fromarray(image, mode="L")
        elif image.ndim == 3:
            if image.shape[2] == 4:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
            elif image.shape[2] == 3:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        raise ValueError(f"Unsupported image array shape: {image.shape}")

    @staticmethod
    def load_image(source: Union[str, Path, Image.Image, np.ndarray, bytes]) -> Image.Image:
        """Load an image from various input types into a PIL Image."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if not p.is_file():
                raise FileNotFoundError(f"Input sketch file does not exist: {p}")
            return Image.open(p).convert("RGB")
        elif isinstance(source, np.ndarray):
            return SketchPreprocessor.cv2_to_pil(source).convert("RGB")
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(source)}")

    # -------------------------------------------------------------------------
    # Aspect Ratio Normalization and Letterboxing
    # -------------------------------------------------------------------------

    def letterbox_image(
        self,
        image_cv: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        border_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """
        Resize image with preserved aspect ratio and smart letterboxing.
        
        Args:
            image_cv: Input image in OpenCV format (BGR or grayscale).
            target_size: Target (width, height), defaults to self.target_size.
            border_color: Color to fill borders (default: white).
            
        Returns:
            Tuple of (letterboxed_image, scale_factor, (top, bottom, left, right_padding))
        """
        target_w, target_h = target_size or self.target_size
        h, w = image_cv.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if not HAS_CV2:
            pil_img = Image.fromarray(image_cv)
            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            fill = border_color if pil_img.mode != "L" else border_color[0]
            canvas = Image.new(pil_img.mode, (target_w, target_h), fill)
            canvas.paste(resized, (left, top))
            return np.array(canvas), scale, (top, bottom, left, right)

        # Use Lanczos / Area interpolation for clean down/upscaling
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image_cv, (new_w, new_h), interpolation=interp)

        if image_cv.ndim == 2:
            fill_val = border_color[0] if isinstance(border_color, (list, tuple)) else border_color
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_val
            )
        else:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color
            )

        return padded, scale, (top, bottom, left, right)

    def unletterbox_image(
        self,
        image_cv: np.ndarray,
        original_size: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop and restore letterboxed image back to original dimensions."""
        top, bottom, left, right = padding
        h, w = image_cv.shape[:2]
        crop_h = h - bottom
        crop_w = w - right
        cropped = image_cv[top:crop_h, left:crop_w]
        orig_w, orig_h = original_size
        if not HAS_CV2:
            return np.array(Image.fromarray(cropped).resize((orig_w, orig_h), Image.Resampling.LANCZOS))
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    # -------------------------------------------------------------------------
    # Lighting and Contrast Enhancement for Faint Sketches
    # -------------------------------------------------------------------------

    @staticmethod
    def is_dark_background(gray_img: np.ndarray) -> bool:
        """
        Determine if the sketch has a dark background (white strokes on dark canvas).
        Returns True if dark background, False if paper-white background.
        """
        # Exclude borders to focus on actual canvas content
        h, w = gray_img.shape[:2]
        center_crop = gray_img[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]
        mean_val = np.mean(center_crop)
        return bool(mean_val < 127)

    def enhance_contrast(
        self,
        gray_img: np.ndarray,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Enhance faint pencil strokes using CLAHE and bilateral edge-preserving smoothing.
        
        Args:
            gray_img: Grayscale input image.
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.
            
        Returns:
            Enhanced contrast grayscale image with preserved line boundaries.
        """
        if not HAS_CV2:
            pil_img = Image.fromarray(gray_img)
            enhancer = ImageEnhance.Contrast(pil_img)
            return np.array(enhancer.enhance(1.4))

        # Bilateral filter reduces paper grain noise while retaining sharp pencil edges
        filtered = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=50)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(filtered)

        # Subtle unsharp masking for stroke crispness
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        unsharp = cv2.addWeighted(enhanced, 1.4, gaussian, -0.4, 0)
        return np.clip(unsharp, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Edge Extraction with Adaptive Thresholding and Line Dilation
    # -------------------------------------------------------------------------

    def extract_canny(
        self,
        gray_img: np.ndarray,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        dilation_kernel: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract Canny edges with adaptive Otsu/median thresholds and stroke bridging.
        
        ControlNet standard expects white edges (255) on a black background (0).
        """
        if not HAS_CV2:
            gy, gx = np.gradient(gray_img.astype(np.float32))
            mag = np.hypot(gx, gy)
            edges = (mag > 28).astype(np.uint8) * 255
            return edges

        # Compute adaptive thresholds if not explicitly provided
        if low_threshold is None or high_threshold is None:
            # Otsu's thresholding provides a robust bimodal luminance estimate
            otsu_val, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            med = np.median(gray_img)
            
            # Combine median sigma and Otsu for optimal packaging contour detection
            calc_low = int(max(10, (1.0 - self.sigma) * med))
            calc_high = int(min(245, max(otsu_val * 0.9, (1.0 + self.sigma) * med)))
            
            low = low_threshold if low_threshold is not None else calc_low
            high = high_threshold if high_threshold is not None else calc_high
        else:
            low = low_threshold
            high = high_threshold

        # Ensure high > low
        if high <= low:
            high = low + 50

        # Run Canny Edge Detector
        edges = cv2.Canny(gray_img, low, high)

        # Line dilation to connect broken pencil strokes
        d_ksize = dilation_kernel if dilation_kernel is not None else self.dilation_kernel_size
        if d_ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_ksize, d_ksize))
            edges = cv2.dilate(edges, kernel, iterations=1)
            # Morphological close bridges small micro-gaps without ballooning stroke width
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    # -------------------------------------------------------------------------
    # Packaging Silhouette Segmentation & Pseudo-Depth Map Estimation
    # -------------------------------------------------------------------------

    def extract_silhouette_mask(self, gray_img: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Segment the main packaging object silhouette from the background.
        
        Returns a binary mask (uint8) where 255 indicates the package interior.
        """
        h, w = gray_img.shape[:2]
        if not HAS_CV2:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[gray_img < 240] = 255
            return mask

        # Combine inverted threshold and edge closure
        if not self.is_dark_background(gray_img):
            # White background -> invert so package markings and body are non-zero
            _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresh = cv2.threshold(gray_img, 25, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(thresh, edges)

        # Close contours aggressively to form a solid packaging envelope
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # Find external contours and select the largest candidate
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            # Filter contours that occupy reasonable area (> 2% of frame)
            min_area = (h * w) * 0.02
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if valid_contours:
                # Fill convex hulls of main contours to solidify package faces
                for c in valid_contours:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)
            else:
                c_max = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [cv2.convexHull(c_max)], -1, 255, thickness=cv2.FILLED)
        else:
            # Fallback mask: central 80% rectangular region
            mask[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)] = 255

        return mask

    def extract_pseudo_depth(
        self,
        gray_img: np.ndarray,
        edges: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a photorealistic 3D pseudo-depth map from 2D sketch contours and gradients.
        
        Depth encoding convention:
        - 255 = Closest to camera (front package face, prominent edges)
        - 0 = Farthest from camera (background wall / infinite studio plane)
        
        Uses Euclidean Distance Transform combined with vertical studio perspective
        ramp and Sobel normal gradients for bevel highlights.
        """
        h, w = gray_img.shape[:2]
        if not HAS_CV2:
            depth = np.full((h, w, 3), 35, dtype=np.uint8)
            if mask is not None:
                depth[mask > 0] = 190
            return depth

        if mask is None:
            mask = self.extract_silhouette_mask(gray_img, edges)

        # 1. Interior Euclidean Distance Transform (convex package volume depth)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        if max_dist > 0:
            dist_norm = (dist_transform / max_dist) * 200.0  # Range 0-200
        else:
            dist_norm = np.zeros((h, w), dtype=np.float32)

        # 2. Studio Perspective Y-Ramp (studio packaging stands upright on pedestal floor)
        # Objects slightly lean toward the 3-point studio camera; bottom grounded
        y_coords = np.linspace(0.8, 1.2, h, dtype=np.float32)[:, None]
        y_gradient = np.tile(y_coords, (1, w))
        dist_norm = dist_norm * y_gradient
        dist_norm = np.clip(dist_norm, 0, 210)

        # 3. High-frequency structural crease depth (edges define panel folds and boundaries)
        sobel_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        edge_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        max_grad = edge_gradient.max()
        if max_grad > 0:
            edge_relief = (edge_gradient / max_grad) * 45.0
        else:
            edge_relief = np.zeros((h, w), dtype=np.float32)

        # 4. Composite Depth Map
        # Base ambient floor depth is 30; interior package ranges 60 - 255
        depth = np.full((h, w), 35.0, dtype=np.float32)
        inside_mask = mask > 0
        depth[inside_mask] = 55.0 + dist_norm[inside_mask] + edge_relief[inside_mask]

        # Smooth transitions
        depth = cv2.GaussianBlur(depth, (7, 7), sigmaX=2.0)
        depth = np.clip(depth, 0, 255).astype(np.uint8)

        # Return 3-channel depth representation for standard pipeline ingest
        return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

    # -------------------------------------------------------------------------
    # Main Preprocessing Entrypoint
    # -------------------------------------------------------------------------

    def preprocess(
        self,
        input_image: Union[str, Path, Image.Image, np.ndarray, bytes],
        invert_mode: str = "auto",  # 'auto', 'invert', 'keep'
        enhance_faint_strokes: bool = True,
    ) -> PreprocessResult:
        """
        Execute the complete preprocessing pipeline on an input sketch.
        
        Args:
            input_image: Filepath, PIL Image, or numpy array.
            invert_mode: 'auto' (detect light/dark), 'invert', or 'keep'.
            enhance_faint_strokes: Whether to apply CLAHE and unsharp filtering.
            
        Returns:
            PreprocessResult containing 1024x1024 normalized image, Canny edge map,
            pseudo-depth map, silhouette mask, and transformation coordinates.
        """
        pil_raw = self.load_image(input_image)
        orig_w, orig_h = pil_raw.size

        # Convert to OpenCV BGR
        cv_raw = self.pil_to_cv2(pil_raw)

        # Letterbox to target resolution (1024x1024)
        padded_bgr, scale, padding = self.letterbox_image(
            cv_raw, target_size=self.target_size, border_color=(255, 255, 255)
        )
        if not HAS_CV2:
            gray = padded_bgr if padded_bgr.ndim == 2 else np.mean(padded_bgr, axis=2).astype(np.uint8)
        else:
            gray = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2GRAY)

        # Determine inversion state
        is_dark = self.is_dark_background(gray)
        should_invert = False
        if invert_mode == "auto":
            should_invert = not is_dark  # Black strokes on white paper -> invert for ControlNet
        elif invert_mode == "invert":
            should_invert = True
        elif invert_mode == "keep":
            should_invert = False

        # Enhance faint pencil strokes
        working_gray = gray
        if enhance_faint_strokes:
            working_gray = self.enhance_contrast(gray)

        # Extract Canny edges (white edges on black background)
        canny_edges = self.extract_canny(working_gray)

        # Extract package silhouette mask
        mask = self.extract_silhouette_mask(working_gray, canny_edges)

        # Generate 3D pseudo-depth map
        depth_bgr = self.extract_pseudo_depth(working_gray, canny_edges, mask=mask)

        # Normalized conditioning image (high contrast line art)
        if should_invert:
            # ControlNet canny receives white lines on black background
            conditioning_arr = canny_edges
        else:
            conditioning_arr = cv2.bitwise_not(canny_edges)

        return PreprocessResult(
            image=self.cv2_to_pil(padded_bgr),
            canny=self.cv2_to_pil(canny_edges),
            depth=self.cv2_to_pil(depth_bgr),
            mask=self.cv2_to_pil(mask),
            original_size=(orig_w, orig_h),
            target_size=self.target_size,
            scale=scale,
            padding=padding,
            metadata={
                "is_dark_background": is_dark,
                "inverted": should_invert,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "dilation_kernel_size": self.dilation_kernel_size,
            },
        )
