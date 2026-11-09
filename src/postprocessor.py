"""
PackVision AI - Logo & Artwork Compositing Engine (Stage 2)
===========================================================
Performs precision 4-point perspective homography warping of vector/raster logos
onto 3D packaging panels with CIELAB luminance blending, material grain integration,
multi-mode ink simulation (Normal, Multiply, Overlay, Hard Light), and edge sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from PIL import Image

from src.preprocessor import SketchPreprocessor


@dataclass
class CompositorResult:
    """Output container for logo compositing operation."""
    image: Image.Image
    warped_logo: Image.Image
    warped_mask: Image.Image
    destination_points: List[Tuple[float, float]]
    blend_mode: str
    opacity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> Tuple[int, int]:
        """Convenience property matching PIL image size."""
        return self.image.size


class LogoCompositor:
    """
    Stage 2 Hybrid Pipeline Compositor.
    
    Transforms 2D brand logos into 3D packaging face coordinates using projective
    homography, modulating ink values with underlying CIELAB surface luminance so
    ambient studio lighting, shadows, and substrate textures naturally show through.
    """

    SUPPORTED_BLEND_MODES = ("normal", "multiply", "overlay", "hard_light")

    def __init__(self) -> None:
        pass

    @staticmethod
    def _ensure_bgr_and_alpha(
        logo_img: Union[Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract BGR color channels and normalized float alpha mask from logo.
        """
        if isinstance(logo_img, Image.Image):
            logo_rgba = logo_img.convert("RGBA")
            arr = np.array(logo_rgba)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            alpha = arr[:, :, 3].astype(np.float32) / 255.0
            return bgr, alpha
        elif isinstance(logo_img, np.ndarray):
            if logo_img.ndim == 3 and logo_img.shape[2] == 4:
                bgr = logo_img[:, :, :3]
                alpha = logo_img[:, :, 3].astype(np.float32) / 255.0
                return bgr, alpha
            elif logo_img.ndim == 3 and logo_img.shape[2] == 3:
                # Opaque logo: estimate white/black background as transparent
                gray = cv2.cvtColor(logo_img, cv2.COLOR_BGR2GRAY)
                # Assume white background is transparent if corners are white
                if np.mean(gray[:5, :5]) > 240:
                    _, alpha_u8 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                else:
                    alpha_u8 = np.full(gray.shape, 255, dtype=np.uint8)
                return logo_img, alpha_u8.astype(np.float32) / 255.0
            else:
                raise ValueError("Logo array must have 3 or 4 channels.")
        else:
            raise TypeError(f"Unsupported logo image type: {type(logo_img)}")

    @staticmethod
    def estimate_package_face(
        package_image: Union[Image.Image, np.ndarray],
        panel_location: str = "center",
        inset_ratio: float = 0.22,
    ) -> List[Tuple[float, float]]:
        """
        Heuristic estimator providing default 4-point quadrilateral coordinates
        for the front panel of the package if manual UV coordinates are not supplied.
        
        Args:
            package_image: Rendered packaging image.
            panel_location: 'center', 'upper', or 'lower'.
            inset_ratio: Inset percentage from package edges.
            
        Returns:
            List of 4 points: [top-left, top-right, bottom-right, bottom-left].
        """
        if isinstance(package_image, Image.Image):
            w, h = package_image.size
        else:
            h, w = package_image.shape[:2]

        # Standard studio hero placement in central 50% bounding zone
        left = w * (0.5 - (0.5 - inset_ratio) * 0.7)
        right = w * (0.5 + (0.5 - inset_ratio) * 0.7)
        
        if panel_location == "upper":
            top = h * 0.28
            bottom = h * 0.50
        elif panel_location == "lower":
            top = h * 0.52
            bottom = h * 0.74
        else:  # center
            top = h * 0.35
            bottom = h * 0.62

        # Add subtle three-quarter perspective tilt
        tilt_y = (bottom - top) * 0.04
        return [
            (float(left), float(top + tilt_y)),
            (float(right), float(top - tilt_y)),
            (float(right), float(bottom - tilt_y)),
            (float(left), float(bottom + tilt_y)),
        ]

    def warp_logo_perspective(
        self,
        logo_bgr: np.ndarray,
        logo_alpha: np.ndarray,
        dst_points: List[Tuple[float, float]],
        target_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp flat logo onto arbitrary 3D quadrilateral using projective homography.
        
        Args:
            logo_bgr: Source logo BGR image.
            logo_alpha: Normalized float alpha mask [0.0, 1.0].
            dst_points: 4 destination points [(x0,y0), (x1,y1), (x2,y2), (x3,y3)].
            target_shape: (height, width) of output canvas.
            
        Returns:
            Tuple of (warped_bgr, warped_alpha).
        """
        lh, lw = logo_bgr.shape[:2]
        target_h, target_w = target_shape

        src_pts = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
        dst_pts = np.float32(dst_points)

        # Compute 3x3 homography matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp BGR with Lanczos / Cubic interpolation for vector edge sharpness
        warped_bgr = cv2.warpPerspective(
            logo_bgr,
            matrix,
            (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Warp Alpha mask
        warped_alpha = cv2.warpPerspective(
            logo_alpha,
            matrix,
            (target_w, target_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Anti-aliasing feather on warped alpha border to eliminate staircase edges
        warped_alpha = np.clip(warped_alpha, 0.0, 1.0)
        feathered_alpha = cv2.GaussianBlur(warped_alpha, (3, 3), sigmaX=0.8)
        # Preserve solid interior
        final_alpha = np.where(warped_alpha > 0.8, warped_alpha, feathered_alpha)

        return warped_bgr, final_alpha

    def blend_cielab_luminance(
        self,
        bg_bgr: np.ndarray,
        warped_logo_bgr: np.ndarray,
        warped_alpha: np.ndarray,
        texture_strength: float = 0.65,
    ) -> np.ndarray:
        """
        CIELAB Luminance Blending.
        
        Modulates the warped logo's luminance (L*) with the underlying substrate's
        ambient lighting, directional shadows, and micro-surface texture grain,
        ensuring that studio highlights and paper fibers naturally show through.
        """
        # Convert background and logo to CIELAB color space
        bg_lab = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        logo_lab = cv2.cvtColor(warped_logo_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        bg_l = bg_lab[:, :, 0]  # Background luminance [0, 255]
        logo_l = logo_lab[:, :, 0]  # Logo luminance [0, 255]

        # Decompose background L* into:
        # 1. Low-frequency shading (studio lights, soft gradients)
        low_freq_l = cv2.GaussianBlur(bg_l, (41, 41), sigmaX=15.0)
        # 2. High-frequency texture (paper fibers, substrate grain)
        high_freq_texture = bg_l - low_freq_l

        # Compute relative lighting ratio relative to local mean
        mean_l = np.mean(low_freq_l[warped_alpha > 0.1]) if np.any(warped_alpha > 0.1) else 180.0
        shading_ratio = np.clip(low_freq_l / max(mean_l, 1.0), 0.5, 1.5)

        # Modulate logo L* channel
        # Base luminance modulated by lighting ratio + high-frequency texture injection
        modulated_logo_l = logo_l * shading_ratio + (high_freq_texture * texture_strength)
        modulated_logo_l = np.clip(modulated_logo_l, 0.0, 255.0)

        # Recombine with original logo color chrominance (a*, b*)
        modulated_logo_lab = np.zeros_like(logo_lab)
        modulated_logo_lab[:, :, 0] = modulated_logo_l
        modulated_logo_lab[:, :, 1] = logo_lab[:, :, 1]
        modulated_logo_lab[:, :, 2] = logo_lab[:, :, 2]

        modulated_bgr = cv2.cvtColor(
            modulated_logo_lab.astype(np.uint8), cv2.COLOR_LAB2BGR
        )
        return modulated_bgr

    def apply_blend_mode(
        self,
        bg_bgr: np.ndarray,
        logo_bgr: np.ndarray,
        mode: str = "normal",
    ) -> np.ndarray:
        """
        Apply print ink simulation blend modes:
        - 'normal': standard physical ink film
        - 'multiply': dark ink absorbed into paper fibers (kraft/corrugated)
        - 'overlay': dynamic contrast blend
        - 'hard_light': intense specular brand color
        """
        bg_f = bg_bgr.astype(np.float32) / 255.0
        logo_f = logo_bgr.astype(np.float32) / 255.0

        if mode == "multiply":
            # Multiply: (A * B)
            blended = bg_f * logo_f
        elif mode == "overlay":
            # Overlay: 2 * A * B if A < 0.5 else 1 - 2 * (1 - A) * (1 - B)
            blended = np.where(
                bg_f < 0.5,
                2.0 * bg_f * logo_f,
                1.0 - 2.0 * (1.0 - bg_f) * (1.0 - logo_f),
            )
        elif mode == "hard_light":
            # Hard Light: overlay with roles reversed
            blended = np.where(
                logo_f < 0.5,
                2.0 * bg_f * logo_f,
                1.0 - 2.0 * (1.0 - bg_f) * (1.0 - logo_f),
            )
        else:  # 'normal'
            blended = logo_f

        return np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)

    def composite(
        self,
        package_image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
        logo_image: Optional[Union[Image.Image, np.ndarray, str, Path]] = None,
        destination_points: Optional[List[Tuple[float, float]]] = None,
        blend_mode: str = "normal",
        opacity: float = 0.95,
        preserve_texture_strength: float = 0.65,
        unsharp_strength: float = 0.35,
        **kwargs,
    ) -> CompositorResult:
        """
        Execute Stage 2 Logo Compositing onto rendered packaging.
        
        Args:
            package_image: Rendered 3D package (PIL Image, numpy array, or path).
            logo_image: 2D vector/raster brand logo with transparency.
            destination_points: 4 quadrilateral points on package face. If None, estimated.
            blend_mode: 'normal', 'multiply', 'overlay', or 'hard_light'.
            opacity: Logo layer opacity [0.0, 1.0].
            preserve_texture_strength: Influence of underlying substrate grain [0.0, 1.0].
            unsharp_strength: Edge sharpening factor for vector crispness.
            **kwargs: Flexible keyword aliases (rendered_image, logo_path, target_box, etc.).
            
        Returns:
            CompositorResult containing final composite, warped logo, and metadata.
        """
        # Resolve flexible keyword aliases
        if package_image is None:
            package_image = kwargs.get("rendered_image") or kwargs.get("rendered") or kwargs.get("image")
        if logo_image is None:
            logo_image = kwargs.get("logo") or kwargs.get("logo_path")
        if destination_points is None:
            box = kwargs.get("target_box") or kwargs.get("box")
            if box and len(box) == 4 and isinstance(box[0], (int, float)):
                x1, y1, x2, y2 = box
                destination_points = [(float(x1), float(y1)), (float(x2), float(y1)), (float(x2), float(y2)), (float(x1), float(y2))]
            elif "placement" in kwargs:
                destination_points = kwargs["placement"]

        # Load and normalize package image
        bg_pil = SketchPreprocessor.load_image(package_image)
        bg_bgr = SketchPreprocessor.pil_to_cv2(bg_pil)
        h, w = bg_bgr.shape[:2]

        # Load logo image
        if isinstance(logo_image, (str, Path)):
            logo_pil = Image.open(logo_image)
        elif isinstance(logo_image, Image.Image):
            logo_pil = logo_image
        else:
            logo_pil = SketchPreprocessor.cv2_to_pil(logo_image)

        logo_bgr, logo_alpha = self._ensure_bgr_and_alpha(logo_pil)

        # Default destination points if not provided
        if destination_points is None or len(destination_points) != 4:
            destination_points = self.estimate_package_face(bg_pil)

        # 1. Perspective Homography Warping
        warped_bgr, warped_alpha = self.warp_logo_perspective(
            logo_bgr, logo_alpha, destination_points, target_shape=(h, w)
        )

        # 2. CIELAB Luminance & Texture Blending
        modulated_logo_bgr = self.blend_cielab_luminance(
            bg_bgr, warped_bgr, warped_alpha, texture_strength=preserve_texture_strength
        )

        # 3. Apply Selected Blend Mode
        blended_bgr = self.apply_blend_mode(bg_bgr, modulated_logo_bgr, mode=blend_mode)

        # 4. Alpha Composition
        alpha_effective = (warped_alpha * opacity)[:, :, None]
        composite_bgr = np.clip(
            blended_bgr.astype(np.float32) * alpha_effective
            + bg_bgr.astype(np.float32) * (1.0 - alpha_effective),
            0.0,
            255.0,
        ).astype(np.uint8)

        # 5. High-Frequency Edge Sharpening (Unsharp Mask on Composited Zone)
        if unsharp_strength > 0:
            blurred = cv2.GaussianBlur(composite_bgr, (0, 0), sigmaX=1.2)
            sharpened = cv2.addWeighted(
                composite_bgr, 1.0 + unsharp_strength, blurred, -unsharp_strength, 0
            )
            # Confine sharpening to logo footprint to avoid amplifying background noise
            sharpen_mask = cv2.dilate(warped_alpha, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            composite_bgr = np.where(
                sharpen_mask[:, :, None] > 0.1, sharpened, composite_bgr
            )

        composite_pil = SketchPreprocessor.cv2_to_pil(composite_bgr)
        warped_logo_pil = SketchPreprocessor.cv2_to_pil(warped_bgr)
        warped_mask_pil = Image.fromarray((warped_alpha * 255.0).astype(np.uint8), mode="L")

        return CompositorResult(
            image=composite_pil,
            warped_logo=warped_logo_pil,
            warped_mask=warped_mask_pil,
            destination_points=destination_points,
            blend_mode=blend_mode,
            opacity=opacity,
            metadata={
                "preserve_texture_strength": preserve_texture_strength,
                "unsharp_strength": unsharp_strength,
            },
        )
