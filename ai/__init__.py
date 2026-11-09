"""
PackVision AI - AI/ML Engine Alias (`ai/`)
===========================================
Re-exports the core packaging synthesis and image processing pipeline components
from `src/` to provide a clean, modern top-level namespace.
"""

from src.preprocessor import PreprocessResult, SketchPreprocessor
from src.prompt_engine import (
    BACKGROUNDS,
    CAMERA_ANGLES,
    CATEGORIES,
    FINISHES,
    LIGHTING_PRESETS,
    SUBSTRATES,
    PromptEngine,
    PromptPackage,
    build_packaging_prompt,
)
from src.pipeline import (
    PackVisionRenderer,
    ProceduralPackagingRenderer,
    RenderResult,
)
from src.postprocessor import CompositorResult, LogoCompositor
from src.evaluation import (
    PackagingEvaluator,
    PackagingMetricResult,
    compute_edge_iou,
    compute_edge_ssim,
)

__all__ = [
    # Preprocessing
    "SketchPreprocessor",
    "PreprocessResult",
    # Prompt Engine & Taxonomy
    "PromptEngine",
    "PromptPackage",
    "build_packaging_prompt",
    "CATEGORIES",
    "SUBSTRATES",
    "FINISHES",
    "LIGHTING_PRESETS",
    "CAMERA_ANGLES",
    "BACKGROUNDS",
    # Pipeline & Renderer
    "PackVisionRenderer",
    "ProceduralPackagingRenderer",
    "RenderResult",
    # Postprocessing
    "LogoCompositor",
    "CompositorResult",
    # Evaluation
    "PackagingEvaluator",
    "PackagingMetricResult",
    "compute_edge_iou",
    "compute_edge_ssim",
]
