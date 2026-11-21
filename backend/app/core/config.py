from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    PROJECT_NAME: str = "PackVision AI"
    APP_ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # Database & Storage
    DATABASE_URL: str = "sqlite:///./packvision.db"
    STORAGE_PATH: str = "./storage"
    MAX_UPLOAD_MB: int = 25

    # Inference Configuration
    USE_MOCK_INFERENCE: bool = True
    MODEL_DEVICE: str = "auto"
    SDXL_BASE_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"
    CONTROLNET_MODEL: str = "diffusers/controlnet-canny-sdxl-1.0"
    VAE_MODEL: str = "madebyollin/sdxl-vae-fp16-fix"
    LORA_WEIGHTS_PATH: str = ""

    # Hybrid Render Engine (cloud-first rendering with local fallback)
    RENDER_ENGINE: str = "auto"  # "cloud" | "local" | "auto" (cloud first, local fallback)
    RENDER_ENGINE_CLOUD_PROVIDER: str = "auto"  # "gemini" | "openrouter" | "openai" | "auto"
    CLOUD_IMAGE_PROVIDER: str = "auto"  # alias of RENDER_ENGINE_CLOUD_PROVIDER
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_IMAGE_MODEL: str = "google/gemini-2.5-flash-image"
    OPENAI_IMAGE_MODEL: str = "gpt-image-1"
    GEMINI_IMAGE_MODEL: str = "gemini-2.5-flash-image"
    GEMINI_VLM_MODEL: str = "gemini-3.8-flash"
    OPENROUTER_VLM_MODEL: str = "nvidia/nemotron-3.5-super-vl:free"
    OPENAI_VLM_MODEL: str = "gpt-4o-mini"
    VLM_PROVIDER: str = "auto"  # "gemini" | "openrouter" | "openai" | "auto"

    # Image Validation
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    ]
    MIN_IMAGE_DIMENSION: int = 128
    MAX_IMAGE_DIMENSION: int = 4096

    @field_validator("ALLOWED_IMAGE_TYPES", mode="before")
    @classmethod
    def parse_allowed_image_types(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def base_storage_dir(self) -> Path:
        return Path(self.STORAGE_PATH).resolve()

    @property
    def inputs_dir(self) -> Path:
        return self.base_storage_dir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.base_storage_dir / "outputs"

    @property
    def thumbnails_dir(self) -> Path:
        return self.base_storage_dir / "thumbnails"

    def ensure_storage_dirs(self) -> None:
        """Create storage directories if they do not exist."""
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
# Ensure directories are created on module import / startup
settings.ensure_storage_dirs()
