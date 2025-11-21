import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings."""

    # Application
    APP_NAME: str = "ARCHINZA Search Pipeline"
    DEBUG: bool = False

    # CLIP Model
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    EMBEDDING_DIM: int = 512

    # ONNX Optimization (2-3x faster CPU inference)
    USE_ONNX: bool = False
    ONNX_VISION_MODEL_PATH: str = "models/onnx/clip_vision.onnx"
    ONNX_TEXT_MODEL_PATH: str = "models/onnx/clip_text.onnx"

    # FAISS
    FAISS_INDEX_PATH: str = "data/index/faiss.index"
    FAISS_METADATA_PATH: str = "data/index/metadata.json"

    # S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""
    S3_URL_EXPIRATION: int = 3600  # 1 hour

    # OpenAI
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_MAX_TOKENS: int = 500

    # Search
    TOP_K_RESULTS: int = 10
    IMAGE_TEXT_FUSION_WEIGHT: float = 0.7  # Weight for image embedding in fusion

    # Architecture labels for CLIP zero-shot
    ARCHITECTURE_LABELS: list[str] = [
        "modern architecture",
        "classical architecture",
        "minimalist design",
        "brutalist architecture",
        "art deco",
        "gothic architecture",
        "contemporary design",
        "sustainable architecture",
        "industrial design",
        "residential building",
        "commercial building",
        "interior design",
        "landscape architecture",
        "urban planning",
        "facade design",
        "structural design",
    ]

    # Furniture and object labels for interior detection
    FURNITURE_OBJECT_LABELS: list[str] = [
        # Seating
        "tuxedo sofa",
        "sectional sofa",
        "chesterfield sofa",
        "loveseat",
        "armchair",
        "dining chair",
        "wingback chair",
        "accent chair",
        "lounge chair",
        "bar stool",
        "counter stool",
        "ottoman",
        "bench",
        # Tables
        "coffee table",
        "dining table",
        "side table",
        "console table",
        "desk",
        "nightstand",
        "end table",
        # Storage
        "bookshelf",
        "cabinet",
        "wardrobe",
        "dresser",
        "sideboard",
        "credenza",
        "shelving unit",
        # Beds
        "bed",
        "platform bed",
        "canopy bed",
        # Lighting
        "chandelier",
        "pendant light",
        "floor lamp",
        "table lamp",
        "wall sconce",
        "ceiling light",
        # Decorative
        "area rug",
        "carpet",
        "mirror",
        "artwork",
        "wall art",
        "decorative pillow",
        "throw blanket",
        "vase",
        "sculpture",
        "indoor plant",
        "potted plant",
        "curtains",
        "drapes",
        # Kitchen & Dining
        "kitchen island",
        "bar cart",
        "wine rack",
        # Architectural Elements
        "fireplace",
        "built-in shelving",
        "window",
        "door",
        "archway",
        "column",
        "staircase",
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Create data directories on import
def init_directories():
    """Initialize required directories."""
    base_path = Path(__file__).parent.parent
    data_index_path = base_path / "data" / "index"
    data_index_path.mkdir(parents=True, exist_ok=True)


init_directories()