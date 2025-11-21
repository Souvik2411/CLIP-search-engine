import numpy as np
from PIL import Image
from io import BytesIO
from typing import Union
import base64


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def fuse_embeddings(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    image_weight: float = 0.7
) -> np.ndarray:
    """
    Fuse image and text embeddings using weighted average.

    Args:
        image_embedding: Normalized image embedding
        text_embedding: Normalized text embedding
        image_weight: Weight for image embedding (0-1)

    Returns:
        Fused and normalized embedding
    """
    text_weight = 1.0 - image_weight
    fused = image_weight * image_embedding + text_weight * text_embedding
    return normalize_embedding(fused)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_image_from_base64(base64_string: str) -> Image.Image:
    """Load PIL Image from base64 string."""
    image_bytes = base64.b64decode(base64_string)
    return load_image_from_bytes(image_bytes)


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def batch_list(items: list, batch_size: int):
    """Yield successive batches from a list."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]