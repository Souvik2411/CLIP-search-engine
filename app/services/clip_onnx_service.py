"""ONNX-optimized CLIP service for faster CPU inference."""
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from typing import Optional
import logging
import onnxruntime as ort
from pathlib import Path

from app.config import get_settings
from app.utils.helpers import normalize_embedding

logger = logging.getLogger(__name__)


class CLIPONNXService:
    """Service for CLIP model operations using ONNX Runtime for faster CPU inference."""

    def __init__(self):
        self.settings = get_settings()
        self.processor: Optional[CLIPProcessor] = None
        self.vision_session: Optional[ort.InferenceSession] = None
        self.text_session: Optional[ort.InferenceSession] = None
        self._loaded = False

    def load_model(self):
        """Load CLIP ONNX models and processor."""
        if self._loaded:
            return

        logger.info("Loading CLIP ONNX models...")

        # Load processor (for preprocessing)
        self.processor = CLIPProcessor.from_pretrained(self.settings.CLIP_MODEL_NAME)

        # Load ONNX models
        vision_path = Path(self.settings.ONNX_VISION_MODEL_PATH)
        text_path = Path(self.settings.ONNX_TEXT_MODEL_PATH)

        if not vision_path.exists() or not text_path.exists():
            raise FileNotFoundError(
                f"ONNX models not found. Please run: python scripts/convert_to_onnx.py\n"
                f"Vision model: {vision_path}\n"
                f"Text model: {text_path}"
            )

        # Create ONNX sessions with CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Adjust based on your CPU
        sess_options.inter_op_num_threads = 4

        self.vision_session = ort.InferenceSession(
            str(vision_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        self.text_session = ort.InferenceSession(
            str(text_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )

        self._loaded = True
        logger.info("CLIP ONNX models loaded successfully")
        logger.info(f"Vision model: {vision_path}")
        logger.info(f"Text model: {text_path}")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for an image using ONNX model.

        Args:
            image: PIL Image object

        Returns:
            Normalized embedding vector
        """
        if not self._loaded:
            self.load_model()

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="np")
        pixel_values = inputs['pixel_values']

        # Run ONNX inference
        outputs = self.vision_session.run(
            None,
            {'pixel_values': pixel_values}
        )

        embedding = outputs[0].flatten()
        return normalize_embedding(embedding)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using ONNX model.

        Args:
            text: Text string to embed

        Returns:
            Normalized embedding vector
        """
        if not self._loaded:
            self.load_model()

        # Preprocess text
        inputs = self.processor(text=[text], return_tensors="np", padding=True, truncation=True)

        # Run ONNX inference
        outputs = self.text_session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )

        embedding = outputs[0].flatten()
        return normalize_embedding(embedding)

    def get_image_labels(
        self,
        image: Image.Image,
        labels: Optional[list[str]] = None,
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Perform zero-shot classification on image using ONNX models.

        Args:
            image: PIL Image object
            labels: List of candidate labels (uses default architecture labels if None)
            top_k: Number of top labels to return

        Returns:
            List of (label, probability) tuples
        """
        if not self._loaded:
            self.load_model()

        if labels is None:
            labels = self.settings.ARCHITECTURE_LABELS

        # Get image embedding
        image_embedding = self.get_image_embedding(image)

        # Get text embeddings for all labels
        text_embeddings = []
        for label in labels:
            text_emb = self.get_text_embedding(label)
            text_embeddings.append(text_emb)

        text_embeddings = np.array(text_embeddings)

        # Calculate similarities (cosine similarity with normalized vectors = dot product)
        similarities = np.dot(text_embeddings, image_embedding)

        # Apply softmax
        exp_sims = np.exp(similarities - np.max(similarities))
        probs = exp_sims / exp_sims.sum()

        # Get top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [(labels[i], float(probs[i])) for i in top_indices]

        return results

    def get_image_embedding_and_labels(
        self,
        image: Image.Image,
        labels: Optional[list[str]] = None,
        top_k: int = 5
    ) -> tuple[np.ndarray, list[tuple[str, float]]]:
        """
        Get both embedding and labels in a single call for efficiency.

        Args:
            image: PIL Image object
            labels: List of candidate labels
            top_k: Number of top labels to return

        Returns:
            Tuple of (embedding, [(label, probability), ...])
        """
        if labels is None:
            labels = self.settings.ARCHITECTURE_LABELS

        # Get image embedding
        embedding = self.get_image_embedding(image)

        # Get labels using the embedding
        label_results = self.get_image_labels(image, labels, top_k)

        return embedding, label_results

    def get_image_embedding_and_dual_labels(
        self,
        image: Image.Image,
        top_k_arch: int = 5,
        top_k_objects: int = 7
    ) -> tuple[np.ndarray, list[tuple[str, float]], list[tuple[str, float]]]:
        """
        Get embedding, architectural labels, and object labels in a single efficient call.

        Args:
            image: PIL Image object
            top_k_arch: Number of top architectural labels to return
            top_k_objects: Number of top object labels to return

        Returns:
            Tuple of (embedding, [(arch_label, probability), ...], [(object_label, probability), ...])
        """
        if not self._loaded:
            self.load_model()

        arch_labels = self.settings.ARCHITECTURE_LABELS
        object_labels = self.settings.FURNITURE_OBJECT_LABELS

        # Get image embedding once
        embedding = self.get_image_embedding(image)

        # Get architectural labels
        arch_results = self.get_image_labels(image, arch_labels, top_k_arch)

        # Get object labels
        object_results = self.get_image_labels(image, object_labels, top_k_objects)

        return embedding, arch_results, object_results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.settings.EMBEDDING_DIM
