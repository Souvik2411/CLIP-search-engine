import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Optional
import logging

from app.config import get_settings
from app.utils.helpers import normalize_embedding

logger = logging.getLogger(__name__)


class CLIPService:
    """Service for CLIP model operations - embeddings and zero-shot labels."""

    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def load_model(self):
        """Load CLIP model and processor."""
        if self._loaded:
            return

        logger.info(f"Loading CLIP model: {self.settings.CLIP_MODEL_NAME}")
        logger.info(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained(self.settings.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(self.settings.CLIP_MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info("CLIP model loaded successfully")

    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for an image.

        Args:
            image: PIL Image object

        Returns:
            Normalized embedding vector
        """
        if not self._loaded:
            self.load_model()

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        embedding = image_features.cpu().numpy().flatten()
        return normalize_embedding(embedding)

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text string to embed

        Returns:
            Normalized embedding vector
        """
        if not self._loaded:
            self.load_model()

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        embedding = text_features.cpu().numpy().flatten()
        return normalize_embedding(embedding)

    def get_image_labels(
        self,
        image: Image.Image,
        labels: Optional[list[str]] = None,
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Perform zero-shot classification on image.

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

        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        probs = probs.cpu().numpy().flatten()

        # Get top-k labels
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
        if not self._loaded:
            self.load_model()

        if labels is None:
            labels = self.settings.ARCHITECTURE_LABELS

        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Process text labels
        text_inputs = self.processor(
            text=labels,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # Get image features
            image_features = self.model.get_image_features(**image_inputs)
            embedding = image_features.cpu().numpy().flatten()
            embedding = normalize_embedding(embedding)

            # Get text features for labels
            text_features = self.model.get_text_features(**text_inputs)

            # Calculate similarities
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features_norm @ text_features_norm.T).softmax(dim=-1)
            probs = similarities.cpu().numpy().flatten()

        # Get top-k labels
        top_indices = np.argsort(probs)[::-1][:top_k]
        label_results = [(labels[i], float(probs[i])) for i in top_indices]

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

        # Process image once
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        # Process architectural labels
        arch_text_inputs = self.processor(
            text=arch_labels,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        arch_text_inputs = {k: v.to(self.device) for k, v in arch_text_inputs.items()}

        # Process object labels
        object_text_inputs = self.processor(
            text=object_labels,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        object_text_inputs = {k: v.to(self.device) for k, v in object_text_inputs.items()}

        with torch.no_grad():
            # Get image features (for embedding)
            image_features = self.model.get_image_features(**image_inputs)
            embedding = image_features.cpu().numpy().flatten()
            embedding = normalize_embedding(embedding)

            # Normalize image features for similarity calculation
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

            # Get architectural label similarities
            arch_text_features = self.model.get_text_features(**arch_text_inputs)
            arch_text_features_norm = arch_text_features / arch_text_features.norm(dim=-1, keepdim=True)
            arch_similarities = (image_features_norm @ arch_text_features_norm.T).softmax(dim=-1)
            arch_probs = arch_similarities.cpu().numpy().flatten()

            # Get object label similarities
            object_text_features = self.model.get_text_features(**object_text_inputs)
            object_text_features_norm = object_text_features / object_text_features.norm(dim=-1, keepdim=True)
            object_similarities = (image_features_norm @ object_text_features_norm.T).softmax(dim=-1)
            object_probs = object_similarities.cpu().numpy().flatten()

        # Get top-k architectural labels
        arch_top_indices = np.argsort(arch_probs)[::-1][:top_k_arch]
        arch_results = [(arch_labels[i], float(arch_probs[i])) for i in arch_top_indices]

        # Get top-k object labels
        object_top_indices = np.argsort(object_probs)[::-1][:top_k_objects]
        object_results = [(object_labels[i], float(object_probs[i])) for i in object_top_indices]

        return embedding, arch_results, object_results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.settings.EMBEDDING_DIM