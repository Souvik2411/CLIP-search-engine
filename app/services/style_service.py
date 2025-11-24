import numpy as np
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StyleService:
    """Service for classifying style and aesthetic features using CLIP zero-shot."""

    # Interior design and architectural styles
    STYLE_LABELS = [
        # Classic styles
        "modern interior design",
        "contemporary design",
        "minimalist design",
        "scandinavian design",
        "industrial design",
        "bohemian style",
        "rustic style",
        "farmhouse style",
        "mid-century modern",
        "art deco style",
        "traditional design",
        "transitional design",
        "coastal style",
        "mediterranean style",
        "french country style",

        # Modern variations
        "modern farmhouse",
        "modern rustic",
        "urban modern",
        "luxury modern",
        "eco-friendly design",

        # Specific characteristics
        "vintage decor",
        "retro design",
        "eclectic style",
        "glam style",
        "zen style",
        "tropical style",
        "asian-inspired design",
        "moroccan style",
        "shabby chic",
    ]

    # Room types and spaces
    SCENE_LABELS = [
        # Residential
        "living room",
        "bedroom",
        "kitchen",
        "dining room",
        "bathroom",
        "home office",
        "nursery",
        "kids room",
        "master bedroom",
        "guest room",
        "hallway",
        "entryway",
        "mudroom",
        "laundry room",
        "closet",
        "walk-in closet",

        # Special spaces
        "library",
        "home theater",
        "game room",
        "gym",
        "studio",
        "sunroom",
        "conservatory",
        "balcony",
        "patio",
        "terrace",
        "garden",

        # Commercial
        "office space",
        "reception area",
        "conference room",
        "lobby",
        "retail store",
        "restaurant interior",
        "cafe interior",
        "hotel room",
        "hotel lobby",
    ]

    # Ambiance and mood
    AMBIANCE_LABELS = [
        "cozy atmosphere",
        "elegant space",
        "spacious room",
        "bright and airy",
        "dark and moody",
        "warm and inviting",
        "cool and calm",
        "luxurious interior",
        "minimalist space",
        "cluttered space",
        "organized space",
        "colorful interior",
        "neutral palette",
        "natural lighting",
        "dramatic lighting",
    ]

    def __init__(self, clip_service):
        """
        Initialize style service.

        Args:
            clip_service: Instance of CLIPService for zero-shot classification
        """
        self.clip_service = clip_service

    def classify_style(
        self,
        image: Image.Image,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> list[tuple[str, float]]:
        """
        Classify interior/architectural style.

        Args:
            image: PIL Image object
            top_k: Number of top styles to return
            threshold: Minimum confidence threshold

        Returns:
            List of (style, confidence) tuples
        """
        results = self.clip_service.get_image_labels(
            image=image,
            labels=self.STYLE_LABELS,
            top_k=len(self.STYLE_LABELS)
        )

        # Filter by threshold and get top-k
        filtered_results = [
            (label, score) for label, score in results
            if score >= threshold
        ][:top_k]

        return filtered_results

    def detect_scene_type(
        self,
        image: Image.Image,
        top_k: int = 3,
        threshold: float = 0.1
    ) -> list[tuple[str, float]]:
        """
        Detect room/scene type.

        Args:
            image: PIL Image object
            top_k: Number of top scenes to return
            threshold: Minimum confidence threshold

        Returns:
            List of (scene, confidence) tuples
        """
        results = self.clip_service.get_image_labels(
            image=image,
            labels=self.SCENE_LABELS,
            top_k=len(self.SCENE_LABELS)
        )

        # Filter by threshold and get top-k
        filtered_results = [
            (label, score) for label, score in results
            if score >= threshold
        ][:top_k]

        return filtered_results

    def detect_ambiance(
        self,
        image: Image.Image,
        top_k: int = 5,
        threshold: float = 0.1
    ) -> list[tuple[str, float]]:
        """
        Detect ambiance and mood.

        Args:
            image: PIL Image object
            top_k: Number of top ambiance labels to return
            threshold: Minimum confidence threshold

        Returns:
            List of (ambiance, confidence) tuples
        """
        results = self.clip_service.get_image_labels(
            image=image,
            labels=self.AMBIANCE_LABELS,
            top_k=len(self.AMBIANCE_LABELS)
        )

        # Filter by threshold and get top-k
        filtered_results = [
            (label, score) for label, score in results
            if score >= threshold
        ][:top_k]

        return filtered_results

    def extract_style_features(
        self,
        image: Image.Image,
        top_k_styles: int = 3,
        top_k_scenes: int = 2,
        top_k_ambiance: int = 4
    ) -> dict:
        """
        Extract comprehensive style and scene features.

        Args:
            image: PIL Image object
            top_k_styles: Number of top styles to return
            top_k_scenes: Number of top scenes to return
            top_k_ambiance: Number of top ambiance labels to return

        Returns:
            Dictionary with all style features
        """
        # Classify style
        styles = self.classify_style(image, top_k=top_k_styles)

        # Detect scene type
        scenes = self.detect_scene_type(image, top_k=top_k_scenes)

        # Detect ambiance
        ambiance = self.detect_ambiance(image, top_k=top_k_ambiance)

        # Extract primary values
        primary_style = styles[0][0] if styles else "unknown"
        primary_scene = scenes[0][0] if scenes else "unknown"

        # Create simple tags for easy filtering
        style_tags = [s[0] for s in styles]
        scene_tags = [s[0] for s in scenes]
        ambiance_tags = [a[0] for a in ambiance]

        return {
            "styles": [
                {"name": name, "confidence": float(score)}
                for name, score in styles
            ],
            "scenes": [
                {"name": name, "confidence": float(score)}
                for name, score in scenes
            ],
            "ambiance": [
                {"name": name, "confidence": float(score)}
                for name, score in ambiance
            ],
            "primary_style": primary_style,
            "primary_scene": primary_scene,
            "style_tags": style_tags,
            "scene_tags": scene_tags,
            "ambiance_tags": ambiance_tags
        }


# Note: This service requires CLIPService instance, so it will be initialized in routes
# style_service = StyleService(clip_service)
