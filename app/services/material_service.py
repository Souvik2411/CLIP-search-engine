import numpy as np
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MaterialService:
    """Service for detecting materials in images using CLIP zero-shot."""

    # Comprehensive list of materials commonly found in architecture and interior design
    MATERIAL_LABELS = [
        # Wood types
        "wooden surface",
        "natural wood",
        "dark wood",
        "light wood",
        "oak wood",
        "walnut wood",
        "pine wood",
        "reclaimed wood",
        "polished wood",
        "rough wood",

        # Metals
        "metal surface",
        "stainless steel",
        "brushed metal",
        "polished metal",
        "brass",
        "copper",
        "bronze",
        "iron",
        "chrome",
        "aluminum",

        # Fabrics and textiles
        "fabric material",
        "linen fabric",
        "cotton fabric",
        "velvet fabric",
        "leather material",
        "genuine leather",
        "suede",
        "upholstery fabric",
        "woven textile",
        "silk fabric",

        # Stone and ceramic
        "stone surface",
        "marble",
        "granite",
        "limestone",
        "sandstone",
        "slate",
        "ceramic material",
        "porcelain",
        "terracotta",
        "concrete",

        # Glass
        "glass surface",
        "clear glass",
        "frosted glass",
        "tinted glass",
        "mirror glass",

        # Plastics and synthetics
        "plastic material",
        "acrylic",
        "resin",
        "laminate",

        # Other materials
        "brick",
        "plaster",
        "paint",
        "wallpaper",
        "tile",
        "bamboo",
        "rattan",
        "wicker",
        "cork",
        "rubber",
    ]

    # Simplified material categories for easier filtering
    MATERIAL_CATEGORIES = {
        "wood": ["wooden surface", "natural wood", "dark wood", "light wood", "polished wood"],
        "metal": ["metal surface", "stainless steel", "brushed metal", "brass", "copper"],
        "fabric": ["fabric material", "linen fabric", "cotton fabric", "velvet fabric", "upholstery fabric"],
        "leather": ["leather material", "genuine leather", "suede"],
        "stone": ["stone surface", "marble", "granite", "concrete"],
        "glass": ["glass surface", "clear glass", "mirror glass"],
        "ceramic": ["ceramic material", "porcelain", "tile"],
        "plastic": ["plastic material", "acrylic", "laminate"],
    }

    def __init__(self, clip_service):
        """
        Initialize material service.

        Args:
            clip_service: Instance of CLIPService for zero-shot classification
        """
        self.clip_service = clip_service

    def detect_materials(
        self,
        image: Image.Image,
        top_k: int = 5,
        threshold: float = 0.1
    ) -> list[tuple[str, float]]:
        """
        Detect materials in image using CLIP zero-shot classification.

        Args:
            image: PIL Image object
            top_k: Number of top materials to return
            threshold: Minimum confidence threshold (0-1)

        Returns:
            List of (material, confidence) tuples
        """
        # Get CLIP predictions for all material labels
        results = self.clip_service.get_image_labels(
            image=image,
            labels=self.MATERIAL_LABELS,
            top_k=len(self.MATERIAL_LABELS)
        )

        # Filter by threshold and get top-k
        filtered_results = [
            (label, score) for label, score in results
            if score >= threshold
        ][:top_k]

        return filtered_results

    def get_material_categories(
        self,
        image: Image.Image,
        threshold: float = 0.15
    ) -> dict[str, float]:
        """
        Get material categories with aggregated confidence scores.

        Args:
            image: PIL Image object
            threshold: Minimum confidence threshold

        Returns:
            Dictionary mapping category to max confidence score
        """
        # Detect all materials
        all_materials = self.detect_materials(
            image=image,
            top_k=20,
            threshold=0.05  # Lower threshold for aggregation
        )

        # Map to categories
        category_scores = {}

        for material, score in all_materials:
            # Find which category this material belongs to
            for category, labels in self.MATERIAL_CATEGORIES.items():
                if material in labels:
                    # Use max score for each category
                    if category not in category_scores:
                        category_scores[category] = score
                    else:
                        category_scores[category] = max(category_scores[category], score)

        # Filter by threshold
        filtered_categories = {
            cat: score for cat, score in category_scores.items()
            if score >= threshold
        }

        # Sort by confidence
        sorted_categories = dict(
            sorted(filtered_categories.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_categories

    def extract_material_features(
        self,
        image: Image.Image,
        top_k_materials: int = 5,
        top_k_categories: int = 3
    ) -> dict:
        """
        Extract comprehensive material features.

        Args:
            image: PIL Image object
            top_k_materials: Number of top materials to return
            top_k_categories: Number of top categories to return

        Returns:
            Dictionary with material features
        """
        # Get detailed materials
        materials = self.detect_materials(image, top_k=top_k_materials)

        # Get material categories
        categories = self.get_material_categories(image)

        # Get top categories
        top_categories = dict(list(categories.items())[:top_k_categories])

        # Get primary material
        primary_material = materials[0][0] if materials else "unknown"

        # Get primary category
        primary_category = list(top_categories.keys())[0] if top_categories else "unknown"

        return {
            "materials": [
                {"name": name, "confidence": float(score)}
                for name, score in materials
            ],
            "categories": {k: float(v) for k, v in top_categories.items()},
            "primary_material": primary_material,
            "primary_category": primary_category
        }


# Note: This service requires CLIPService instance, so it will be initialized in routes
# material_service = MaterialService(clip_service)
