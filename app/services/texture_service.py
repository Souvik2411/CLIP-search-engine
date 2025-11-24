import cv2
import numpy as np
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TextureService:
    """Service for analyzing texture features from images using OpenCV."""

    def __init__(self):
        pass

    def compute_glcm_features(self, gray_image: np.ndarray, distance: int = 1) -> dict:
        """
        Compute Gray-Level Co-occurrence Matrix (GLCM) texture features.

        This is a simplified version that computes basic texture properties.

        Args:
            gray_image: Grayscale image as numpy array
            distance: Pixel pair distance for GLCM computation

        Returns:
            Dictionary with texture features
        """
        # Reduce image size for faster computation
        if gray_image.shape[0] > 512 or gray_image.shape[1] > 512:
            scale = 512 / max(gray_image.shape)
            new_width = int(gray_image.shape[1] * scale)
            new_height = int(gray_image.shape[0] * scale)
            gray_image = cv2.resize(gray_image, (new_width, new_height))

        # Reduce number of gray levels to 32 for efficiency
        levels = 32
        gray_image = (gray_image / 256.0 * levels).astype(np.uint8)

        # Initialize GLCM matrix
        glcm = np.zeros((levels, levels), dtype=np.float32)

        # Compute GLCM for horizontal direction (0 degrees)
        rows, cols = gray_image.shape
        for i in range(rows):
            for j in range(cols - distance):
                i_val = gray_image[i, j]
                j_val = gray_image[i, j + distance]
                glcm[i_val, j_val] += 1

        # Normalize GLCM
        glcm = glcm / np.sum(glcm)

        # Calculate texture properties
        # Contrast: Measure of local variations
        contrast = 0
        for i in range(levels):
            for j in range(levels):
                contrast += glcm[i, j] * (i - j) ** 2

        # Homogeneity: Measure of closeness of distribution
        homogeneity = 0
        for i in range(levels):
            for j in range(levels):
                homogeneity += glcm[i, j] / (1 + abs(i - j))

        # Energy: Measure of uniformity
        energy = np.sum(glcm ** 2)

        # Correlation: Measure of linear dependency
        mean_i = np.sum(np.arange(levels)[:, np.newaxis] * glcm)
        mean_j = np.sum(np.arange(levels)[np.newaxis, :] * glcm)
        std_i = np.sqrt(np.sum((np.arange(levels)[:, np.newaxis] - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((np.arange(levels)[np.newaxis, :] - mean_j) ** 2 * glcm))

        correlation = 0
        if std_i > 0 and std_j > 0:
            for i in range(levels):
                for j in range(levels):
                    correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)

        return {
            "contrast": float(contrast),
            "homogeneity": float(homogeneity),
            "energy": float(energy),
            "correlation": float(correlation)
        }

    def compute_lbp_features(self, gray_image: np.ndarray, radius: int = 1, n_points: int = 8) -> dict:
        """
        Compute Local Binary Pattern (LBP) features.

        Args:
            gray_image: Grayscale image as numpy array
            radius: Radius of circular LBP
            n_points: Number of sampling points

        Returns:
            Dictionary with LBP statistics
        """
        # Resize if too large
        if gray_image.shape[0] > 512 or gray_image.shape[1] > 512:
            scale = 512 / max(gray_image.shape)
            new_width = int(gray_image.shape[1] * scale)
            new_height = int(gray_image.shape[0] * scale)
            gray_image = cv2.resize(gray_image, (new_width, new_height))

        rows, cols = gray_image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)

        # Compute basic LBP (8-neighbors)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray_image[i, j]
                code = 0

                # Compare with 8 neighbors
                code |= (gray_image[i-1, j-1] >= center) << 0
                code |= (gray_image[i-1, j] >= center) << 1
                code |= (gray_image[i-1, j+1] >= center) << 2
                code |= (gray_image[i, j+1] >= center) << 3
                code |= (gray_image[i+1, j+1] >= center) << 4
                code |= (gray_image[i+1, j] >= center) << 5
                code |= (gray_image[i+1, j-1] >= center) << 6
                code |= (gray_image[i, j-1] >= center) << 7

                lbp[i, j] = code

        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)

        # Compute statistics
        uniformity = float(np.sum(hist ** 2))  # Higher = more uniform texture
        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))  # Higher = more complex texture

        return {
            "uniformity": uniformity,
            "entropy": entropy
        }

    def classify_texture(self, glcm_features: dict, lbp_features: dict) -> dict:
        """
        Classify texture type based on computed features.

        Args:
            glcm_features: GLCM texture features
            lbp_features: LBP texture features

        Returns:
            Dictionary with texture classifications
        """
        # Contrast-based classification
        if glcm_features["contrast"] > 15:
            roughness = "rough"
        elif glcm_features["contrast"] > 5:
            roughness = "moderate"
        else:
            roughness = "smooth"

        # Homogeneity-based classification
        if glcm_features["homogeneity"] > 0.7:
            pattern_type = "uniform"
        elif glcm_features["homogeneity"] > 0.5:
            pattern_type = "regular"
        else:
            pattern_type = "irregular"

        # Energy-based classification
        if glcm_features["energy"] > 0.15:
            texture_complexity = "simple"
        elif glcm_features["energy"] > 0.05:
            texture_complexity = "moderate"
        else:
            texture_complexity = "complex"

        # LBP entropy classification
        if lbp_features["entropy"] > 6:
            detail_level = "high_detail"
        elif lbp_features["entropy"] > 4:
            detail_level = "medium_detail"
        else:
            detail_level = "low_detail"

        return {
            "roughness": roughness,
            "pattern": pattern_type,
            "complexity": texture_complexity,
            "detail_level": detail_level
        }

    def analyze_edge_density(self, gray_image: np.ndarray) -> dict:
        """
        Analyze edge density to understand texture characteristics.

        Args:
            gray_image: Grayscale image as numpy array

        Returns:
            Dictionary with edge statistics
        """
        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 50, 150)

        # Calculate edge density
        edge_density = float(np.sum(edges > 0) / edges.size)

        # Classify edge density
        if edge_density > 0.15:
            edge_level = "high_edges"
        elif edge_density > 0.05:
            edge_level = "medium_edges"
        else:
            edge_level = "low_edges"

        return {
            "edge_density": edge_density,
            "edge_level": edge_level
        }

    def extract_texture_features(self, image: Image.Image) -> dict:
        """
        Extract comprehensive texture features from image.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with all texture features
        """
        # Convert to grayscale for texture analysis
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Compute GLCM features
        glcm_features = self.compute_glcm_features(gray)

        # Compute LBP features
        lbp_features = self.compute_lbp_features(gray)

        # Classify texture
        texture_classification = self.classify_texture(glcm_features, lbp_features)

        # Analyze edges
        edge_features = self.analyze_edge_density(gray)

        # Combine all features
        return {
            "glcm": {
                "contrast": round(glcm_features["contrast"], 3),
                "homogeneity": round(glcm_features["homogeneity"], 3),
                "energy": round(glcm_features["energy"], 3),
                "correlation": round(glcm_features["correlation"], 3)
            },
            "lbp": {
                "uniformity": round(lbp_features["uniformity"], 4),
                "entropy": round(lbp_features["entropy"], 3)
            },
            "classification": texture_classification,
            "edges": {
                "density": round(edge_features["edge_density"], 4),
                "level": edge_features["edge_level"]
            },
            # Simplified tags for easy searching
            "texture_tags": [
                texture_classification["roughness"],
                texture_classification["pattern"],
                texture_classification["complexity"],
                edge_features["edge_level"]
            ]
        }


# Global instance
texture_service = TextureService()
